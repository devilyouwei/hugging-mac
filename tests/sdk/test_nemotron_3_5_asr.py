from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import SpeechTranscription, StreamingSpeechTranscription
from hugging_mac_sdk.models.nemotron_3_5_asr import NEMOTRON_3_5_ASR_DEFINITION
from hugging_mac_sdk.models.nemotron_3_5_asr.config import (
    NemotronCoreMlInstanceConfig,
    variant_source_path,
)
from hugging_mac_sdk.models.nemotron_3_5_asr.coreml import CoreMlNemotronEngine
from hugging_mac_sdk.models.nemotron_3_5_asr.instance import NemotronCoreMlInstance
from hugging_mac_sdk.models.nemotron_3_5_asr.utils.types import (
    AsrEngineOutput,
    PreparedAudio,
    StreamingAsrEngineOutput,
)
from hugging_mac_sdk.schemas.streaming_transcription import StreamingTranscriptionRequest
from hugging_mac_sdk.schemas.transcription import AudioInput, TranscriptionRequest


def test_manifest_declares_all_upstream_bundles() -> None:
    definition = NEMOTRON_3_5_ASR_DEFINITION
    assert definition.manifest.model_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert "streaming-speech-transcription" in definition.manifest.capabilities
    assert definition.manifest.default_variant == "multilingual-2240ms"
    assert {item.name for item in definition.manifest.variants} == {
        f"{script}-{tier}ms"
        for script in ("latin", "multilingual")
        for tier in (560, 1120, 2240, 4480)
    }
    assert set(definition.runtime_factories) == {"coreml"}
    assert len(definition.artifacts) == 8


def test_variant_source_path() -> None:
    assert variant_source_path("latin-2240ms") == "latin/2240ms"


@pytest.mark.asyncio
async def test_instance_exposes_transcription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = AsyncMock()
    engine.runtime_name = "coreml"
    engine.device = "cpu-and-neural-engine"
    engine.infer.return_value = AsrEngineOutput(" hello ", 3, "en-US")
    config = NemotronCoreMlInstanceConfig(artifact_path=tmp_path)
    instance = NemotronCoreMlInstance(config, engine)
    assert instance.supports(SpeechTranscription)
    engine.resolve.return_value = tmp_path
    monkeypatch.setattr(
        "hugging_mac_sdk.models.nemotron_3_5_asr.instance.prepare_audio",
        lambda *_args, **_kwargs: PreparedAudio(np.zeros(1600, dtype=np.float32), 16000, 0.1),
    )
    await instance.load()
    response = await instance.require(SpeechTranscription).transcribe(  # type: ignore[type-abstract]
        TranscriptionRequest(audio=AudioInput(data=b"audio"))
    )
    assert response.text == "hello"
    assert response.generated_tokens == 3
    assert response.model_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    await instance.unload()


@pytest.mark.asyncio
async def test_instance_preserves_a_streaming_session_across_audio_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = AsyncMock()
    engine.runtime_name = "coreml"
    engine.device = "cpu-and-neural-engine"
    engine.resolve.return_value = tmp_path
    engine.infer_stream.side_effect = (
        StreamingAsrEngineOutput("hello", "hello", 1, "en-US"),
        StreamingAsrEngineOutput("hello world", " world", 2, "en-US"),
    )
    engine.finish_stream.return_value = StreamingAsrEngineOutput(
        "hello world", "", 2, "en-US"
    )
    instance = NemotronCoreMlInstance(
        NemotronCoreMlInstanceConfig(artifact_path=tmp_path), engine
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.nemotron_3_5_asr.instance.prepare_audio",
        lambda *_args, **_kwargs: PreparedAudio(
            np.zeros(1600, dtype=np.float32), 16000, 0.1
        ),
    )
    await instance.load()

    assert instance.supports(StreamingSpeechTranscription)
    streamer = instance.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
    session = await streamer.start_stream()
    first = await streamer.transcribe_stream(
        StreamingTranscriptionRequest(
            session_id=session.session_id,
            audio=AudioInput(data=b"first"),
        )
    )
    second = await streamer.transcribe_stream(
        StreamingTranscriptionRequest(
            session_id=session.session_id,
            audio=AudioInput(data=b"second"),
        )
    )
    final = await streamer.finish_stream(session.session_id)

    engine.start_stream.assert_awaited_once_with(session.session_id)
    assert first.delta == "hello"
    assert second.text == "hello world"
    assert second.delta == " world"
    assert second.audio_seconds == pytest.approx(0.2)
    assert final.is_final
    await instance.unload()


@pytest.mark.asyncio
async def test_coreml_engine_buffers_partial_chunks_and_keeps_decoder_state(
    tmp_path: Path,
) -> None:
    audio_lengths: list[int] = []

    class FakeSession:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.decoder_calls = 0

        def run(self, inputs: dict[str, object]) -> dict[str, np.ndarray]:
            if self.kind == "preprocessor":
                audio_lengths.append(int(np.asarray(inputs["audio_length"])[0]))
                return {"mel": np.zeros((1, 1, 1), dtype=np.float32)}
            if self.kind == "encoder":
                return {
                    "encoded": np.zeros((1, 1, 1), dtype=np.float32),
                    "cache_channel_out": np.asarray(inputs["cache_channel"]),
                    "cache_time_out": np.asarray(inputs["cache_time"]),
                    "cache_len_out": np.asarray(inputs["cache_len"]),
                }
            self.decoder_calls += 1
            predicted = 1 if self.decoder_calls % 2 else 0
            logits = np.zeros((2,), dtype=np.float32)
            logits[predicted] = 1
            return {
                "logits": logits,
                "h_out": np.asarray(inputs["h_in"]),
                "c_out": np.asarray(inputs["c_in"]),
            }

        async def close(self) -> None:
            return None

    engine = CoreMlNemotronEngine(
        NemotronCoreMlInstanceConfig(artifact_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )
    engine._metadata = {
        "chunk_mel_frames": 1,
        "pre_encode_cache": 1,
        "cache_channel_shape": [1],
        "cache_time_shape": [1],
        "mel_features": 1,
        "decoder_layers": 1,
        "decoder_hidden": 1,
        "blank_idx": 0,
        "lang_tag_token_ids": [],
    }
    engine._vocab = {1: "▁hello"}
    engine._sessions = cast(
        Any,
        {
            "preprocessor": FakeSession("preprocessor"),
            "encoder": FakeSession("encoder"),
            "decoder_joint": FakeSession("decoder_joint"),
        },
    )
    await engine.start_stream("stream")

    first = await engine.infer_stream(
        "stream", PreparedAudio(np.zeros(80, dtype=np.float32), 16000, 0.005)
    )
    second = await engine.infer_stream(
        "stream", PreparedAudio(np.zeros(80, dtype=np.float32), 16000, 0.005)
    )
    final = await engine.finish_stream("stream")

    assert first.text == ""
    assert second.text == "hello"
    assert second.delta == "hello"
    assert final.text == "hello"
    assert final.delta == ""
    assert audio_lengths == [160]

    await engine.start_stream("short")
    partial = await engine.infer_stream(
        "short", PreparedAudio(np.zeros(80, dtype=np.float32), 16000, 0.005)
    )
    await engine.finish_stream("short")

    assert partial.text == ""
    assert audio_lengths[-1] == 80
    await engine.close()
