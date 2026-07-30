from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import SpeechTranscription
from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.models.audio8_asr import (
    AUDIO8_ASR_MANIFEST,
    register_audio8_asr,
)
from hugging_mac_sdk.models.audio8_asr.config import Audio8AsrInstanceConfig
from hugging_mac_sdk.models.audio8_asr.instance import Audio8AsrInstance
from hugging_mac_sdk.models.audio8_asr.resources import Audio8AsrResourceProvider
from hugging_mac_sdk.models.audio8_asr.utils.types import (
    AsrEngineOutput,
    PreparedAudio,
)
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.transcription import AudioInput, TranscriptionRequest


class FakeAsrEngine:
    runtime_name = "pytorch-mps"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False
        self.requests: list[TranscriptionRequest] = []

    @property
    def device(self) -> str:
        return "mps"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(
        self,
        prepared: PreparedAudio,
        request: TranscriptionRequest,
    ) -> AsrEngineOutput:
        assert prepared.sample_rate == 16000
        self.requests.append(request)
        return AsrEngineOutput(
            text="  hello from audio8  ",
            prompt_tokens=42,
            generated_tokens=5,
        )

    async def close(self) -> None:
        self.closed = True


def test_audio8_manifest_downloads_only_weights_and_json() -> None:
    assert AUDIO8_ASR_MANIFEST.default_variant == "base"
    assert AUDIO8_ASR_MANIFEST.capabilities == {"speech-transcription"}
    assert AUDIO8_ASR_MANIFEST.license == "CC-BY-NC-4.0"
    assert {runtime.name for runtime in AUDIO8_ASR_MANIFEST.runtimes} == {
        "pytorch-mps",
        "coreml",
    }
    source = AUDIO8_ASR_MANIFEST.get_variant().resources[0]
    assert source.allow_patterns == ("model.safetensors", "*.json")  # type: ignore[union-attr]


def test_registered_factory_exposes_speech_transcription() -> None:
    registry = ModelRegistry()
    converters = ConverterRegistry()
    definition = register_audio8_asr(registry, converters)

    pytorch = definition.create(runtime="pytorch-mps", variant="base")
    coreml = definition.create(
        runtime="coreml",
        variant="base",
        options={"device": "cpu-and-neural-engine"},
    )

    assert isinstance(pytorch, Audio8AsrInstance)
    assert isinstance(coreml, Audio8AsrInstance)
    assert pytorch.supports(SpeechTranscription)  # type: ignore[type-abstract]
    assert coreml.supports(SpeechTranscription)  # type: ignore[type-abstract]
    assert pytorch.info().variant == "base"
    assert pytorch.info().runtime == "pytorch-mps"
    assert coreml.info().runtime == "coreml"
    assert coreml.info().device == "cpu-and-neural-engine"
    assert converters.get("audio8.audio8-asr-0.1b").converter_id == (
        "audio8.audio8-asr-0.1b"
    )


async def test_audio8_instance_orchestrates_transcription(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeAsrEngine(artifact)
    config = Audio8AsrInstanceConfig()
    prepared = PreparedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.audio8_asr.instance.prepare_audio",
        lambda *_args, **_kwargs: prepared,
    )
    instance = Audio8AsrInstance(config, engine)

    await instance.load()
    transcriber = instance.require(SpeechTranscription)  # type: ignore[type-abstract]
    response = await transcriber.transcribe(
        TranscriptionRequest(audio=AudioInput(data=b"encoded audio"))
    )

    assert instance.state is ModelState.READY
    assert engine.loaded
    assert response.text == "hello from audio8"
    assert response.runtime == "pytorch-mps"
    assert response.device == "mps"
    assert response.duration_seconds == 1.0
    assert response.prompt_tokens == 42
    assert response.generated_tokens == 5
    assert response.timings.inference_ms is not None

    await instance.unload()
    assert engine.closed
    assert instance.state is ModelState.UNLOADED


async def test_audio8_resource_provider_rejects_unsupported_conversion() -> None:
    source = AUDIO8_ASR_MANIFEST.get_variant().resources[0]
    provider = Audio8AsrResourceProvider(source)  # type: ignore[arg-type]

    with pytest.raises(UnsupportedRuntimeError, match="not implemented"):
        await provider.convert("base", ArtifactFormat.ONNX)
