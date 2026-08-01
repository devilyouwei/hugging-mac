from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.audio8_tts_onnx_int4 import (
    AUDIO8_TTS_ONNX_INT4_MANIFEST,
    register_audio8_tts_onnx_int4,
)
from hugging_mac_sdk.models.audio8_tts_onnx_int4.config import (
    AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
    AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES,
    AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES,
    Audio8TtsOnnxInt4InstanceConfig,
)
from hugging_mac_sdk.models.audio8_tts_onnx_int4.instance import (
    Audio8TtsOnnxInt4Instance,
)
from hugging_mac_sdk.models.audio8_tts_onnx_int4.resources import (
    Audio8TtsOnnxInt4ResourceResolver,
)
from hugging_mac_sdk.models.audio8_tts_onnx_int4.utils.runtime import _sample
from hugging_mac_sdk.models.audio8_tts_onnx_int4.utils.types import TtsEngineOutput
from hugging_mac_sdk.models.audio8_tts_onnx_int4.utils.voices import VoiceStore
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest


def _source() -> HuggingFaceSource:
    source = AUDIO8_TTS_ONNX_INT4_MANIFEST.get_variant().resources[0]
    assert isinstance(source, HuggingFaceSource)
    return source


def _write_snapshot(path: Path, *, registration: bool = True) -> None:
    for name in AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    (path / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "model_family": "audio8_tts",
                "default_precision": "int4",
                "codec_sample_rate": 44100,
                "num_codebooks": 10,
                "model_fingerprint": AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
            }
        )
    )
    if registration:
        for name in AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        (path / "registration/registration_manifest.json").write_text(
            json.dumps({"model_fingerprint": AUDIO8_TTS_ONNX_INT4_FINGERPRINT})
        )


def test_manifest_is_pinned_to_official_onnx_int4_snapshot() -> None:
    assert AUDIO8_TTS_ONNX_INT4_MANIFEST.default_variant == "int4"
    assert AUDIO8_TTS_ONNX_INT4_MANIFEST.default_runtime == "onnx"
    assert AUDIO8_TTS_ONNX_INT4_MANIFEST.capabilities == {"speech-synthesis"}
    source = _source()
    assert source.revision == "7af3196fed72de44708f5d095d42cbf8085f6123"
    assert "registration/*.onnx" in source.allow_patterns


def test_registered_factory_exposes_speech_synthesis() -> None:
    registry = ModelRegistry()
    definition = register_audio8_tts_onnx_int4(registry)
    instance = definition.create(runtime="onnx", variant="int4")

    assert isinstance(instance, Audio8TtsOnnxInt4Instance)
    assert instance.supports(SpeechSynthesis)
    assert instance.info().device == "cpu"


async def test_resource_resolver_accepts_official_layout(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    _write_snapshot(artifact)
    config = Audio8TtsOnnxInt4InstanceConfig(source_path=artifact)
    resolver = Audio8TtsOnnxInt4ResourceResolver(_source(), config)

    resolved = await resolver.resolve_source()

    assert resolved.path == artifact
    assert resolver.status().artifacts[0].available


async def test_registration_files_must_be_complete(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    _write_snapshot(artifact, registration=False)
    partial = artifact / AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES[0]
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.touch()
    resolver = Audio8TtsOnnxInt4ResourceResolver(
        _source(), Audio8TtsOnnxInt4InstanceConfig(source_path=artifact)
    )

    with pytest.raises(Exception, match="registration files are incomplete"):
        await resolver.resolve_source()


def test_voice_profiles_round_trip_and_check_fingerprint(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path, 10, AUDIO8_TTS_ONNX_INT4_FINGERPRINT)
    codes = np.arange(30, dtype=np.int64).reshape(10, 3)

    store.save("speaker_a", codes, " reference   transcript ")
    restored, text = store.load("speaker_a")

    np.testing.assert_array_equal(restored, codes)
    assert text == "reference transcript"


def test_greedy_sampling_does_not_consume_rng() -> None:
    rng = np.random.default_rng(42)
    logits = np.asarray([0.1, 2.0, 1.0])

    assert _sample(logits, 0.7, 0.9, 2, rng, do_sample=False) == 1


async def test_instance_preserves_actionable_inference_reason(tmp_path: Path) -> None:
    class FailingEngine:
        runtime_name = "onnx"
        device = "cpu"

        async def resolve(self) -> Path:
            return tmp_path

        async def load(self, artifact: Path) -> None:
            assert artifact == tmp_path

        async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
            raise ValueError("reference audio duration must be between 0.5 and 30 seconds")

        async def close(self) -> None:
            return None

    instance = Audio8TtsOnnxInt4Instance(
        Audio8TtsOnnxInt4InstanceConfig(source_path=tmp_path),
        FailingEngine(),
    )
    await instance.load()

    with pytest.raises(InferenceError) as caught:
        await instance.synthesize(SpeechSynthesisRequest(text="hello", voice="speaker_a"))

    assert caught.value.details["reason"] == (
        "reference audio duration must be between 0.5 and 30 seconds"
    )
