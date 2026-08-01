from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.audio8_tts import (
    AUDIO8_TTS_MANIFEST,
    register_audio8_tts,
)
from hugging_mac_sdk.models.audio8_tts.config import Audio8TtsInstanceConfig
from hugging_mac_sdk.models.audio8_tts.instance import Audio8TtsInstance
from hugging_mac_sdk.models.audio8_tts.torch import TorchAudio8TtsEngine
from hugging_mac_sdk.models.audio8_tts.utils.types import TtsEngineOutput
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest


class FakeTtsEngine:
    runtime_name = "pytorch-mps"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    @property
    def device(self) -> str:
        return "mps"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        assert request.text == "你好, Audio8."
        return TtsEngineOutput(
            audio=b"\x00\x00\x00\x00" * 44100,
            sample_rate=44100,
            duration_seconds=1.0,
            generated_tokens=22,
        )

    async def close(self) -> None:
        self.closed = True


def test_audio8_tts_manifest_is_pinned() -> None:
    assert AUDIO8_TTS_MANIFEST.default_variant == "preview"
    assert AUDIO8_TTS_MANIFEST.capabilities == {"speech-synthesis"}
    assert AUDIO8_TTS_MANIFEST.license == "Apache-2.0"
    source = AUDIO8_TTS_MANIFEST.get_variant().resources[0]
    assert source.revision == "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17"  # type: ignore[union-attr]
    assert source.allow_patterns == (  # type: ignore[union-attr]
        "*.json",
        "*.py",
        "*.pth",
        "*.safetensors",
    )


def test_registered_factory_exposes_speech_synthesis() -> None:
    registry = ModelRegistry()
    definition = register_audio8_tts(registry)
    instance = definition.create(runtime="pytorch-mps", variant="preview")

    assert isinstance(instance, Audio8TtsInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


def test_synthesis_request_requires_complete_reference_pair() -> None:
    with pytest.raises(ValueError, match="provided together"):
        SpeechSynthesisRequest(text="hello", reference_text="reference")


def test_audio8_tts_uses_checkpoint_precision_on_mps() -> None:
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(),
        resources=object(),  # type: ignore[arg-type]
    )
    engine._device = "mps"

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

    assert engine._resolve_dtype(FakeTorch) == "bf16"


def test_audio8_tts_uses_cpu_by_default() -> None:
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(),
        resources=object(),  # type: ignore[arg-type]
    )

    assert engine.device == "cpu"


async def test_audio8_tts_instance_orchestrates_synthesis(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeTtsEngine(artifact)
    instance = Audio8TtsInstance(Audio8TtsInstanceConfig(), engine)

    await instance.load()
    synthesizer = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
    response = await synthesizer.synthesize(
        SpeechSynthesisRequest(text="你好, Audio8.")
    )

    assert instance.state is ModelState.READY
    assert engine.loaded
    assert response.sample_rate == 44100
    assert response.duration_seconds == 1.0
    assert response.generated_tokens == 22
    assert response.audio_format == "f32le"

    await instance.unload()
    assert engine.closed
