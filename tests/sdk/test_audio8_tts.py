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
from hugging_mac_sdk.models.audio8_tts.torch import (
    TorchAudio8TtsEngine,
    _restore_rope_buffers,
)
from hugging_mac_sdk.models.audio8_tts.utils.types import TtsEngineOutput
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from pydantic import ValidationError


class FakeTtsEngine:
    runtime_name = "pytorch"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    @property
    def device(self) -> str:
        return "cpu"

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
    runtime = AUDIO8_TTS_MANIFEST.runtimes[0]
    assert runtime.name == "pytorch"
    assert runtime.devices == ("cpu",)
    assert runtime.dtypes == ("float32",)
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
    instance = definition.create(runtime="pytorch", variant="preview")

    assert isinstance(instance, Audio8TtsInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


def test_synthesis_request_requires_complete_reference_pair() -> None:
    with pytest.raises(ValueError, match="provided together"):
        SpeechSynthesisRequest(text="hello", reference_text="reference")


def test_audio8_tts_rejects_mps() -> None:
    with pytest.raises(ValidationError, match="device"):
        Audio8TtsInstanceConfig(device="mps")  # type: ignore[arg-type]


def test_audio8_tts_uses_cpu_by_default() -> None:
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(),
        resources=object(),  # type: ignore[arg-type]
    )

    assert engine.device == "cpu"

    class FakeTorch:
        float32 = "fp32"

    assert engine._resolve_dtype(FakeTorch) == "fp32"


def test_audio8_tts_restores_transformers_v5_rope_buffers() -> None:
    torch = pytest.importorskip("torch")

    class FakeConfig:
        max_seq_len = 32
        head_dim = 8
        rope_base = 1_000_000.0
        num_codebooks = 10
        fast_head_dim = 8

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = FakeConfig()
            self.register_buffer("freqs_cis", torch.full((32, 4, 2), torch.nan))
            self.register_buffer("fast_freqs_cis", torch.full((10, 4, 2), torch.nan))

    model = FakeModel()
    _restore_rope_buffers(torch, model)

    assert model.freqs_cis.shape == (32, 4, 2)
    assert model.fast_freqs_cis.shape == (10, 4, 2)
    assert model.freqs_cis.dtype == torch.bfloat16
    assert torch.isfinite(model.freqs_cis).all()
    assert model.freqs_cis[0, :, 0].tolist() == [1.0] * 4
    assert model.freqs_cis[0, :, 1].tolist() == [0.0] * 4


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
