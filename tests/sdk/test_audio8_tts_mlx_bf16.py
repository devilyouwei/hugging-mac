from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.audio8_tts_mlx_bf16 import (
    AUDIO8_TTS_MLX_BF16_MANIFEST,
    register_audio8_tts_mlx_bf16,
)
from hugging_mac_sdk.models.audio8_tts_mlx_bf16.config import (
    AUDIO8_TTS_MLX_BF16_REQUIRED_FILES,
    Audio8TtsMlxBf16InstanceConfig,
)
from hugging_mac_sdk.models.audio8_tts_mlx_bf16.instance import Audio8TtsMlxBf16Instance
from hugging_mac_sdk.models.audio8_tts_mlx_bf16.resources import (
    Audio8TtsMlxBf16ResourceResolver,
)
from hugging_mac_sdk.models.audio8_tts_mlx_bf16.utils.types import TtsEngineOutput
from hugging_mac_sdk.models.audio8_tts_mlx_bf16.utils.voices import VoiceStore
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput


def _source() -> HuggingFaceSource:
    source = AUDIO8_TTS_MLX_BF16_MANIFEST.get_variant().resources[0]
    assert isinstance(source, HuggingFaceSource)
    return source


def _write_snapshot(path: Path) -> None:
    for name in AUDIO8_TTS_MLX_BF16_REQUIRED_FILES:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def test_manifest_is_pinned_to_mlx_bf16_snapshot() -> None:
    assert AUDIO8_TTS_MLX_BF16_MANIFEST.default_variant == "bf16"
    assert AUDIO8_TTS_MLX_BF16_MANIFEST.default_runtime == "mlx"
    assert AUDIO8_TTS_MLX_BF16_MANIFEST.capabilities == {"speech-synthesis"}
    source = _source()
    assert source.revision == "f7be312aaaed724b6ecb8e916b21c9fd0842db02"
    assert "codec.safetensors" in source.allow_patterns


def test_registered_factory_exposes_gpu_speech_synthesis() -> None:
    registry = ModelRegistry()
    definition = register_audio8_tts_mlx_bf16(registry)
    instance = definition.create(runtime="mlx", variant="bf16")

    assert isinstance(instance, Audio8TtsMlxBf16Instance)
    assert instance.supports(SpeechSynthesis)
    assert instance.info().device == "gpu"


async def test_resource_resolver_accepts_mlx_layout(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    _write_snapshot(artifact)
    resolver = Audio8TtsMlxBf16ResourceResolver(
        _source(), Audio8TtsMlxBf16InstanceConfig(source_path=artifact)
    )

    resolved = await resolver.resolve_source()

    assert resolved.path == artifact
    assert resolver.status().artifacts[0].available


def test_voice_profiles_store_reference_audio_and_text(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path)
    audio = b"RIFF-test-wave"

    path, text = store.save("speaker_a", AudioInput(data=audio), " reference   transcript ")
    restored_path, restored_text = store.load("speaker_a")

    assert path.read_bytes() == audio
    assert restored_path == path
    assert text == restored_text == "reference transcript"


async def test_instance_preserves_actionable_inference_reason(tmp_path: Path) -> None:
    class FailingEngine:
        runtime_name = "mlx"
        device = "gpu"

        async def resolve(self) -> Path:
            return tmp_path

        async def load(self, artifact: Path) -> None:
            assert artifact == tmp_path

        async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
            raise ValueError("reference audio could not be decoded")

        async def close(self) -> None:
            return None

    instance = Audio8TtsMlxBf16Instance(
        Audio8TtsMlxBf16InstanceConfig(source_path=tmp_path), FailingEngine()
    )
    await instance.load()

    with pytest.raises(InferenceError) as caught:
        await instance.synthesize(SpeechSynthesisRequest(text="hello", voice="speaker_a"))

    assert caught.value.details["reason"] == "reference audio could not be decoded"
