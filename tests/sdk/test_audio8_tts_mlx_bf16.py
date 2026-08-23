from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.audio8_tts import (
    AUDIO8_TTS_DEFINITION,
    AUDIO8_TTS_MANIFEST,
    register_audio8_tts,
)
from hugging_mac_sdk.models.audio8_tts.config import Audio8TtsMlxInstanceConfig
from hugging_mac_sdk.models.audio8_tts.instance import Audio8TtsInstance
from hugging_mac_sdk.models.audio8_tts.mlx_resources import Audio8TtsMlxResourceResolver
from hugging_mac_sdk.models.audio8_tts.utils.types import TtsEngineOutput
from hugging_mac_sdk.models.audio8_tts.utils.voices import VoiceStore
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput


def _source() -> HuggingFaceSource:
    source = AUDIO8_TTS_DEFINITION.get_artifact("mlx", "mlx-bf16", "0.6b-preview").source
    assert isinstance(source, HuggingFaceSource)
    return source


def _write_snapshot(path: Path) -> None:
    for name in _source().allow_patterns:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def test_manifest_contains_mlx_bf16_artifact_source() -> None:
    assert AUDIO8_TTS_MANIFEST.default_variant == "0.1b-preview"
    assert Audio8TtsMlxInstanceConfig().variant == "0.6b-preview"
    assert "mlx" in {runtime.name for runtime in AUDIO8_TTS_MANIFEST.runtimes}
    assert AUDIO8_TTS_MANIFEST.capabilities == {"speech-synthesis"}
    source = _source()
    assert source.revision == "main"
    assert source.allow_patterns == (
        "config.json",
        "generation_config.json",
        "codec.safetensors",
        "model.safetensors",
    )


def test_registered_factory_exposes_gpu_speech_synthesis() -> None:
    registry = ModelRegistry()
    definition = register_audio8_tts(registry)
    instance = definition.create(runtime="mlx", variant="0.6b-preview")

    assert isinstance(instance, Audio8TtsInstance)
    assert instance.supports(SpeechSynthesis)
    assert instance.info().device == "gpu"


async def test_resource_resolver_accepts_mlx_layout(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    _write_snapshot(artifact)
    resolver = Audio8TtsMlxResourceResolver(
        _source(),
        Audio8TtsMlxInstanceConfig(source_path=artifact),
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=next(
            artifact
            for artifact in register_audio8_tts(ModelRegistry()).artifacts
            if artifact.artifact_id == "mlx-bf16"
        ),
        tokenizer_artifact=register_audio8_tts(ModelRegistry()).shared_artifacts[0],
    )

    resolved = await resolver.resolve_source()

    assert resolved.path == artifact
    assert resolver.status().artifacts[0].available


async def test_mlx_engine_keeps_merged_model_view_until_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hugging_mac_sdk.models.audio8_tts.mlx import MlxAudio8TtsEngine

    loaded_paths: list[Path] = []
    loaded_model = object()
    utils = ModuleType("mlx_audio.tts.utils")
    mlx_core = ModuleType("mlx.core")

    def fake_load(path: str) -> object:
        loaded_paths.append(Path(path))
        return loaded_model

    utils.load = fake_load  # type: ignore[attr-defined]
    mlx_core.clear_cache = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()

    class Resources:
        pass

    resources = Resources()
    resources.tokenizer_path = tokenizer_path
    engine = MlxAudio8TtsEngine(
        Audio8TtsMlxInstanceConfig(source_path=tmp_path),
        resources,  # type: ignore[arg-type]
    )

    await engine.load(tmp_path)

    assert engine._model is loaded_model
    assert loaded_paths[0].is_dir()

    await engine.close()

    assert not loaded_paths[0].exists()


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

    instance = Audio8TtsInstance(
        Audio8TtsMlxInstanceConfig(source_path=tmp_path),
        FailingEngine(),
        AUDIO8_TTS_MANIFEST,
    )
    await instance.load()

    with pytest.raises(InferenceError) as caught:
        await instance.synthesize(SpeechSynthesisRequest(text="hello", voice="speaker_a"))

    assert caught.value.details["reason"] == "reference audio could not be decoded"
