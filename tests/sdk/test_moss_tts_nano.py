from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.moss_tts_nano import (
    MOSS_TTS_NANO_DEFINITION,
    MOSS_TTS_NANO_MANIFEST,
    register_moss_tts_nano,
)
from hugging_mac_sdk.models.moss_tts_nano.config import MossTtsNanoInstanceConfig
from hugging_mac_sdk.models.moss_tts_nano.instance import MossTtsNanoInstance
from hugging_mac_sdk.models.moss_tts_nano.mlx import MlxMossTtsNanoEngine
from hugging_mac_sdk.models.moss_tts_nano.resources import MossTtsNanoResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

MODEL_ARTIFACT = MOSS_TTS_NANO_DEFINITION.get_artifact("mlx", "mlx-fp16", "nano-100m")
TOKENIZER_ARTIFACT = MOSS_TTS_NANO_DEFINITION.shared_artifacts[0]


def test_moss_tts_nano_manifest_and_registration() -> None:
    manifest = MOSS_TTS_NANO_MANIFEST
    assert manifest.model_id == "openmoss-team/moss-tts-nano-100m"
    assert manifest.display_name == "MOSS-TTS-Nano"
    assert manifest.default_variant == "nano-100m"
    assert manifest.default_runtime == "mlx"
    assert manifest.capabilities == {"speech-synthesis"}
    assert MODEL_ARTIFACT.required_shares == ("audio-tokenizer",)
    assert isinstance(MODEL_ARTIFACT.source, HuggingFaceSource)
    assert MODEL_ARTIFACT.source.repo_id == "mlx-community/MOSS-TTS-Nano-100M"
    assert isinstance(TOKENIZER_ARTIFACT.source, HuggingFaceSource)
    assert TOKENIZER_ARTIFACT.source.repo_id == "mlx-community/MOSS-Audio-Tokenizer-Nano"

    definition = register_moss_tts_nano(ModelRegistry())
    instance = definition.create(runtime="mlx", variant="nano-100m")
    assert isinstance(instance, MossTtsNanoInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


def test_moss_tts_nano_maps_generation_and_downmixes_stereo(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        def generate(self, **kwargs: object):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.ones((48_000, 2), dtype=np.float32),
                sample_rate=48_000,
                token_count=125,
            )

    resources = SimpleNamespace(audio_tokenizer_path=tmp_path / "codec")
    engine = MlxMossTtsNanoEngine(
        MossTtsNanoInstanceConfig(source_path=tmp_path),
        resources,  # type: ignore[arg-type]
    )
    engine._model = FakeModel()
    reference = tmp_path / "speaker.wav"
    reference.touch()
    output = engine._infer_sync(
        SpeechSynthesisRequest(
            text="Hello, MOSS.",
            reference_audio=AudioInput(path=reference),
            max_new_tokens=375,
        ),
        Event(),
    )

    assert calls == [
        {
            "text": "Hello, MOSS.",
            "mode": "voice_clone",
            "ref_audio": str(reference),
            "max_tokens": 375,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "audio_tokenizer_source": str(tmp_path / "codec"),
            "audio_tokenizer_device": "cpu",
        }
    ]
    assert output.sample_rate == 48_000
    assert output.duration_seconds == 1.0
    assert output.generated_tokens == 125
    assert len(output.audio) == 48_000 * 4


def test_moss_tts_nano_omits_reference_for_direct_generation(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        def generate(self, **kwargs: object):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.ones((24_000, 2), dtype=np.float32),
                sample_rate=48_000,
                token_count=40,
            )

    resources = SimpleNamespace(audio_tokenizer_path=tmp_path / "codec")
    engine = MlxMossTtsNanoEngine(
        MossTtsNanoInstanceConfig(source_path=tmp_path),
        resources,  # type: ignore[arg-type]
    )
    engine._model = FakeModel()
    output = engine._infer_sync(
        SpeechSynthesisRequest(text="Hello without a reference.", max_new_tokens=40),
        Event(),
    )

    assert calls == [
        {
            "text": "Hello without a reference.",
            "mode": "continuation",
            "max_tokens": 40,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "audio_tokenizer_source": str(tmp_path / "codec"),
            "audio_tokenizer_device": "cpu",
        }
    ]
    assert "ref_audio" not in calls[0]
    assert output.duration_seconds == 0.5


async def test_moss_tts_nano_resource_validation_and_status(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    tokenizer_path = tmp_path / "tokenizer"
    for root, artifact, model_type in (
        (model_path, MODEL_ARTIFACT, "moss_tts_nano"),
        (tokenizer_path, TOKENIZER_ARTIFACT, "moss-audio-tokenizer"),
    ):
        root.mkdir()
        for name in artifact.required_files:
            (root / name).touch()
        (root / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    assert isinstance(MODEL_ARTIFACT.source, HuggingFaceSource)
    assert isinstance(TOKENIZER_ARTIFACT.source, HuggingFaceSource)
    resolver = MossTtsNanoResourceResolver(
        MODEL_ARTIFACT.source,
        TOKENIZER_ARTIFACT.source,
        MossTtsNanoInstanceConfig(source_path=model_path, audio_tokenizer_path=tokenizer_path),
        MOSS_TTS_NANO_MANIFEST,
        MODEL_ARTIFACT,
        TOKENIZER_ARTIFACT,
    )

    assert (await resolver.resolve_source()).path == model_path
    assert (await resolver.resolve_audio_tokenizer()).path == tokenizer_path
    status = resolver.status()
    assert all(item.available for item in status.artifacts)
    assert status.runtimes[0].available
