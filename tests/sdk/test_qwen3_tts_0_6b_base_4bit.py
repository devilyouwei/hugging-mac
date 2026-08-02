from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Event
from types import ModuleType

import numpy as np
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.qwen3_tts_0_6b_base_4bit import (
    QWEN3_TTS_0_6B_BASE_4BIT_MANIFEST,
    register_qwen3_tts_0_6b_base_4bit,
)
from hugging_mac_sdk.models.qwen3_tts_0_6b_base_4bit.config import (
    QWEN3_TTS_MODEL_ID,
    QWEN3_TTS_REQUIRED_FILES,
    QWEN3_TTS_REVISION,
    Qwen3TtsInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_tts_0_6b_base_4bit.instance import Qwen3TtsInstance
from hugging_mac_sdk.models.qwen3_tts_0_6b_base_4bit.mlx import MlxQwen3TtsEngine
from hugging_mac_sdk.models.qwen3_tts_0_6b_base_4bit.resources import (
    Qwen3TtsResourceResolver,
)
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest


class FakeResult:
    audio = np.zeros(24000, dtype=np.float32)
    sample_rate = 24000
    token_count = 25


class FakeModel:
    def generate(self, **kwargs: object):
        assert kwargs == {
            "text": "Hello from Qwen.",
            "voice": None,
            "speed": 1.0,
            "lang_code": "english",
            "ref_audio": None,
            "ref_text": None,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": 512,
            "verbose": False,
        }
        yield FakeResult()


def test_qwen3_tts_manifest_and_registration() -> None:
    manifest = QWEN3_TTS_0_6B_BASE_4BIT_MANIFEST
    assert manifest.model_id == QWEN3_TTS_MODEL_ID
    assert manifest.default_variant == "4bit"
    assert manifest.default_runtime == "mlx"
    assert manifest.revision == QWEN3_TTS_REVISION == "0d6bb6f"
    assert manifest.capabilities == {"speech-synthesis"}

    definition = register_qwen3_tts_0_6b_base_4bit(ModelRegistry())
    instance = definition.create(runtime="mlx", variant="4bit")

    assert isinstance(instance, Qwen3TtsInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


async def test_qwen3_tts_load_explicitly_selects_model_type(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    loaded_model = object()
    utils = ModuleType("mlx_audio.tts.utils")

    def fake_load(path: str, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return loaded_model

    utils.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    engine = MlxQwen3TtsEngine(
        Qwen3TtsInstanceConfig(source_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )

    await engine.load(tmp_path)

    assert engine._model is loaded_model
    assert calls == [(str(tmp_path), {"model_type": "qwen3_tts"})]


def test_qwen3_tts_engine_maps_generation_request(tmp_path: Path) -> None:
    engine = MlxQwen3TtsEngine(
        Qwen3TtsInstanceConfig(source_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )
    engine._model = FakeModel()

    output = engine._infer_sync(
        SpeechSynthesisRequest(
            text="Hello from Qwen.",
            language="english",
            max_new_tokens=512,
        ),
        Event(),
    )

    assert output.sample_rate == 24000
    assert output.duration_seconds == 1.0
    assert output.generated_tokens == 25
    assert len(output.audio) == 24000 * 4


def test_qwen3_tts_resource_validation_accepts_weight_shards(tmp_path: Path) -> None:
    (tmp_path / "speech_tokenizer").mkdir()
    for name in QWEN3_TTS_REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    resolver = Qwen3TtsResourceResolver(
        HuggingFaceSource(repo_id="test/qwen", revision="main"),
        Qwen3TtsInstanceConfig(source_path=tmp_path),
    )

    resolver._validate(tmp_path)
    assert resolver.status().artifacts[0].available
