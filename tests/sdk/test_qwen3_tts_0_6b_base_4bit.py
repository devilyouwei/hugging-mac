from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Event
from types import ModuleType

import numpy as np
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.qwen3_tts import (
    QWEN3_TTS_DEFINITION,
    QWEN3_TTS_MANIFEST,
    register_qwen3_tts,
)
from hugging_mac_sdk.models.qwen3_tts.config import (
    QWEN3_TTS_COREML_REQUIRED_FILES,
    QWEN3_TTS_MODEL_ID,
    QWEN3_TTS_REQUIRED_FILES,
    QWEN3_TTS_REVISION,
    QWEN3_TTS_TOKENIZER_REQUIRED_FILES,
    Qwen3TtsInstanceConfig,
)
from hugging_mac_sdk.models.qwen3_tts.coreml import CoreMlQwen3TtsEngine
from hugging_mac_sdk.models.qwen3_tts.instance import Qwen3TtsInstance
from hugging_mac_sdk.models.qwen3_tts.mlx import MlxQwen3TtsEngine
from hugging_mac_sdk.models.qwen3_tts.resources import (
    Qwen3TtsCoreMlResourceResolver,
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
    manifest = QWEN3_TTS_MANIFEST
    assert manifest.model_id == QWEN3_TTS_MODEL_ID
    assert manifest.default_variant == "0.6b-base"
    assert manifest.default_runtime == "mlx"
    assert {runtime.name for runtime in manifest.runtimes} == {"mlx", "coreml"}
    assert manifest.revision == QWEN3_TTS_REVISION == "0d6bb6f"
    assert manifest.capabilities == {"speech-synthesis"}

    definition = register_qwen3_tts(ModelRegistry())
    instance = definition.create(runtime="mlx", variant="0.6b-base")

    assert isinstance(instance, Qwen3TtsInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]

    coreml_instance = definition.create(runtime="coreml", variant="0.6b-base")
    assert isinstance(coreml_instance, Qwen3TtsInstance)
    assert isinstance(coreml_instance._engine, CoreMlQwen3TtsEngine)
    assert coreml_instance.info().device == "cpu-and-neural-engine"


async def test_qwen3_tts_load_explicitly_selects_model_type(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    loaded_model = object()
    utils = ModuleType("mlx_audio.tts.utils")

    def fake_load(path: str, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return loaded_model

    utils.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    tokenizer_path = tmp_path / "tokenizers"
    tokenizer_path.mkdir()

    class Resources:
        pass

    resources = Resources()
    resources.tokenizer_path = tokenizer_path
    engine = MlxQwen3TtsEngine(
        Qwen3TtsInstanceConfig(source_path=tmp_path),
        resources,  # type: ignore[arg-type]
    )

    await engine.load(tmp_path)

    assert engine._model is loaded_model
    assert len(calls) == 1
    assert Path(calls[0][0]).name.startswith("hugging-mac-model-view-")
    assert calls[0][1] == {"model_type": "qwen3_tts"}


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


async def test_qwen3_tts_coreml_resource_validation(tmp_path: Path) -> None:
    for name in QWEN3_TTS_COREML_REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    tokenizer_path = tmp_path / "tokenizers"
    for name in QWEN3_TTS_TOKENIZER_REQUIRED_FILES:
        path = tokenizer_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    source = HuggingFaceSource(repo_id="test/qwen-coreml", revision="pinned")
    resolver = Qwen3TtsCoreMlResourceResolver(
        source,
        source,
        Qwen3TtsInstanceConfig(
            runtime="coreml",
            coreml_path=tmp_path,
            tokenizer_path=tokenizer_path,
            device="cpu-and-neural-engine",
        ),
    )

    resolved = await resolver.resolve_source()
    assert resolved.path == tmp_path
    assert resolver.status().artifacts[0].available


async def test_qwen3_tts_resource_status_exposes_both_runtimes(tmp_path: Path) -> None:
    provider = QWEN3_TTS_DEFINITION.resource_provider
    assert provider is not None
    status = await provider.status("0.6b-base", {"model_home": tmp_path})

    assert {runtime.runtime for runtime in status.runtimes} == {"mlx", "coreml"}
    assert {artifact.artifact_id for artifact in status.artifacts} == {
        "mlx-4bit",
        "coreml-w8a16",
        "tokenizers",
    }
