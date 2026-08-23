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
from hugging_mac_sdk.models.qwen3_tts.config import Qwen3TtsInstanceConfig
from hugging_mac_sdk.models.qwen3_tts.coreml import CoreMlQwen3TtsEngine
from hugging_mac_sdk.models.qwen3_tts.instance import Qwen3TtsInstance
from hugging_mac_sdk.models.qwen3_tts.mlx import MlxQwen3TtsEngine
from hugging_mac_sdk.models.qwen3_tts.resources import (
    Qwen3TtsCoreMlResourceResolver,
    Qwen3TtsResourceResolver,
)
from hugging_mac_sdk.models.qwen3_tts.utils.tokenizer import Qwen3CoreMlTokenizer
from hugging_mac_sdk.schemas.resources import HuggingFaceSource
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

QWEN3_TTS_MODEL_ID = QWEN3_TTS_MANIFEST.model_id
QWEN3_TTS_REVISION = QWEN3_TTS_MANIFEST.revision
QWEN3_TTS_MLX_ARTIFACT = next(
    artifact for artifact in QWEN3_TTS_DEFINITION.artifacts if artifact.artifact_id == "mlx-4bit"
)
QWEN3_TTS_TOKENIZER_ARTIFACT = next(
    artifact for artifact in QWEN3_TTS_DEFINITION.artifacts if artifact.artifact_id == "tokenizers"
)
QWEN3_TTS_COREML_ARTIFACT = next(
    artifact
    for artifact in QWEN3_TTS_DEFINITION.artifacts
    if artifact.artifact_id == "coreml-w8a16"
)
QWEN3_TTS_REQUIRED_FILES = QWEN3_TTS_MLX_ARTIFACT.required_files
QWEN3_TTS_TOKENIZER_REQUIRED_FILES = QWEN3_TTS_TOKENIZER_ARTIFACT.required_files
QWEN3_TTS_COREML_REQUIRED_FILES = QWEN3_TTS_COREML_ARTIFACT.required_files
assert isinstance(QWEN3_TTS_COREML_ARTIFACT.source, HuggingFaceSource)
QWEN3_TTS_COREML_REPO_ID = QWEN3_TTS_COREML_ARTIFACT.source.repo_id
QWEN3_TTS_COREML_REVISION = QWEN3_TTS_COREML_ARTIFACT.source.revision


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
    assert manifest.revision == QWEN3_TTS_REVISION == "main"
    assert manifest.capabilities == {"speech-synthesis"}
    coreml_source = QWEN3_TTS_COREML_ARTIFACT.source
    assert isinstance(coreml_source, HuggingFaceSource)
    assert coreml_source.repo_id == QWEN3_TTS_COREML_REPO_ID == "aufklarer/Qwen3-TTS-CoreML"
    assert coreml_source.revision == QWEN3_TTS_COREML_REVISION
    coreml_runtime = next(runtime for runtime in manifest.runtimes if runtime.name == "coreml")
    assert coreml_runtime.required_modules == ("coremltools", "numpy")

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
    mlx_core = ModuleType("mlx.core")
    mlx_core.clear_cache = lambda: None  # type: ignore[attr-defined]

    def fake_load(path: str, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return loaded_model

    utils.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
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


def test_qwen3_tts_engine_passes_reference_pair_only_when_present(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class ReferenceModel:
        def generate(self, **kwargs: object):
            calls.append(kwargs)
            yield FakeResult()

    engine = MlxQwen3TtsEngine(
        Qwen3TtsInstanceConfig(source_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )
    engine._model = ReferenceModel()
    reference = tmp_path / "speaker.wav"
    reference.touch()

    engine._infer_sync(
        SpeechSynthesisRequest(
            text="Hello in the cloned voice.",
            language="english",
            reference_audio=AudioInput(path=reference),
            reference_text="This is the reference voice.",
        ),
        Event(),
    )

    assert calls[0]["ref_audio"] == str(reference)
    assert calls[0]["ref_text"] == "This is the reference voice."


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
        QWEN3_TTS_MANIFEST,
        QWEN3_TTS_MLX_ARTIFACT,
        QWEN3_TTS_TOKENIZER_ARTIFACT,
    )

    resolver._validate(tmp_path)
    assert resolver.status().artifacts[0].available


async def test_qwen3_tts_coreml_resource_validation(tmp_path: Path) -> None:
    for name in QWEN3_TTS_COREML_REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts_coreml",
                "models": [
                    "TextProjector",
                    "CodeEmbedder",
                    "MultiCodeEmbedder",
                    "CodeDecoder",
                    "MultiCodeDecoder",
                    "SpeechDecoder",
                ],
            }
        ),
        encoding="utf-8",
    )
    source = HuggingFaceSource(repo_id="test/qwen-coreml", revision="pinned")
    tokenizer_path = tmp_path / "tokenizers"
    for name in QWEN3_TTS_TOKENIZER_REQUIRED_FILES:
        path = tokenizer_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    resolver = Qwen3TtsCoreMlResourceResolver(
        source,
        source,
        Qwen3TtsInstanceConfig(
            runtime="coreml",
            coreml_path=tmp_path,
            tokenizer_path=tokenizer_path,
            device="cpu-and-neural-engine",
        ),
        QWEN3_TTS_MANIFEST,
        QWEN3_TTS_COREML_ARTIFACT,
        QWEN3_TTS_TOKENIZER_ARTIFACT,
    )

    resolved = await resolver.resolve_source()
    assert resolved.path == tmp_path
    status = resolver.status()
    assert status.artifacts[0].available
    assert status.runtimes[0].runtime == "coreml"
    assert status.runtimes[0].available
    assert status.runtimes[0].artifact_ids == ("coreml-w8a16", "tokenizers")


def test_qwen3_tts_coreml_tokenizer_matches_bundle_bpe(tmp_path: Path) -> None:
    vocab = {"Hello": 10, "Ġworld": 11}
    merges = "\n".join(
        (
            "H e",
            "He l",
            "Hel l",
            "Hell o",
            "Ġ w",
            "Ġw o",
            "Ġwo r",
            "Ġwor l",
            "Ġworl d",
        )
    )
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text(merges, encoding="utf-8")

    tokenizer = Qwen3CoreMlTokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.txt")

    assert tokenizer.encode("Hello world") == [10, 11]


def test_qwen3_tts_coreml_decoder_uses_mlstate_and_normalizes_rank() -> None:
    seen: dict[str, object] = {}
    state = object()

    class StatefulSession:
        def run_with_state(self, inputs: dict[str, object], received_state: object):
            seen.update(inputs)
            seen["state"] = received_state
            return {"logits": np.zeros((1, 1, 3072), dtype=np.float16)}

    result, key, value = CoreMlQwen3TtsEngine._decode_step(
        StatefulSession(),  # type: ignore[arg-type]
        np.zeros((1024, 1, 1), dtype=np.float16),
        3,
        None,
        None,
        16,
        state=state,
    )

    assert result["logits"].shape == (1, 1, 3072)
    assert key is value is None
    assert seen["state"] is state
    assert np.asarray(seen["input_embeds"]).shape == (1, 1024, 1, 1)
    assert "key_cache" not in seen and "value_cache" not in seen
    assert np.asarray(seen["key_padding_mask"])[0, :4].tolist() == [0.0] * 4


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
    runtimes = {runtime.runtime: runtime for runtime in status.runtimes}
    assert runtimes["mlx"].artifact_ids == ("mlx-4bit", "tokenizers")
    assert runtimes["coreml"].artifact_ids == ("coreml-w8a16", "tokenizers")


async def test_qwen3_tts_coreml_requires_shared_tokenizers(tmp_path: Path) -> None:
    coreml_path = QWEN3_TTS_COREML_ARTIFACT.resolve(tmp_path)
    for name in QWEN3_TTS_COREML_REQUIRED_FILES:
        path = coreml_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (coreml_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts_coreml",
                "models": [
                    "TextProjector",
                    "CodeEmbedder",
                    "MultiCodeEmbedder",
                    "CodeDecoder",
                    "MultiCodeDecoder",
                    "SpeechDecoder",
                ],
            }
        ),
        encoding="utf-8",
    )
    provider = QWEN3_TTS_DEFINITION.resource_provider
    assert provider is not None

    status = await provider.status("0.6b-base", {"model_home": tmp_path})
    runtimes = {runtime.runtime: runtime for runtime in status.runtimes}

    assert not runtimes["coreml"].available
    assert not runtimes["mlx"].available

    tokenizer_path = QWEN3_TTS_TOKENIZER_ARTIFACT.resolve(tmp_path)
    for name in QWEN3_TTS_TOKENIZER_REQUIRED_FILES:
        path = tokenizer_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    status = await provider.status("0.6b-base", {"model_home": tmp_path})
    runtimes = {runtime.runtime: runtime for runtime in status.runtimes}
    assert runtimes["coreml"].available
