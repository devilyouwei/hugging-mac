"""Qwen3-TTS 0.6B Base definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen3TtsInstanceConfig
from .coreml import CoreMlQwen3TtsEngine
from .instance import Qwen3TtsInstance
from .mlx import MlxQwen3TtsEngine
from .resources import (
    Qwen3TtsCoreMlResourceResolver,
    Qwen3TtsResourceProvider,
    Qwen3TtsResourceResolver,
)

QWEN3_TTS_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_TTS_MANIFEST = QWEN3_TTS_CONFIG.manifest


def _artifact(runtime: str) -> ModelArtifact:
    artifact_id = "mlx-4bit" if runtime == "mlx" else "coreml-w8a16"
    return QWEN3_TTS_CONFIG.get_artifact(artifact_id, variant="0.6b-base", runtime=runtime)


def _source() -> HuggingFaceSource:
    source = _artifact("mlx").source
    if not isinstance(source, HuggingFaceSource):
        raise TypeError("Qwen3-TTS needs an MLX Hugging Face source")
    return source


def _coreml_source() -> HuggingFaceSource:
    source = _artifact("coreml").source
    if not isinstance(source, HuggingFaceSource):
        raise TypeError("Qwen3-TTS needs one Core ML Hugging Face source")
    return source


def _tokenizer_source() -> HuggingFaceSource:
    shared = QWEN3_TTS_CONFIG.get_artifact(artifact_id="tokenizers")
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Qwen3-TTS needs one shared tokenizer source")
    return shared.source


def _create_mlx(options: dict[str, object]) -> Qwen3TtsInstance:
    config = Qwen3TtsInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Qwen3TtsResourceResolver(
        _source(),
        config,
        QWEN3_TTS_MANIFEST,
        _artifact("mlx"),
        QWEN3_TTS_CONFIG.get_artifact(artifact_id="tokenizers"),
        tokenizer_source=_tokenizer_source(),
    )
    return Qwen3TtsInstance(config, MlxQwen3TtsEngine(config, resources), QWEN3_TTS_MANIFEST)


def _create_coreml(options: dict[str, object]) -> Qwen3TtsInstance:
    normalized = options | {"runtime": "coreml"}
    normalized.setdefault("device", "cpu-and-neural-engine")
    config = Qwen3TtsInstanceConfig.model_validate(normalized)
    resources = Qwen3TtsCoreMlResourceResolver(
        _coreml_source(),
        _tokenizer_source(),
        config,
        QWEN3_TTS_MANIFEST,
        _artifact("coreml"),
        QWEN3_TTS_CONFIG.get_artifact(artifact_id="tokenizers"),
    )
    return Qwen3TtsInstance(config, CoreMlQwen3TtsEngine(config, resources), QWEN3_TTS_MANIFEST)


QWEN3_TTS_DEFINITION = ModelDefinition(
    manifest=QWEN3_TTS_MANIFEST,
    runtime_factories={"mlx": _create_mlx, "coreml": _create_coreml},
    artifacts=QWEN3_TTS_CONFIG.artifacts,
    resource_provider=Qwen3TtsResourceProvider(
        _source(),
        _tokenizer_source(),
        QWEN3_TTS_MANIFEST,
        _artifact("mlx"),
        QWEN3_TTS_CONFIG.get_artifact(artifact_id="tokenizers"),
        _coreml_source(),
        _artifact("coreml"),
    ),
)


def register_qwen3_tts(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(QWEN3_TTS_DEFINITION, replace=replace)
    return QWEN3_TTS_DEFINITION
