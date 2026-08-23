"""Qwen3-ASR Core ML definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen3AsrCoreMlInstanceConfig
from .coreml import CoreMlQwen3AsrEngine
from .instance import Qwen3AsrCoreMlInstance
from .resources import Qwen3AsrCoreMlResourceProvider, Qwen3AsrCoreMlResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_ASR_MANIFEST = _CONFIG.manifest
_ARTIFACT = _CONFIG.get_artifact("coreml-int8", variant="0.6b", runtime="coreml")
_TOKENIZER_ARTIFACT = _CONFIG.get_artifact("tokenizer", shared=True)
if not isinstance(_ARTIFACT.source, HuggingFaceSource) or not isinstance(
    _TOKENIZER_ARTIFACT.source, HuggingFaceSource
):
    raise TypeError("Qwen3-ASR requires Core ML and tokenizer Hugging Face sources")
_SOURCE = _ARTIFACT.source
_TOKENIZER_SOURCE = _TOKENIZER_ARTIFACT.source


def _create(options: dict[str, object]) -> Qwen3AsrCoreMlInstance:
    config = Qwen3AsrCoreMlInstanceConfig.model_validate(options | {"runtime": "coreml"})
    resources = Qwen3AsrCoreMlResourceResolver(
        _SOURCE,
        _TOKENIZER_SOURCE,
        config,
        manifest=QWEN3_ASR_MANIFEST,
        artifact=_ARTIFACT,
        tokenizer_artifact=_TOKENIZER_ARTIFACT,
    )
    return Qwen3AsrCoreMlInstance(
        config, CoreMlQwen3AsrEngine(config, resources), QWEN3_ASR_MANIFEST
    )


QWEN3_ASR_DEFINITION = ModelDefinition(
    manifest=QWEN3_ASR_MANIFEST,
    runtime_factories={"coreml": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=Qwen3AsrCoreMlResourceProvider(
        _SOURCE,
        _TOKENIZER_SOURCE,
        QWEN3_ASR_MANIFEST,
        _ARTIFACT,
        _TOKENIZER_ARTIFACT,
    ),
)


def register_qwen3_asr(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(QWEN3_ASR_DEFINITION, replace=replace)
    return QWEN3_ASR_DEFINITION
