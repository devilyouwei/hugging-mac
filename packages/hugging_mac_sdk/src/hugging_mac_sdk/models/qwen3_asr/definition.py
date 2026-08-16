"""Qwen3-ASR Core ML definition and registration."""

from pathlib import Path
from typing import cast

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen3AsrCoreMlInstanceConfig
from .coreml import CoreMlQwen3AsrEngine
from .instance import Qwen3AsrCoreMlInstance
from .resources import Qwen3AsrCoreMlResourceProvider, Qwen3AsrCoreMlResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_ASR_MANIFEST = _CONFIG.manifest
_RESOURCES = QWEN3_ASR_MANIFEST.get_variant("0.6b").resources
if len(_RESOURCES) != 2 or not all(isinstance(item, HuggingFaceSource) for item in _RESOURCES):
    raise TypeError("Qwen3-ASR requires Core ML and tokenizer Hugging Face resources")
_SOURCE = cast(HuggingFaceSource, _RESOURCES[0])
_TOKENIZER_SOURCE = cast(HuggingFaceSource, _RESOURCES[1])


def _create(options: dict[str, object]) -> Qwen3AsrCoreMlInstance:
    config = Qwen3AsrCoreMlInstanceConfig.model_validate(options | {"runtime": "coreml"})
    resources = Qwen3AsrCoreMlResourceResolver(_SOURCE, _TOKENIZER_SOURCE, config)
    return Qwen3AsrCoreMlInstance(config, CoreMlQwen3AsrEngine(config, resources))


QWEN3_ASR_DEFINITION = ModelDefinition(
    manifest=QWEN3_ASR_MANIFEST,
    runtime_factories={"coreml": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=Qwen3AsrCoreMlResourceProvider(_SOURCE, _TOKENIZER_SOURCE),
)


def register_qwen3_asr(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(QWEN3_ASR_DEFINITION, replace=replace)
    return QWEN3_ASR_DEFINITION
