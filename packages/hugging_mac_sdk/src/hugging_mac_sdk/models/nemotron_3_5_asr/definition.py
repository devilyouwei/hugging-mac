"""Nemotron 3.5 ASR definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import NemotronCoreMlInstanceConfig
from .coreml import CoreMlNemotronEngine
from .instance import NemotronCoreMlInstance
from .resources import NemotronCoreMlResourceProvider, NemotronCoreMlResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
NEMOTRON_3_5_ASR_MANIFEST = _CONFIG.manifest
_SOURCES: dict[str, HuggingFaceSource] = {}
for variant in NEMOTRON_3_5_ASR_MANIFEST.variants:
    if len(variant.resources) != 1 or not isinstance(variant.resources[0], HuggingFaceSource):
        raise TypeError(f"Nemotron variant {variant.name} requires one Hugging Face resource")
    _SOURCES[variant.name] = variant.resources[0]


def _create(options: dict[str, object]) -> NemotronCoreMlInstance:
    config = NemotronCoreMlInstanceConfig.model_validate(options | {"runtime": "coreml"})
    resources = NemotronCoreMlResourceResolver(_SOURCES[config.variant], config)
    return NemotronCoreMlInstance(config, CoreMlNemotronEngine(config, resources))


NEMOTRON_3_5_ASR_DEFINITION = ModelDefinition(
    manifest=NEMOTRON_3_5_ASR_MANIFEST,
    runtime_factories={"coreml": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=NemotronCoreMlResourceProvider(_SOURCES),
)


def register_nemotron_3_5_asr(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(NEMOTRON_3_5_ASR_DEFINITION, replace=replace)
    return NEMOTRON_3_5_ASR_DEFINITION
