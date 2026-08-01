"""SenseVoiceSmall definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import SenseVoiceSmallInstanceConfig
from .instance import SenseVoiceSmallInstance
from .resources import (
    SenseVoiceSmallResourceProvider,
    SenseVoiceSmallResourceResolver,
)
from .torch import TorchSenseVoiceSmallEngine

SENSEVOICE_SMALL_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
SENSEVOICE_SMALL_MANIFEST = SENSEVOICE_SMALL_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = SENSEVOICE_SMALL_MANIFEST.get_variant("small").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("SenseVoiceSmall needs one Hugging Face snapshot source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> SenseVoiceSmallInstance:
    config = SenseVoiceSmallInstanceConfig.model_validate(
        options | {"runtime": "pytorch-mps"}
    )
    resources = SenseVoiceSmallResourceResolver(_source(), config)
    return SenseVoiceSmallInstance(
        config,
        TorchSenseVoiceSmallEngine(config, resources),
    )


SENSEVOICE_SMALL_DEFINITION = ModelDefinition(
    manifest=SENSEVOICE_SMALL_MANIFEST,
    runtime_factories={"pytorch-mps": _create_pytorch_mps},
    artifacts=SENSEVOICE_SMALL_CONFIG.artifacts,
    resource_provider=SenseVoiceSmallResourceProvider(_source()),
)


def register_sensevoice_small(
    models: ModelRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned SenseVoiceSmall model."""

    models.register(SENSEVOICE_SMALL_DEFINITION, replace=replace)
    return SENSEVOICE_SMALL_DEFINITION
