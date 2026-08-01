"""Kokoro-82M definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Kokoro82mInstanceConfig
from .instance import Kokoro82mInstance
from .resources import Kokoro82mResourceProvider, Kokoro82mResourceResolver
from .torch import TorchKokoro82mEngine

KOKORO_82M_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
KOKORO_82M_MANIFEST = KOKORO_82M_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = KOKORO_82M_MANIFEST.get_variant("v1.0").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Kokoro-82M needs one Hugging Face snapshot source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> Kokoro82mInstance:
    config = Kokoro82mInstanceConfig.model_validate(
        options | {"runtime": "pytorch-mps"}
    )
    resources = Kokoro82mResourceResolver(_source(), config)
    return Kokoro82mInstance(
        config,
        TorchKokoro82mEngine(config, resources),
    )


KOKORO_82M_DEFINITION = ModelDefinition(
    manifest=KOKORO_82M_MANIFEST,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
    },
    artifacts=KOKORO_82M_CONFIG.artifacts,
    resource_provider=Kokoro82mResourceProvider(_source()),
)


def register_kokoro_82m(
    models: ModelRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned Kokoro-82M model."""

    models.register(KOKORO_82M_DEFINITION, replace=replace)
    return KOKORO_82M_DEFINITION
