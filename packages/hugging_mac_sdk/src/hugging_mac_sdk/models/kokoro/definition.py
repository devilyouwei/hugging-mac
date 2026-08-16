"""Kokoro-82M definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Kokoro82mInstanceConfig
from .coreml import CoreMlKokoro82mEngine
from .instance import Kokoro82mInstance
from .resources import Kokoro82mResourceProvider, Kokoro82mResourceResolver
from .torch import TorchKokoro82mEngine

KOKORO_82M_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
KOKORO_82M_MANIFEST = KOKORO_82M_CONFIG.manifest


def _sources() -> tuple[HuggingFaceSource, HuggingFaceSource]:
    resources = KOKORO_82M_MANIFEST.get_variant("v1.0").resources
    if len(resources) != 2 or not all(isinstance(item, HuggingFaceSource) for item in resources):
        raise TypeError("Kokoro-82M needs PyTorch and Core ML Hugging Face sources")
    return resources[0], resources[1]  # type: ignore[return-value]


def _create_pytorch_mps(options: dict[str, object]) -> Kokoro82mInstance:
    config = Kokoro82mInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = Kokoro82mResourceResolver(*_sources(), config)
    return Kokoro82mInstance(
        config,
        TorchKokoro82mEngine(config, resources),
    )


def _create_coreml(options: dict[str, object]) -> Kokoro82mInstance:
    config = Kokoro82mInstanceConfig.model_validate(options | {"runtime": "coreml"})
    resources = Kokoro82mResourceResolver(*_sources(), config)
    return Kokoro82mInstance(config, CoreMlKokoro82mEngine(config, resources))


KOKORO_82M_DEFINITION = ModelDefinition(
    manifest=KOKORO_82M_MANIFEST,
    runtime_factories={
        "coreml": _create_coreml,
        "pytorch-mps": _create_pytorch_mps,
    },
    artifacts=KOKORO_82M_CONFIG.artifacts,
    resource_provider=Kokoro82mResourceProvider(*_sources()),
)


def register_kokoro(
    models: ModelRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned Kokoro-82M model."""

    models.register(KOKORO_82M_DEFINITION, replace=replace)
    return KOKORO_82M_DEFINITION
