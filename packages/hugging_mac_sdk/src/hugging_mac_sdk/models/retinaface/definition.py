"""py-feat RetinaFace definition, factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import RetinaFaceInstanceConfig
from .converter import RetinaFaceConverter
from .coreml import CoreMlRetinaFaceEngine
from .instance import RetinaFaceInstance
from .resources import RetinaFaceResourceProvider, RetinaFaceResourceResolver
from .torch import TorchRetinaFaceEngine

RETINAFACE_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
RETINAFACE_MANIFEST = RETINAFACE_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = RETINAFACE_MANIFEST.get_variant("mobilenet0.25").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("RetinaFace requires one Hugging Face PyTorch source")
    return resources[0]


def _create_pytorch(options: dict[str, object]) -> RetinaFaceInstance:
    config = RetinaFaceInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = RetinaFaceResourceResolver(_source(), config)
    return RetinaFaceInstance(config, TorchRetinaFaceEngine(config, resources))


def _create_coreml(options: dict[str, object]) -> RetinaFaceInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = RetinaFaceInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = RetinaFaceResourceResolver(_source(), config)
    return RetinaFaceInstance(config, CoreMlRetinaFaceEngine(config, resources))


RETINAFACE_DEFINITION = ModelDefinition(
    manifest=RETINAFACE_MANIFEST,
    runtime_factories={"pytorch-mps": _create_pytorch, "coreml": _create_coreml},
    artifacts=RETINAFACE_CONFIG.artifacts,
    converter_ids=("py-feat.retinaface",),
    resource_provider=RetinaFaceResourceProvider(_source()),
)


def register_retinaface(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(RetinaFaceConverter(), replace=replace)
    models.register(RETINAFACE_DEFINITION, replace=replace)
    return RETINAFACE_DEFINITION
