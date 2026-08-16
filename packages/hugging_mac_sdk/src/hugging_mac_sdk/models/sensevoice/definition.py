"""SenseVoiceSmall definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import SenseVoiceSmallInstanceConfig
from .converter import SenseVoiceSmallConverter
from .coreml import CoreMlSenseVoiceSmallEngine
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


def _tokenizer_source() -> HuggingFaceSource:
    shared = SENSEVOICE_SMALL_CONFIG.artifacts[-1]
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("SenseVoiceSmall needs one shared tokenizer source")
    return shared.source


def _create_pytorch_mps(options: dict[str, object]) -> SenseVoiceSmallInstance:
    config = SenseVoiceSmallInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = SenseVoiceSmallResourceResolver(
        _source(), config, tokenizer_source=_tokenizer_source()
    )
    return SenseVoiceSmallInstance(
        config,
        TorchSenseVoiceSmallEngine(config, resources),
    )


def _create_coreml(options: dict[str, object]) -> SenseVoiceSmallInstance:
    normalized = dict(options)
    requested_device = normalized.pop("device", None)
    if requested_device is not None:
        normalized["compute_units"] = requested_device
    config = SenseVoiceSmallInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = SenseVoiceSmallResourceResolver(
        _source(), config, tokenizer_source=_tokenizer_source()
    )
    return SenseVoiceSmallInstance(config, CoreMlSenseVoiceSmallEngine(config, resources))


SENSEVOICE_SMALL_DEFINITION = ModelDefinition(
    manifest=SENSEVOICE_SMALL_MANIFEST,
    runtime_factories={"pytorch-mps": _create_pytorch_mps, "coreml": _create_coreml},
    artifacts=SENSEVOICE_SMALL_CONFIG.artifacts,
    converter_ids=("funaudiollm.sensevoice-small",),
    resource_provider=SenseVoiceSmallResourceProvider(_source(), _tokenizer_source()),
)


def register_sensevoice(
    models: ModelRegistry,
    converters: ConverterRegistry | None = None,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned SenseVoiceSmall model."""

    if converters is not None:
        converters.register(SenseVoiceSmallConverter(), replace=replace)
    models.register(SENSEVOICE_SMALL_DEFINITION, replace=replace)
    return SENSEVOICE_SMALL_DEFINITION
