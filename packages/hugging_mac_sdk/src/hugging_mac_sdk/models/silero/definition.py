"""Silero definition, ONNX factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, UrlFileSource

from .config import SileroInstanceConfig
from .coreml import CoreMlSileroEngine
from .instance import SileroInstance
from .onnx import OnnxSileroEngine
from .resources import SileroResourceProvider, SileroResourceResolver

SILERO_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
SILERO_MANIFEST = SILERO_CONFIG.manifest


def _sources() -> tuple[UrlFileSource, HuggingFaceSource]:
    resources = SILERO_MANIFEST.get_variant("v6.2.1").resources
    if (
        len(resources) != 2
        or not isinstance(resources[0], UrlFileSource)
        or not isinstance(resources[1], HuggingFaceSource)
    ):
        raise TypeError("Silero needs ONNX URL and Core ML Hugging Face sources")
    return resources[0], resources[1]


def _create_onnx(options: dict[str, object]) -> SileroInstance:
    config = SileroInstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = SileroResourceResolver(*_sources(), config)
    return SileroInstance(config, OnnxSileroEngine(config, resources))


def _create_coreml(options: dict[str, object]) -> SileroInstance:
    normalized = dict(options)
    requested_device = normalized.get("device")
    if requested_device == "coreml":
        normalized["device"] = "all"
    config = SileroInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = SileroResourceResolver(*_sources(), config)
    return SileroInstance(config, CoreMlSileroEngine(config, resources))


SILERO_DEFINITION = ModelDefinition(
    manifest=SILERO_MANIFEST,
    runtime_factories={"coreml": _create_coreml, "onnx": _create_onnx},
    artifacts=SILERO_CONFIG.artifacts,
    resource_provider=SileroResourceProvider(*_sources()),
)


def register_silero(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(SILERO_DEFINITION, replace=replace)
    return SILERO_DEFINITION
