"""YOLOv8n model manifest and registration entry point."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.models.yolov8.assets import (
    YoloV8AssetResolver,
    YoloV8ResourceProvider,
)
from hugging_mac_sdk.models.yolov8.config import YoloV8InstanceConfig
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.models.yolov8.instance import (
    BaseYoloV8Instance,
    CoreMlYoloV8Instance,
    PyTorchMpsYoloV8Instance,
)
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

YOLOV8N_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8N_MANIFEST = YOLOV8N_CONFIG.manifest

def _create_pytorch_mps(options: dict[str, object]) -> PyTorchMpsYoloV8Instance:
    config = YoloV8InstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    source = cast(HuggingFaceSource, YOLOV8N_MANIFEST.resources[0])
    assets = YoloV8AssetResolver(source, config)
    return PyTorchMpsYoloV8Instance(config, assets)


def _create_coreml(options: dict[str, object]) -> CoreMlYoloV8Instance:
    normalized = dict(options)
    requested_device = normalized.pop("device", None)
    if requested_device is not None:
        normalized["compute_units"] = requested_device
    config = YoloV8InstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    source = cast(HuggingFaceSource, YOLOV8N_MANIFEST.resources[0])
    assets = YoloV8AssetResolver(source, config)
    return CoreMlYoloV8Instance(config, assets)


def _create_yolov8_legacy(options: dict[str, object]) -> BaseYoloV8Instance:
    """Compatibility dispatcher; new code uses ``runtime_factories`` directly."""

    runtime = str(options.get("runtime", YOLOV8N_MANIFEST.default_runtime))
    if runtime == "pytorch-mps":
        return _create_pytorch_mps(options)
    if runtime == "coreml":
        return _create_coreml(options)
    raise UnsupportedRuntimeError(f"Unsupported YOLOv8 runtime: {runtime}")


YOLOV8N_DEFINITION = ModelDefinition(
    manifest=YOLOV8N_MANIFEST,
    factory=_create_yolov8_legacy,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
        "coreml": _create_coreml,
    },
    artifacts=YOLOV8N_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8",),
    resource_provider=YoloV8ResourceProvider(
        cast(HuggingFaceSource, YOLOV8N_MANIFEST.resources[0]),
        revision=YOLOV8N_MANIFEST.revision,
    ),
)


def register_yolov8(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned source model and both conversion paths."""

    converters.register(UltralyticsExportConverter(), replace=replace)
    converters.register(YoloV8Converter(), replace=replace)
    models.register(YOLOV8N_DEFINITION, replace=replace)
    return YOLOV8N_DEFINITION
