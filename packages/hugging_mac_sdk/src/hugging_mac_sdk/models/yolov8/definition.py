"""YOLOv8 model definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import YoloV8InstanceConfig
from .converter import YoloV8Converter
from .coreml import CoreMlYoloV8Engine
from .instance import YoloV8Instance
from .onnx import OnnxYoloV8Engine
from .resources import YoloV8ResourceProvider, YoloV8ResourceResolver
from .torch import TorchYoloV8Engine

YOLOV8_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8_MANIFEST = YOLOV8_CONFIG.manifest


def _variant_source(variant: str) -> HuggingFaceSource:
    resources = YOLOV8_MANIFEST.get_variant(variant).resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError(f"YOLOv8 variant {variant} needs one Hugging Face source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> YoloV8Instance:
    config = YoloV8InstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = YoloV8ResourceResolver(_variant_source(config.variant), config)
    return YoloV8Instance(config, TorchYoloV8Engine(config, resources))


def _create_coreml(options: dict[str, object]) -> YoloV8Instance:
    normalized = dict(options)
    requested_device = normalized.pop("device", None)
    if requested_device is not None:
        normalized["compute_units"] = requested_device
    config = YoloV8InstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = YoloV8ResourceResolver(_variant_source(config.variant), config)
    return YoloV8Instance(config, CoreMlYoloV8Engine(config, resources))


def _create_onnx(options: dict[str, object]) -> YoloV8Instance:
    config = YoloV8InstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = YoloV8ResourceResolver(_variant_source(config.variant), config)
    return YoloV8Instance(config, OnnxYoloV8Engine(config, resources))


YOLOV8_DEFINITION = ModelDefinition(
    manifest=YOLOV8_MANIFEST,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
        "coreml": _create_coreml,
        "onnx": _create_onnx,
    },
    artifacts=YOLOV8_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8",),
    resource_provider=YoloV8ResourceProvider(
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_MANIFEST.variants},
        revision=YOLOV8_MANIFEST.revision,
    ),
)


def register_yolov8(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned source model and conversion implementation."""

    converters.register(YoloV8Converter(), replace=replace)
    models.register(YOLOV8_DEFINITION, replace=replace)
    return YOLOV8_DEFINITION
