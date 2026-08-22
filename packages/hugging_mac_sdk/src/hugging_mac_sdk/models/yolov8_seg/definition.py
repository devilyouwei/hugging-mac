"""YOLOv8 Seg model definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import ResourceSource

from .config import YoloV8SegInstanceConfig
from .converter import YoloV8SegConverter
from .coreml import CoreMlYoloV8SegEngine
from .instance import YoloV8SegInstance
from .onnx import OnnxYoloV8SegEngine
from .resources import YoloV8SegResourceProvider, YoloV8SegResourceResolver
from .torch import TorchYoloV8SegEngine
from .utils.checkpoint import inspect_yolov8_checkpoint

YOLOV8_SEG_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8_SEG_MANIFEST = YOLOV8_SEG_CONFIG.manifest


def _variant_source(variant: str) -> ResourceSource:
    resources = YOLOV8_SEG_MANIFEST.get_variant(variant).resources
    if len(resources) != 1:
        raise TypeError(f"YOLOv8 Seg variant {variant} needs one source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> YoloV8SegInstance:
    config = YoloV8SegInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = YoloV8SegResourceResolver(_variant_source(config.variant), config)
    return YoloV8SegInstance(config, TorchYoloV8SegEngine(config, resources))


def _create_coreml(options: dict[str, object]) -> YoloV8SegInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = YoloV8SegInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = YoloV8SegResourceResolver(_variant_source(config.variant), config)
    return YoloV8SegInstance(config, CoreMlYoloV8SegEngine(config, resources))


def _create_onnx(options: dict[str, object]) -> YoloV8SegInstance:
    config = YoloV8SegInstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = YoloV8SegResourceResolver(_variant_source(config.variant), config)
    return YoloV8SegInstance(config, OnnxYoloV8SegEngine(config, resources))


YOLOV8_SEG_DEFINITION = ModelDefinition(
    manifest=YOLOV8_SEG_MANIFEST,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
        "coreml": _create_coreml,
        "onnx": _create_onnx,
    },
    artifact_inspectors={"pytorch-mps": inspect_yolov8_checkpoint},
    artifacts=YOLOV8_SEG_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8-seg",),
    resource_provider=YoloV8SegResourceProvider(
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_SEG_MANIFEST.variants}
    ),
)


def register_yolov8_seg(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(YoloV8SegConverter(), replace=replace)
    models.register(YOLOV8_SEG_DEFINITION, replace=replace)
    return YOLOV8_SEG_DEFINITION
