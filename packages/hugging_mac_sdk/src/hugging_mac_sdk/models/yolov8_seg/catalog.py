"""YOLOv8 Seg registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.models.yolov8_seg.assets import (
    YoloV8SegAssetResolver,
    YoloV8SegResourceProvider,
)
from hugging_mac_sdk.models.yolov8_seg.config import YoloV8SegInstanceConfig
from hugging_mac_sdk.models.yolov8_seg.converter import YoloV8SegConverter
from hugging_mac_sdk.models.yolov8_seg.instance import (
    BaseYoloV8SegInstance,
    CoreMlYoloV8SegInstance,
    PyTorchMpsYoloV8SegInstance,
)
from hugging_mac_sdk.schemas.resources import ResourceSource

YOLOV8_SEG_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8_SEG_MANIFEST = YOLOV8_SEG_CONFIG.manifest


def _variant_source(variant: str) -> ResourceSource:
    resources = YOLOV8_SEG_MANIFEST.get_variant(variant).resources
    if len(resources) != 1:
        raise TypeError(f"YOLOv8 Seg variant {variant} needs one source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> PyTorchMpsYoloV8SegInstance:
    config = YoloV8SegInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    return PyTorchMpsYoloV8SegInstance(
        config, YoloV8SegAssetResolver(_variant_source(config.variant), config)
    )


def _create_coreml(options: dict[str, object]) -> CoreMlYoloV8SegInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = YoloV8SegInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    return CoreMlYoloV8SegInstance(
        config, YoloV8SegAssetResolver(_variant_source(config.variant), config)
    )


def _legacy_factory(options: dict[str, object]) -> BaseYoloV8SegInstance:
    runtime = str(options.get("runtime", YOLOV8_SEG_MANIFEST.default_runtime))
    if runtime == "pytorch-mps":
        return _create_pytorch_mps(options)
    if runtime == "coreml":
        return _create_coreml(options)
    raise UnsupportedRuntimeError(f"Unsupported YOLOv8 Seg runtime: {runtime}")


YOLOV8_SEG_DEFINITION = ModelDefinition(
    manifest=YOLOV8_SEG_MANIFEST,
    factory=_legacy_factory,
    runtime_factories={"pytorch-mps": _create_pytorch_mps, "coreml": _create_coreml},
    artifacts=YOLOV8_SEG_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8-seg",),
    resource_provider=YoloV8SegResourceProvider(
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_SEG_MANIFEST.variants}
    ),
)


def register_yolov8_seg(
    models: ModelRegistry, converters: ConverterRegistry, *, replace: bool = False
) -> ModelDefinition:
    # The generic exporter is shared by all Ultralytics task packages.
    converters.register(UltralyticsExportConverter(), replace=True)
    converters.register(YoloV8SegConverter(), replace=replace)
    models.register(YOLOV8_SEG_DEFINITION, replace=replace)
    return YOLOV8_SEG_DEFINITION
