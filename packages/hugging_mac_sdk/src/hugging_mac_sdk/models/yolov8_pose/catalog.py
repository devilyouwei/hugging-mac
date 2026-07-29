"""YOLOv8 Pose registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.models.yolov8_pose.assets import (
    YoloV8PoseAssetResolver,
    YoloV8PoseResourceProvider,
)
from hugging_mac_sdk.models.yolov8_pose.config import YoloV8PoseInstanceConfig
from hugging_mac_sdk.models.yolov8_pose.converter import YoloV8PoseConverter
from hugging_mac_sdk.models.yolov8_pose.instance import (
    BaseYoloV8PoseInstance,
    CoreMlYoloV8PoseInstance,
    PyTorchMpsYoloV8PoseInstance,
)
from hugging_mac_sdk.schemas.resources import ResourceSource

YOLOV8_POSE_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8_POSE_MANIFEST = YOLOV8_POSE_CONFIG.manifest


def _variant_source(variant: str) -> ResourceSource:
    resources = YOLOV8_POSE_MANIFEST.get_variant(variant).resources
    if len(resources) != 1:
        raise TypeError(f"YOLOv8 Pose variant {variant} needs one source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> PyTorchMpsYoloV8PoseInstance:
    config = YoloV8PoseInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    return PyTorchMpsYoloV8PoseInstance(
        config, YoloV8PoseAssetResolver(_variant_source(config.variant), config)
    )


def _create_coreml(options: dict[str, object]) -> CoreMlYoloV8PoseInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = YoloV8PoseInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    return CoreMlYoloV8PoseInstance(
        config, YoloV8PoseAssetResolver(_variant_source(config.variant), config)
    )


def _legacy_factory(options: dict[str, object]) -> BaseYoloV8PoseInstance:
    runtime = str(options.get("runtime", YOLOV8_POSE_MANIFEST.default_runtime))
    if runtime == "pytorch-mps":
        return _create_pytorch_mps(options)
    if runtime == "coreml":
        return _create_coreml(options)
    raise UnsupportedRuntimeError(f"Unsupported YOLOv8 Pose runtime: {runtime}")


YOLOV8_POSE_DEFINITION = ModelDefinition(
    manifest=YOLOV8_POSE_MANIFEST,
    factory=_legacy_factory,
    runtime_factories={"pytorch-mps": _create_pytorch_mps, "coreml": _create_coreml},
    artifacts=YOLOV8_POSE_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8-pose",),
    resource_provider=YoloV8PoseResourceProvider(
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_POSE_MANIFEST.variants}
    ),
)


def register_yolov8_pose(
    models: ModelRegistry, converters: ConverterRegistry, *, replace: bool = False
) -> ModelDefinition:
    # The generic exporter is shared by all Ultralytics task packages.
    converters.register(UltralyticsExportConverter(), replace=True)
    converters.register(YoloV8PoseConverter(), replace=replace)
    models.register(YOLOV8_POSE_DEFINITION, replace=replace)
    return YOLOV8_POSE_DEFINITION
