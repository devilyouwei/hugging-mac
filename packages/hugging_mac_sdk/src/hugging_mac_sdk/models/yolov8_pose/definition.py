"""YOLOv8 Pose model definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import ResourceSource

from .config import YoloV8PoseInstanceConfig
from .converter import YoloV8PoseConverter
from .coreml import CoreMlYoloV8PoseEngine
from .instance import YoloV8PoseInstance
from .onnx import OnnxYoloV8PoseEngine
from .resources import YoloV8PoseResourceProvider, YoloV8PoseResourceResolver
from .torch import TorchYoloV8PoseEngine
from .utils.checkpoint import inspect_yolov8_checkpoint

YOLOV8_POSE_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
YOLOV8_POSE_MANIFEST = YOLOV8_POSE_CONFIG.manifest


def _variant_source(variant: str) -> ResourceSource:
    resources = YOLOV8_POSE_MANIFEST.get_variant(variant).resources
    if len(resources) != 1:
        raise TypeError(f"YOLOv8 Pose variant {variant} needs one source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> YoloV8PoseInstance:
    config = YoloV8PoseInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = YoloV8PoseResourceResolver(_variant_source(config.variant), config)
    return YoloV8PoseInstance(config, TorchYoloV8PoseEngine(config, resources))


def _create_coreml(options: dict[str, object]) -> YoloV8PoseInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = YoloV8PoseInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = YoloV8PoseResourceResolver(_variant_source(config.variant), config)
    return YoloV8PoseInstance(config, CoreMlYoloV8PoseEngine(config, resources))


def _create_onnx(options: dict[str, object]) -> YoloV8PoseInstance:
    config = YoloV8PoseInstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = YoloV8PoseResourceResolver(_variant_source(config.variant), config)
    return YoloV8PoseInstance(config, OnnxYoloV8PoseEngine(config, resources))


YOLOV8_POSE_DEFINITION = ModelDefinition(
    manifest=YOLOV8_POSE_MANIFEST,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
        "coreml": _create_coreml,
        "onnx": _create_onnx,
    },
    artifact_inspectors={"pytorch-mps": inspect_yolov8_checkpoint},
    artifacts=YOLOV8_POSE_CONFIG.artifacts,
    converter_ids=("ultralytics.yolov8-pose",),
    resource_provider=YoloV8PoseResourceProvider(
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_POSE_MANIFEST.variants}
    ),
)


def register_yolov8_pose(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(YoloV8PoseConverter(), replace=replace)
    models.register(YOLOV8_POSE_DEFINITION, replace=replace)
    return YOLOV8_POSE_DEFINITION
