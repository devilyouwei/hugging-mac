"""YOLOv8 Pose model definition, runtime factories, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.artifact import ModelArtifact
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
YOLOV8_POSE_CONVERTER = YoloV8PoseConverter(YOLOV8_POSE_MANIFEST.model_id)


def _variant_source(variant: str) -> ResourceSource:
    source = YOLOV8_POSE_CONFIG.get_artifact(
        "source", variant=variant, runtime="pytorch-mps"
    ).source
    if source is None:
        raise TypeError(f"YOLOv8 Pose variant {variant} needs one source")
    return source


def _variant_artifacts(variant: str) -> dict[str, ModelArtifact]:
    return {
        artifact.artifact_id: artifact
        for artifact in YOLOV8_POSE_CONFIG.get_artifacts(variant=variant)
    }


def _create_pytorch_mps(options: dict[str, object]) -> YoloV8PoseInstance:
    config = YoloV8PoseInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = YoloV8PoseResourceResolver(
        _variant_source(config.variant),
        config,
        YOLOV8_POSE_MANIFEST,
        _variant_artifacts(config.variant),
    )
    return YoloV8PoseInstance(
        config, TorchYoloV8PoseEngine(config, resources), YOLOV8_POSE_MANIFEST
    )


def _create_coreml(options: dict[str, object]) -> YoloV8PoseInstance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = YoloV8PoseInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = YoloV8PoseResourceResolver(
        _variant_source(config.variant),
        config,
        YOLOV8_POSE_MANIFEST,
        _variant_artifacts(config.variant),
    )
    return YoloV8PoseInstance(
        config, CoreMlYoloV8PoseEngine(config, resources), YOLOV8_POSE_MANIFEST
    )


def _create_onnx(options: dict[str, object]) -> YoloV8PoseInstance:
    config = YoloV8PoseInstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = YoloV8PoseResourceResolver(
        _variant_source(config.variant),
        config,
        YOLOV8_POSE_MANIFEST,
        _variant_artifacts(config.variant),
    )
    return YoloV8PoseInstance(config, OnnxYoloV8PoseEngine(config, resources), YOLOV8_POSE_MANIFEST)


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
        {variant.name: _variant_source(variant.name) for variant in YOLOV8_POSE_MANIFEST.variants},
        YOLOV8_POSE_MANIFEST,
        YOLOV8_POSE_CONFIG.artifacts,
    ),
)


def register_yolov8_pose(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(YOLOV8_POSE_CONVERTER, replace=replace)
    models.register(YOLOV8_POSE_DEFINITION, replace=replace)
    return YOLOV8_POSE_DEFINITION
