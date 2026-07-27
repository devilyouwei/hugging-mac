"""YOLOv8n model manifest and registration entry point."""

from __future__ import annotations

from typing import cast

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.models.yolov8.assets import (
    YoloV8AssetResolver,
    YoloV8ResourceProvider,
)
from hugging_mac_sdk.models.yolov8.config import (
    YOLOV8_REPO_ID,
    YOLOV8_REPO_REVISION,
    YOLOV8N_FILENAME,
    YOLOV8N_SHA256,
    YoloV8InstanceConfig,
)
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.models.yolov8.instance import (
    CoreMlYoloV8Instance,
    PyTorchMpsYoloV8Instance,
)
from hugging_mac_sdk.schemas.manifest import ModelManifest, RuntimeSpec
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

YOLOV8N_MANIFEST = ModelManifest(
    model_id="ultralytics/yolov8n",
    revision=f"{YOLOV8_REPO_REVISION}-{YOLOV8N_FILENAME.removesuffix('.pt')}",
    display_name="Ultralytics YOLOv8n",
    description="面向实时场景的轻量级 COCO 目标检测模型。",
    tags=frozenset({"vision", "object-detection", "yolo"}),
    family="yolov8",
    capabilities=frozenset({"object-detection"}),
    runtimes=(
        RuntimeSpec(
            name="pytorch-mps",
            devices=("mps", "cpu"),
            dtypes=("float32",),
        ),
        RuntimeSpec(
            name="coreml",
            devices=("all", "cpu-and-gpu", "cpu-and-neural-engine", "cpu"),
            dtypes=("float16", "float32"),
            quantizations=("float16", "int8"),
        ),
    ),
    resources=(
        HuggingFaceSource(
            repo_id=YOLOV8_REPO_ID,
            revision=YOLOV8_REPO_REVISION,
            filename=YOLOV8N_FILENAME,
            expected_sha256=YOLOV8N_SHA256,
        ),
    ),
    license="AGPL-3.0",
    source_url="https://huggingface.co/Ultralytics/YOLOv8",
    default_runtime="coreml",
)

def _create_yolov8(options: dict[str, object]) -> BaseModelInstance:
    config = YoloV8InstanceConfig.model_validate(options)
    source = cast(HuggingFaceSource, YOLOV8N_MANIFEST.resources[0])
    assets = YoloV8AssetResolver(source, config)
    if config.runtime == "pytorch-mps":
        return PyTorchMpsYoloV8Instance(config, assets)
    if config.runtime == "coreml":
        return CoreMlYoloV8Instance(config, assets)
    raise UnsupportedRuntimeError(f"Unsupported YOLOv8 runtime: {config.runtime}")


YOLOV8N_DEFINITION = ModelDefinition(
    manifest=YOLOV8N_MANIFEST,
    factory=_create_yolov8,
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
