"""YOLOv8 Pose conversion defaults and model-specific selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.converters.yolov8 import YoloV8ExportConverter
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .config import (
    YOLOV8_POSE_MODEL_ID,
    YoloV8PoseCoreMlConfig,
    YoloV8PoseVariant,
)
from .utils.checkpoint import load_yolov8_checkpoint


class YoloV8PoseConverter(YoloV8ExportConverter):
    @property
    def converter_id(self) -> str:
        return "ultralytics.yolov8-pose"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            super().supports(request)
            and request.model_id == YOLOV8_POSE_MODEL_ID
            and request.variant in {"n", "s", "m"}
            and request.source.path.name == f"yolov8{request.variant}-pose.pt"
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            defaults = YoloV8PoseCoreMlConfig(
                variant=cast(YoloV8PoseVariant, request.variant)
            ).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            # Core ML only supports Ultralytics' embedded NMS pipeline for
            # detection models. Keep Pose outputs raw so keypoint decoding
            # remains available to the SDK runtime.
            request = request.model_copy(
                update={"options": defaults | request.options | {"nms": False}}
            )
        return await super().convert(request)

    def _load_checkpoint(self, source: Path, torch: Any) -> Any:
        return load_yolov8_checkpoint(source, torch)
