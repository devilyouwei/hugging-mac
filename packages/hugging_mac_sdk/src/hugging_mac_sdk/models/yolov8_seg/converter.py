"""YOLOv8 Seg conversion defaults and model-specific selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.converters.yolov8 import YoloV8ExportConverter
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .config import (
    YOLOV8_SEG_MODEL_ID,
    YoloV8SegCoreMlConfig,
    YoloV8SegVariant,
)
from .utils.checkpoint import load_yolov8_checkpoint


class YoloV8SegConverter(YoloV8ExportConverter):
    @property
    def converter_id(self) -> str:
        return "ultralytics.yolov8-seg"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            super().supports(request)
            and request.model_id == YOLOV8_SEG_MODEL_ID
            and request.variant in {"n", "s", "m"}
            and request.source.path.name == f"yolov8{request.variant}-seg.pt"
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            defaults = YoloV8SegCoreMlConfig(
                variant=cast(YoloV8SegVariant, request.variant)
            ).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            # Core ML only supports Ultralytics' embedded NMS pipeline for
            # detection models. Segmentation must preserve its raw mask
            # prototypes for task-specific post-processing.
            request = request.model_copy(
                update={"options": defaults | request.options | {"nms": False}}
            )
        return await super().convert(request)

    def _load_checkpoint(self, source: Path, torch: Any) -> Any:
        return load_yolov8_checkpoint(source, torch)
