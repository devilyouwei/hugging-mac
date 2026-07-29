"""YOLOv8 Seg conversion defaults and model-specific selection."""

from __future__ import annotations

from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.models.yolov8_seg.config import (
    YOLOV8_SEG_MODEL_ID,
    YoloV8SegCoreMlConfig,
)
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult


class YoloV8SegConverter(UltralyticsExportConverter):
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
            defaults = YoloV8SegCoreMlConfig(variant=request.variant).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            # Core ML only supports Ultralytics' embedded NMS pipeline for
            # detection models. Segmentation must preserve its raw mask
            # prototypes for task-specific post-processing.
            request = request.model_copy(
                update={"options": defaults | request.options | {"nms": False}}
            )
        return await super().convert(request)
