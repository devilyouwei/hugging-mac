"""YOLOv8-specific conversion defaults and validation."""

from __future__ import annotations

from typing import cast

from hugging_mac_sdk.converters.ultralytics import UltralyticsExportConverter
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)

from .config import YoloV8CoreMlConfig
from .config import YOLOV8_FILENAMES, YOLOV8_MODEL_ID, YoloV8Variant


class YoloV8Converter(UltralyticsExportConverter):
    @property
    def converter_id(self) -> str:
        return "ultralytics.yolov8"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        variant = cast(YoloV8Variant, request.variant)
        return (
            super().supports(request)
            and request.model_id == YOLOV8_MODEL_ID
            and request.variant in YOLOV8_FILENAMES
            and request.source.path.name == YOLOV8_FILENAMES[variant]
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            defaults = YoloV8CoreMlConfig(
                variant=cast(YoloV8Variant, request.variant)
            ).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            request = request.model_copy(update={"options": defaults | request.options})
        return await super().convert(request)
