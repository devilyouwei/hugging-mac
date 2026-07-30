"""YOLOv8-specific conversion defaults and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.converters.yolov8 import YoloV8ExportConverter
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)

from .config import YOLOV8_FILENAMES, YOLOV8_MODEL_ID, YoloV8CoreMlConfig, YoloV8Variant
from .utils.checkpoint import load_yolov8_checkpoint


class YoloV8Converter(YoloV8ExportConverter):
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
            defaults = YoloV8CoreMlConfig(variant=cast(YoloV8Variant, request.variant)).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            request = request.model_copy(
                update={"options": defaults | request.options | {"nms": False}}
            )
        return await super().convert(request)

    def _load_checkpoint(self, source: Path, torch: Any) -> Any:
        return load_yolov8_checkpoint(source, torch)
