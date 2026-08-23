"""YOLOv8-specific conversion defaults and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hugging_mac_sdk.converters.yolov8 import YoloV8ExportConverter
from hugging_mac_sdk.core.config import ModelPackageConfig
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)

from .config import YoloV8CoreMlConfig
from .utils.checkpoint import load_yolov8_checkpoint


class YoloV8Converter(YoloV8ExportConverter):
    def __init__(self, package: ModelPackageConfig) -> None:
        self._package = package

    @property
    def converter_id(self) -> str:
        return "ultralytics.yolov8"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        source = self._package.get_artifact(variant=request.variant, artifact_id="source")
        return (
            super().supports(request)
            and request.model_id == self._package.manifest.model_id
            and request.source.path.name == source.path.name
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            defaults = YoloV8CoreMlConfig(variant=request.variant).model_dump()
            defaults.pop("compute_units")
            defaults.pop("variant")
            request = request.model_copy(
                update={"options": defaults | request.options | {"nms": False}}
            )
        return await super().convert(request)

    def _load_checkpoint(self, source: Path, torch: Any) -> Any:
        return load_yolov8_checkpoint(source, torch)
