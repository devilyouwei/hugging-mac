"""YOLOv8 Pose conversion defaults and model-specific selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hugging_mac_sdk.converters.yolov8 import YoloV8ExportConverter
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .config import YoloV8PoseCoreMlConfig
from .utils.checkpoint import load_yolov8_checkpoint


class YoloV8PoseConverter(YoloV8ExportConverter):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def converter_id(self) -> str:
        return "ultralytics.yolov8-pose"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        return (
            super().supports(request)
            and request.model_id == self._model_id
            and request.variant in {"n", "s", "m"}
            and request.source.path.name == f"yolov8{request.variant}-pose.pt"
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if request.target_format is ArtifactFormat.COREML:
            defaults = YoloV8PoseCoreMlConfig(variant=request.variant).model_dump()
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
