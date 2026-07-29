"""Segmentation-specific resource facade over the shared Ultralytics asset lifecycle."""

from __future__ import annotations

from collections.abc import Mapping

from hugging_mac_sdk.models._ultralytics_assets import (
    UltralyticsTaskAssetResolver,
    UltralyticsTaskResourceProvider,
)
from hugging_mac_sdk.models.yolov8_seg.config import (
    YOLOV8_SEG_MODEL_ID,
    YOLOV8_SEG_REVISION,
    YoloV8SegInstanceConfig,
)
from hugging_mac_sdk.models.yolov8_seg.converter import YoloV8SegConverter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.resources import ResourceSource


class YoloV8SegAssetResolver(UltralyticsTaskAssetResolver):
    def __init__(
        self,
        source: ResourceSource,
        config: YoloV8SegInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: YoloV8SegConverter | None = None,
    ) -> None:
        super().__init__(
            model_id=YOLOV8_SEG_MODEL_ID,
            model_revision=YOLOV8_SEG_REVISION,
            source=source,
            config=config,
            downloader=downloader,
            converter=converter or YoloV8SegConverter(),
        )


class YoloV8SegResourceProvider(UltralyticsTaskResourceProvider):
    def __init__(self, sources: Mapping[str, ResourceSource]) -> None:
        super().__init__(
            model_id=YOLOV8_SEG_MODEL_ID,
            model_revision=YOLOV8_SEG_REVISION,
            sources=sources,
            config_factory=lambda values: YoloV8SegInstanceConfig.model_validate(values),
            converter_factory=YoloV8SegConverter,
        )
