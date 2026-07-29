"""Pose-specific resource facade over the shared Ultralytics asset lifecycle."""

from __future__ import annotations

from collections.abc import Mapping

from hugging_mac_sdk.models._ultralytics_assets import (
    UltralyticsTaskAssetResolver,
    UltralyticsTaskResourceProvider,
)
from hugging_mac_sdk.models.yolov8_pose.config import (
    YOLOV8_POSE_MODEL_ID,
    YOLOV8_POSE_REVISION,
    YoloV8PoseInstanceConfig,
)
from hugging_mac_sdk.models.yolov8_pose.converter import YoloV8PoseConverter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.resources import ResourceSource


class YoloV8PoseAssetResolver(UltralyticsTaskAssetResolver):
    def __init__(
        self,
        source: ResourceSource,
        config: YoloV8PoseInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: YoloV8PoseConverter | None = None,
    ) -> None:
        super().__init__(
            model_id=YOLOV8_POSE_MODEL_ID,
            model_revision=YOLOV8_POSE_REVISION,
            source=source,
            config=config,
            downloader=downloader,
            converter=converter or YoloV8PoseConverter(),
        )


class YoloV8PoseResourceProvider(UltralyticsTaskResourceProvider):
    def __init__(self, sources: Mapping[str, ResourceSource]) -> None:
        super().__init__(
            model_id=YOLOV8_POSE_MODEL_ID,
            model_revision=YOLOV8_POSE_REVISION,
            sources=sources,
            config_factory=lambda values: YoloV8PoseInstanceConfig.model_validate(values),
            converter_factory=YoloV8PoseConverter,
        )
