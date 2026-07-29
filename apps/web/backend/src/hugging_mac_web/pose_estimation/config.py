"""Typed YOLOv8 Pose App configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoseEstimationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_POSE_ESTIMATION_",
        env_file=".env",
        extra="ignore",
    )

    model_id: str = "ultralytics/yolov8-pose"
    model_variant: str = "n"
    default_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    default_iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    default_max_detections: int = Field(default=100, ge=1, le=1000)
