"""Runtime-neutral human pose estimation schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    DetectionRequest,
    DetectionTimings,
    ImageSize,
)

PoseRequest = DetectionRequest


class Keypoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Pose(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    label: str
    keypoints: tuple[Keypoint, ...]


class PoseEstimationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSize
    poses: tuple[Pose, ...]
    timings: DetectionTimings = DetectionTimings()
