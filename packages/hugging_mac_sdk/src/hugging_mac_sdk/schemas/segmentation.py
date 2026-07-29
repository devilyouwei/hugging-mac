"""Runtime-neutral instance segmentation schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    DetectionRequest,
    DetectionTimings,
    ImageSize,
)

SegmentationRequest = DetectionRequest


class PolygonPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class Segmentation(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    label: str
    polygons: tuple[tuple[PolygonPoint, ...], ...] = ()


class SegmentationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSize
    segments: tuple[Segmentation, ...]
    timings: DetectionTimings = DetectionTimings()
