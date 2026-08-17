"""Runtime-neutral hand detection and landmark schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.detection import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    BoundingBox,
    DetectionTimings,
    ImageInput,
    ImageSize,
)


class HandDetectionRequest(BaseModel):
    """Detect hands and optionally run the landmark submodel on every hand ROI."""

    model_config = ConfigDict(frozen=True)

    image: ImageInput
    confidence: float = Field(default=DEFAULT_DETECTION_CONFIDENCE, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_DETECTION_IOU_THRESHOLD, ge=0.0, le=1.0)
    max_detections: int = Field(default=DEFAULT_MAX_DETECTIONS, ge=1)
    include_landmarks: bool = False
    landmark_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    input_mirrored: bool = False


class HandLandmark(BaseModel):
    """One landmark with source-image pixel x/y and model-relative depth z."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float


class HandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    handedness: Literal["left", "right"] | None = None
    handedness_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    landmark_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    landmarks: tuple[HandLandmark, ...] = ()


class HandDetectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSize
    hands: tuple[HandResult, ...]
    landmarks_enabled: bool
    timings: DetectionTimings = DetectionTimings()
