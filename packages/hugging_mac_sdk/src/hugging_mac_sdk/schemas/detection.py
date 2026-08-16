"""Runtime-neutral object detection schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_DETECTION_CONFIDENCE = 0.5
DEFAULT_DETECTION_IOU_THRESHOLD = 0.7
DEFAULT_MAX_DETECTIONS = 300


class ImageInput(BaseModel):
    """An image supplied as a local path or encoded bytes."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path | None = None
    data: bytes | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> ImageInput:
        if (self.path is None) == (self.data is None):
            raise ValueError("Exactly one of path or data must be provided")
        return self


class DetectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: ImageInput
    confidence: float = Field(default=DEFAULT_DETECTION_CONFIDENCE, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_DETECTION_IOU_THRESHOLD, ge=0.0, le=1.0)
    max_detections: int = Field(default=DEFAULT_MAX_DETECTIONS, ge=1)
    classes: tuple[int, ...] | None = None


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box maximums must be >= minimums")
        return self


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    label: str


class ImageSize(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectionTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float | None = Field(default=None, ge=0.0)
    inference_ms: float | None = Field(default=None, ge=0.0)
    postprocess_ms: float | None = Field(default=None, ge=0.0)


class DetectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSize
    detections: tuple[Detection, ...]
    timings: DetectionTimings = DetectionTimings()
