"""Runtime-neutral document layout analysis schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    DetectionTimings,
    ImageInput,
    ImageSize,
    Point2D,
)


class DocumentLayoutRequest(BaseModel):
    """Locate document regions and recover their reading order."""

    model_config = ConfigDict(frozen=True)

    image: ImageInput
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_regions: int = Field(default=300, ge=1, le=300)
    include_polygons: bool = True


class DocumentLayoutRegion(BaseModel):
    """One source-image region in model-predicted reading order."""

    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    polygon: tuple[Point2D, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    label: str
    order: int = Field(ge=0)


class DocumentLayoutResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSize
    regions: tuple[DocumentLayoutRegion, ...]
    timings: DetectionTimings = DetectionTimings()
