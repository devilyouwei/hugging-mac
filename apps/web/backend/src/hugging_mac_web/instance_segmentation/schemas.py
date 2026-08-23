"""HTTP schemas owned by the YOLOv8 Seg App."""

from __future__ import annotations

from enum import StrEnum

from hugging_mac_sdk.schemas.detection import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
)
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from hugging_mac_sdk.schemas.segmentation import SegmentationResponse
from pydantic import BaseModel, ConfigDict, Field


class RuntimeChoice(StrEnum):
    AUTO = "auto"
    COREML = "coreml"
    PYTORCH_MPS = "pytorch-mps"
    ONNX = "onnx"


class SegmentationCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: RuntimeChoice = RuntimeChoice.AUTO
    variant: str | None = None
    confidence: float = Field(default=DEFAULT_DETECTION_CONFIDENCE, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_DETECTION_IOU_THRESHOLD, ge=0.0, le=1.0)
    max_detections: int = Field(default=DEFAULT_MAX_DETECTIONS, ge=1, le=1000)


class BoxView(BaseModel):
    model_config = ConfigDict(frozen=True)
    x1: float
    y1: float
    x2: float
    y2: float


class PolygonPointView(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: float
    y: float


class SegmentView(BaseModel):
    model_config = ConfigDict(frozen=True)
    box: BoxView
    confidence: float
    class_id: int
    label: str
    polygons: tuple[tuple[PolygonPointView, ...], ...]


class ImageSizeView(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int
    height: int


class TimingsView(BaseModel):
    model_config = ConfigDict(frozen=True)
    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None


class SegmentationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cache_id: str | None
    model_id: str
    variant: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSizeView
    segments: tuple[SegmentView, ...]
    timings: TimingsView

    @classmethod
    def from_sdk(
        cls,
        response: SegmentationResponse,
        *,
        input_cache_id: str | None,
        variant: str,
    ) -> SegmentationResult:
        return cls(
            input_cache_id=input_cache_id,
            model_id=response.model_id,
            variant=variant,
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            image_size=ImageSizeView(
                width=response.image_size.width,
                height=response.image_size.height,
            ),
            segments=tuple(
                SegmentView(
                    box=BoxView(
                        x1=segment.box.x1,
                        y1=segment.box.y1,
                        x2=segment.box.x2,
                        y2=segment.box.y2,
                    ),
                    confidence=segment.confidence,
                    class_id=segment.class_id,
                    label=segment.label,
                    polygons=tuple(
                        tuple(PolygonPointView(x=point.x, y=point.y) for point in polygon)
                        for polygon in segment.polygons
                    ),
                )
                for segment in response.segments
            ),
            timings=TimingsView(
                preprocess_ms=response.timings.preprocess_ms,
                inference_ms=response.timings.inference_ms,
                postprocess_ms=response.timings.postprocess_ms,
            ),
        )


class ArtifactResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)
    artifact_id: str
    format: str
    runtime: str | None
    available: bool
    size_bytes: int | None


class VariantView(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    display_name: str
    description: str
    default: bool


class ResourceStatusView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    default_runtime: str | None
    variants: tuple[VariantView, ...]
    artifacts: tuple[ArtifactResourceView, ...]

    @classmethod
    def from_sdk(
        cls,
        status: ModelResourceStatus,
        *,
        default_runtime: str | None,
        variants: tuple[VariantView, ...],
    ) -> ResourceStatusView:
        return cls(
            model_id=status.model_id,
            revision=status.revision,
            variant=status.variant,
            default_runtime=default_runtime,
            variants=variants,
            artifacts=tuple(
                ArtifactResourceView(
                    artifact_id=artifact.artifact_id,
                    format=artifact.format,
                    runtime=artifact.runtime,
                    available=artifact.available,
                    size_bytes=artifact.size_bytes,
                )
                for artifact in status.artifacts
            ),
        )
