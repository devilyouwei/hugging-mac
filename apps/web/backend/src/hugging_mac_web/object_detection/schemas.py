"""HTTP schemas owned by the Object Detection App."""

from __future__ import annotations

from enum import StrEnum

from hugging_mac_sdk.schemas.detection import DetectionResponse
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from pydantic import BaseModel, ConfigDict, Field


class RuntimeChoice(StrEnum):
    AUTO = "auto"
    COREML = "coreml"
    PYTORCH_MPS = "pytorch-mps"


class DetectCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: RuntimeChoice = RuntimeChoice.AUTO
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)


class BoxView(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionView(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: BoxView
    confidence: float
    class_id: int
    label: str


class ImageSizeView(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int
    height: int


class TimingsView(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None


class DetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cache_id: str | None
    model_id: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSizeView
    detections: tuple[DetectionView, ...]
    timings: TimingsView

    @classmethod
    def from_sdk(
        cls,
        response: DetectionResponse,
        *,
        input_cache_id: str | None,
    ) -> DetectionResult:
        return cls(
            input_cache_id=input_cache_id,
            model_id=response.model_id,
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            image_size=ImageSizeView(
                width=response.image_size.width,
                height=response.image_size.height,
            ),
            detections=tuple(
                DetectionView(
                    box=BoxView(
                        x1=detection.box.x1,
                        y1=detection.box.y1,
                        x2=detection.box.x2,
                        y2=detection.box.y2,
                    ),
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    label=detection.label,
                )
                for detection in response.detections
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


class ResourceStatusView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    default_runtime: str | None
    artifacts: tuple[ArtifactResourceView, ...]

    @classmethod
    def from_sdk(
        cls,
        status: ModelResourceStatus,
        *,
        default_runtime: str | None,
    ) -> ResourceStatusView:
        return cls(
            model_id=status.model_id,
            revision=status.revision,
            default_runtime=default_runtime,
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
