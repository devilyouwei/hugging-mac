"""HTTP schemas owned by the YOLOv8 Pose App."""

from __future__ import annotations

from enum import StrEnum

from hugging_mac_sdk.schemas.detection import (
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    FaceDetectionResponse,
    Point2D,
)
from hugging_mac_sdk.schemas.hand import HandDetectionResponse
from hugging_mac_sdk.schemas.pose import PoseEstimationResponse
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from pydantic import BaseModel, ConfigDict, Field


class RuntimeChoice(StrEnum):
    AUTO = "auto"
    COREML = "coreml"
    PYTORCH_MPS = "pytorch-mps"


class PoseCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: RuntimeChoice = RuntimeChoice.AUTO
    variant: str | None = None
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_DETECTION_IOU_THRESHOLD, ge=0.0, le=1.0)
    max_detections: int = Field(default=DEFAULT_MAX_DETECTIONS, ge=1, le=1000)
    pose_enabled: bool = True
    face_enabled: bool = True
    face_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    hand_enabled: bool = True
    hand_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    hand_landmarks_enabled: bool = False
    hand_landmark_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hand_input_mirrored: bool = False


class BoxView(BaseModel):
    model_config = ConfigDict(frozen=True)
    x1: float
    y1: float
    x2: float
    y2: float


class KeypointView(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: float
    y: float
    confidence: float | None


class PoseView(BaseModel):
    model_config = ConfigDict(frozen=True)
    box: BoxView
    confidence: float
    class_id: int
    label: str
    keypoints: tuple[KeypointView, ...]


class FaceLandmarksView(BaseModel):
    model_config = ConfigDict(frozen=True)
    left_eye: KeypointView
    right_eye: KeypointView
    nose: KeypointView
    left_mouth: KeypointView
    right_mouth: KeypointView


class FaceView(BaseModel):
    model_config = ConfigDict(frozen=True)
    box: BoxView
    confidence: float
    landmarks: FaceLandmarksView


class HandLandmarkView(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: float
    y: float
    z: float


class HandView(BaseModel):
    model_config = ConfigDict(frozen=True)
    box: BoxView
    confidence: float
    label: str
    handedness: str | None = None
    handedness_confidence: float | None = None
    landmark_confidence: float | None = None
    landmarks: tuple[HandLandmarkView, ...] = ()


class ImageSizeView(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int
    height: int


class TimingsView(BaseModel):
    model_config = ConfigDict(frozen=True)
    preprocess_ms: float | None
    inference_ms: float | None
    postprocess_ms: float | None


class ParallelTimingsView(BaseModel):
    model_config = ConfigDict(frozen=True)
    pose_ms: float | None
    face_ms: float | None
    hand_ms: float | None
    sum_ms: float
    round_ms: float


class PoseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cache_id: str | None
    model_id: str
    variant: str
    instance_id: str
    runtime: str
    device: str
    image_size: ImageSizeView
    poses: tuple[PoseView, ...]
    faces: tuple[FaceView, ...] = ()
    hands: tuple[HandView, ...] = ()
    timings: TimingsView
    parallel_timings: ParallelTimingsView

    @classmethod
    def from_sdk(
        cls,
        response: PoseEstimationResponse,
        *,
        input_cache_id: str | None,
        variant: str,
        parallel_timings: ParallelTimingsView,
        face_response: FaceDetectionResponse | None = None,
        hand_response: HandDetectionResponse | None = None,
    ) -> PoseResult:
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
            poses=tuple(
                PoseView(
                    box=BoxView(
                        x1=pose.box.x1,
                        y1=pose.box.y1,
                        x2=pose.box.x2,
                        y2=pose.box.y2,
                    ),
                    confidence=pose.confidence,
                    class_id=pose.class_id,
                    label=pose.label,
                    keypoints=tuple(
                        KeypointView(
                            x=keypoint.x,
                            y=keypoint.y,
                            confidence=keypoint.confidence,
                        )
                        for keypoint in pose.keypoints
                    ),
                )
                for pose in response.poses
            ),
            faces=cls._faces(face_response),
            hands=cls._hands(hand_response),
            timings=TimingsView(
                preprocess_ms=response.timings.preprocess_ms,
                inference_ms=response.timings.inference_ms,
                postprocess_ms=response.timings.postprocess_ms,
            ),
            parallel_timings=parallel_timings,
        )

    @classmethod
    def from_face_sdk(
        cls,
        response: FaceDetectionResponse,
        *,
        input_cache_id: str | None,
        parallel_timings: ParallelTimingsView,
        hand_response: HandDetectionResponse | None = None,
    ) -> PoseResult:
        return cls(
            input_cache_id=input_cache_id,
            model_id=response.model_id,
            variant="mobilenet0.25",
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            image_size=ImageSizeView(
                width=response.image_size.width,
                height=response.image_size.height,
            ),
            poses=(),
            faces=cls._faces(response),
            hands=cls._hands(hand_response),
            timings=TimingsView(
                preprocess_ms=response.timings.preprocess_ms,
                inference_ms=response.timings.inference_ms,
                postprocess_ms=response.timings.postprocess_ms,
            ),
            parallel_timings=parallel_timings,
        )

    @classmethod
    def from_hand_sdk(
        cls,
        response: HandDetectionResponse,
        *,
        input_cache_id: str | None,
        parallel_timings: ParallelTimingsView,
    ) -> PoseResult:
        return cls(
            input_cache_id=input_cache_id,
            model_id=response.model_id,
            variant="float",
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            image_size=ImageSizeView(
                width=response.image_size.width,
                height=response.image_size.height,
            ),
            poses=(),
            faces=(),
            hands=cls._hands(response),
            timings=TimingsView(
                preprocess_ms=response.timings.preprocess_ms,
                inference_ms=response.timings.inference_ms,
                postprocess_ms=response.timings.postprocess_ms,
            ),
            parallel_timings=parallel_timings,
        )

    @staticmethod
    def _faces(response: FaceDetectionResponse | None) -> tuple[FaceView, ...]:
        if response is None:
            return ()

        def point(value: Point2D) -> KeypointView:
            return KeypointView(
                x=value.x,
                y=value.y,
                confidence=None,
            )

        return tuple(
            FaceView(
                box=BoxView(
                    x1=face.box.x1,
                    y1=face.box.y1,
                    x2=face.box.x2,
                    y2=face.box.y2,
                ),
                confidence=face.confidence,
                landmarks=FaceLandmarksView(
                    left_eye=point(face.landmarks.left_eye),
                    right_eye=point(face.landmarks.right_eye),
                    nose=point(face.landmarks.nose),
                    left_mouth=point(face.landmarks.left_mouth),
                    right_mouth=point(face.landmarks.right_mouth),
                ),
            )
            for face in response.faces
        )

    @staticmethod
    def _hands(response: HandDetectionResponse | None) -> tuple[HandView, ...]:
        if response is None:
            return ()
        return tuple(
            HandView(
                box=BoxView(
                    x1=hand.box.x1,
                    y1=hand.box.y1,
                    x2=hand.box.x2,
                    y2=hand.box.y2,
                ),
                confidence=hand.confidence,
                label="hand",
                handedness=hand.handedness,
                handedness_confidence=hand.handedness_confidence,
                landmark_confidence=hand.landmark_confidence,
                landmarks=tuple(
                    HandLandmarkView(x=point.x, y=point.y, z=point.z) for point in hand.landmarks
                ),
            )
            for hand in response.hands
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
