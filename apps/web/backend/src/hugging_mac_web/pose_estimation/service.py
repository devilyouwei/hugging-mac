"""YOLOv8 Pose business use case."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from contextlib import AsyncExitStack

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import FaceDetection, HandDetection, PoseEstimation
from hugging_mac_sdk.models.mediapipe_hand_detection.config import (
    MEDIAPIPE_HAND_DETECTION_MODEL_ID,
)
from hugging_mac_sdk.models.retinaface.config import RETINAFACE_MODEL_ID
from hugging_mac_sdk.models.yolov8_pose.config import YOLOV8_POSE_MODEL_ID
from hugging_mac_sdk.schemas.detection import (
    DetectionRequest,
    FaceDetectionResponse,
    ImageInput,
)
from hugging_mac_sdk.schemas.hand import HandDetectionRequest, HandDetectionResponse
from hugging_mac_sdk.schemas.pose import PoseEstimationResponse

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.pose_estimation.schemas import (
    ParallelTimingsView,
    PoseCommand,
    PoseResult,
    ResourceStatusView,
    RuntimeChoice,
    VariantView,
)


class PoseEstimationService:
    def __init__(self, context: PlatformContext) -> None:
        self._context = context

    async def resource_status(self, *, variant: str | None = None) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        status = await self._context.models.resources.status(
            YOLOV8_POSE_MODEL_ID,
            variant=selected_variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == YOLOV8_POSE_MODEL_ID),
            None,
        )
        return ResourceStatusView.from_sdk(
            status,
            default_runtime=model.default_runtime if model is not None else None,
            variants=tuple(
                VariantView(
                    name=item.name,
                    display_name=item.display_name,
                    description=item.description,
                    default=item.default,
                )
                for item in (model.variants if model is not None else ())
            ),
        )

    async def face_resource_status(self) -> ResourceStatusView:
        variant = "mobilenet0.25"
        status = await self._context.models.resources.status(
            RETINAFACE_MODEL_ID,
            variant=variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == RETINAFACE_MODEL_ID),
            None,
        )
        return ResourceStatusView.from_sdk(
            status,
            default_runtime=model.default_runtime if model is not None else "coreml",
            variants=tuple(
                VariantView(
                    name=item.name,
                    display_name=item.display_name,
                    description=item.description,
                    default=item.default,
                )
                for item in (model.variants if model is not None else ())
            ),
        )

    async def hand_resource_status(self) -> ResourceStatusView:
        variant = "float"
        status = await self._context.models.resources.status(
            MEDIAPIPE_HAND_DETECTION_MODEL_ID,
            variant=variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == MEDIAPIPE_HAND_DETECTION_MODEL_ID),
            None,
        )
        return ResourceStatusView.from_sdk(
            status,
            default_runtime="coreml",
            variants=tuple(
                VariantView(
                    name=item.name,
                    display_name=item.display_name,
                    description=item.description,
                    default=item.default,
                )
                for item in (model.variants if model is not None else ())
            ),
        )

    async def estimate(
        self,
        command: PoseCommand,
        image: bytes,
        *,
        filename: str,
        content_type: str | None,
        suffix: str,
        image_metadata: dict[str, object],
        cache_input: bool = True,
    ) -> PoseResult:
        if not command.pose_enabled and not command.face_enabled and not command.hand_enabled:
            raise ValueError("At least one pose, face or hand model must be enabled")
        input_cache_id: str | None = None
        if cache_input:
            cached = self._context.cache.put_bytes(
                "pose-estimation-inputs",
                image,
                suffix=suffix,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    **image_metadata,
                },
            )
            input_cache_id = cached.cache_id
        runtime = None if command.runtime is RuntimeChoice.AUTO else command.runtime.value
        # ``variant`` belongs to the pose model on this aggregate endpoint.  Hand-only
        # clients use the MediaPipe variant name (``float``), so do not validate it
        # against YOLO Pose when pose inference is disabled.
        selected_variant = (
            self._resolve_variant(command.variant) if command.pose_enabled else "n"
        )
        pose_response = None
        face_response = None
        hand_response = None
        async with AsyncExitStack() as stack:
            jobs: list[
                Awaitable[PoseEstimationResponse | FaceDetectionResponse | HandDetectionResponse]
            ] = []
            if command.pose_enabled:
                pose_handle = await stack.enter_async_context(
                    await self._context.models.acquire(
                        YOLOV8_POSE_MODEL_ID,
                        variant=selected_variant,
                        runtime=runtime,
                        options=self._model_options,
                        reuse=ReusePolicy.SHARED,
                    )
                )
                estimator = pose_handle.require(PoseEstimation)  # type: ignore[type-abstract]
                jobs.append(
                    estimator.estimate_pose(
                        DetectionRequest(
                            image=ImageInput(data=image),
                            confidence=command.confidence,
                            iou_threshold=command.iou_threshold,
                            max_detections=command.max_detections,
                        )
                    )
                )
            if command.face_enabled:
                face_handle = await stack.enter_async_context(
                    await self._context.models.acquire(
                        RETINAFACE_MODEL_ID,
                        variant="mobilenet0.25",
                        runtime=None,
                        options=self._model_options,
                        reuse=ReusePolicy.SHARED,
                    )
                )
                detector = face_handle.require(FaceDetection)  # type: ignore[type-abstract]
                jobs.append(
                    detector.detect_faces(
                        DetectionRequest(
                            image=ImageInput(data=image),
                            confidence=command.face_confidence,
                            iou_threshold=command.iou_threshold,
                            max_detections=command.max_detections,
                        )
                    )
                )
            if command.hand_enabled:
                hand_handle = await stack.enter_async_context(
                    await self._context.models.acquire(
                        MEDIAPIPE_HAND_DETECTION_MODEL_ID,
                        variant="float",
                        runtime=None,
                        options=self._model_options,
                        reuse=ReusePolicy.SHARED,
                    )
                )
                hand_detector = hand_handle.require(HandDetection)  # type: ignore[type-abstract]
                jobs.append(
                    hand_detector.detect_hands(
                        HandDetectionRequest(
                            image=ImageInput(data=image),
                            confidence=command.hand_confidence,
                            iou_threshold=command.iou_threshold,
                            max_detections=command.max_detections,
                            include_landmarks=command.hand_landmarks_enabled,
                            landmark_confidence=command.hand_landmark_confidence,
                            input_mirrored=command.hand_input_mirrored,
                        )
                    )
                )

            async def measure(
                job: Awaitable[
                    PoseEstimationResponse | FaceDetectionResponse | HandDetectionResponse
                ],
            ) -> tuple[
                PoseEstimationResponse | FaceDetectionResponse | HandDetectionResponse, float
            ]:
                started_at = time.perf_counter()
                result = await job
                return result, (time.perf_counter() - started_at) * 1000

            round_started_at = time.perf_counter()
            measured_results = await asyncio.gather(*(measure(job) for job in jobs))
            round_ms = (time.perf_counter() - round_started_at) * 1000
            pose_ms = None
            face_ms = None
            hand_ms = None
            for result, elapsed_ms in measured_results:
                if isinstance(result, PoseEstimationResponse):
                    pose_ms = elapsed_ms
                elif isinstance(result, FaceDetectionResponse):
                    face_ms = elapsed_ms
                else:
                    hand_ms = elapsed_ms
            parallel_timings = ParallelTimingsView(
                pose_ms=pose_ms,
                face_ms=face_ms,
                hand_ms=hand_ms,
                sum_ms=sum(value for value in (pose_ms, face_ms, hand_ms) if value is not None),
                round_ms=round_ms,
            )
            results = [result for result, _ in measured_results]
            for result in results:
                if isinstance(result, PoseEstimationResponse):
                    pose_response = result
                elif isinstance(result, FaceDetectionResponse):
                    face_response = result
                else:
                    hand_response = result

        if pose_response is not None:
            return PoseResult.from_sdk(
                pose_response,
                input_cache_id=input_cache_id,
                variant=selected_variant,
                parallel_timings=parallel_timings,
                face_response=face_response,
                hand_response=hand_response,
            )
        if face_response is not None:
            return PoseResult.from_face_sdk(
                face_response,
                input_cache_id=input_cache_id,
                parallel_timings=parallel_timings,
                hand_response=hand_response,
            )
        assert hand_response is not None
        return PoseResult.from_hand_sdk(
            hand_response,
            input_cache_id=input_cache_id,
            parallel_timings=parallel_timings,
        )

    def _resolve_variant(self, variant: str | None) -> str:
        return (
            self._context.models.registry.get(YOLOV8_POSE_MODEL_ID)
            .manifest.get_variant(variant)
            .name
        )

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
