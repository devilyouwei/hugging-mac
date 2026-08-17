"""YOLOv8 Pose REST routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Query, UploadFile, status
from hugging_mac_sdk.schemas.detection import (
    DEFAULT_DETECTION_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
)

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.pose_estimation.schemas import (
    PoseCommand,
    PoseResult,
    ResourceStatusView,
    RuntimeChoice,
)
from hugging_mac_web.pose_estimation.service import PoseEstimationService
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.media_util import inspect_image, safe_suffix
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited

SUPPORTED_IMAGES = {"image/jpeg", "image/png", "image/webp"}


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/apps/pose-estimation", tags=["pose-estimation"])

    @router.get("/resources", response_model=ApiResponse[ResourceStatusView])
    async def resource_status(
        context: ContextDependency,
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ResourceStatusView]:
        data = await PoseEstimationService(context).resource_status(variant=variant)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.get("/resources/face", response_model=ApiResponse[ResourceStatusView])
    async def face_resource_status(
        context: ContextDependency,
    ) -> ApiResponse[ResourceStatusView]:
        data = await PoseEstimationService(context).face_resource_status()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.get("/resources/hand", response_model=ApiResponse[ResourceStatusView])
    async def hand_resource_status(
        context: ContextDependency,
    ) -> ApiResponse[ResourceStatusView]:
        data = await PoseEstimationService(context).hand_resource_status()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/estimate", response_model=ApiResponse[PoseResult])
    async def estimate_pose(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
        runtime: Annotated[RuntimeChoice, Form()] = RuntimeChoice.AUTO,
        variant: Annotated[str | None, Form()] = None,
        confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.65,
        iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_DETECTION_IOU_THRESHOLD,
        max_detections: Annotated[int, Form(ge=1, le=1000)] = DEFAULT_MAX_DETECTIONS,
        cache_input: Annotated[bool, Form()] = True,
        pose_enabled: Annotated[bool, Form()] = True,
        face_enabled: Annotated[bool, Form()] = True,
        face_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.90,
        hand_enabled: Annotated[bool, Form()] = True,
        hand_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.85,
        hand_landmarks_enabled: Annotated[bool, Form()] = False,
        hand_landmark_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
        hand_input_mirrored: Annotated[bool, Form()] = False,
    ) -> ApiResponse[PoseResult]:
        if not pose_enabled and not face_enabled and not hand_enabled:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one pose, face or hand model must be enabled",
            )
        if file.content_type not in SUPPORTED_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Pose Estimation supports JPEG, PNG and WebP images",
            )
        image = await read_upload_limited(file, context.settings.max_upload_bytes)
        try:
            image_metadata = inspect_image(image, max_pixels=context.settings.max_image_pixels)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        result = await PoseEstimationService(context).estimate(
            PoseCommand(
                runtime=runtime,
                variant=variant,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                pose_enabled=pose_enabled,
                face_enabled=face_enabled,
                face_confidence=face_confidence,
                hand_enabled=hand_enabled,
                hand_confidence=hand_confidence,
                hand_landmarks_enabled=hand_landmarks_enabled,
                hand_landmark_confidence=hand_landmark_confidence,
                hand_input_mirrored=hand_input_mirrored,
            ),
            image,
            filename=file.filename or "pose-image",
            content_type=file.content_type,
            suffix=safe_suffix(file.filename or "image.bin"),
            image_metadata=image_metadata,
            cache_input=cache_input,
        )
        return ApiResponse(data=result, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/estimate/frame", response_model=ApiResponse[PoseResult])
    async def estimate_frame(
        context: ContextDependency,
        image: Annotated[bytes, Body(media_type="image/jpeg")],
        content_type: Annotated[str, Header(alias="Content-Type")],
        runtime: Annotated[RuntimeChoice, Query()] = RuntimeChoice.AUTO,
        variant: Annotated[str | None, Query()] = None,
        confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.65,
        iou_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = DEFAULT_DETECTION_IOU_THRESHOLD,
        max_detections: Annotated[int, Query(ge=1, le=1000)] = DEFAULT_MAX_DETECTIONS,
        pose_enabled: Annotated[bool, Query()] = True,
        face_enabled: Annotated[bool, Query()] = True,
        face_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.90,
        hand_enabled: Annotated[bool, Query()] = True,
        hand_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.85,
        hand_landmarks_enabled: Annotated[bool, Query()] = False,
        hand_landmark_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
        hand_input_mirrored: Annotated[bool, Query()] = False,
    ) -> ApiResponse[PoseResult]:
        if not pose_enabled and not face_enabled and not hand_enabled:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one pose, face or hand model must be enabled",
            )
        normalized_content_type = content_type.split(";", maxsplit=1)[0]
        if normalized_content_type not in SUPPORTED_IMAGES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Live pose estimation supports JPEG, PNG and WebP frames",
            )
        if len(image) > context.settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Frame exceeds {context.settings.max_upload_bytes} bytes",
            )
        try:
            image_metadata = inspect_image(
                image,
                max_pixels=context.settings.max_image_pixels,
                verify=False,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        result = await PoseEstimationService(context).estimate(
            PoseCommand(
                runtime=runtime,
                variant=variant,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                pose_enabled=pose_enabled,
                face_enabled=face_enabled,
                face_confidence=face_confidence,
                hand_enabled=hand_enabled,
                hand_confidence=hand_confidence,
                hand_landmarks_enabled=hand_landmarks_enabled,
                hand_landmark_confidence=hand_landmark_confidence,
                hand_input_mirrored=hand_input_mirrored,
            ),
            image,
            filename="latest-frame.jpg",
            content_type=normalized_content_type,
            suffix=".jpg",
            image_metadata=image_metadata,
            cache_input=False,
        )
        return ApiResponse(data=result, meta=ResponseMeta(generated_at=utc_now()))

    return router
