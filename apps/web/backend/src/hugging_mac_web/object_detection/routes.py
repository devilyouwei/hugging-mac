"""Object Detection REST routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.object_detection.config import ObjectDetectionSettings
from hugging_mac_web.object_detection.schemas import (
    DetectCommand,
    DetectionResult,
    ResourceStatusView,
    RuntimeChoice,
)
from hugging_mac_web.object_detection.service import ObjectDetectionService
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.media_util import inspect_image, safe_suffix
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited


def create_router(settings: ObjectDetectionSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/apps/object-detection",
        tags=["object-detection"],
    )

    @router.get("/resources", response_model=ApiResponse[ResourceStatusView])
    async def resource_status(
        context: ContextDependency,
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ResourceStatusView]:
        data = await ObjectDetectionService(context, settings).resource_status(variant=variant)
        return ApiResponse(
            data=data,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/resources/source/download",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def download_source(
        context: ContextDependency,
        variant: str | None = Query(default=None),
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await ObjectDetectionService(context, settings).download_source(
            variant=variant,
            overwrite=overwrite,
        )
        return ApiResponse(
            data=data,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/resources/coreml/convert",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def convert_coreml(
        context: ContextDependency,
        variant: str | None = Query(default=None),
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await ObjectDetectionService(context, settings).convert_coreml(
            variant=variant,
            overwrite=overwrite,
        )
        return ApiResponse(
            data=data,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post("/detect", response_model=ApiResponse[DetectionResult])
    async def detect_objects(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
        runtime: Annotated[RuntimeChoice, Form()] = RuntimeChoice.AUTO,
        variant: Annotated[str | None, Form()] = None,
        confidence: Annotated[
            float,
            Form(ge=0.0, le=1.0),
        ] = settings.default_confidence,
        iou_threshold: Annotated[
            float,
            Form(ge=0.0, le=1.0),
        ] = settings.default_iou_threshold,
        max_detections: Annotated[
            int,
            Form(ge=1, le=1000),
        ] = settings.default_max_detections,
        cache_input: Annotated[
            bool,
            Form(description="Persist the input image in the local content cache"),
        ] = True,
    ) -> ApiResponse[DetectionResult]:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Object Detection supports JPEG, PNG and WebP images",
            )
        image = await read_upload_limited(file, context.settings.max_upload_bytes)
        try:
            image_metadata = inspect_image(
                image,
                max_pixels=context.settings.max_image_pixels,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        result = await ObjectDetectionService(context, settings).detect(
            DetectCommand(
                runtime=runtime,
                variant=variant,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            ),
            image,
            filename=file.filename or "detection-image",
            content_type=file.content_type,
            suffix=safe_suffix(file.filename or "image.bin"),
            image_metadata=image_metadata,
            cache_input=cache_input,
        )
        return ApiResponse(
            data=result,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post("/detect/frame", response_model=ApiResponse[DetectionResult])
    async def detect_latest_frame(
        context: ContextDependency,
        image: Annotated[
            bytes,
            Body(media_type="image/jpeg", description="Raw JPEG, PNG or WebP frame"),
        ],
        content_type: Annotated[str, Header(alias="Content-Type")],
        runtime: Annotated[RuntimeChoice, Query()] = RuntimeChoice.AUTO,
        variant: Annotated[str | None, Query()] = None,
        confidence: Annotated[
            float,
            Query(ge=0.0, le=1.0),
        ] = settings.default_confidence,
        iou_threshold: Annotated[
            float,
            Query(ge=0.0, le=1.0),
        ] = settings.default_iou_threshold,
        max_detections: Annotated[
            int,
            Query(ge=1, le=1000),
        ] = settings.default_max_detections,
    ) -> ApiResponse[DetectionResult]:
        normalized_content_type = content_type.split(";", maxsplit=1)[0]
        if normalized_content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Live detection supports JPEG, PNG and WebP frames",
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
        result = await ObjectDetectionService(context, settings).detect(
            DetectCommand(
                runtime=runtime,
                variant=variant,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            ),
            image,
            filename="latest-frame.jpg",
            content_type=normalized_content_type,
            suffix=".jpg",
            image_metadata=image_metadata,
            cache_input=False,
        )
        return ApiResponse(
            data=result,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
