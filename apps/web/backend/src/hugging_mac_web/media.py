"""Shared media upload endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.media_util import (
    classify_media,
    inspect_image,
    safe_suffix,
)
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited


class MediaUploadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_id: str
    digest: str
    media_type: str
    content_type: str | None
    filename: str
    size_bytes: int
    metadata: dict[str, Any]


def create_media_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/media", tags=["media"])

    @router.post(
        "/uploads",
        response_model=ApiResponse[MediaUploadResult],
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_media(
        file: Annotated[UploadFile, File()],
        context: ContextDependency,
    ) -> ApiResponse[MediaUploadResult]:
        data = await read_upload_limited(file, context.settings.max_image_upload_bytes)
        filename = file.filename or "upload.bin"
        media_type = classify_media(filename, file.content_type)
        if media_type == "unknown":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported media type",
            )
        metadata: dict[str, Any] = {}
        if media_type == "image":
            try:
                metadata = inspect_image(
                    data,
                    max_pixels=context.settings.max_image_pixels,
                )
            except ValueError as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(error),
                ) from error
        entry = context.cache.put_bytes(
            "uploads",
            data,
            suffix=safe_suffix(filename),
            metadata={
                "filename": filename,
                "content_type": file.content_type,
                "media_type": media_type,
                **metadata,
            },
        )
        return ApiResponse(
            data=MediaUploadResult(
                cache_id=entry.cache_id,
                digest=entry.digest,
                media_type=media_type,
                content_type=file.content_type,
                filename=filename,
                size_bytes=entry.size_bytes,
                metadata=metadata,
            ),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
