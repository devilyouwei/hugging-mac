"""Live transcription REST routes."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.schemas import (
    ResourceStatusView,
    TranscriptionResultView,
)
from hugging_mac_web.live_transcription.service import LiveTranscriptionService
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited

SUPPORTED_AUDIO = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
}


def create_router(settings: LiveTranscriptionSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/apps/live-transcription",
        tags=["live-transcription"],
    )

    @router.get("/resources", response_model=ApiResponse[ResourceStatusView])
    async def resource_status(
        context: ContextDependency,
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).resource_status()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post(
        "/resources/source/download",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def download_source(
        context: ContextDependency,
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).download_source(
            overwrite=overwrite
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post(
        "/resources/coreml/convert",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def convert_coreml(
        context: ContextDependency,
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).convert_coreml(
            overwrite=overwrite
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/transcribe", response_model=ApiResponse[TranscriptionResultView])
    async def transcribe(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="A short WAV or FLAC utterance")],
    ) -> ApiResponse[TranscriptionResultView]:
        if file.content_type not in SUPPORTED_AUDIO:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Live Transcription supports WAV and FLAC audio",
            )
        audio = await read_upload_limited(file, context.settings.max_upload_bytes)
        data = await LiveTranscriptionService(context, settings).transcribe(audio)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    return router
