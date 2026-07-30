"""Live transcription REST routes."""

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.schemas import (
    AsrModelView,
    LoadedModelView,
    LoadModelRequest,
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
DEFAULT_MODEL_ID = "audio8/audio8-asr-0.1b"


def create_router(settings: LiveTranscriptionSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/apps/live-transcription",
        tags=["live-transcription"],
    )

    @router.get("/models", response_model=ApiResponse[tuple[AsrModelView, ...]])
    async def models(
        context: ContextDependency,
    ) -> ApiResponse[tuple[AsrModelView, ...]]:
        data = await LiveTranscriptionService(context, settings).list_models()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.get("/resources", response_model=ApiResponse[ResourceStatusView])
    async def resource_status(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).resource_status(model_id)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post(
        "/resources/source/download",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def download_source(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).download_source(
            model_id,
            overwrite=overwrite,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post(
        "/resources/coreml/convert",
        response_model=ApiResponse[ResourceStatusView],
    )
    async def convert_coreml(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
        overwrite: bool = False,
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).convert_coreml(
            model_id,
            overwrite=overwrite,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/models/load", response_model=ApiResponse[LoadedModelView])
    async def load_model(
        request: LoadModelRequest,
        context: ContextDependency,
    ) -> ApiResponse[LoadedModelView]:
        data = await LiveTranscriptionService(context, settings).load_model(request.model_id)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/transcribe", response_model=ApiResponse[TranscriptionResultView])
    async def transcribe(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="A short WAV or FLAC utterance")],
        model_id: Annotated[str, Form()],
        instance_id: Annotated[str, Form()],
    ) -> ApiResponse[TranscriptionResultView]:
        if file.content_type not in SUPPORTED_AUDIO:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Live Transcription supports WAV and FLAC audio",
            )
        audio = await read_upload_limited(file, context.settings.max_upload_bytes)
        data = await LiveTranscriptionService(context, settings).transcribe(
            audio,
            model_id=model_id,
            instance_id=instance_id,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    return router
