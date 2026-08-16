"""Live transcription REST routes."""

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.schemas import (
    AsrModelView,
    LoadedModelView,
    LoadModelRequest,
    PipelineComponentView,
    ResourceStatusView,
    StreamingFinishRequest,
    StreamingSessionRequest,
    StreamingSessionView,
    StreamingTranscriptionView,
    TranscriptionResultView,
    VadDetectionView,
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
DEFAULT_MODEL_ID = "audio8/audio8-asr"


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

    @router.get(
        "/pipeline/components",
        response_model=ApiResponse[tuple[PipelineComponentView, ...]],
    )
    async def pipeline_components(
        context: ContextDependency,
    ) -> ApiResponse[tuple[PipelineComponentView, ...]]:
        data = await LiveTranscriptionService(context, settings).pipeline_components()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/models/load", response_model=ApiResponse[LoadedModelView])
    async def load_model(
        request: LoadModelRequest,
        context: ContextDependency,
    ) -> ApiResponse[LoadedModelView]:
        data = await LiveTranscriptionService(context, settings).load_model(
            request.model_id,
            runtime=request.runtime,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/vad/model/load", response_model=ApiResponse[LoadedModelView])
    async def load_vad_model(
        context: ContextDependency,
    ) -> ApiResponse[LoadedModelView]:
        data = await LiveTranscriptionService(context, settings).load_vad()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/enhancement/model/load", response_model=ApiResponse[LoadedModelView])
    async def load_enhancement_model(
        context: ContextDependency,
    ) -> ApiResponse[LoadedModelView]:
        data = await LiveTranscriptionService(context, settings).load_enhancement()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/vad/detect", response_model=ApiResponse[VadDetectionView])
    async def detect_voice_activity(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="A short WAV VAD window")],
        instance_id: Annotated[str, Form()],
        threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    ) -> ApiResponse[VadDetectionView]:
        if file.content_type not in SUPPORTED_AUDIO:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Silero VAD supports WAV and FLAC audio",
            )
        audio = await read_upload_limited(file, context.settings.max_upload_bytes)
        data = await LiveTranscriptionService(context, settings).detect_voice_activity(
            audio,
            instance_id=instance_id,
            threshold=threshold,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/transcribe", response_model=ApiResponse[TranscriptionResultView])
    async def transcribe(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="A short WAV or FLAC utterance")],
        model_id: Annotated[str, Form()],
        instance_id: Annotated[str, Form()],
        vad_instance_id: Annotated[str | None, Form()] = None,
        enhancement_instance_id: Annotated[str | None, Form()] = None,
        vad_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
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
            vad_instance_id=vad_instance_id,
            enhancement_instance_id=enhancement_instance_id,
            vad_threshold=vad_threshold,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/stream/start", response_model=ApiResponse[StreamingSessionView])
    async def start_stream(
        request: StreamingSessionRequest,
        context: ContextDependency,
    ) -> ApiResponse[StreamingSessionView]:
        data = await LiveTranscriptionService(context, settings).start_stream(
            model_id=request.model_id,
            instance_id=request.instance_id,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/stream/chunk", response_model=ApiResponse[StreamingTranscriptionView])
    async def stream_chunk(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="A consecutive streaming WAV chunk")],
        model_id: Annotated[str, Form()],
        instance_id: Annotated[str, Form()],
        session_id: Annotated[str, Form()],
    ) -> ApiResponse[StreamingTranscriptionView]:
        if file.content_type not in SUPPORTED_AUDIO:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Streaming transcription supports WAV and FLAC audio",
            )
        audio = await read_upload_limited(file, context.settings.max_upload_bytes)
        data = await LiveTranscriptionService(context, settings).transcribe_stream_chunk(
            audio,
            model_id=model_id,
            instance_id=instance_id,
            session_id=session_id,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/stream/finish", response_model=ApiResponse[StreamingTranscriptionView])
    async def finish_stream(
        request: StreamingFinishRequest,
        context: ContextDependency,
    ) -> ApiResponse[StreamingTranscriptionView]:
        data = await LiveTranscriptionService(context, settings).finish_stream(
            model_id=request.model_id,
            instance_id=request.instance_id,
            session_id=request.session_id,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/stream/cancel", status_code=status.HTTP_204_NO_CONTENT)
    async def cancel_stream(
        request: StreamingFinishRequest,
        context: ContextDependency,
    ) -> None:
        await LiveTranscriptionService(context, settings).cancel_stream(
            model_id=request.model_id,
            instance_id=request.instance_id,
            session_id=request.session_id,
        )

    return router
