"""Text-to-speech REST routes."""

from fastapi import APIRouter, Query, Response

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.text_to_speech.config import TextToSpeechSettings
from hugging_mac_web.text_to_speech.schemas import (
    LoadedTtsModelView,
    LoadTtsModelRequest,
    SynthesizeSpeechRequest,
    TtsModelView,
    TtsResourceView,
)
from hugging_mac_web.text_to_speech.service import TextToSpeechService

DEFAULT_MODEL_ID = "hexgrad/kokoro-82m"


def create_router(settings: TextToSpeechSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/apps/text-to-speech",
        tags=["text-to-speech"],
    )

    @router.get("/models", response_model=ApiResponse[tuple[TtsModelView, ...]])
    async def models(context: ContextDependency) -> ApiResponse[tuple[TtsModelView, ...]]:
        data = await TextToSpeechService(context, settings).list_models()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.get("/resources", response_model=ApiResponse[TtsResourceView])
    async def resources(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
    ) -> ApiResponse[TtsResourceView]:
        data = await TextToSpeechService(context, settings).resource_status(model_id)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/resources/source/download", response_model=ApiResponse[TtsResourceView])
    async def download_source(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
        overwrite: bool = False,
    ) -> ApiResponse[TtsResourceView]:
        data = await TextToSpeechService(context, settings).download_source(
            model_id,
            overwrite=overwrite,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/resources/coreml/convert", response_model=ApiResponse[TtsResourceView])
    async def convert_coreml(
        context: ContextDependency,
        model_id: str = Query(default=DEFAULT_MODEL_ID),
        overwrite: bool = False,
    ) -> ApiResponse[TtsResourceView]:
        data = await TextToSpeechService(context, settings).convert_coreml(
            model_id,
            overwrite=overwrite,
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/models/load", response_model=ApiResponse[LoadedTtsModelView])
    async def load_model(
        request: LoadTtsModelRequest,
        context: ContextDependency,
    ) -> ApiResponse[LoadedTtsModelView]:
        data = await TextToSpeechService(context, settings).load_model(request.model_id)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/synthesize")
    async def synthesize(
        request: SynthesizeSpeechRequest,
        context: ContextDependency,
    ) -> Response:
        audio, headers = await TextToSpeechService(context, settings).synthesize(request)
        return Response(content=audio, media_type="audio/wav", headers=headers)

    return router
