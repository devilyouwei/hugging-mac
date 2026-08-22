"""Live transcription management and realtime WebSocket routes."""

import asyncio
import contextlib
import json
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.schemas import (
    AsrModelView,
    LoadedModelView,
    LoadModelRequest,
    PipelineComponentView,
    ResourceStatusView,
)
from hugging_mac_web.live_transcription.service import LiveTranscriptionService
from hugging_mac_web.live_transcription.session import (
    LiveSessionStart,
    LiveTranscriptionSession,
)
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.log_util import get_logger
from hugging_mac_web.shared.utils.time_util import utc_now

DEFAULT_MODEL_ID = "audio8/audio8-asr"
logger = get_logger("live_transcription.routes")


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
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ResourceStatusView]:
        data = await LiveTranscriptionService(context, settings).resource_status(
            model_id, variant=variant
        )
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
        service = LiveTranscriptionService(context, settings)
        if request.variant is None:
            data = await service.load_model(request.model_id, runtime=request.runtime)
        else:
            data = await service.load_model(
                request.model_id,
                variant=request.variant,
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

    @router.websocket("/live/ws")
    async def live_websocket(websocket: WebSocket, context: ContextDependency) -> None:
        connection_id = uuid4().hex
        client = websocket.client
        log = logger.bind(
            connection_id=connection_id,
            client_host=client.host if client else None,
            client_port=client.port if client else None,
        )
        connected_at = perf_counter()
        received_chunks = received_bytes = 0
        await websocket.accept()
        log.info("live_websocket_connected")
        session: LiveTranscriptionSession | None = None
        send_lock = asyncio.Lock()

        async def emit(payload: dict[str, object] | bytes) -> None:
            async with send_lock:
                if isinstance(payload, bytes):
                    await websocket.send_bytes(payload)
                else:
                    await websocket.send_json(payload)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    log.info(
                        "live_websocket_disconnected",
                        received_chunks=received_chunks,
                        received_bytes=received_bytes,
                        session_active=session is not None and not session.closed,
                    )
                    break
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    message_type = payload.get("type")
                    if message_type == "start":
                        if session is not None:
                            raise ValueError("A live transcription session is already active")
                        start = LiveSessionStart.from_payload(payload)
                        log.info(
                            "live_websocket_start_requested",
                            model_id=start.model_id,
                            instance_id=start.instance_id,
                            input_sample_rate=start.sample_rate,
                            use_vad=start.use_vad,
                            vad_instance_id=start.vad_instance_id,
                            use_enhancement=start.use_enhancement,
                            enhancement_instance_id=start.enhancement_instance_id,
                        )
                        session = await LiveTranscriptionSession.create(
                            context,
                            settings,
                            start,
                            emit,
                        )
                        log = log.bind(session_id=session.session_id)
                        log.info(
                            "live_websocket_ready",
                            model_id=start.model_id,
                            streaming=session.streaming,
                        )
                        await emit(
                            {
                                "type": "ready",
                                "sample_rate": 16_000,
                                "streaming": LiveTranscriptionService(context, settings)
                                ._profile(start.model_id)
                                .streaming,
                            }
                        )
                    elif message_type == "stop":
                        if session is None:
                            raise ValueError("No live transcription session is active")
                        log.info(
                            "live_websocket_stop_requested",
                            received_chunks=received_chunks,
                            received_bytes=received_bytes,
                        )
                        await session.stop()
                        await emit({"type": "stopped"})
                        break
                    else:
                        raise ValueError("Unsupported live transcription WebSocket message")
                    continue

                audio = message.get("bytes")
                if audio is None or session is None:
                    raise ValueError("Binary PCM requires an active live session")
                received_chunks += 1
                received_bytes += len(audio)
                await session.enqueue(audio)
        except WebSocketDisconnect:
            log.info(
                "live_websocket_disconnected",
                received_chunks=received_chunks,
                received_bytes=received_bytes,
                session_active=session is not None and not session.closed,
            )
        except Exception as error:
            log.exception(
                "live_websocket_failed",
                error_type=type(error).__name__,
                received_chunks=received_chunks,
                received_bytes=received_bytes,
            )
            with contextlib.suppress(RuntimeError):
                await emit({"type": "error", "message": str(error)[:500]})
        finally:
            if session is not None:
                await session.cancel()
            with contextlib.suppress(RuntimeError):
                await websocket.close()
            log.info(
                "live_websocket_closed",
                elapsed_ms=round((perf_counter() - connected_at) * 1000, 3),
                received_chunks=received_chunks,
                received_bytes=received_bytes,
            )

    return router
