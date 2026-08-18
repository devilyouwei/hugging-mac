"""REST routes for local multimodal chat."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter, ValidationError

from hugging_mac_web.chat.config import ChatSettings
from hugging_mac_web.chat.schemas import (
    ChatHistoryMessage,
    ChatModelView,
    ChatReplyView,
    LoadChatModelRequest,
    LoadedChatModelView,
)
from hugging_mac_web.chat.service import ChatService
from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.live_transcription.config import (
    NEMOTRON_3_5_ASR_PROFILE,
    LiveTranscriptionSettings,
)
from hugging_mac_web.live_transcription.schemas import StreamingSessionView
from hugging_mac_web.live_transcription.service import LiveTranscriptionService
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.sse_util import SseEvent
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited

_HISTORY_ADAPTER = TypeAdapter(tuple[ChatHistoryMessage, ...])
_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def create_router(settings: ChatSettings) -> APIRouter:
    router = APIRouter(prefix="/api/v1/apps/chat", tags=["chat"])

    @router.get("/models", response_model=ApiResponse[tuple[ChatModelView, ...]])
    async def models(context: ContextDependency) -> ApiResponse[tuple[ChatModelView, ...]]:
        data = await ChatService(context, settings).list_models()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.get("/model", response_model=ApiResponse[LoadedChatModelView | None])
    async def model(
        context: ContextDependency,
        model_id: str | None = Query(default=None),
    ) -> ApiResponse[LoadedChatModelView | None]:
        data = await ChatService(context, settings).ready_model(model_id or settings.model_variant)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/model/load", response_model=ApiResponse[LoadedChatModelView])
    async def load_model(
        context: ContextDependency,
        request: LoadChatModelRequest | None = None,
    ) -> ApiResponse[LoadedChatModelView]:
        data = await ChatService(context, settings).load_model(
            request.model_id if request is not None else settings.model_variant
        )
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.websocket("/asr/ws")
    async def asr_websocket(websocket: WebSocket, context: ContextDependency) -> None:
        await websocket.accept()
        live = LiveTranscriptionService(context, LiveTranscriptionSettings())
        session: StreamingSessionView | None = None
        utterance_id: int | None = None
        sample_rate = 0
        utterance_bytes = 0
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    message_type = payload.get("type")
                    if message_type == "start_utterance":
                        if session is not None:
                            raise ValueError("An ASR utterance is already active")
                        sample_rate = int(payload.get("sample_rate", 0))
                        utterance_id = int(payload.get("utterance_id", 0))
                        if not 8_000 <= sample_rate <= 192_000:
                            raise ValueError("Invalid PCM sample rate")
                        session = await live.start_stream(
                            model_id=NEMOTRON_3_5_ASR_PROFILE.model_id,
                            instance_id=str(payload.get("instance_id", "")),
                        )
                        utterance_bytes = 0
                        await websocket.send_json(
                            {
                                "type": "ready",
                                "utterance_id": utterance_id,
                                **session.model_dump(mode="json"),
                            }
                        )
                    elif message_type == "finish_utterance":
                        if session is None:
                            raise ValueError("No ASR utterance is active")
                        response = await live.finish_stream(
                            model_id=NEMOTRON_3_5_ASR_PROFILE.model_id,
                            instance_id=session.instance_id,
                            session_id=session.session_id,
                        )
                        session = None
                        utterance_bytes = 0
                        await websocket.send_json(
                            {
                                "type": "final",
                                "utterance_id": utterance_id,
                                **response.model_dump(mode="json"),
                            }
                        )
                        utterance_id = None
                    elif message_type == "cancel_utterance":
                        if session is not None:
                            await live.cancel_stream(
                                model_id=NEMOTRON_3_5_ASR_PROFILE.model_id,
                                instance_id=session.instance_id,
                                session_id=session.session_id,
                            )
                            session = None
                            utterance_bytes = 0
                        await websocket.send_json(
                            {
                                "type": "cancelled",
                                "utterance_id": payload.get("utterance_id"),
                            }
                        )
                    else:
                        raise ValueError("Unsupported ASR WebSocket message")
                    continue
                audio = message.get("bytes")
                if audio is None or session is None:
                    raise ValueError("Binary PCM requires an active ASR utterance")
                utterance_bytes += len(audio)
                if utterance_bytes > context.settings.max_upload_bytes:
                    raise ValueError("ASR utterance exceeds the configured audio limit")
                response = await live.transcribe_stream_chunk(
                    _pcm16_to_wav(audio, sample_rate),
                    model_id=NEMOTRON_3_5_ASR_PROFILE.model_id,
                    instance_id=session.instance_id,
                    session_id=session.session_id,
                )
                await websocket.send_json(
                    {
                        "type": "partial",
                        "utterance_id": utterance_id,
                        **response.model_dump(mode="json"),
                    }
                )
        except WebSocketDisconnect:
            pass
        except Exception as error:
            with contextlib.suppress(RuntimeError):
                await websocket.send_json({"type": "error", "message": str(error)[:500]})
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await live.cancel_stream(
                        model_id=NEMOTRON_3_5_ASR_PROFILE.model_id,
                        instance_id=session.instance_id,
                        session_id=session.session_id,
                    )
            with contextlib.suppress(RuntimeError):
                await websocket.close()

    @router.post("/messages", response_model=ApiResponse[ChatReplyView])
    async def send_message(
        context: ContextDependency,
        instance_id: Annotated[str, Form(min_length=1)],
        prompt: Annotated[str, Form(min_length=1, max_length=32_000)],
        history_json: Annotated[str, Form()] = "[]",
        max_tokens: Annotated[int, Form(ge=1, le=4096)] = 512,
        temperature: Annotated[float, Form(ge=0.0, le=2.0)] = 0.0,
        enable_thinking: Annotated[bool, Form()] = False,
        images: Annotated[list[UploadFile] | None, File()] = None,
    ) -> ApiResponse[ChatReplyView]:
        uploads = images or []
        if len(uploads) > settings.max_images:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"At most {settings.max_images} images are allowed",
            )
        history = _parse_history(history_json, settings.max_prompt_characters)
        upload_root = context.settings.cache_dir / "chat-uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        try:
            for upload in uploads:
                paths.append(await _store_image(upload, upload_root, settings, context))
            data = await ChatService(context, settings).chat(
                instance_id,
                history,
                prompt,
                tuple(paths),
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))
        finally:
            for path in paths:
                with contextlib.suppress(OSError):
                    path.unlink()

    @router.post("/messages/stream", response_class=StreamingResponse)
    async def stream_message(
        context: ContextDependency,
        instance_id: Annotated[str, Form(min_length=1)],
        prompt: Annotated[str, Form(min_length=1, max_length=32_000)],
        history_json: Annotated[str, Form()] = "[]",
        max_tokens: Annotated[int, Form(ge=1, le=4096)] = 512,
        temperature: Annotated[float, Form(ge=0.0, le=2.0)] = 0.0,
        enable_thinking: Annotated[bool, Form()] = False,
        images: Annotated[list[UploadFile] | None, File()] = None,
    ) -> StreamingResponse:
        uploads = images or []
        if len(uploads) > settings.max_images:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"At most {settings.max_images} images are allowed",
            )
        history = _parse_history(history_json, settings.max_prompt_characters)
        upload_root = context.settings.cache_dir / "chat-uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        try:
            for upload in uploads:
                paths.append(await _store_image(upload, upload_root, settings, context))
        except BaseException:
            for path in paths:
                with contextlib.suppress(OSError):
                    path.unlink()
            raise

        async def events() -> AsyncIterator[str]:
            try:
                async for event in ChatService(context, settings).stream_chat(
                    instance_id,
                    history,
                    prompt,
                    tuple(paths),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                ):
                    yield SseEvent(
                        event="done" if event.finish_reason is not None else "delta",
                        data=event.model_dump(mode="json"),
                    ).encode()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                yield SseEvent(
                    event="error",
                    data={"message": str(error)[:500]},
                ).encode()
            finally:
                for path in paths:
                    with contextlib.suppress(OSError):
                        path.unlink()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _pcm16_to_wav(audio: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)
    return output.getvalue()


def _parse_history(value: str, character_limit: int) -> tuple[ChatHistoryMessage, ...]:
    try:
        history = _HISTORY_ADAPTER.validate_python(json.loads(value))
    except (json.JSONDecodeError, ValidationError) as error:
        raise HTTPException(status_code=422, detail="Invalid chat history") from error
    if sum(len(item.content) for item in history) > character_limit:
        raise HTTPException(status_code=422, detail="Chat history is too long")
    return history


async def _store_image(
    upload: UploadFile,
    root: Path,
    settings: ChatSettings,
    context: ContextDependency,
) -> Path:
    suffix = _IMAGE_TYPES.get(upload.content_type or "")
    if suffix is None:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported")
    data = await read_upload_limited(upload, settings.max_image_bytes)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > context.settings.max_image_pixels:
                raise HTTPException(status_code=413, detail="Image pixel count exceeds the limit")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Uploaded image is invalid") from error

    def write() -> Path:
        with tempfile.NamedTemporaryFile(dir=root, suffix=suffix, delete=False) as output:
            output.write(data)
            return Path(output.name)

    return await asyncio.to_thread(write)
