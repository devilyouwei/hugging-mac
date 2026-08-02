"""HTTP routes for the Digital Human game."""

from __future__ import annotations

import contextlib
import io
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter, ValidationError

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.digital_human_clone.config import DigitalHumanSettings
from hugging_mac_web.digital_human_clone.schemas import (
    ConversationMessage,
    LoadedModelsView,
    SetupView,
    TranscriptView,
)
from hugging_mac_web.digital_human_clone.service import DigitalHumanService
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.sse_util import SseEvent
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.shared.utils.upload_util import read_upload_limited

_HISTORY = TypeAdapter(tuple[ConversationMessage, ...])
_AUDIO_TYPES = {
    "audio/mp4",
    "audio/wav",
    "audio/wave",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}
_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def create_router(settings: DigitalHumanSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/games/digital-human",
        tags=["digital-human"],
    )

    @router.get("/setup", response_model=ApiResponse[SetupView])
    async def setup(context: ContextDependency) -> ApiResponse[SetupView]:
        data = await DigitalHumanService(context, settings).setup()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/setup/prepare", response_model=ApiResponse[LoadedModelsView])
    async def prepare(context: ContextDependency) -> ApiResponse[LoadedModelsView]:
        data = await DigitalHumanService(context, settings).prepare()
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/transcribe", response_model=ApiResponse[TranscriptView])
    async def transcribe(
        context: ContextDependency,
        file: Annotated[UploadFile, File(description="One VAD utterance")],
        instance_id: Annotated[str, Form(min_length=1)],
    ) -> ApiResponse[TranscriptView]:
        _require_audio(file)
        audio = await read_upload_limited(file, settings.max_audio_bytes)
        data = await DigitalHumanService(context, settings).transcribe(audio, instance_id)
        return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))

    @router.post("/chat/stream", response_class=StreamingResponse)
    async def chat_stream(
        context: ContextDependency,
        image: Annotated[UploadFile, File(description="Current webcam frame")],
        instance_id: Annotated[str, Form(min_length=1)],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        history_json: Annotated[str, Form()] = "[]",
    ) -> StreamingResponse:
        history = _parse_history(history_json, settings.max_history_messages)
        image_path = await _store_image(image, context.settings.cache_dir, settings)

        async def events() -> AsyncIterator[str]:
            try:
                async for event in DigitalHumanService(context, settings).stream_chat(
                    instance_id, prompt, history, image_path
                ):
                    yield SseEvent(
                        event="done" if event.finish_reason is not None else "delta",
                        data=event.model_dump(mode="json"),
                    ).encode()
            finally:
                with contextlib.suppress(OSError):
                    image_path.unlink()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/synthesize")
    async def synthesize(
        context: ContextDependency,
        instance_id: Annotated[str, Form(min_length=1)],
        voice: Annotated[str, Form(min_length=1, max_length=64)],
        language: Annotated[str, Form(pattern="^[abefhipjz]$")],
        text: Annotated[str, Form(min_length=1, max_length=1000)],
    ) -> Response:
        audio = await DigitalHumanService(context, settings).synthesize(
            instance_id, voice, language, text
        )
        return Response(content=audio, media_type="audio/wav")

    return router


def _require_audio(upload: UploadFile) -> None:
    if upload.content_type not in _AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only WAV, WebM, or MP4 audio is supported",
        )


def _parse_history(value: str, limit: int) -> tuple[ConversationMessage, ...]:
    try:
        history = _HISTORY.validate_python(json.loads(value))
    except (json.JSONDecodeError, ValidationError) as error:
        raise HTTPException(status_code=422, detail="Invalid conversation history") from error
    return history[-limit:]


async def _store_image(
    upload: UploadFile,
    cache_dir: Path,
    settings: DigitalHumanSettings,
) -> Path:
    suffix = _IMAGE_TYPES.get(upload.content_type or "")
    if suffix is None:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG and WebP images are supported")
    data = await read_upload_limited(upload, settings.max_image_bytes)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Invalid webcam image") from error
    root = cache_dir / "digital-human"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path
