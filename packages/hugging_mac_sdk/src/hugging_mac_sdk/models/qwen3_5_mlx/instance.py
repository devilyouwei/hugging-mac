"""Runtime-independent Qwen3.5 MLX chat instance."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ChatTimings,
)

from .config import QWEN3_5_MLX_MODEL_ID, Qwen35MlxInstanceConfig
from .utils.types import GenerationOutput


class Qwen35Engine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(self, request: ChatRequest) -> GenerationOutput: ...
    def stream(self, request: ChatRequest) -> AsyncIterator[GenerationOutput]: ...
    async def close(self) -> None: ...


class Qwen35MlxInstance(BaseModelInstance):
    def __init__(self, config: Qwen35MlxInstanceConfig, engine: Qwen35Engine) -> None:
        super().__init__(
            model_id=QWEN3_5_MLX_MODEL_ID,
            revision="variant-pinned",
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._engine, self._inference_lock = engine, asyncio.Lock()
        self.register_capability(Chat, cast(Chat, self))  # type: ignore[type-abstract]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Qwen3.5 instance must be READY before chat")
        async with self._inference_lock:
            started = perf_counter()
            try:
                output = await self._engine.infer(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Qwen3.5 chat failed", details={"reason": str(error)[:500]}, cause=error
                ) from error
        return ChatResponse(
            model_id=QWEN3_5_MLX_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            message=ChatMessage(role="assistant", content=output.text),
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            timings=ChatTimings(inference_ms=(perf_counter() - started) * 1000),
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        if self.state is not ModelState.READY:
            raise InferenceError("Qwen3.5 instance must be READY before stream_chat")
        generated_tokens: int | None = None
        async with self._inference_lock:
            try:
                async for output in self._engine.stream(request):
                    generated_tokens = output.generated_tokens
                    yield ChatStreamEvent(delta=output.text, generated_tokens=generated_tokens)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Qwen3.5 streaming chat failed",
                    details={"reason": str(error)[:500]},
                    cause=error,
                ) from error
        yield ChatStreamEvent(finish_reason="stop", generated_tokens=generated_tokens)

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
