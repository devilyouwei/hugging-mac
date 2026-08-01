"""Local multimodal chat use case."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.chat import ChatImage, ChatMessage, ChatRequest

from hugging_mac_web.chat.config import ChatSettings
from hugging_mac_web.chat.schemas import (
    ChatHistoryMessage,
    ChatReplyView,
    ChatStreamEventView,
    LoadedChatModelView,
)
from hugging_mac_web.context import PlatformContext


class ChatService:
    def __init__(self, context: PlatformContext, settings: ChatSettings) -> None:
        self._context = context
        self._settings = settings

    async def load_model(self) -> LoadedChatModelView:
        handle = await self._context.models.load(
            self._settings.model_id,
            variant=self._settings.variant,
            runtime=self._settings.runtime,
            options={"model_home": self._context.settings.model_home},
            reuse=ReusePolicy.SHARED,
        )
        try:
            return LoadedChatModelView.from_sdk(handle.info())
        finally:
            await handle.close()

    async def ready_model(self) -> LoadedChatModelView | None:
        for snapshot in await self._context.models.instances.snapshots():
            if (
                snapshot.model_id == self._settings.model_id
                and snapshot.variant == self._settings.variant
                and snapshot.runtime == self._settings.runtime
                and snapshot.state is ModelState.READY
            ):
                instance = await self._context.models.instances.require(snapshot.instance_id)
                return LoadedChatModelView.from_sdk(instance.info())
        return None

    async def chat(
        self,
        instance_id: str,
        history: tuple[ChatHistoryMessage, ...],
        prompt: str,
        image_paths: tuple[Path, ...],
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> ChatReplyView:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != self._settings.model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected chat model instance is not ready",
                details={"instance_id": instance_id},
            )
        messages = [ChatMessage(role=item.role, content=item.content) for item in history]
        messages.append(
            ChatMessage(
                role="user",
                content=prompt,
                images=tuple(ChatImage(path=path) for path in image_paths),
            )
        )
        capability = instance.require(Chat)  # type: ignore[type-abstract]
        response = await capability.chat(
            ChatRequest(
                messages=tuple(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
        )
        return ChatReplyView(
            content=response.message.content,
            model_id=response.model_id,
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            prompt_tokens=response.prompt_tokens,
            generated_tokens=response.generated_tokens,
            inference_ms=response.timings.inference_ms,
        )

    async def stream_chat(
        self,
        instance_id: str,
        history: tuple[ChatHistoryMessage, ...],
        prompt: str,
        image_paths: tuple[Path, ...],
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> AsyncIterator[ChatStreamEventView]:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != self._settings.model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected chat model instance is not ready",
                details={"instance_id": instance_id},
            )
        messages = [ChatMessage(role=item.role, content=item.content) for item in history]
        messages.append(
            ChatMessage(
                role="user",
                content=prompt,
                images=tuple(ChatImage(path=path) for path in image_paths),
            )
        )
        capability = instance.require(Chat)  # type: ignore[type-abstract]
        started = perf_counter()
        async for event in capability.stream_chat(
            ChatRequest(
                messages=tuple(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
        ):
            yield ChatStreamEventView(
                delta=event.delta,
                finish_reason=event.finish_reason,
                model_id=info.model_id or self._settings.model_id,
                instance_id=info.instance_id,
                runtime=info.runtime or self._settings.runtime,
                device=info.device or "gpu",
                generated_tokens=event.generated_tokens,
                inference_ms=(
                    (perf_counter() - started) * 1000 if event.finish_reason is not None else None
                ),
            )
