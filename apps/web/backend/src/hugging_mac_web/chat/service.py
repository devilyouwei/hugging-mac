"""Local multimodal chat use case."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedCapabilityError
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.chat import ChatImage, ChatMessage, ChatRequest

from hugging_mac_web.chat.config import CHAT_MODEL_IDS, ChatModelProfile, ChatSettings
from hugging_mac_web.chat.schemas import (
    ChatHistoryMessage,
    ChatModelView,
    ChatReplyView,
    ChatResourceView,
    ChatStreamEventView,
    LoadedChatModelView,
)
from hugging_mac_web.context import PlatformContext


class ChatService:
    def __init__(self, context: PlatformContext, settings: ChatSettings) -> None:
        self._context = context
        self._settings = settings

    async def list_models(self) -> tuple[ChatModelView, ...]:
        return tuple(
            await asyncio.gather(
                *(self.model_view(profile.profile_id) for profile in self._profiles())
            )
        )

    async def model_view(self, model_id: str) -> ChatModelView:
        configured = self._profile(model_id)
        resource = await self.resource_status(model_id)
        profile = self._select_runtime(configured, resource)
        return ChatModelView.from_profile(
            profile,
            resource,
            ready_instance_id=await self._ready_instance_id(profile),
        )

    async def resource_status(self, model_id: str) -> ChatResourceView:
        profile = self._profile(model_id)
        status = await self._context.models.resources.status(
            profile.model_id,
            variant=profile.variant,
            options=self._model_options(),
        )
        return ChatResourceView.from_sdk(status)

    async def ready_model(self, model_id: str) -> LoadedChatModelView | None:
        profile = self._select_runtime(
            self._profile(model_id), await self.resource_status(model_id)
        )
        for snapshot in await self._context.models.instances.snapshots():
            if (
                snapshot.model_id == profile.model_id
                and snapshot.variant == profile.variant
                and snapshot.runtime == profile.runtime
                and snapshot.state is ModelState.READY
            ):
                instance = await self._context.models.instances.require(snapshot.instance_id)
                return LoadedChatModelView.from_sdk(instance.info())
        return None

    async def load_model(self, model_id: str) -> LoadedChatModelView:
        profile = self._select_runtime(
            self._profile(model_id), await self.resource_status(model_id)
        )
        handle = await self._context.models.load(
            profile.model_id,
            variant=profile.variant,
            runtime=profile.runtime,
            options=self._model_options(),
            reuse=ReusePolicy.SHARED,
        )
        try:
            return LoadedChatModelView.from_sdk(handle.info())
        finally:
            await handle.close()

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
        profile = self._profile_for_instance(info.model_id, info.variant)
        if info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected chat model instance is not ready",
                details={"instance_id": instance_id},
            )
        if image_paths and not profile.supports_images:
            raise UnsupportedCapabilityError(
                "The selected chat model does not support images",
                details={"model_id": profile.model_id},
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
        profile = self._profile_for_instance(info.model_id, info.variant)
        if info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected chat model instance is not ready",
                details={"instance_id": instance_id},
            )
        if image_paths and not profile.supports_images:
            raise UnsupportedCapabilityError(
                "The selected chat model does not support images",
                details={"model_id": profile.model_id},
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
                model_id=info.model_id or profile.model_id,
                instance_id=info.instance_id,
                runtime=info.runtime or profile.runtime,
                device=info.device or "gpu",
                generated_tokens=event.generated_tokens,
                inference_ms=(
                    (perf_counter() - started) * 1000 if event.finish_reason is not None else None
                ),
            )

    def _profile(self, model_id: str) -> ChatModelProfile:
        for profile in self._profiles():
            if profile.profile_id == model_id:
                return profile
        raise ResourceNotFoundError(f"Chat model variant is not supported: {model_id}")

    def _profile_for_instance(
        self,
        model_id: str | None,
        variant: str | None,
    ) -> ChatModelProfile:
        for profile in self._profiles():
            if profile.model_id == model_id and profile.variant == variant:
                return profile
        raise ResourceNotFoundError(f"Chat model is not supported: {model_id}/{variant}")

    def _profiles(self) -> tuple[ChatModelProfile, ...]:
        return tuple(
            profile for model_id in CHAT_MODEL_IDS for profile in self._model_profiles(model_id)
        )

    def _model_profiles(self, model_id: str) -> tuple[ChatModelProfile, ...]:
        definition = self._context.models.registry.get(model_id)
        manifest = definition.manifest
        runtime = manifest.default_runtime or "mlx"
        return tuple(
            ChatModelProfile(
                profile_id=self._profile_id(manifest.model_id, variant.name),
                model_id=manifest.model_id,
                display_name=variant.display_name,
                short_name=variant.display_name,
                description=variant.description or manifest.description,
                variant=variant.name,
                runtime=runtime,
                required_artifact_id=self._artifact_id(
                    definition.artifacts,
                    variant=variant.name,
                    runtime=runtime,
                ),
                disk_size_bytes=int(variant.metadata.get("disk_size_bytes", 0)),
                supports_images=bool(variant.metadata.get("multimodal", False)),
            )
            for variant in manifest.variants
        )

    @staticmethod
    def _profile_id(model_id: str, variant: str) -> str:
        if model_id == "qwen/qwen3.5":
            return variant
        return f"{model_id.rsplit('/', 1)[-1]}-{variant}"

    @staticmethod
    def _artifact_id(artifacts: tuple[ModelArtifact, ...], *, variant: str, runtime: str) -> str:
        matching = [
            artifact.artifact_id
            for artifact in artifacts
            if not artifact.shared and artifact.variant == variant and artifact.runtime == runtime
        ]
        if len(matching) != 1:
            raise ResourceNotFoundError(
                "Chat model runtime artifact is not uniquely declared",
                details={"variant": variant, "runtime": runtime},
            )
        return matching[0]

    @staticmethod
    def _select_runtime(profile: ChatModelProfile, resource: ChatResourceView) -> ChatModelProfile:
        available = {
            item.runtime: item for item in resource.artifacts if item.available and item.runtime
        }
        runtime = next((item for item in ("coreml", "pytorch", "mlx") if item in available), None)
        runtime = runtime or next(iter(available), None)
        if runtime is None:
            return profile
        return replace(
            profile,
            runtime=runtime,
            required_artifact_id=available[runtime].artifact_id,
        )

    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}

    async def _ready_instance_id(self, profile: ChatModelProfile) -> str | None:
        for snapshot in await self._context.models.instances.snapshots():
            if (
                snapshot.model_id == profile.model_id
                and snapshot.variant == profile.variant
                and snapshot.runtime == profile.runtime
                and snapshot.state is ModelState.READY
            ):
                return snapshot.instance_id
        return None
