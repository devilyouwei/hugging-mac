"""Three-model orchestration for the Digital Human game."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter

from hugging_mac_sdk import ArtifactFormat, AudioInput, ReusePolicy, TranscriptionRequest
from hugging_mac_sdk.capabilities import Chat, SpeechSynthesis, SpeechTranscription
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.chat import ChatImage, ChatMessage, ChatRequest, ChatStreamEvent
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.digital_human_clone.config import DigitalHumanSettings
from hugging_mac_web.digital_human_clone.schemas import (
    ConversationMessage,
    LoadedModelsView,
    LlmVariantView,
    ModelStateView,
    SetupView,
    TranscriptView,
)
from hugging_mac_web.text_to_speech.audio import float32le_to_wav

ASR_MODEL_ID = "audio8/audio8-asr-0.1b"
LLM_MODEL_ID = "mlx-community/qwen3.5-mlx"
TTS_MODEL_ID = "mlx-community/kokoro-82m-bf16"

_MODEL_SPECS = (
    ("asr", ASR_MODEL_ID, "base", "coreml", "coreml"),
    ("llm", LLM_MODEL_ID, "4b-optiq-4bit", "mlx", "model"),
    ("tts", TTS_MODEL_ID, "bf16", "mlx", "model"),
)
_UNSPEAKABLE = re.compile(r"[`*_#>|~\[\]{}]+")


class DigitalHumanService:
    def __init__(self, context: PlatformContext, settings: DigitalHumanSettings) -> None:
        self._context = context
        self._settings = settings

    async def setup(self, llm_variant: str | None = None) -> SetupView:
        llm_variant = self._llm_variant(llm_variant)
        snapshots = await self._context.models.instances.snapshots()
        models: list[ModelStateView] = []
        for role, model_id, variant, runtime, required_artifact in self._model_specs(llm_variant):
            status = await self._context.models.resources.status(
                model_id,
                variant=variant,
                options=self._options(role),
            )
            resources_ready = any(
                artifact.artifact_id == required_artifact and artifact.available
                for artifact in status.artifacts
            )
            ready = next(
                (
                    item.instance_id
                    for item in snapshots
                    if item.model_id == model_id
                    and item.variant == variant
                    and item.runtime == runtime
                    and item.state is ModelState.READY
                ),
                None,
            )
            models.append(
                ModelStateView(
                    role=role,  # type: ignore[arg-type]
                    model_id=model_id,
                    runtime=runtime,
                    resources_ready=resources_ready,
                    ready_instance_id=ready,
                )
            )
        return SetupView(
            models=tuple(models),
            llm_variants=self._llm_variants(),
            selected_llm_variant=llm_variant,
        )

    async def prepare(self, llm_variant: str | None = None) -> LoadedModelsView:
        llm_variant = self._llm_variant(llm_variant)
        await self._prepare_resources(llm_variant)

        async def load(role: str, model_id: str, variant: str, runtime: str) -> str:
            handle = await self._context.models.load(
                model_id,
                variant=variant,
                runtime=runtime,
                options=self._options(role),
                reuse=ReusePolicy.SHARED,
            )
            try:
                return handle.info().instance_id
            finally:
                await handle.close()

        asr, llm, tts = await asyncio.gather(
            *(
                load(role, model_id, variant, runtime)
                for role, model_id, variant, runtime, _ in self._model_specs(llm_variant)
            )
        )
        return LoadedModelsView(
            asr_instance_id=asr,
            llm_instance_id=llm,
            tts_instance_id=tts,
        )

    async def transcribe(self, audio: bytes, instance_id: str) -> TranscriptView:
        instance = await self._ready_instance(instance_id, ASR_MODEL_ID)
        capability = instance.require(SpeechTranscription)  # type: ignore[type-abstract]
        started = perf_counter()
        response = await capability.transcribe(
            TranscriptionRequest(
                audio=AudioInput(data=audio),
                prompt="Transcribe the speech accurately in its original language.",
                max_new_tokens=128,
            )
        )
        return TranscriptView(
            text=response.text.strip(),
            inference_ms=(perf_counter() - started) * 1000,
        )

    async def stream_chat(
        self,
        instance_id: str,
        prompt: str,
        history: tuple[ConversationMessage, ...],
        image_path: Path,
    ) -> AsyncIterator[ChatStreamEvent]:
        instance = await self._ready_instance(instance_id, LLM_MODEL_ID)
        capability = instance.require(Chat)  # type: ignore[type-abstract]
        recent = history[-self._settings.max_history_messages :]
        messages = [ChatMessage(role="system", content=self._settings.system_prompt)]
        messages.extend(ChatMessage(role=item.role, content=item.content) for item in recent)
        messages.append(
            ChatMessage(
                role="user",
                content=prompt,
                images=(ChatImage(path=image_path),),
            )
        )
        async for event in capability.stream_chat(
            ChatRequest(
                messages=tuple(messages),
                max_tokens=self._settings.max_reply_tokens,
                temperature=0.6,
                top_p=0.9,
                enable_thinking=False,
            )
        ):
            yield event

    async def synthesize(
        self, instance_id: str, voice: str, language: str, text: str
    ) -> bytes:
        cleaned = _clean_for_speech(text)
        if not cleaned:
            raise ValueError("The assistant reply contains no speakable text")
        return await self._synthesize(
            instance_id,
            voice,
            language,
            cleaned[:240],
            max_new_tokens=256,
        )

    async def _synthesize(
        self,
        instance_id: str,
        voice: str,
        language: str,
        text: str,
        *,
        max_new_tokens: int,
    ) -> bytes:
        instance = await self._ready_instance(instance_id, TTS_MODEL_ID)
        capability = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
        response = await capability.synthesize(
            SpeechSynthesisRequest(
                text=text,
                voice=voice,
                language=language,
                max_new_tokens=max_new_tokens,
            )
        )
        return float32le_to_wav(response.audio, response.sample_rate)

    async def _prepare_resources(self, llm_variant: str) -> None:
        for role, model_id, variant, _, required_artifact in self._model_specs(llm_variant):
            options = self._options(role)
            status = await self._context.models.resources.status(
                model_id, variant=variant, options=options
            )
            ready = any(
                artifact.artifact_id == required_artifact and artifact.available
                for artifact in status.artifacts
            )
            if ready:
                continue
            source = next((item for item in status.artifacts if item.artifact_id == "source"), None)
            if source is None or not source.available:
                await self._context.models.resources.download_source(
                    model_id, variant=variant, options=options
                )
            if role == "asr":
                await self._context.models.resources.convert(
                    model_id,
                    ArtifactFormat.COREML,
                    variant=variant,
                    options=options,
                )

    async def _ready_instance(self, instance_id: str, model_id: str):
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected model instance is not ready",
                details={"instance_id": instance_id, "model_id": model_id},
            )
        return instance

    def _options(self, role: str) -> dict[str, object]:
        options: dict[str, object] = {"model_home": self._context.settings.model_home}
        return options

    @staticmethod
    def _model_specs(llm_variant: str):
        return (
            ("asr", ASR_MODEL_ID, "base", "coreml", "coreml"),
            ("llm", LLM_MODEL_ID, llm_variant, "mlx", "model"),
            ("tts", TTS_MODEL_ID, "bf16", "mlx", "model"),
        )

    def _llm_variants(self) -> tuple[LlmVariantView, ...]:
        manifest = self._context.models.registry.get(LLM_MODEL_ID).manifest
        return tuple(
            LlmVariantView(
                name=variant.name,
                display_name=variant.display_name,
                description=variant.description,
            )
            for variant in manifest.variants
        )

    def _llm_variant(self, value: str | None) -> str:
        variants = self._llm_variants()
        selected = value or "4b-optiq-4bit"
        if selected not in {variant.name for variant in variants}:
            raise ResourceNotFoundError("The selected Qwen3.5 variant is not registered")
        return selected


def _clean_for_speech(text: str) -> str:
    return " ".join(_UNSPEAKABLE.sub("", text).split())
