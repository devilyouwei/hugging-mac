"""Three-model orchestration for the Digital Human game."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import Chat, SpeechSynthesis
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.catalog import InstanceSnapshot
from hugging_mac_sdk.schemas.chat import ChatImage, ChatMessage, ChatRequest, ChatStreamEvent
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.digital_human_clone.config import DigitalHumanSettings
from hugging_mac_web.digital_human_clone.schemas import (
    AsrModelOptionView,
    ConversationMessage,
    LlmVariantView,
    LoadedModelsView,
    ModelStateView,
    SetupView,
    TranscriptView,
)
from hugging_mac_web.live_transcription.config import (
    AUDIO8_PROFILE,
    NEMOTRON_3_5_ASR_PROFILE,
    AsrModelProfile,
    LiveTranscriptionSettings,
)
from hugging_mac_web.live_transcription.service import (
    DEEPFILTERNET3_MODEL_ID,
    SILERO_MODEL_ID,
    LiveTranscriptionService,
)
from hugging_mac_web.text_to_speech.audio import float32le_to_wav

NEMOTRON_ASR_MODEL_ID = NEMOTRON_3_5_ASR_PROFILE.model_id
ASR_MODEL_ID = NEMOTRON_ASR_MODEL_ID
LLM_MODEL_ID = "qwen/qwen3.5"
GEMMA_4_MODEL_ID = "google/gemma-4"
LLM_MODEL_IDS = (LLM_MODEL_ID, GEMMA_4_MODEL_ID)
TTS_MODEL_ID = "hexgrad/kokoro"

type ModelRole = Literal["asr", "llm", "tts"]
type ModelSpec = tuple[ModelRole, str, str, str]

_UNSPEAKABLE = re.compile(r"[`*_#>|~\[\]{}]+")
logger = logging.getLogger(__name__)


class DigitalHumanService:
    def __init__(self, context: PlatformContext, settings: DigitalHumanSettings) -> None:
        self._context = context
        self._settings = settings

    async def setup(
        self,
        llm_variant: str | None = None,
        asr_model_id: str | None = None,
        llm_model_id: str | None = None,
    ) -> SetupView:
        selected_llm_model, llm_variant = self._llm_selection(llm_model_id, llm_variant)
        asr_profile = self._asr_profile(asr_model_id)
        snapshots = await self._context.models.instances.snapshots()
        models: list[ModelStateView] = []
        for role, model_id, variant, runtime in self._model_specs(
            llm_variant, asr_profile.model_id, selected_llm_model
        ):
            status = await self._context.models.resources.status(
                model_id,
                variant=variant,
                options=self._options(role),
            )
            resources_ready = any(
                item.runtime == runtime and item.available for item in status.runtimes
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
                    role=role,
                    model_id=model_id,
                    runtime=runtime,
                    resources_ready=resources_ready,
                    ready_instance_id=ready,
                )
            )
        return SetupView(
            models=tuple(models),
            asr_models=self._asr_models(),
            selected_asr_model_id=asr_profile.model_id,
            llm_variants=self._llm_variants(),
            selected_llm_model_id=selected_llm_model,
            selected_llm_variant=llm_variant,
            vad_instance_id=self._ready_snapshot_id(snapshots, SILERO_MODEL_ID),
            enhancement_instance_id=self._ready_snapshot_id(
                snapshots, DEEPFILTERNET3_MODEL_ID
            ),
        )

    async def load_models(
        self,
        llm_variant: str | None = None,
        asr_model_id: str | None = None,
        llm_model_id: str | None = None,
    ) -> LoadedModelsView:
        selected_llm_model, selected_variant = self._llm_selection(
            llm_model_id, llm_variant
        )
        selected_asr = self._asr_profile(asr_model_id)
        asr, llm, tts = await asyncio.gather(
            *(
                self._load(role, model_id, variant, runtime)
                for role, model_id, variant, runtime in self._model_specs(
                    selected_variant, selected_asr.model_id, selected_llm_model
                )
            )
        )
        if selected_asr.streaming:
            # Nemotron performs stateful chunk inference directly. The utterance-level
            # VAD and enhancement pipeline is only relevant to non-streaming ASR.
            vad, enhancement = None, None
        else:
            vad, enhancement = await asyncio.gather(
                self._load_optional_pipeline_component("vad"),
                self._load_optional_pipeline_component("enhancement"),
            )
        return LoadedModelsView(
            asr_instance_id=asr,
            llm_instance_id=llm,
            tts_instance_id=tts,
            vad_instance_id=vad,
            enhancement_instance_id=enhancement,
        )

    async def load_model(
        self,
        role: str,
        llm_variant: str | None = None,
        asr_model_id: str | None = None,
        llm_model_id: str | None = None,
    ) -> str:
        selected_llm_model, selected_variant = self._llm_selection(
            llm_model_id, llm_variant
        )
        selected_asr = self._asr_profile(asr_model_id)
        for spec_role, model_id, variant, runtime in self._model_specs(
            selected_variant, selected_asr.model_id, selected_llm_model
        ):
            if spec_role == role:
                return await self._load(spec_role, model_id, variant, runtime)
        raise ResourceNotFoundError(f"Digital Human model role is not supported: {role}")

    async def _load(self, role: str, model_id: str, variant: str, runtime: str) -> str:
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

    async def transcribe(
        self,
        audio: bytes,
        instance_id: str,
        *,
        model_id: str = ASR_MODEL_ID,
        vad_instance_id: str | None = None,
        enhancement_instance_id: str | None = None,
    ) -> TranscriptView:
        profile = self._asr_profile(model_id)
        if profile.streaming:
            raise ResourceNotFoundError(
                "Streaming ASR must use the stream start/chunk/finish endpoints",
                details={"model_id": profile.model_id},
            )
        response = await LiveTranscriptionService(
            self._context,
            LiveTranscriptionSettings(
                prompt="Transcribe the speech accurately in its original language.",
                max_new_tokens=128,
            ),
        ).transcribe(
            audio,
            model_id=profile.model_id,
            instance_id=instance_id,
            vad_instance_id=vad_instance_id if self._settings.enable_vad else None,
            enhancement_instance_id=(
                enhancement_instance_id
                if self._settings.enable_speech_enhancement
                else None
            ),
            vad_threshold=self._settings.vad_threshold,
        )
        return TranscriptView(
            text=response.text.strip(),
            inference_ms=response.inference_ms,
            source_duration_seconds=response.source_duration_seconds,
            speech_duration_seconds=response.speech_duration_seconds,
            speech_segment_count=response.speech_segment_count,
            vad_inference_ms=response.vad_inference_ms,
            enhancement_inference_ms=response.enhancement_inference_ms,
        )

    async def _load_optional_pipeline_component(self, component: str) -> str | None:
        if component == "vad":
            if not self._settings.enable_vad:
                return None
            model_id = SILERO_MODEL_ID
        else:
            if not self._settings.enable_speech_enhancement:
                return None
            model_id = DEEPFILTERNET3_MODEL_ID
        try:
            manifest = self._context.models.registry.get(model_id).manifest
            status = await self._context.models.resources.status(
                model_id,
                variant=manifest.default_variant,
                options={"model_home": self._context.settings.model_home},
            )
            if not any(
                item.runtime == manifest.default_runtime and item.available
                for item in status.runtimes
            ):
                return None
            live = LiveTranscriptionService(self._context, LiveTranscriptionSettings())
            loaded = await (
                live.load_vad() if component == "vad" else live.load_enhancement()
            )
            return loaded.instance_id
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Digital Human optional ASR component unavailable component=%s model_id=%s",
                component,
                model_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _ready_snapshot_id(
        snapshots: tuple[InstanceSnapshot, ...], model_id: str
    ) -> str | None:
        return next(
            (
                snapshot.instance_id
                for snapshot in snapshots
                if snapshot.model_id == model_id and snapshot.state is ModelState.READY
            ),
            None,
        )

    async def stream_chat(
        self,
        instance_id: str,
        prompt: str,
        history: tuple[ConversationMessage, ...],
        image_path: Path,
    ) -> AsyncIterator[ChatStreamEvent]:
        instance = await self._ready_llm_instance(instance_id)
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

    async def _ready_instance(
        self, instance_id: str, model_id: str
    ) -> BaseModelInstance:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected model instance is not ready",
                details={"instance_id": instance_id, "model_id": model_id},
            )
        return instance

    async def _ready_llm_instance(self, instance_id: str) -> BaseModelInstance:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id not in LLM_MODEL_IDS or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected Digital Human LLM instance is not ready",
                details={"instance_id": instance_id, "model_id": info.model_id},
            )
        return instance

    def _options(self, role: str) -> dict[str, object]:
        options: dict[str, object] = {"model_home": self._context.settings.model_home}
        return options

    def _model_specs(
        self,
        llm_variant: str,
        asr_model_id: str = ASR_MODEL_ID,
        llm_model_id: str = LLM_MODEL_ID,
    ) -> tuple[ModelSpec, ...]:
        """Resolve every role through the shared WEB_RUNTIME_PREFERENCE policy."""

        selections = (
            ("asr", asr_model_id, self._asr_profile(asr_model_id).variant),
            ("llm", llm_model_id, llm_variant),
            ("tts", TTS_MODEL_ID, "v1.0"),
        )
        return tuple(
            (
                role,
                model_id,
                variant,
                self._context.models.runtime_policy.select(
                    self._context.models.registry.get(model_id).manifest
                ),
            )
            for role, model_id, variant in selections
        )

    def _llm_variants(self) -> tuple[LlmVariantView, ...]:
        return tuple(
            LlmVariantView(
                model_id=model_id,
                name=variant.name,
                display_name=variant.display_name,
                description=variant.description,
            )
            for model_id in LLM_MODEL_IDS
            for variant in self._context.models.registry.get(model_id).manifest.variants
        )

    @staticmethod
    def _asr_models() -> tuple[AsrModelOptionView, ...]:
        return tuple(
            AsrModelOptionView(
                model_id=profile.model_id,
                display_name=profile.display_name,
                description=profile.description,
                streaming=profile.streaming,
                streaming_chunk_seconds=profile.streaming_chunk_seconds,
            )
            for profile in (AUDIO8_PROFILE, NEMOTRON_3_5_ASR_PROFILE)
        )

    @staticmethod
    def _asr_profile(value: str | None) -> AsrModelProfile:
        selected = value or ASR_MODEL_ID
        profiles = {
            profile.model_id: profile
            for profile in (AUDIO8_PROFILE, NEMOTRON_3_5_ASR_PROFILE)
        }
        try:
            return profiles[selected]
        except KeyError as error:
            raise ResourceNotFoundError(
                "The selected Digital Human ASR model is not supported",
                details={"model_id": selected, "supported_models": list(profiles)},
            ) from error

    def _llm_selection(
        self, model_id: str | None, variant: str | None
    ) -> tuple[str, str]:
        selected_model = model_id or LLM_MODEL_ID
        if selected_model not in LLM_MODEL_IDS:
            raise ResourceNotFoundError("The selected Digital Human LLM is not registered")
        manifest = self._context.models.registry.get(selected_model).manifest
        selected_variant = variant or manifest.default_variant
        if selected_variant not in {item.name for item in manifest.variants}:
            raise ResourceNotFoundError(
                "The selected Digital Human LLM variant is not registered",
                details={"model_id": selected_model, "variant": selected_variant},
            )
        assert selected_variant is not None
        return selected_model, selected_variant


def _clean_for_speech(text: str) -> str:
    return " ".join(_UNSPEAKABLE.sub("", text).split())
