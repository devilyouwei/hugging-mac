"""Multi-model live transcription use case."""

import asyncio
from dataclasses import replace
from time import perf_counter
from typing import Literal

from hugging_mac_sdk import (
    AudioInput,
    ReusePolicy,
    SpeechEnhancementRequest,
    StreamingTranscriptionRequest,
    TranscriptionRequest,
    VoiceActivityRequest,
)
from hugging_mac_sdk.capabilities import (
    SpeechEnhancement,
    SpeechTranscription,
    SpeechUnderstanding,
    StreamingSpeechTranscription,
    VoiceActivityDetection,
)
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.speech_understanding import SpeechUnderstandingRequest

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.live_transcription.audio_pipeline import prepare_detected_speech
from hugging_mac_web.live_transcription.config import (
    ASR_MODEL_PROFILES,
    QWEN3_ASR_PROFILE,
    SENSEVOICE_PROFILE,
    AsrModelProfile,
    LiveTranscriptionSettings,
)
from hugging_mac_web.live_transcription.schemas import (
    AsrModelView,
    AsrVariantView,
    LoadedModelView,
    PipelineComponentView,
    ReadyAsrInstanceView,
    ResourceStatusView,
    StreamingSessionView,
    StreamingTranscriptionView,
    TranscriptionResultView,
    VadDetectionView,
    encoded_input_audio,
)
from hugging_mac_web.shared.utils.log_util import get_logger

SILERO_MODEL_ID = "snakers4/silero-vad"
DEEPFILTERNET3_MODEL_ID = "deepfilternet/deepfilternet3"
logger = get_logger("live_transcription.service")


class LiveTranscriptionService:
    def __init__(
        self,
        context: PlatformContext,
        settings: LiveTranscriptionSettings,
    ) -> None:
        self._context = context
        self._settings = settings

    async def list_models(self) -> tuple[AsrModelView, ...]:
        return tuple(
            await asyncio.gather(*(self.model_view(model_id) for model_id in ASR_MODEL_PROFILES))
        )

    async def model_view(self, model_id: str) -> AsrModelView:
        configured = self._profile(model_id)
        manifest = self._context.models.registry.get(model_id).manifest
        resources = await asyncio.gather(
            *(self.resource_status(model_id, variant=item.name) for item in manifest.variants)
        )
        available_variant = next(
            (
                item.variant
                for item in resources
                if any(runtime.available for runtime in item.runtimes)
            ),
            None,
        )
        selected_variant = (
            configured.variant
            if any(
                item.variant == configured.variant
                and any(runtime.available for runtime in item.runtimes)
                for item in resources
            )
            else available_variant or configured.variant
        )
        resource = next(item for item in resources if item.variant == selected_variant)
        configured = replace(configured, variant=selected_variant)
        profile = self._select_runtime(configured, resource)
        ready_instances = await self._ready_instances(profile.model_id)
        return AsrModelView.from_profile(
            profile,
            resource,
            ready_instance_id=next(
                (
                    item.instance_id
                    for item in ready_instances
                    if item.variant == profile.variant and item.runtime == profile.runtime
                ),
                None,
            ),
            ready_instances=ready_instances,
            variants=tuple(
                AsrVariantView(
                    name=variant.name,
                    display_name=variant.display_name,
                    available=any(runtime.available for runtime in status.runtimes),
                    available_runtimes=tuple(
                        runtime.runtime for runtime in status.runtimes if runtime.available
                    ),
                    streaming_chunk_seconds=(
                        float(variant.metadata["chunk_ms"]) / 1000
                        if "chunk_ms" in variant.metadata
                        else None
                    ),
                )
                for variant, status in zip(manifest.variants, resources, strict=True)
            ),
        )

    async def resource_status(
        self, model_id: str, *, variant: str | None = None
    ) -> ResourceStatusView:
        profile = self._profile(model_id)
        selected_variant = variant or profile.variant
        status = await self._context.models.resources.status(
            profile.model_id,
            variant=selected_variant,
            options=self._model_options_for(profile.model_id, profile.runtime),
        )
        return ResourceStatusView.from_sdk(status)

    async def load_model(
        self,
        model_id: str,
        *,
        variant: str | None = None,
        runtime: str | None = None,
    ) -> LoadedModelView:
        started = perf_counter()
        logger.info(
            "asr_model_load_requested",
            model_id=model_id,
            requested_variant=variant,
            requested_runtime=runtime,
        )
        configured = self._profile(model_id)
        if variant is not None:
            manifest = self._context.models.registry.get(model_id).manifest
            if variant not in {item.name for item in manifest.variants}:
                raise ResourceNotFoundError(
                    f"Variant {variant} is not supported by {model_id}",
                    details={"model_id": model_id, "variant": variant},
                )
            configured = replace(configured, variant=variant)
        resource = await self.resource_status(model_id, variant=configured.variant)
        profile = self._select_runtime(configured, resource)
        if runtime is not None:
            selected = next(
                (item for item in resource.runtimes if item.runtime == runtime),
                None,
            )
            if selected is None:
                raise ResourceNotFoundError(
                    f"Runtime {runtime} is not supported by {model_id}",
                    details={"model_id": model_id, "runtime": runtime},
                )
            if not selected.available:
                raise ResourceNotFoundError(
                    f"Runtime {runtime} assets are not available for {model_id}",
                    details={"model_id": model_id, "runtime": runtime},
                )
            profile = replace(
                profile,
                runtime=runtime,
                required_artifact_id=self._resource_artifact_id(resource, runtime),
            )
        handle = await self._context.models.load(
            profile.model_id,
            variant=profile.variant,
            runtime=profile.runtime,
            options=self._model_options_for(profile.model_id, profile.runtime),
            reuse=ReusePolicy.SHARED,
        )
        try:
            loaded = LoadedModelView.from_sdk(handle.info())
            logger.info(
                "asr_model_load_completed",
                model_id=loaded.model_id,
                instance_id=loaded.instance_id,
                variant=loaded.variant,
                runtime=loaded.runtime,
                device=loaded.device,
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )
            return loaded
        finally:
            await handle.close()

    async def load_vad(self) -> LoadedModelView:
        started = perf_counter()
        logger.info("pipeline_model_load_requested", component="vad", model_id=SILERO_MODEL_ID)
        manifest = self._context.models.registry.get(SILERO_MODEL_ID).manifest
        handle = await self._context.models.load(
            SILERO_MODEL_ID,
            variant=manifest.default_variant,
            runtime=manifest.default_runtime,
            options=self._base_model_options,
            reuse=ReusePolicy.SHARED,
        )
        try:
            loaded = LoadedModelView.from_sdk(handle.info())
            logger.info(
                "pipeline_model_load_completed",
                component="vad",
                model_id=loaded.model_id,
                instance_id=loaded.instance_id,
                runtime=loaded.runtime,
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )
            return loaded
        finally:
            await handle.close()

    async def pipeline_components(self) -> tuple[PipelineComponentView, ...]:
        """Describe optional VAD and enhancement resources without loading them."""

        return tuple(
            await asyncio.gather(
                self._pipeline_component(
                    component_id="vad",
                    model_id=SILERO_MODEL_ID,
                    description=(
                        "Neural voice activity detection and pause-based utterance splitting."
                    ),
                ),
                self._pipeline_component(
                    component_id="enhancement",
                    model_id=DEEPFILTERNET3_MODEL_ID,
                    description="Core ML speech denoising before ASR inference.",
                ),
            )
        )

    async def load_enhancement(self) -> LoadedModelView:
        """Load the shared DeepFilterNet3 instance using its Core ML default."""
        started = perf_counter()
        logger.info(
            "pipeline_model_load_requested",
            component="enhancement",
            model_id=DEEPFILTERNET3_MODEL_ID,
        )
        manifest = self._context.models.registry.get(DEEPFILTERNET3_MODEL_ID).manifest
        handle = await self._context.models.load(
            DEEPFILTERNET3_MODEL_ID,
            variant=manifest.default_variant,
            runtime=manifest.default_runtime,
            options=self._base_model_options,
            reuse=ReusePolicy.SHARED,
        )
        try:
            loaded = LoadedModelView.from_sdk(handle.info())
            logger.info(
                "pipeline_model_load_completed",
                component="enhancement",
                model_id=loaded.model_id,
                instance_id=loaded.instance_id,
                runtime=loaded.runtime,
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )
            return loaded
        finally:
            await handle.close()

    async def detect_voice_activity(
        self,
        audio: bytes,
        *,
        instance_id: str,
        threshold: float,
    ) -> VadDetectionView:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != SILERO_MODEL_ID or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The Silero VAD instance is not ready",
                details={"instance_id": instance_id, "model_id": info.model_id},
            )
        detector = instance.require(VoiceActivityDetection)  # type: ignore[type-abstract]
        response = await detector.detect_voice_activity(
            VoiceActivityRequest(
                audio=AudioInput(data=audio),
                threshold=threshold,
                min_speech_ms=96,
                min_silence_ms=160,
                speech_pad_ms=64,
            )
        )
        trailing_silence_seconds = (
            response.duration_seconds - response.segments[-1].end_seconds
            if response.segments
            else response.duration_seconds
        )
        return VadDetectionView(
            # The client sends a rolling context window. Only speech close to
            # its tail describes the current microphone state.
            voiced=bool(response.segments) and trailing_silence_seconds <= 0.24,
            speech_seconds=response.speech_seconds,
            duration_seconds=response.duration_seconds,
            inference_ms=response.timings.inference_ms,
        )

    async def start_stream(
        self,
        *,
        model_id: str,
        instance_id: str,
    ) -> StreamingSessionView:
        instance = await self._require_streaming_instance(model_id, instance_id)
        streamer = instance.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
        return StreamingSessionView.from_sdk(await streamer.start_stream())

    async def transcribe_stream_chunk(
        self,
        audio: bytes,
        *,
        model_id: str,
        instance_id: str,
        session_id: str,
    ) -> StreamingTranscriptionView:
        instance = await self._require_streaming_instance(model_id, instance_id)
        streamer = instance.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
        response = await streamer.transcribe_stream(
            StreamingTranscriptionRequest(
                session_id=session_id,
                audio=AudioInput(data=audio),
            )
        )
        return StreamingTranscriptionView.from_sdk(response, input_audio=audio)

    async def finish_stream(
        self,
        *,
        model_id: str,
        instance_id: str,
        session_id: str,
    ) -> StreamingTranscriptionView:
        instance = await self._require_streaming_instance(model_id, instance_id)
        streamer = instance.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
        return StreamingTranscriptionView.from_sdk(await streamer.finish_stream(session_id))

    async def cancel_stream(
        self,
        *,
        model_id: str,
        instance_id: str,
        session_id: str,
    ) -> None:
        instance = await self._require_streaming_instance(model_id, instance_id)
        streamer = instance.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
        await streamer.cancel_stream(session_id)

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str,
        instance_id: str,
        use_vad: bool = True,
        use_enhancement: bool = True,
        include_input_audio: bool = True,
        vad_instance_id: str | None = None,
        enhancement_instance_id: str | None = None,
        vad_threshold: float = 0.5,
    ) -> TranscriptionResultView:
        started = perf_counter()
        source_bytes = len(audio)
        logger.info(
            "asr_inference_started",
            model_id=model_id,
            instance_id=instance_id,
            input_bytes=source_bytes,
            offline_vad_enabled=use_vad and bool(vad_instance_id),
            inline_enhancement_enabled=use_enhancement and bool(enhancement_instance_id),
            preprocessed_upstream=not use_vad and not use_enhancement,
            vad_instance_id=vad_instance_id,
            enhancement_instance_id=enhancement_instance_id,
        )
        profile = self._profile(model_id)
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != profile.model_id:
            raise ResourceNotFoundError(
                "The selected ASR instance does not belong to this model",
                details={
                    "instance_id": instance_id,
                    "expected_model_id": profile.model_id,
                    "actual_model_id": info.model_id,
                },
            )
        if info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                f"ASR model instance is not ready: {instance_id}",
                details={"instance_id": instance_id, "state": info.state.value},
            )

        source_duration_seconds: float | None = None
        speech_duration_seconds: float | None = None
        speech_segment_count = 0
        vad_inference_ms: float | None = None
        enhancement_inference_ms: float | None = None
        if use_vad and vad_instance_id:
            try:
                logger.info(
                    "offline_vad_started",
                    instance_id=vad_instance_id,
                    input_bytes=len(audio),
                    threshold=vad_threshold,
                )
                vad_instance = await self._require_vad_instance(vad_instance_id)
                detector = vad_instance.require(VoiceActivityDetection)  # type: ignore[type-abstract]
                vad_response = await detector.detect_voice_activity(
                    VoiceActivityRequest(
                        audio=AudioInput(data=audio),
                        threshold=vad_threshold,
                        min_speech_ms=120,
                        min_silence_ms=300,
                        speech_pad_ms=80,
                        max_speech_seconds=25.0,
                    )
                )
                vad_inference_ms = vad_response.timings.inference_ms
                source_duration_seconds = vad_response.duration_seconds
                if not vad_response.segments:
                    result = TranscriptionResultView(
                        text="",
                        model_id=profile.model_id,
                        instance_id=info.instance_id,
                        runtime=info.runtime or profile.runtime,
                        device=info.device or "",
                        sample_rate=vad_response.sample_rate,
                        duration_seconds=vad_response.duration_seconds,
                        generated_tokens=0,
                        inference_ms=0.0,
                        source_duration_seconds=vad_response.duration_seconds,
                        speech_duration_seconds=0.0,
                        speech_segment_count=0,
                        vad_inference_ms=vad_inference_ms,
                        enhancement_inference_ms=None,
                    )
                    logger.info(
                        "offline_vad_completed",
                        instance_id=vad_instance_id,
                        segments=0,
                        speech_seconds=0.0,
                        duration_seconds=vad_response.duration_seconds,
                        inference_ms=vad_inference_ms,
                    )
                    self._log_transcription_completed(
                        result,
                        started=started,
                        source_bytes=source_bytes,
                        prepared_bytes=0,
                        path="vad_no_speech",
                    )
                    return result
                prepared = await asyncio.to_thread(
                    prepare_detected_speech,
                    audio,
                    vad_response.segments,
                    vad_sample_rate=vad_response.sample_rate,
                )
                audio = prepared.wav
                source_duration_seconds = prepared.source_duration_seconds
                speech_duration_seconds = prepared.speech_duration_seconds
                speech_segment_count = prepared.segment_count
                logger.info(
                    "offline_vad_completed",
                    instance_id=vad_instance_id,
                    segments=speech_segment_count,
                    speech_seconds=speech_duration_seconds,
                    duration_seconds=source_duration_seconds,
                    inference_ms=vad_inference_ms,
                    prepared_bytes=len(audio),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "offline_vad_failed_continuing_without_vad",
                    instance_id=vad_instance_id,
                    exc_info=True,
                )
        elif use_vad:
            logger.warning("offline_vad_skipped", reason="missing_instance_id")

        if use_enhancement and enhancement_instance_id:
            try:
                enhancement_instance = await self._require_enhancement_instance(
                    enhancement_instance_id
                )
                enhancer = enhancement_instance.require(SpeechEnhancement)  # type: ignore[type-abstract]
                logger.info(
                    "inline_enhancement_started",
                    instance_id=enhancement_instance_id,
                    input_bytes=len(audio),
                )
                enhanced = await enhancer.enhance_speech(
                    SpeechEnhancementRequest(audio=AudioInput(data=audio))
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "inline_enhancement_failed_continuing_without_enhancement",
                    instance_id=enhancement_instance_id,
                    exc_info=True,
                )
            else:
                audio = enhanced.audio
                enhancement_inference_ms = enhanced.timings.inference_ms
                logger.info(
                    "inline_enhancement_completed",
                    instance_id=enhancement_instance_id,
                    output_bytes=len(audio),
                    duration_seconds=enhanced.duration_seconds,
                    inference_ms=enhancement_inference_ms,
                    sample_rate=enhanced.sample_rate,
                )
        elif use_enhancement:
            logger.warning("inline_enhancement_skipped", reason="missing_instance_id")

        audio_input = AudioInput(data=audio)
        if profile.rich_understanding and instance.supports(SpeechUnderstanding):
            understanding = instance.require(SpeechUnderstanding)  # type: ignore[type-abstract]
            understanding_response = await understanding.understand_speech(
                SpeechUnderstandingRequest(audio=audio_input)
            )
            result = TranscriptionResultView.from_understanding(understanding_response)
            completed = result.model_copy(
                update={
                    "source_duration_seconds": source_duration_seconds,
                    "speech_duration_seconds": speech_duration_seconds,
                    "speech_segment_count": speech_segment_count,
                    "vad_inference_ms": vad_inference_ms,
                    "enhancement_inference_ms": enhancement_inference_ms,
                    **(encoded_input_audio(audio) if include_input_audio else {}),
                }
            )
            self._log_transcription_completed(
                completed,
                started=started,
                source_bytes=source_bytes,
                prepared_bytes=len(audio),
                path="speech_understanding",
            )
            return completed

        transcriber = instance.require(SpeechTranscription)  # type: ignore[type-abstract]
        transcription_response = await transcriber.transcribe(
            TranscriptionRequest(
                audio=audio_input,
                prompt=self._settings.prompt,
                max_new_tokens=self._settings.max_new_tokens,
            )
        )
        result = TranscriptionResultView.from_sdk(transcription_response)
        completed = result.model_copy(
            update={
                "source_duration_seconds": source_duration_seconds,
                "speech_duration_seconds": speech_duration_seconds,
                "speech_segment_count": speech_segment_count,
                "vad_inference_ms": vad_inference_ms,
                "enhancement_inference_ms": enhancement_inference_ms,
                **(encoded_input_audio(audio) if include_input_audio else {}),
            }
        )
        self._log_transcription_completed(
            completed,
            started=started,
            source_bytes=source_bytes,
            prepared_bytes=len(audio),
            path="speech_transcription",
        )
        return completed

    @staticmethod
    def _log_transcription_completed(
        result: TranscriptionResultView,
        *,
        started: float,
        source_bytes: int,
        prepared_bytes: int,
        path: str,
    ) -> None:
        logger.info(
            "asr_inference_completed",
            model_id=result.model_id,
            instance_id=result.instance_id,
            path=path,
            source_bytes=source_bytes,
            prepared_bytes=prepared_bytes,
            text_length=len(result.text),
            generated_tokens=result.generated_tokens,
            model_inference_ms=result.inference_ms,
            wall_ms=round((perf_counter() - started) * 1000, 3),
            runtime=result.runtime,
            device=result.device,
            speech_segments=result.speech_segment_count,
        )

    async def _require_vad_instance(self, instance_id: str) -> BaseModelInstance:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != SILERO_MODEL_ID or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The Silero VAD instance is not ready",
                details={"instance_id": instance_id, "model_id": info.model_id},
            )
        return instance

    async def _require_enhancement_instance(self, instance_id: str) -> BaseModelInstance:
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != DEEPFILTERNET3_MODEL_ID or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The DeepFilterNet3 enhancement instance is not ready",
                details={"instance_id": instance_id, "model_id": info.model_id},
            )
        return instance

    async def _require_streaming_instance(
        self, model_id: str, instance_id: str
    ) -> BaseModelInstance:
        profile = self._profile(model_id)
        if not profile.streaming:
            raise ResourceNotFoundError(
                "The selected ASR model does not support streaming transcription",
                details={"model_id": model_id},
            )
        instance = await self._context.models.instances.require(instance_id)
        info = instance.info()
        if info.model_id != model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The streaming ASR instance is not ready",
                details={
                    "instance_id": instance_id,
                    "expected_model_id": model_id,
                    "actual_model_id": info.model_id,
                    "state": info.state.value,
                },
            )
        if not instance.supports(StreamingSpeechTranscription):
            raise ResourceNotFoundError(
                "The ASR instance does not expose streaming transcription",
                details={"instance_id": instance_id, "model_id": model_id},
            )
        return instance

    async def _ready_instances(self, model_id: str) -> tuple[ReadyAsrInstanceView, ...]:
        return tuple(
            ReadyAsrInstanceView(
                instance_id=snapshot.instance_id,
                variant=snapshot.variant,
                runtime=snapshot.runtime,
            )
            for snapshot in await self._context.models.instances.snapshots()
            if snapshot.model_id == model_id and snapshot.state is ModelState.READY
        )

    async def _pipeline_component(
        self,
        *,
        component_id: str,
        model_id: str,
        description: str,
    ) -> PipelineComponentView:
        definition = self._context.models.registry.get(model_id)
        manifest = definition.manifest
        runtime = manifest.default_runtime
        if runtime is None:
            raise ResourceNotFoundError(
                "Optional pipeline component has no default runtime",
                details={"model_id": model_id},
            )
        status = await self._context.models.resources.status(
            model_id,
            variant=manifest.default_variant,
            options=self._base_model_options,
        )
        downloaded = any(item.runtime == runtime and item.available for item in status.runtimes)
        ready = next(
            (
                snapshot
                for snapshot in await self._context.models.instances.snapshots()
                if snapshot.model_id == model_id
                and snapshot.variant == manifest.default_variant
                and snapshot.runtime == runtime
                and snapshot.state is ModelState.READY
            ),
            None,
        )
        loaded_model = None
        if ready is not None:
            instance = await self._context.models.instances.require(ready.instance_id)
            loaded_model = LoadedModelView.from_sdk(instance.info())
        state: Literal["not-downloaded", "not-loaded", "loaded"] = (
            "loaded"
            if loaded_model is not None
            else "not-loaded"
            if downloaded
            else "not-downloaded"
        )
        return PipelineComponentView(
            component_id=component_id,
            model_id=model_id,
            display_name=manifest.display_name,
            description=description,
            runtime=runtime,
            downloaded=downloaded,
            state=state,
            loaded_model=loaded_model,
        )

    @staticmethod
    def _select_runtime(
        profile: AsrModelProfile,
        resource: ResourceStatusView,
    ) -> AsrModelProfile:
        available = {runtime.runtime: runtime for runtime in resource.runtimes if runtime.available}
        selected = available.get(profile.runtime)
        if selected is None and available:
            selected = next(iter(available.values()))
        if selected is None:
            return profile
        return replace(
            profile,
            runtime=selected.runtime,
            required_artifact_id=LiveTranscriptionService._resource_artifact_id(
                resource, selected.runtime
            ),
        )

    @staticmethod
    def _resource_artifact_id(resource: ResourceStatusView, runtime: str) -> str:
        matching = tuple(
            artifact.artifact_id for artifact in resource.artifacts if artifact.runtime == runtime
        )
        if len(matching) != 1:
            raise ResourceNotFoundError(
                "Live Transcription runtime artifact is not uniquely declared",
                details={"model_id": resource.model_id, "runtime": runtime},
            )
        return matching[0]

    def _profile(self, model_id: str) -> AsrModelProfile:
        profile = ASR_MODEL_PROFILES.get(model_id)
        if profile is None:
            raise ResourceNotFoundError(
                f"Live Transcription does not support model: {model_id}",
                details={"supported_models": list(ASR_MODEL_PROFILES)},
            )
        manifest = self._context.models.registry.get(profile.model_id).manifest
        runtime = (
            profile.runtime
            if profile.runtime in {item.name for item in manifest.runtimes}
            else manifest.default_runtime
        )
        if runtime is None:
            raise ResourceNotFoundError(
                "Live Transcription model has no usable runtime",
                details={"model_id": profile.model_id},
            )
        return replace(
            profile,
            variant=manifest.default_variant,
            runtime=runtime,
        )

    @property
    def _base_model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}

    def _runtime_artifact_id(self, model_id: str, variant: str, runtime: str) -> str:
        definition = self._context.models.registry.get(model_id)
        return definition.get_artifact(runtime, variant=variant).artifact_id

    def _model_options_for(self, model_id: str, runtime: str = "coreml") -> dict[str, object]:
        options = self._base_model_options
        if runtime == "coreml" and model_id == QWEN3_ASR_PROFILE.model_id:
            options["device"] = self._settings.coreml_compute_units
        elif runtime == "coreml" and model_id == SENSEVOICE_PROFILE.model_id:
            options["compute_units"] = self._settings.coreml_compute_units
        return options
