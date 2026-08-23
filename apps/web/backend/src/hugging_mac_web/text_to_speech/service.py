"""Multi-model text-to-speech use case."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.text_to_speech.audio import float32le_to_wav
from hugging_mac_web.text_to_speech.config import (
    TTS_MODEL_PROFILES,
    TextToSpeechSettings,
    TtsModelProfile,
)
from hugging_mac_web.text_to_speech.schemas import (
    LoadedTtsModelView,
    ReadyTtsInstanceView,
    SynthesizeSpeechRequest,
    TtsModelView,
    TtsResourceView,
    TtsVariantView,
)


class TextToSpeechService:
    def __init__(
        self,
        context: PlatformContext,
        settings: TextToSpeechSettings,
    ) -> None:
        self._context = context
        self._settings = settings

    async def list_models(self) -> tuple[TtsModelView, ...]:
        return tuple(
            await asyncio.gather(*(self.model_view(model_id) for model_id in TTS_MODEL_PROFILES))
        )

    async def model_view(self, model_id: str) -> TtsModelView:
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
        return TtsModelView.from_profile(
            profile,
            resource,
            ready_instance_id=next(
                (
                    item.instance_id for item in ready_instances
                    if item.variant == profile.variant and item.runtime == profile.runtime
                ),
                None,
            ),
            ready_instances=ready_instances,
            variants=tuple(
                TtsVariantView(
                    name=variant.name,
                    display_name=variant.display_name,
                    available=any(runtime.available for runtime in status.runtimes),
                    available_runtimes=tuple(
                        runtime.runtime for runtime in status.runtimes if runtime.available
                    ),
                )
                for variant, status in zip(manifest.variants, resources, strict=True)
            ),
        )

    async def resource_status(
        self, model_id: str, *, variant: str | None = None
    ) -> TtsResourceView:
        profile = self._profile(model_id, variant=variant)
        status = await self._context.models.resources.status(
            profile.model_id,
            variant=profile.variant,
            options=self._model_options(profile),
        )
        return TtsResourceView.from_sdk(status)

    async def load_model(
        self,
        model_id: str,
        *,
        variant: str | None = None,
        runtime: str | None = None,
    ) -> LoadedTtsModelView:
        configured = self._profile(model_id, variant=variant)
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
            options=self._model_options(profile),
            reuse=ReusePolicy.SHARED,
        )
        try:
            return LoadedTtsModelView.from_sdk(handle.info())
        finally:
            await handle.close()

    async def synthesize(
        self,
        request: SynthesizeSpeechRequest,
        *,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        profile = self._profile(request.model_id)
        if profile.requires_reference_voice and not request.voice:
            raise ResourceNotFoundError(
                "Audio8 MLX requires a voice profile name",
                details={"model_id": profile.model_id},
            )
        if (reference_audio is None) != (reference_text is None):
            raise ValueError("reference_audio and reference_text must be provided together")
        instance = await self._context.models.instances.require(request.instance_id)
        info = instance.info()
        if info.model_id != profile.model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected TTS instance is not ready for this model",
                details={"instance_id": request.instance_id},
            )
        if (
            profile.requires_reference_audio
            and info.runtime != "coreml"
            and (reference_audio is None or reference_text is None)
        ):
            raise ResourceNotFoundError(
                "Qwen3-TTS MLX requires reference audio and its transcript",
                details={"model_id": profile.model_id, "runtime": info.runtime},
            )
        synthesizer = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
        response = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text=request.text,
                voice=request.voice,
                language=request.language,
                speed=request.speed,
                reference_audio=(
                    AudioInput(data=reference_audio) if reference_audio is not None else None
                ),
                reference_text=reference_text,
                max_new_tokens=profile.max_new_tokens,
            )
        )
        return float32le_to_wav(response.audio, response.sample_rate), {
            "x-model-id": response.model_id,
            "x-runtime": response.runtime,
            "x-device": response.device,
            "x-sample-rate": str(response.sample_rate),
            "x-duration-seconds": f"{response.duration_seconds:.4f}",
            "x-inference-ms": f"{response.timings.inference_ms or 0:.3f}",
        }

    async def _ready_instances(self, model_id: str) -> tuple[ReadyTtsInstanceView, ...]:
        return tuple(
            ReadyTtsInstanceView(
                instance_id=snapshot.instance_id,
                variant=snapshot.variant,
                runtime=snapshot.runtime,
            )
            for snapshot in await self._context.models.instances.snapshots()
            if snapshot.model_id == model_id
            and snapshot.state is ModelState.READY
        )

    @staticmethod
    def _select_runtime(
        profile: TtsModelProfile,
        resource: TtsResourceView,
    ) -> TtsModelProfile:
        available = {runtime.runtime: runtime for runtime in resource.runtimes if runtime.available}
        selected = available.get(profile.runtime)
        if selected is None and available:
            selected = next(iter(available.values()))
        if selected is None:
            return profile
        return replace(
            profile,
            runtime=selected.runtime,
            required_artifact_id=TextToSpeechService._resource_artifact_id(
                resource, selected.runtime
            ),
        )

    @staticmethod
    def _resource_artifact_id(resource: TtsResourceView, runtime: str) -> str:
        matching = tuple(
            artifact.artifact_id
            for artifact in resource.artifacts
            if artifact.runtime == runtime and artifact.format != ArtifactFormat.TOKENIZER.value
        )
        if len(matching) != 1:
            raise ResourceNotFoundError(
                "Text to Speech runtime artifact is not uniquely declared",
                details={"model_id": resource.model_id, "runtime": runtime},
            )
        return matching[0]

    def _profile(self, model_id: str, *, variant: str | None = None) -> TtsModelProfile:
        profile = TTS_MODEL_PROFILES.get(model_id)
        if profile is None:
            raise ResourceNotFoundError(
                f"Text to Speech does not support model: {model_id}",
                details={"supported_models": list(TTS_MODEL_PROFILES)},
            )
        definition = self._context.models.registry.get(profile.model_id)
        manifest = definition.manifest
        selected_variant = manifest.get_variant(variant or manifest.default_variant).name
        runtime = (
            profile.runtime
            if profile.runtime in {item.name for item in manifest.runtimes}
            else manifest.default_runtime
        )
        if runtime is None:
            raise ResourceNotFoundError(
                "Text to Speech model has no usable runtime",
                details={"model_id": profile.model_id},
            )
        return replace(
            profile,
            variant=selected_variant,
            runtime=runtime,
            required_artifact_id=self._runtime_artifact_id(
                profile.model_id, selected_variant, runtime
            ),
        )

    def _runtime_artifact_id(self, model_id: str, variant: str, runtime: str) -> str:
        definition = self._context.models.registry.get(model_id)
        matching = tuple(
            artifact
            for artifact in definition.get_artifacts(runtime, variant=variant)
            if artifact.format is not ArtifactFormat.TOKENIZER
        )
        if len(matching) != 1:
            raise ResourceNotFoundError(
                "Text to Speech runtime artifact is not uniquely declared",
                details={"model_id": model_id, "runtime": runtime},
            )
        return matching[0].artifact_id

    def _model_options(self, profile: TtsModelProfile) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
