"""Multi-model text-to-speech use case."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from hugging_mac_sdk import ArtifactFormat, ReusePolicy
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.text_to_speech.audio import float32le_to_wav
from hugging_mac_web.text_to_speech.config import (
    TTS_MODEL_PROFILES,
    TextToSpeechSettings,
    TtsModelProfile,
)
from hugging_mac_web.text_to_speech.schemas import (
    LoadedTtsModelView,
    SynthesizeSpeechRequest,
    TtsModelView,
    TtsResourceView,
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
            await asyncio.gather(
                *(self.model_view(model_id) for model_id in TTS_MODEL_PROFILES)
            )
        )

    async def model_view(self, model_id: str) -> TtsModelView:
        profile = self._profile(model_id)
        resource = await self.resource_status(model_id)
        profile = self._effective_profile(profile, resource)
        return TtsModelView.from_profile(
            profile,
            resource,
            ready_instance_id=await self._ready_instance_id(profile),
        )

    async def resource_status(self, model_id: str) -> TtsResourceView:
        profile = self._profile(model_id)
        status = await self._context.models.resources.status(
            profile.model_id,
            variant=profile.variant,
            options=self._model_options,
        )
        return TtsResourceView.from_sdk(status)

    async def download_source(
        self,
        model_id: str,
        *,
        overwrite: bool = False,
    ) -> TtsResourceView:
        profile = self._profile(model_id)
        await self._context.models.resources.download_source(
            profile.model_id,
            variant=profile.variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status(model_id)

    async def convert_coreml(
        self,
        model_id: str,
        *,
        overwrite: bool = False,
    ) -> TtsResourceView:
        profile = self._profile(model_id)
        if not profile.supports_coreml_conversion:
            raise ResourceNotFoundError(
                f"Core ML conversion is not available for {model_id}"
            )
        await self._context.models.resources.convert(
            profile.model_id,
            ArtifactFormat.COREML,
            variant=profile.variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status(model_id)

    async def load_model(self, model_id: str) -> LoadedTtsModelView:
        profile = self._profile(model_id)
        resource = await self.resource_status(model_id)
        profile = self._effective_profile(profile, resource)
        handle = await self._context.models.load(
            profile.model_id,
            variant=profile.variant,
            runtime=profile.runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        )
        try:
            return LoadedTtsModelView.from_sdk(handle.info())
        finally:
            await handle.close()

    async def synthesize(self, request: SynthesizeSpeechRequest) -> tuple[bytes, dict[str, str]]:
        profile = self._profile(request.model_id)
        instance = await self._context.models.instances.require(request.instance_id)
        info = instance.info()
        if info.model_id != profile.model_id or info.state is not ModelState.READY:
            raise ResourceNotFoundError(
                "The selected TTS instance is not ready for this model",
                details={"instance_id": request.instance_id},
            )
        synthesizer = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
        response = await synthesizer.synthesize(
            SpeechSynthesisRequest(
                text=request.text,
                voice=request.voice,
                language=request.language,
                speed=request.speed,
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

    async def _ready_instance_id(self, profile: TtsModelProfile) -> str | None:
        for snapshot in await self._context.models.instances.snapshots():
            if (
                snapshot.model_id == profile.model_id
                and snapshot.variant == profile.variant
                and snapshot.runtime == profile.runtime
                and snapshot.state is ModelState.READY
            ):
                return snapshot.instance_id
        return None

    @staticmethod
    def _effective_profile(
        profile: TtsModelProfile,
        resource: TtsResourceView,
    ) -> TtsModelProfile:
        if profile.model_id != "hexgrad/kokoro-82m":
            return profile
        coreml_ready = any(
            item.artifact_id == "coreml" and item.available
            for item in resource.artifacts
        )
        return replace(
            profile,
            runtime="coreml" if coreml_ready else "pytorch-mps",
            required_artifact_id="coreml" if coreml_ready else "source",
        )

    @staticmethod
    def _profile(model_id: str) -> TtsModelProfile:
        profile = TTS_MODEL_PROFILES.get(model_id)
        if profile is None:
            raise ResourceNotFoundError(
                f"Text to Speech does not support model: {model_id}",
                details={"supported_models": list(TTS_MODEL_PROFILES)},
            )
        return profile

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
