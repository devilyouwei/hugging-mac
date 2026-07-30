"""Multi-model live transcription use case."""

import asyncio

from hugging_mac_sdk import ArtifactFormat, AudioInput, ReusePolicy, TranscriptionRequest
from hugging_mac_sdk.capabilities import SpeechTranscription, SpeechUnderstanding
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.speech_understanding import SpeechUnderstandingRequest

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.live_transcription.config import (
    ASR_MODEL_PROFILES,
    AsrModelProfile,
    LiveTranscriptionSettings,
)
from hugging_mac_web.live_transcription.schemas import (
    AsrModelView,
    LoadedModelView,
    ResourceStatusView,
    TranscriptionResultView,
)


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
        profile = self._profile(model_id)
        resource = await self.resource_status(model_id)
        return AsrModelView.from_profile(
            profile,
            resource,
            ready_instance_id=await self._ready_instance_id(profile),
        )

    async def resource_status(self, model_id: str) -> ResourceStatusView:
        profile = self._profile(model_id)
        status = await self._context.models.resources.status(
            profile.model_id,
            variant=profile.variant,
            options=self._model_options,
        )
        return ResourceStatusView.from_sdk(status)

    async def download_source(
        self,
        model_id: str,
        *,
        overwrite: bool = False,
    ) -> ResourceStatusView:
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
    ) -> ResourceStatusView:
        profile = self._profile(model_id)
        if not profile.supports_coreml_conversion:
            raise ResourceNotFoundError(f"Core ML conversion is not available for {model_id}")
        await self._context.models.resources.convert(
            profile.model_id,
            ArtifactFormat.COREML,
            variant=profile.variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status(model_id)

    async def load_model(self, model_id: str) -> LoadedModelView:
        profile = self._profile(model_id)
        handle = await self._context.models.load(
            profile.model_id,
            variant=profile.variant,
            runtime=profile.runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        )
        try:
            return LoadedModelView.from_sdk(handle.info())
        finally:
            await handle.close()

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str,
        instance_id: str,
    ) -> TranscriptionResultView:
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

        audio_input = AudioInput(data=audio)
        if profile.rich_understanding and instance.supports(SpeechUnderstanding):
            understanding = instance.require(SpeechUnderstanding)  # type: ignore[type-abstract]
            response = await understanding.understand_speech(
                SpeechUnderstandingRequest(audio=audio_input)
            )
            return TranscriptionResultView.from_understanding(response)

        transcriber = instance.require(SpeechTranscription)  # type: ignore[type-abstract]
        response = await transcriber.transcribe(
            TranscriptionRequest(
                audio=audio_input,
                prompt=self._settings.prompt,
                max_new_tokens=self._settings.max_new_tokens,
            )
        )
        return TranscriptionResultView.from_sdk(response)

    async def _ready_instance_id(self, profile: AsrModelProfile) -> str | None:
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
    def _profile(model_id: str) -> AsrModelProfile:
        profile = ASR_MODEL_PROFILES.get(model_id)
        if profile is None:
            raise ResourceNotFoundError(
                f"Live Transcription does not support model: {model_id}",
                details={"supported_models": list(ASR_MODEL_PROFILES)},
            )
        return profile

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
