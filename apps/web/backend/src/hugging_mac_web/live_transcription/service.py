"""Audio8-ASR live transcription use case."""

from hugging_mac_sdk import ArtifactFormat, AudioInput, ReusePolicy, TranscriptionRequest
from hugging_mac_sdk.capabilities import SpeechTranscription

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.schemas import (
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

    async def resource_status(self) -> ResourceStatusView:
        status = await self._context.models.resources.status(
            self._settings.model_id,
            variant=self._settings.model_variant,
            options=self._model_options,
        )
        return ResourceStatusView.from_sdk(status)

    async def download_source(self, *, overwrite: bool = False) -> ResourceStatusView:
        await self._context.models.resources.download_source(
            self._settings.model_id,
            variant=self._settings.model_variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status()

    async def convert_coreml(self, *, overwrite: bool = False) -> ResourceStatusView:
        await self._context.models.resources.convert(
            self._settings.model_id,
            ArtifactFormat.COREML,
            variant=self._settings.model_variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status()

    async def transcribe(self, audio: bytes) -> TranscriptionResultView:
        async with await self._context.models.acquire(
            self._settings.model_id,
            variant=self._settings.model_variant,
            runtime=self._settings.runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        ) as handle:
            transcriber = handle.require(SpeechTranscription)  # type: ignore[type-abstract]
            response = await transcriber.transcribe(
                TranscriptionRequest(
                    audio=AudioInput(data=audio),
                    prompt=self._settings.prompt,
                    max_new_tokens=self._settings.max_new_tokens,
                )
            )
        return TranscriptionResultView.from_sdk(response)

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
