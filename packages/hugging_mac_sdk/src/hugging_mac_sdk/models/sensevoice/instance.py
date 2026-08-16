"""SenseVoiceSmall instance and composed speech capabilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import SpeechTranscription, SpeechUnderstanding
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.speech_understanding import (
    SpeechUnderstandingRequest,
    SpeechUnderstandingResponse,
)
from hugging_mac_sdk.schemas.transcription import (
    AudioInput,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionTimings,
)

from .config import (
    SENSEVOICE_SMALL_MODEL_ID,
    SENSEVOICE_SMALL_REVISION,
    SenseVoiceSmallInstanceConfig,
)
from .utils.audio import prepare_audio
from .utils.types import (
    PreparedAudio,
    SenseVoiceEngineOutput,
    SenseVoiceInferenceOptions,
)


class SenseVoiceSmallEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def infer(
        self,
        prepared: PreparedAudio,
        options: SenseVoiceInferenceOptions,
    ) -> SenseVoiceEngineOutput: ...

    async def close(self) -> None: ...


class SenseVoiceSmallInstance(BaseModelInstance):
    """Compose generic transcription and rich speech understanding."""

    def __init__(
        self,
        config: SenseVoiceSmallInstanceConfig,
        engine: SenseVoiceSmallEngine,
    ) -> None:
        super().__init__(
            model_id=SENSEVOICE_SMALL_MODEL_ID,
            revision=SENSEVOICE_SMALL_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._config = config
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            SpeechTranscription,  # type: ignore[type-abstract]
            cast(SpeechTranscription, self),
        )
        self.register_capability(
            SpeechUnderstanding,  # type: ignore[type-abstract]
            cast(SpeechUnderstanding, self),
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        output, duration, timings = await self._execute(
            request.audio,
            SenseVoiceInferenceOptions(language="auto", use_itn=True),
        )
        return TranscriptionResponse(
            model_id=SENSEVOICE_SMALL_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=output.text,
            sample_rate=self._config.sample_rate,
            duration_seconds=duration,
            generated_tokens=output.token_count,
            timings=timings,
        )

    async def understand_speech(
        self,
        request: SpeechUnderstandingRequest,
    ) -> SpeechUnderstandingResponse:
        output, duration, timings = await self._execute(
            request.audio,
            SenseVoiceInferenceOptions(
                language=request.language,
                use_itn=request.use_itn,
                ban_unknown_emotion=request.ban_unknown_emotion,
            ),
        )
        return SpeechUnderstandingResponse(
            model_id=SENSEVOICE_SMALL_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=output.text,
            raw_text=output.raw_text,
            languages=output.languages,
            emotion=output.emotion,
            events=output.events,
            sample_rate=self._config.sample_rate,
            duration_seconds=duration,
            token_count=output.token_count,
            timings=timings,
        )

    async def _execute(
        self,
        audio: AudioInput,
        options: SenseVoiceInferenceOptions,
    ) -> tuple[SenseVoiceEngineOutput, float, TranscriptionTimings]:
        if self.state is not ModelState.READY:
            raise InferenceError("SenseVoiceSmall instance must be READY before inference")
        preprocess_started = perf_counter()
        try:
            prepared = await asyncio.to_thread(
                prepare_audio,
                audio,
                sample_rate=self._config.sample_rate,
                max_seconds=self._config.max_audio_seconds,
            )
        except Exception as error:
            raise InferenceError(
                "SenseVoiceSmall audio preprocessing failed",
                cause=error,
            ) from error
        preprocess_ms = (perf_counter() - preprocess_started) * 1000

        async with self._inference_lock:
            inference_started = perf_counter()
            try:
                output = await self._engine.infer(prepared, options)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "SenseVoiceSmall speech understanding failed",
                    cause=error,
                ) from error
            inference_ms = (perf_counter() - inference_started) * 1000
        return (
            output,
            prepared.duration_seconds,
            TranscriptionTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=0.0,
            ),
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
