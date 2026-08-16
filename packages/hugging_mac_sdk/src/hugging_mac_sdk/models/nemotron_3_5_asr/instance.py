"""Nemotron 3.5 ASR lifecycle and capability instance."""

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

from hugging_mac_sdk.capabilities import SpeechTranscription, StreamingSpeechTranscription
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.streaming_transcription import (
    StreamingTranscriptionRequest,
    StreamingTranscriptionResponse,
    StreamingTranscriptionSession,
)
from hugging_mac_sdk.schemas.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionTimings,
)

from .config import (
    NEMOTRON_3_5_ASR_MODEL_ID,
    NEMOTRON_3_5_ASR_REVISION,
    NemotronCoreMlInstanceConfig,
)
from .utils.audio import prepare_audio
from .utils.types import AsrEngineOutput, PreparedAudio, StreamingAsrEngineOutput


class NemotronCoreMlEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput: ...
    async def start_stream(self, session_id: str) -> None: ...
    async def infer_stream(
        self, session_id: str, prepared: PreparedAudio
    ) -> StreamingAsrEngineOutput: ...
    async def finish_stream(self, session_id: str) -> StreamingAsrEngineOutput: ...
    async def cancel_stream(self, session_id: str) -> None: ...
    async def close(self) -> None: ...


class NemotronCoreMlInstance(BaseModelInstance):
    def __init__(self, config: NemotronCoreMlInstanceConfig, engine: NemotronCoreMlEngine) -> None:
        super().__init__(
            model_id=NEMOTRON_3_5_ASR_MODEL_ID,
            revision=NEMOTRON_3_5_ASR_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._config, self._engine = config, engine
        self._inference_lock = asyncio.Lock()
        self._stream_durations: dict[str, float] = {}
        self.register_capability(SpeechTranscription, cast(SpeechTranscription, self))  # type: ignore[type-abstract]
        self.register_capability(
            StreamingSpeechTranscription,  # type: ignore[type-abstract]
            cast(StreamingSpeechTranscription, self),
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Nemotron 3.5 ASR instance must be READY before transcribe")
        started = perf_counter()
        try:
            prepared = await asyncio.to_thread(
                prepare_audio,
                request.audio,
                sample_rate=self._config.sample_rate,
                max_seconds=self._config.max_audio_seconds,
            )
        except Exception as error:
            raise InferenceError("Nemotron audio preprocessing failed", cause=error) from error
        preprocess_ms = (perf_counter() - started) * 1000
        async with self._inference_lock:
            started = perf_counter()
            try:
                output = await self._engine.infer(prepared, request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("Nemotron transcription failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        text = output.text.strip()
        postprocess_ms = (perf_counter() - started) * 1000
        return TranscriptionResponse(
            model_id=NEMOTRON_3_5_ASR_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=text,
            sample_rate=prepared.sample_rate,
            duration_seconds=prepared.duration_seconds,
            generated_tokens=output.generated_tokens,
            timings=TranscriptionTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )

    async def start_stream(self) -> StreamingTranscriptionSession:
        self._require_ready()
        session_id = uuid4().hex
        async with self._inference_lock:
            await self._engine.start_stream(session_id)
            self._stream_durations[session_id] = 0.0
        return StreamingTranscriptionSession(
            session_id=session_id,
            model_id=NEMOTRON_3_5_ASR_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            sample_rate=self._config.sample_rate,
        )

    async def transcribe_stream(
        self, request: StreamingTranscriptionRequest
    ) -> StreamingTranscriptionResponse:
        self._require_ready()
        if request.session_id not in self._stream_durations:
            raise InferenceError(f"Unknown Nemotron stream session: {request.session_id}")
        started = perf_counter()
        try:
            prepared = await asyncio.to_thread(
                prepare_audio,
                request.audio,
                sample_rate=self._config.sample_rate,
                max_seconds=self._config.max_audio_seconds,
            )
        except Exception as error:
            raise InferenceError(
                "Nemotron stream audio preprocessing failed", cause=error
            ) from error
        preprocess_ms = (perf_counter() - started) * 1000
        async with self._inference_lock:
            started = perf_counter()
            try:
                output = await self._engine.infer_stream(request.session_id, prepared)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Nemotron streaming transcription failed", cause=error
                ) from error
            inference_ms = (perf_counter() - started) * 1000
            self._stream_durations[request.session_id] += prepared.duration_seconds
            duration = self._stream_durations[request.session_id]
        return self._stream_response(
            request.session_id,
            output,
            duration,
            is_final=False,
            timings=TranscriptionTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
            ),
        )

    async def finish_stream(self, session_id: str) -> StreamingTranscriptionResponse:
        self._require_ready()
        if session_id not in self._stream_durations:
            raise InferenceError(f"Unknown Nemotron stream session: {session_id}")
        async with self._inference_lock:
            started = perf_counter()
            output = await self._engine.finish_stream(session_id)
            inference_ms = (perf_counter() - started) * 1000
            duration = self._stream_durations.pop(session_id)
        return self._stream_response(
            session_id,
            output,
            duration,
            is_final=True,
            timings=TranscriptionTimings(inference_ms=inference_ms),
        )

    async def cancel_stream(self, session_id: str) -> None:
        if session_id not in self._stream_durations:
            return
        async with self._inference_lock:
            await self._engine.cancel_stream(session_id)
            self._stream_durations.pop(session_id, None)

    def _stream_response(
        self,
        session_id: str,
        output: StreamingAsrEngineOutput,
        duration: float,
        *,
        is_final: bool,
        timings: TranscriptionTimings,
    ) -> StreamingTranscriptionResponse:
        return StreamingTranscriptionResponse(
            session_id=session_id,
            model_id=NEMOTRON_3_5_ASR_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=output.text,
            delta=output.delta,
            sample_rate=self._config.sample_rate,
            audio_seconds=duration,
            generated_tokens=output.generated_tokens,
            detected_language=output.detected_language,
            is_final=is_final,
            timings=timings,
        )

    def _require_ready(self) -> None:
        if self.state is not ModelState.READY:
            raise InferenceError("Nemotron 3.5 ASR instance must be READY before streaming")

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
        self._stream_durations.clear()
