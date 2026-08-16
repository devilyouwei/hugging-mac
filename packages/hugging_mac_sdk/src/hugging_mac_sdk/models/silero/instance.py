"""Silero lifecycle and voice activity capability composition."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import VoiceActivityDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.voice_activity import (
    SpeechSegment,
    VoiceActivityRequest,
    VoiceActivityResponse,
    VoiceActivityTimings,
)

from .config import SILERO_MODEL_ID, SILERO_REVISION, SileroInstanceConfig
from .utils.audio import prepare_audio
from .utils.postprocess import probabilities_to_segments
from .utils.types import PreparedAudio, SileroInferenceOptions


class SileroEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def probabilities(self, prepared: PreparedAudio) -> tuple[float, ...]: ...

    async def close(self) -> None: ...


class SileroInstance(BaseModelInstance):
    def __init__(self, config: SileroInstanceConfig, engine: SileroEngine) -> None:
        super().__init__(
            model_id=SILERO_MODEL_ID,
            revision=SILERO_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._config = config
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            VoiceActivityDetection,  # type: ignore[type-abstract]
            cast(VoiceActivityDetection, self),
        )

    async def detect_voice_activity(
        self, request: VoiceActivityRequest
    ) -> VoiceActivityResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Silero instance must be READY before inference")
        started = perf_counter()
        try:
            prepared = await asyncio.to_thread(
                prepare_audio, request.audio, sample_rate=self._config.sample_rate
            )
        except Exception as error:
            raise InferenceError("Silero audio preprocessing failed", cause=error) from error
        preprocess_ms = (perf_counter() - started) * 1000

        async with self._inference_lock:
            started = perf_counter()
            try:
                probabilities = await self._engine.probabilities(prepared)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("Silero VAD inference failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        options = SileroInferenceOptions(
            threshold=request.threshold,
            min_speech_ms=request.min_speech_ms,
            min_silence_ms=request.min_silence_ms,
            speech_pad_ms=request.speech_pad_ms,
            max_speech_seconds=request.max_speech_seconds,
        )
        intervals = probabilities_to_segments(
            probabilities,
            total_samples=len(prepared.samples),  # type: ignore[arg-type]
            sample_rate=prepared.sample_rate,
            chunk_samples=self._config.chunk_samples,
            options=options,
        )
        segments = tuple(
            SpeechSegment(
                start_sample=item.start,
                end_sample=item.end,
                start_seconds=item.start / prepared.sample_rate,
                end_seconds=item.end / prepared.sample_rate,
            )
            for item in intervals
        )
        postprocess_ms = (perf_counter() - started) * 1000
        return VoiceActivityResponse(
            model_id=SILERO_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            sample_rate=prepared.sample_rate,
            duration_seconds=prepared.duration_seconds,
            speech_seconds=sum(item.end_seconds - item.start_seconds for item in segments),
            segments=segments,
            timings=VoiceActivityTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
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
