"""Silero lifecycle and voice activity capability composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import VoiceActivityDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.voice_activity import (
    SpeechSegment,
    VoiceActivityRequest,
    VoiceActivityResponse,
    VoiceActivityTimings,
)

from .config import SileroInstanceConfig
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

    def create_stream(self) -> SileroEngineStream: ...

    async def close(self) -> None: ...


class SileroEngineStream(Protocol):
    async def probability(self, samples: tuple[float, ...]) -> float: ...


@dataclass(frozen=True, slots=True)
class SileroStreamingResult:
    voiced: bool
    speech_started: bool
    speech_ended: bool
    probability: float
    processed_samples: int


class SileroStreamingDetector:
    """Connection-scoped Silero state and speech/silence hysteresis."""

    def __init__(
        self,
        engine: SileroEngineStream,
        inference_lock: asyncio.Lock,
        *,
        chunk_samples: int,
        sample_rate: int,
        threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
    ) -> None:
        self._engine = engine
        self._inference_lock = inference_lock
        self._chunk_samples = chunk_samples
        self._threshold = threshold
        self._min_speech_samples = round(sample_rate * min_speech_ms / 1000)
        self._min_silence_samples = round(sample_rate * min_silence_ms / 1000)
        self._buffer: list[float] = []
        self._candidate_speech_samples = 0
        self._silence_samples = 0
        self._voiced = False
        self._last_probability = 0.0

    async def push(self, samples: tuple[float, ...]) -> SileroStreamingResult:
        self._buffer.extend(samples)
        speech_started = False
        speech_ended = False
        processed_samples = 0
        while len(self._buffer) >= self._chunk_samples:
            frame = tuple(self._buffer[: self._chunk_samples])
            del self._buffer[: self._chunk_samples]
            async with self._inference_lock:
                probability = await self._engine.probability(frame)
            self._last_probability = probability
            processed_samples += self._chunk_samples
            if probability >= self._threshold:
                self._silence_samples = 0
                if not self._voiced:
                    self._candidate_speech_samples += self._chunk_samples
                    if self._candidate_speech_samples >= self._min_speech_samples:
                        self._voiced = True
                        speech_started = True
            else:
                self._candidate_speech_samples = 0
                if self._voiced:
                    self._silence_samples += self._chunk_samples
                    if self._silence_samples >= self._min_silence_samples:
                        self._voiced = False
                        self._silence_samples = 0
                        speech_ended = True

        return SileroStreamingResult(
            voiced=self._voiced,
            speech_started=speech_started,
            speech_ended=speech_ended,
            probability=self._last_probability,
            processed_samples=processed_samples,
        )

    def reset_boundary(self) -> None:
        """Start endpoint hysteresis for a new utterance without dropping audio."""

        self._candidate_speech_samples = 0
        self._silence_samples = 0
        self._voiced = False


class SileroInstance(BaseModelInstance):
    def __init__(
        self, config: SileroInstanceConfig, engine: SileroEngine, manifest: ModelManifest
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._config = config
        self._manifest_model_id = manifest.model_id
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            VoiceActivityDetection,  # type: ignore[type-abstract]
            cast(VoiceActivityDetection, self),
        )

    def create_streaming_detector(
        self,
        *,
        threshold: float,
        min_speech_ms: int = 96,
        min_silence_ms: int = 800,
    ) -> SileroStreamingDetector:
        if self.state is not ModelState.READY:
            raise InferenceError("Silero instance must be READY before streaming")
        return SileroStreamingDetector(
            self._engine.create_stream(),
            self._inference_lock,
            chunk_samples=self._config.chunk_samples,
            sample_rate=self._config.sample_rate,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
        )

    async def detect_voice_activity(self, request: VoiceActivityRequest) -> VoiceActivityResponse:
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
            model_id=self._manifest_model_id,
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
