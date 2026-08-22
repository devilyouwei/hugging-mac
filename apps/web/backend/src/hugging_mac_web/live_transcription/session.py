"""Queue-driven realtime transcription session."""

from __future__ import annotations

import asyncio
import io
import math
import sys
import wave
from array import array
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from hugging_mac_sdk import AudioInput, SpeechEnhancementRequest, StreamingTranscriptionRequest
from hugging_mac_sdk.capabilities import SpeechEnhancement, StreamingSpeechTranscription
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.models.silero.instance import SileroInstance, SileroStreamingResult

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.service import (
    DEEPFILTERNET3_MODEL_ID,
    LiveTranscriptionService,
)
from hugging_mac_web.shared.utils.log_util import get_logger

RATE, FRAME_SAMPLES = 16_000, 512
PRE_ROLL_FRAMES, TAIL_FRAMES, MAX_FRAMES = 8, 6, 782
Outbound = dict[str, object] | bytes
Emitter = Callable[[Outbound], Awaitable[None]]
logger = get_logger("live_transcription.session")


@dataclass(frozen=True, slots=True)
class AudioFrame:
    start_sample: int
    pcm16: bytes


@dataclass(frozen=True, slots=True)
class UtteranceWorkItem:
    utterance_id: int
    frames: tuple[AudioFrame, ...]
    vad_ms: float


@dataclass(frozen=True, slots=True)
class AsrWorkItem:
    utterance: UtteranceWorkItem
    audio: bytes
    enhancement_ms: float | None


@dataclass(frozen=True, slots=True)
class StreamEnd:
    utterance: UtteranceWorkItem


@dataclass(frozen=True, slots=True)
class LiveSessionStart:
    model_id: str
    instance_id: str
    sample_rate: int
    use_vad: bool
    vad_instance_id: str | None
    use_enhancement: bool
    enhancement_instance_id: str | None
    vad_threshold: float
    sensitivity: float
    streaming_chunk_seconds: float

    @classmethod
    def from_payload(cls, data: dict[str, object]) -> LiveSessionStart:
        sample_rate = int(str(data.get("sample_rate", 0)))
        threshold = float(str(data.get("vad_threshold", 0.5)))
        sensitivity = float(str(data.get("sensitivity", 0.65)))
        chunk_seconds = float(str(data.get("streaming_chunk_seconds", 2.24)))
        if not 8_000 <= sample_rate <= 192_000:
            raise ValueError("Invalid microphone sample rate")
        if not 0 <= threshold <= 1 or not 0 <= sensitivity <= 1:
            raise ValueError("Invalid speech sensitivity")
        if not 0.25 <= chunk_seconds <= 10:
            raise ValueError("Invalid streaming chunk duration")
        return cls(
            str(data.get("model_id", "")),
            str(data.get("instance_id", "")),
            sample_rate,
            bool(data.get("use_vad", True)),
            _optional_string(data.get("vad_instance_id")),
            bool(data.get("use_enhancement", True)),
            _optional_string(data.get("enhancement_instance_id")),
            threshold,
            sensitivity,
            chunk_seconds,
        )


class Detector(Protocol):
    async def push(self, samples: tuple[float, ...]) -> SileroStreamingResult: ...
    def reset_boundary(self) -> None: ...


class EnergyDetector:
    def __init__(self, sensitivity: float) -> None:
        self.threshold = 0.035 - sensitivity * 0.029
        self.candidate = self.silence = 0
        self.voiced = False

    async def push(self, samples: tuple[float, ...]) -> SileroStreamingResult:
        speech = math.sqrt(sum(v * v for v in samples) / len(samples)) >= self.threshold
        started = ended = False
        if speech:
            self.silence = 0
            if not self.voiced:
                self.candidate += FRAME_SAMPLES
                if self.candidate >= 1_536:
                    self.voiced, started = True, True
        else:
            self.candidate = 0
            if self.voiced:
                self.silence += FRAME_SAMPLES
                if self.silence >= 10_240:
                    self.voiced, ended, self.silence = False, True, 0
        return SileroStreamingResult(
            self.voiced, started, ended, 1.0 if speech else 0.0, FRAME_SAMPLES
        )

    def reset_boundary(self) -> None:
        self.candidate = self.silence = 0
        self.voiced = False


class FrameResampler:
    """Resample directly into immutable, VAD-aligned frames."""

    def __init__(self, input_rate: int) -> None:
        self.step, self.position = input_rate / RATE, 0.0
        self.tail: list[int] = []
        self.output = array("h")
        self.next_sample = 0

    def push(self, payload: bytes) -> tuple[AudioFrame, ...]:
        if len(payload) % 2:
            raise ValueError("Incomplete PCM16 sample")
        values = array("h")
        values.frombytes(payload)
        if sys.byteorder != "little":
            values.byteswap()
        joined, frames = self.tail + list(values), []
        while self.position + 1 < len(joined):
            left, fraction = int(self.position), self.position % 1
            value = round(joined[left] * (1 - fraction) + joined[left + 1] * fraction)
            self.output.append(max(-32_768, min(32_767, value)))
            self.position += self.step
            if len(self.output) == FRAME_SAMPLES:
                if sys.byteorder != "little":
                    self.output.byteswap()
                frames.append(AudioFrame(self.next_sample, self.output.tobytes()))
                self.next_sample += FRAME_SAMPLES
                self.output = array("h")
        consumed = int(self.position)
        self.tail, self.position = joined[consumed:], self.position - consumed
        return tuple(frames)


class LiveTranscriptionSession:
    def __init__(
        self,
        context: PlatformContext,
        service: LiveTranscriptionService,
        start: LiveSessionStart,
        detector: Detector,
        emitter: Emitter,
        *,
        streaming: bool,
        streamer: StreamingSpeechTranscription | None,
        include_input_audio: bool,
    ) -> None:
        self.context, self.service, self.start = context, service, start
        self.detector, self.emitter, self.streaming = detector, emitter, streaming
        self.streamer = streamer
        self.include_input_audio = include_input_audio
        self.session_id = uuid4().hex
        self.log = logger.bind(
            session_id=self.session_id,
            model_id=start.model_id,
            instance_id=start.instance_id,
        )
        self.resampler = FrameResampler(start.sample_rate)
        self.raw: asyncio.Queue[bytes | None] = asyncio.Queue(32)
        self.utterances: asyncio.Queue[UtteranceWorkItem | None] | None = (
            None if streaming else asyncio.Queue(8)
        )
        self.asr: asyncio.Queue[AsrWorkItem | None] | None = None if streaming else asyncio.Queue(8)
        self.results: asyncio.Queue[Outbound | None] = asyncio.Queue(64)
        self.pre_roll: deque[AudioFrame] = deque(maxlen=PRE_ROLL_FRAMES)
        self.confirmed: list[AudioFrame] = []
        self.holdback: list[AudioFrame] = []
        self.active = self.closed = False
        self.utterance_id, self.vad_ms = 0, 0.0
        self.received_chunks = self.received_bytes = self.processed_frames = 0
        self.started_at = perf_counter()
        self.last_backpressure_log_at = 0.0
        self.stream_tasks: list[asyncio.Task[None]] = []
        self.stream_queue: asyncio.Queue[AudioFrame | StreamEnd] | None = None
        self.audio_task = asyncio.create_task(
            self._guard_worker("audio", self._audio_worker),
            name=f"live-audio-{self.session_id[:8]}",
        )
        self.sender_task = asyncio.create_task(
            self._sender_worker(), name=f"live-sender-{self.session_id[:8]}"
        )
        self.enhancement_task: asyncio.Task[None] | None = None
        self.asr_task: asyncio.Task[None] | None = None
        self.workers = [self.audio_task, self.sender_task]
        if not streaming:
            self.enhancement_task = asyncio.create_task(
                self._guard_worker("enhancement", self._enhancement_worker),
                name=f"live-enhancement-{self.session_id[:8]}",
            )
            self.asr_task = asyncio.create_task(
                self._guard_worker("asr", self._asr_worker),
                name=f"live-asr-{self.session_id[:8]}",
            )
            self.workers.extend((self.enhancement_task, self.asr_task))
        self.log.info(
            "live_session_started",
            input_sample_rate=start.sample_rate,
            pipeline_sample_rate=RATE,
            detector=type(detector).__name__,
            use_vad=start.use_vad,
            vad_instance_id=start.vad_instance_id,
            vad_threshold=start.vad_threshold,
            use_enhancement=start.use_enhancement,
            enhancement_instance_id=start.enhancement_instance_id,
            streaming=streaming,
            streaming_chunk_seconds=start.streaming_chunk_seconds if streaming else None,
        )

    @classmethod
    async def create(
        cls,
        context: PlatformContext,
        settings: LiveTranscriptionSettings,
        start: LiveSessionStart,
        emitter: Emitter,
        *,
        include_input_audio: bool = True,
    ) -> LiveTranscriptionSession:
        service = LiveTranscriptionService(context, settings)
        profile = service._profile(start.model_id)
        asr = await context.models.instances.require(start.instance_id)
        asr_info = asr.info()
        if asr_info.model_id != profile.model_id or asr_info.state is not ModelState.READY:
            raise ValueError("The selected ASR instance is not ready")
        streamer: StreamingSpeechTranscription | None = None
        if profile.streaming:
            if not asr.supports(StreamingSpeechTranscription):
                raise ValueError("The selected ASR instance does not support streaming")
            streamer = asr.require(StreamingSpeechTranscription)  # type: ignore[type-abstract]
        detector: Detector
        if start.use_vad:
            if not start.vad_instance_id:
                raise ValueError("Silero VAD is enabled but not loaded")
            vad = await context.models.instances.require(start.vad_instance_id)
            if not isinstance(vad, SileroInstance) or vad.info().state is not ModelState.READY:
                raise ValueError("The selected Silero VAD instance is not ready")
            detector = vad.create_streaming_detector(
                threshold=start.vad_threshold, min_speech_ms=96, min_silence_ms=640
            )
        else:
            detector = EnergyDetector(start.sensitivity)
        if start.use_enhancement:
            if profile.streaming:
                raise ValueError("DeepFilterNet3 is unavailable for streaming ASR")
            if not start.enhancement_instance_id:
                raise ValueError("DeepFilterNet3 is enabled but not loaded")
            enhancer = await context.models.instances.require(start.enhancement_instance_id)
            if (
                enhancer.info().model_id != DEEPFILTERNET3_MODEL_ID
                or enhancer.info().state is not ModelState.READY
            ):
                raise ValueError("The selected DeepFilterNet3 instance is not ready")
        return cls(
            context,
            service,
            start,
            detector,
            emitter,
            streaming=profile.streaming,
            streamer=streamer,
            include_input_audio=include_input_audio,
        )

    async def enqueue(self, payload: bytes) -> None:
        failed = next((task for task in self.workers if task.done()), None)
        if failed is not None:
            try:
                failure = failed.exception()
            except asyncio.CancelledError as error:
                failure = error
            raise RuntimeError("The backend audio pipeline stopped") from failure
        if self.closed or not payload or len(payload) > 64 * 1024:
            raise ValueError("Invalid audio chunk or closed session")
        self.received_chunks += 1
        self.received_bytes += len(payload)
        queued_at = perf_counter()
        await self.raw.put(payload)
        wait_ms = (perf_counter() - queued_at) * 1000
        now = perf_counter()
        if wait_ms >= 5 and now - self.last_backpressure_log_at >= 1:
            self.last_backpressure_log_at = now
            self.log.warning(
                "raw_audio_queue_backpressure",
                enqueue_wait_ms=round(wait_ms, 3),
                queue_size=self.raw.qsize(),
                queue_capacity=self.raw.maxsize,
                received_chunks=self.received_chunks,
                received_bytes=self.received_bytes,
            )

    async def stop(self) -> None:
        if self.closed:
            return
        self.log.info(
            "live_session_stop_requested",
            active_utterance=self.active,
            raw_queue_size=self.raw.qsize(),
            utterance_queue_size=self.utterances.qsize() if self.utterances else None,
            asr_queue_size=self.asr.qsize() if self.asr else None,
        )
        self.closed = True
        await self.raw.put(None)
        await self.audio_task
        if not self.streaming:
            assert self.utterances is not None
            assert self.asr is not None
            assert self.enhancement_task is not None
            assert self.asr_task is not None
            await self.utterances.put(None)
            await self.enhancement_task
            await self.asr.put(None)
            await self.asr_task
        if self.stream_tasks:
            await asyncio.gather(*self.stream_tasks)
        await self.results.put(None)
        await self.sender_task
        self.log.info(
            "live_session_stopped",
            elapsed_ms=round((perf_counter() - self.started_at) * 1000, 3),
            received_chunks=self.received_chunks,
            received_bytes=self.received_bytes,
            processed_frames=self.processed_frames,
            utterance_count=self.utterance_id,
        )

    async def cancel(self) -> None:
        if self.closed and all(task.done() for task in (*self.workers, *self.stream_tasks)):
            return
        self.closed = True
        for task in (*self.workers, *self.stream_tasks):
            task.cancel()
        await asyncio.gather(*self.workers, *self.stream_tasks, return_exceptions=True)
        self.log.info(
            "live_session_cancelled",
            elapsed_ms=round((perf_counter() - self.started_at) * 1000, 3),
            received_chunks=self.received_chunks,
            received_bytes=self.received_bytes,
            active_utterance=self.active,
        )

    async def _audio_worker(self) -> None:
        self.log.debug("worker_started", worker="audio")
        while (payload := await self.raw.get()) is not None:
            for frame in self.resampler.push(payload):
                self.processed_frames += 1
                await self._process_frame(frame)
        if self.active:
            await self._seal(reason="client_stop")
        self.log.debug("worker_stopped", worker="audio")

    async def _guard_worker(self, name: str, worker: Callable[[], Awaitable[None]]) -> None:
        try:
            await worker()
        except asyncio.CancelledError:
            self.log.debug("worker_cancelled", worker=name)
            raise
        except Exception as error:
            self.closed = True
            self.log.exception("pipeline_worker_failed", worker=name)
            await self._publish({"type": "error", "message": f"{name} worker failed: {error}"})
            raise

    async def _process_frame(self, frame: AudioFrame) -> None:
        pcm = array("h")
        pcm.frombytes(frame.pcm16)
        if sys.byteorder != "little":
            pcm.byteswap()
        started = perf_counter()
        result = await self.detector.push(tuple(value / 32768.0 for value in pcm))
        elapsed = (perf_counter() - started) * 1000
        if not self.active:
            self.pre_roll.append(frame)
            if result.speech_started:
                self.active, self.utterance_id = True, self.utterance_id + 1
                self.confirmed, self.holdback, self.vad_ms = list(self.pre_roll), [], elapsed
                self.pre_roll.clear()
                self.log.info(
                    "speech_started",
                    utterance_id=self.utterance_id,
                    started_at_ms=round(self.confirmed[0].start_sample / 16),
                    pre_roll_frames=len(self.confirmed),
                    vad_probability=round(result.probability, 4),
                    vad_inference_ms=round(elapsed, 3),
                )
                await self._publish(
                    {
                        "type": "speech_start",
                        "utterance_id": self.utterance_id,
                        "started_at_ms": round(self.confirmed[0].start_sample / 16),
                    }
                )
                if self.streaming:
                    self.stream_queue = asyncio.Queue(128)
                    self.stream_tasks.append(
                        asyncio.create_task(
                            self._streaming_worker(self.utterance_id, self.stream_queue),
                            name=f"live-stream-{self.session_id[:8]}-{self.utterance_id}",
                        )
                    )
                    await self._commit_stream(self.confirmed)
            return
        self.vad_ms += elapsed
        if result.probability >= self.start.vad_threshold:
            self.confirmed.extend(self.holdback)
            if self.streaming:
                await self._commit_stream(self.holdback)
            self.holdback.clear()
            self.confirmed.append(frame)
            if self.streaming:
                await self._commit_stream((frame,))
        else:
            self.holdback.append(frame)
        if result.speech_ended:
            await self._seal(reason="vad_endpoint")
        elif len(self.confirmed) + len(self.holdback) >= MAX_FRAMES:
            await self._seal(reason="max_duration")
            self.detector.reset_boundary()

    async def _seal(self, *, reason: str) -> None:
        force = reason != "vad_endpoint"
        retained = self.holdback if force else self.holdback[:TAIL_FRAMES]
        discarded = [] if force else self.holdback[TAIL_FRAMES:]
        self.confirmed.extend(retained)
        if self.streaming:
            await self._commit_stream(retained)
        item = UtteranceWorkItem(self.utterance_id, tuple(self.confirmed), self.vad_ms)
        ended = item.frames[-1].start_sample + FRAME_SAMPLES
        self.active, self.confirmed, self.holdback = False, [], []
        self.pre_roll.clear()
        self.pre_roll.extend(discarded[-PRE_ROLL_FRAMES:])
        self.log.info(
            "speech_ended",
            utterance_id=item.utterance_id,
            reason=reason,
            ended_at_ms=round(ended / 16),
            duration_ms=round(len(item.frames) * FRAME_SAMPLES / RATE * 1000),
            audio_frames=len(item.frames),
            retained_tail_frames=len(retained),
            recycled_pre_roll_frames=len(self.pre_roll),
            vad_inference_ms=round(item.vad_ms, 3),
            streaming=self.streaming,
        )
        await self._publish(
            {
                "type": "speech_end",
                "utterance_id": item.utterance_id,
                "ended_at_ms": round(ended / 16),
            }
        )
        if self.streaming:
            assert self.stream_queue is not None
            await self.stream_queue.put(StreamEnd(item))
            self.stream_queue = None
        else:
            assert self.utterances is not None
            await self.utterances.put(item)

    async def _commit_stream(self, frames: list[AudioFrame] | tuple[AudioFrame, ...]) -> None:
        assert self.stream_queue is not None
        for frame in frames:
            await self.stream_queue.put(frame)

    async def _enhancement_worker(self) -> None:
        assert self.utterances is not None
        assert self.asr is not None
        self.log.debug("worker_started", worker="enhancement")
        while (item := await self.utterances.get()) is not None:
            audio, elapsed = _encode_wav(item.frames), None
            if self.start.use_enhancement:
                started = perf_counter()
                self.log.info(
                    "speech_enhancement_started",
                    utterance_id=item.utterance_id,
                    input_bytes=len(audio),
                    enhancement_instance_id=self.start.enhancement_instance_id,
                )
                instance = await self.context.models.instances.require(
                    self.start.enhancement_instance_id or ""
                )
                enhancer = instance.require(SpeechEnhancement)  # type: ignore[type-abstract]
                response = await enhancer.enhance_speech(
                    SpeechEnhancementRequest(audio=AudioInput(data=audio))
                )
                audio, elapsed = response.audio, response.timings.inference_ms
                self.log.info(
                    "speech_enhancement_completed",
                    utterance_id=item.utterance_id,
                    output_bytes=len(audio),
                    inference_ms=round(elapsed, 3),
                    wall_ms=round((perf_counter() - started) * 1000, 3),
                )
            else:
                self.log.debug(
                    "speech_enhancement_skipped",
                    utterance_id=item.utterance_id,
                    reason="disabled",
                )
            await self.asr.put(AsrWorkItem(item, audio, elapsed))
        self.log.debug("worker_stopped", worker="enhancement")

    async def _asr_worker(self) -> None:
        assert self.asr is not None
        self.log.debug("worker_started", worker="asr")
        while (work := await self.asr.get()) is not None:
            started = perf_counter()
            self.log.info(
                "utterance_asr_started",
                utterance_id=work.utterance.utterance_id,
                input_bytes=len(work.audio),
                duration_ms=round(len(work.utterance.frames) * FRAME_SAMPLES / RATE * 1000),
            )
            result = await self.service.transcribe(
                work.audio,
                model_id=self.start.model_id,
                instance_id=self.start.instance_id,
            )
            self.log.info(
                "utterance_asr_completed",
                utterance_id=work.utterance.utterance_id,
                text_length=len(result.text),
                generated_tokens=result.generated_tokens,
                inference_ms=(
                    round(result.inference_ms, 3) if result.inference_ms is not None else None
                ),
                wall_ms=round((perf_counter() - started) * 1000, 3),
                runtime=result.runtime,
                device=result.device,
            )
            data = result.model_dump()
            data.update(
                type="transcript",
                utterance_id=work.utterance.utterance_id,
                speech_duration_seconds=(len(work.utterance.frames) * FRAME_SAMPLES / RATE),
                vad_inference_ms=work.utterance.vad_ms,
                enhancement_inference_ms=work.enhancement_ms,
                audio_media_type="audio/wav",
                audio_bytes=len(work.audio),
            )
            await self._publish(data)
            if self.include_input_audio:
                await self._publish(_audio_message(work.utterance.utterance_id, work.audio))
        self.log.debug("worker_stopped", worker="asr")

    async def _streaming_worker(
        self,
        utterance_id: int,
        queue: asyncio.Queue[AudioFrame | StreamEnd],
    ) -> None:
        assert self.streamer is not None
        streamer = self.streamer
        session = await streamer.start_stream()
        size = max(1, round(self.start.streaming_chunk_seconds * RATE / FRAME_SAMPLES))
        stream_started = perf_counter()
        chunks = 0
        inference_ms = 0.0
        preprocess_ms = 0.0
        self.log.info(
            "streaming_asr_started",
            utterance_id=utterance_id,
            stream_session_id=session.session_id,
            chunk_frames=size,
            chunk_duration_ms=round(size * FRAME_SAMPLES / RATE * 1000),
        )
        last = None
        pending: list[AudioFrame] = []
        model_input: list[AudioFrame] = []
        item: UtteranceWorkItem | None = None
        try:
            while item is None:
                message = await queue.get()
                if isinstance(message, StreamEnd):
                    item = message.utterance
                else:
                    pending.append(message)
                    model_input.append(message)
                if len(pending) < size and item is None:
                    continue
                if not pending:
                    continue
                last = await streamer.transcribe_stream(
                    StreamingTranscriptionRequest(
                        session_id=session.session_id,
                        audio=AudioInput(data=_encode_wav(tuple(pending))),
                    )
                )
                chunks += 1
                chunk_inference_ms = last.timings.inference_ms or 0.0
                inference_ms += chunk_inference_ms
                preprocess_ms += last.timings.preprocess_ms or 0.0
                self.log.debug(
                    "streaming_asr_chunk_completed",
                    utterance_id=utterance_id,
                    stream_session_id=session.session_id,
                    chunk_number=chunks,
                    inference_ms=round(chunk_inference_ms, 3),
                    text_length=len(last.text),
                    delta_length=len(last.delta),
                )
                pending = []
                await self._publish(
                    {
                        "type": "partial",
                        "utterance_id": utterance_id,
                        "text": last.text,
                        "delta": last.delta,
                        "detected_language": last.detected_language,
                        "inference_ms": last.timings.inference_ms,
                    }
                )
            final = await streamer.finish_stream(session.session_id)
            inference_ms += final.timings.inference_ms or 0.0
            preprocess_ms += final.timings.preprocess_ms or 0.0
        except asyncio.CancelledError:
            self.log.info(
                "streaming_asr_cancelled",
                utterance_id=utterance_id,
                stream_session_id=session.session_id,
                chunks=chunks,
            )
            await streamer.cancel_stream(session.session_id)
            raise
        assert item is not None
        audio = _encode_wav(tuple(model_input))
        self.log.info(
            "streaming_asr_completed",
            utterance_id=item.utterance_id,
            stream_session_id=session.session_id,
            chunks=chunks,
            input_bytes=len(audio),
            text_length=len(final.text),
            generated_tokens=final.generated_tokens,
            inference_ms=round(inference_ms, 3),
            preprocess_ms=round(preprocess_ms, 3),
            wall_ms=round((perf_counter() - stream_started) * 1000, 3),
        )
        await self._publish(
            {
                "type": "transcript",
                "utterance_id": item.utterance_id,
                "text": final.text,
                "model_id": final.model_id,
                "instance_id": final.instance_id,
                "runtime": final.runtime,
                "device": final.device,
                "sample_rate": final.sample_rate,
                "duration_seconds": len(item.frames) * FRAME_SAMPLES / RATE,
                "generated_tokens": final.generated_tokens,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "languages": [last.detected_language] if last and last.detected_language else [],
                "emotion": None,
                "events": [],
                "speech_duration_seconds": len(item.frames) * FRAME_SAMPLES / RATE,
                "vad_inference_ms": item.vad_ms,
                "enhancement_inference_ms": None,
                "audio_media_type": "audio/wav",
                "audio_bytes": len(audio),
            }
        )
        if self.include_input_audio:
            await self._publish(_audio_message(item.utterance_id, audio))

    async def _publish(self, message: Outbound) -> None:
        await self.results.put(message)

    async def _sender_worker(self) -> None:
        self.log.debug("worker_started", worker="sender")
        try:
            while (message := await self.results.get()) is not None:
                await self.emitter(message)
        except asyncio.CancelledError:
            self.log.debug("worker_cancelled", worker="sender")
            raise
        except Exception:
            self.closed = True
            self.log.exception("pipeline_worker_failed", worker="sender")
            raise
        self.log.debug("worker_stopped", worker="sender")


def _encode_wav(frames: tuple[AudioFrame, ...]) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for frame in frames:
            stream.writeframesraw(frame.pcm16)
    return output.getvalue()


def _audio_message(utterance_id: int, wav: bytes) -> bytes:
    return utterance_id.to_bytes(8, "big") + wav


def _optional_string(value: object) -> str | None:
    return str(value) if value else None
