"""MLX-Audio runtime for Audio8-TTS BF16."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import tempfile
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

from .config import Audio8TtsMlxBf16InstanceConfig
from .resources import Audio8TtsMlxBf16ResourceResolver
from .utils.types import TtsEngineOutput
from .utils.voices import VoiceStore


class MlxAudio8TtsBf16Engine:
    runtime_name = "mlx"

    def __init__(
        self,
        config: Audio8TtsMlxBf16InstanceConfig,
        resources: Audio8TtsMlxBf16ResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None
        self._voices = VoiceStore(config.voice_home)

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        try:
            from mlx_audio.tts.utils import load
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Audio8-TTS MLX requires an mlx-audio build with arktts support"
            ) from error
        self._model = await asyncio.to_thread(load, str(artifact))

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        cancelled = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._infer_sync, request, cancelled))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            with contextlib.suppress(Exception):
                await asyncio.shield(worker)
            raise

    async def close(self) -> None:
        self._model = None
        await asyncio.to_thread(gc.collect)
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    def _infer_sync(
        self, request: SpeechSynthesisRequest, cancelled: threading.Event
    ) -> TtsEngineOutput:
        import numpy as np

        if self._model is None:
            raise RuntimeError("Audio8-TTS MLX engine is not loaded")
        reference_path, reference_text, temporary = self._reference(request)
        chunks: list[Any] = []
        sample_rate: int | None = None
        generated_tokens = 0
        try:
            results = self._model.generate(
                text=request.text,
                ref_audio=str(reference_path) if reference_path is not None else None,
                ref_text=reference_text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
            )
            for result in results:
                if cancelled.is_set():
                    break
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    chunks.append(audio)
                result_rate = int(result.sample_rate)
                generated_tokens += int(getattr(result, "token_count", 0))
                if sample_rate is not None and result_rate != sample_rate:
                    raise RuntimeError("Audio8-TTS MLX returned inconsistent sample rates")
                sample_rate = result_rate
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if cancelled.is_set():
            raise RuntimeError("Audio8-TTS MLX synthesis was cancelled")
        if not chunks or sample_rate is None:
            raise RuntimeError("Audio8-TTS MLX produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype="<f4")
        return TtsEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=sample_rate,
            duration_seconds=waveform.size / sample_rate,
            generated_tokens=generated_tokens,
        )

    def _reference(
        self, request: SpeechSynthesisRequest
    ) -> tuple[Path | None, str | None, Path | None]:
        if request.reference_audio is not None:
            assert request.reference_text is not None
            if request.voice is not None:
                path, text = self._voices.save(
                    request.voice, request.reference_audio, request.reference_text
                )
                return path, text, None
            if request.reference_audio.path is not None:
                return request.reference_audio.path, request.reference_text, None
            temporary = self._temporary_reference(request.reference_audio)
            return temporary, request.reference_text, temporary
        if request.voice is not None:
            path, text = self._voices.load(request.voice)
            return path, text, None
        return None, None, None

    @staticmethod
    def _temporary_reference(source: AudioInput) -> Path:
        if source.path is not None:
            return source.path
        assert source.data is not None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(source.data)
            return Path(handle.name)
