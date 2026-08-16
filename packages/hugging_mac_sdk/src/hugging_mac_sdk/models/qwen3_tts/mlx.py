"""MLX-Audio runtime for Qwen3-TTS 0.6B Base 4-bit."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import tempfile
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

from .config import Qwen3TtsInstanceConfig
from .resources import Qwen3TtsResourceResolver
from .utils.types import Qwen3TtsEngineOutput


class MlxQwen3TtsEngine:
    runtime_name = "mlx"

    def __init__(
        self,
        config: Qwen3TtsInstanceConfig,
        resources: Qwen3TtsResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_tokenizers()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        try:
            from mlx_audio.tts.utils import load  # type: ignore[import-untyped]
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Qwen3-TTS requires mlx-audio with qwen3_tts support"
            ) from error

        def _load() -> Any:
            with merged_directory_view(artifact, (self._resources.tokenizer_path,)) as model_view:
                return load(str(model_view), model_type="qwen3_tts")

        self._model = await asyncio.to_thread(_load)

    async def infer(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput:
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
        self,
        request: SpeechSynthesisRequest,
        cancelled: threading.Event,
    ) -> Qwen3TtsEngineOutput:
        import numpy as np

        if self._model is None:
            raise RuntimeError("Qwen3-TTS MLX engine is not loaded")
        reference_path, temporary = self._reference_path(request.reference_audio)
        chunks: list[Any] = []
        sample_rate: int | None = None
        generated_tokens = 0
        try:
            results = self._model.generate(
                text=request.text,
                voice=request.voice,
                speed=request.speed,
                lang_code=request.language or "auto",
                ref_audio=str(reference_path) if reference_path is not None else None,
                ref_text=request.reference_text,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
                verbose=False,
            )
            for result in results:
                if cancelled.is_set():
                    break
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    chunks.append(audio)
                result_rate = int(result.sample_rate)
                if sample_rate is not None and result_rate != sample_rate:
                    raise RuntimeError("Qwen3-TTS returned inconsistent sample rates")
                sample_rate = result_rate
                generated_tokens += int(getattr(result, "token_count", 0))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if cancelled.is_set():
            raise RuntimeError("Qwen3-TTS synthesis was cancelled")
        if not chunks or sample_rate is None:
            raise RuntimeError("Qwen3-TTS produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype="<f4")
        return Qwen3TtsEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=sample_rate,
            duration_seconds=waveform.size / sample_rate,
            generated_tokens=generated_tokens,
        )

    @staticmethod
    def _reference_path(source: AudioInput | None) -> tuple[Path | None, Path | None]:
        if source is None:
            return None, None
        if source.path is not None:
            return source.path, None
        assert source.data is not None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(source.data)
            path = Path(handle.name)
        return path, path
