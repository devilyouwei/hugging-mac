"""MLX-Audio runtime for Kokoro-82M BF16."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Kokoro82mInstanceConfig
from .resources import Kokoro82mResourceResolver
from .utils.types import KokoroEngineOutput


class MlxKokoro82mEngine:
    runtime_name = "mlx"

    def __init__(
        self,
        config: Kokoro82mInstanceConfig,
        resources: Kokoro82mResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None
        self._artifact: Path | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        try:
            from mlx_audio.tts.utils import load  # type: ignore[import-untyped]
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Kokoro-82M MLX requires mlx-audio with Kokoro support"
            ) from error
        # This repository predates mlx-audio's model_type config field. Because
        # our normalized artifact directory is named ``model``, auto-detection
        # would otherwise try to import the unsupported TTS type "model".
        self._model = await asyncio.to_thread(
            load,
            str(artifact),
            model_type="kokoro",
        )
        self._artifact = artifact

    async def infer(self, request: SpeechSynthesisRequest) -> KokoroEngineOutput:
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
        self._artifact = None
        await asyncio.to_thread(gc.collect)
        try:
            import mlx.core as mx  # type: ignore[import-not-found]

            mx.clear_cache()
        except ImportError:
            pass

    def _infer_sync(
        self,
        request: SpeechSynthesisRequest,
        cancelled: threading.Event,
    ) -> KokoroEngineOutput:
        import numpy as np

        if self._model is None or self._artifact is None:
            raise RuntimeError("Kokoro-82M MLX engine is not loaded")
        chunks: list[Any] = []
        sample_rate: int | None = None
        generated_tokens = 0
        for result in self._model.generate(
            text=request.text,
            voice=self._voice_path(request.voice or self._config.default_voice),
            speed=request.speed,
            lang_code=request.language or self._config.default_language,
        ):
            if cancelled.is_set():
                break
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if audio.size:
                chunks.append(audio)
            result_rate = int(result.sample_rate)
            if sample_rate is not None and result_rate != sample_rate:
                raise RuntimeError("Kokoro-82M MLX returned inconsistent sample rates")
            sample_rate = result_rate
            generated_tokens += int(getattr(result, "token_count", 0))
        if cancelled.is_set():
            raise RuntimeError("Kokoro-82M MLX synthesis was cancelled")
        if not chunks or sample_rate is None:
            raise RuntimeError("Kokoro-82M MLX produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype="<f4")
        return KokoroEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=sample_rate,
            duration_seconds=waveform.size / sample_rate,
            generated_tokens=generated_tokens,
        )

    def _voice_path(self, voice: str) -> str:
        assert self._artifact is not None
        name = voice.strip().removesuffix(".safetensors")
        if not name or Path(name).name != name or "," in name:
            raise ValueError("Kokoro voice must be a packaged voice name")
        path = self._artifact / "voices" / f"{name}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Kokoro voice is not downloaded: {name}")
        return str(path)
