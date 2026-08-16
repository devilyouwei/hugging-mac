"""Optional official PyTorch Kokoro-82M runtime."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Kokoro82mInstanceConfig
from .resources import Kokoro82mResourceResolver
from .utils.types import KokoroEngineOutput


class TorchKokoro82mEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self, config: Kokoro82mInstanceConfig, resources: Kokoro82mResourceResolver
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None
        self._pipelines: dict[str, Any] = {}
        self._artifact: Path | None = None
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        await asyncio.to_thread(self._load_sync, artifact)

    def _load_sync(self, artifact: Path) -> None:
        try:
            from kokoro import KModel  # type: ignore[import-untyped]
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Kokoro-82M requires the official kokoro package"
            ) from error
        self._device = TorchProvider().resolve_device(
            self._config.device,
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        self._model = (
            KModel(
                repo_id="hexgrad/Kokoro-82M",
                config=str(artifact / "config.json"),
                model=str(artifact / "kokoro-v1_0.pth"),
            )
            .to(self._device)
            .eval()
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

    def _infer_sync(
        self, request: SpeechSynthesisRequest, cancelled: threading.Event
    ) -> KokoroEngineOutput:
        import numpy as np
        import torch
        from kokoro import KPipeline

        if self._model is None or self._artifact is None:
            raise RuntimeError("Kokoro-82M PyTorch engine is not loaded")
        language = request.language or self._config.default_language
        pipeline = self._pipelines.get(language)
        if pipeline is None:
            pipeline = KPipeline(
                lang_code=language,
                repo_id="hexgrad/Kokoro-82M",
                model=self._model,
                device=self._device,
            )
            self._pipelines[language] = pipeline
        voice = (request.voice or self._config.default_voice).strip().removesuffix(".pt")
        if not voice or Path(voice).name != voice or "," in voice:
            raise ValueError("Kokoro voice must be a packaged voice name")
        voice_path = self._artifact / "voices" / f"{voice}.pt"
        if not voice_path.is_file():
            raise FileNotFoundError(f"Kokoro voice is not downloaded: {voice}")
        chunks: list[Any] = []
        generated_tokens = 0
        with torch.inference_mode():
            for result in pipeline(request.text, voice=str(voice_path), speed=request.speed):
                if cancelled.is_set():
                    raise RuntimeError("Kokoro-82M synthesis was cancelled")
                audio = result.audio
                if audio is not None:
                    chunks.append(audio.detach().float().cpu().numpy().reshape(-1))
                generated_tokens += len(result.phonemes or "")
        if not chunks:
            raise RuntimeError("Kokoro-82M produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype="<f4")
        return KokoroEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=24000,
            duration_seconds=waveform.size / 24000,
            generated_tokens=generated_tokens,
        )

    async def close(self) -> None:
        self._model = None
        self._pipelines = {}
        self._artifact = None
        await asyncio.to_thread(gc.collect)
