"""Kokoro-82M engine using PyTorch with Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import KOKORO_82M_SAMPLE_RATE, Kokoro82mInstanceConfig
from .resources import Kokoro82mResourceResolver
from .utils.modeling import (
    load_kokoro_model,
    load_voice,
    phonemes_to_ids,
    phonemize_chunks,
)
from .utils.types import KokoroEngineOutput


class TorchKokoro82mEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self,
        config: Kokoro82mInstanceConfig,
        resources: Kokoro82mResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._device: str = config.device or "mps"
        self._torch: Any | None = None
        self._model: Any | None = None
        self._artifact: Path | None = None

    @property
    def device(self) -> str:
        return self._device

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        provider = TorchProvider()
        if not provider.is_available():
            raise UnsupportedRuntimeError(
                "PyTorch is not installed; install hugging-mac-sdk[tts]"
            )
        self._device = provider.resolve_device(
            self._config.device,
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        await asyncio.to_thread(self._load_sync, artifact)

    async def infer(self, request: SpeechSynthesisRequest) -> KokoroEngineOutput:
        if self._model is None:
            raise RuntimeError("Kokoro-82M PyTorch engine is not loaded")
        return await asyncio.to_thread(self._infer_sync, request)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._artifact = None
        if not had_model:
            return
        await asyncio.to_thread(gc.collect)
        if self._device == "mps" and self._torch is not None:
            empty_cache = getattr(getattr(self._torch, "mps", None), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()
        self._torch = None

    def _load_sync(self, artifact: Path) -> None:
        torch = importlib.import_module("torch")
        model = load_kokoro_model(
            artifact,
            device=self._device,
            disable_complex=False,
        )
        if self._config.dtype == "float16":
            model = model.half()
        elif self._config.dtype == "float32":
            model = model.float()
        self._torch = torch
        self._model = model
        self._artifact = artifact

    def _infer_sync(self, request: SpeechSynthesisRequest) -> KokoroEngineOutput:
        torch = self._torch
        model = self._model
        artifact = self._artifact
        if torch is None or model is None or artifact is None:
            raise RuntimeError("Kokoro-82M PyTorch engine is not loaded")
        language = request.language or self._config.default_language
        voice = request.voice or self._config.default_voice
        waveforms: list[Any] = []
        generated_tokens = 0
        for phonemes in phonemize_chunks(request.text, language):
            ref_s = load_voice(torch, artifact, voice, len(phonemes)).to(self._device)
            input_ids = phonemes_to_ids(model, torch, phonemes, self._device)
            with torch.inference_mode():
                waveform, durations = model.forward_with_tokens(
                    input_ids,
                    ref_s,
                    speed=request.speed,
                )
            waveforms.append(waveform.detach().float().cpu().reshape(-1))
            generated_tokens += int(durations.sum().item())
        if not waveforms:
            raise ValueError("Kokoro phonemization produced no speech")
        waveform = torch.cat(waveforms).contiguous().numpy()
        return KokoroEngineOutput(
            audio=waveform.astype("<f4", copy=False).tobytes(),
            sample_rate=KOKORO_82M_SAMPLE_RATE,
            duration_seconds=float(waveform.shape[0]) / KOKORO_82M_SAMPLE_RATE,
            generated_tokens=generated_tokens,
        )
