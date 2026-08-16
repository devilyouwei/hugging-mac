"""Runtime for FluidInference's precompiled end-to-end Kokoro Core ML model."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import KOKORO_82M_SAMPLE_RATE, Kokoro82mInstanceConfig
from .resources import Kokoro82mResourceResolver
from .utils.types import KokoroEngineOutput

_LANGUAGE_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}
_MAX_INPUT_IDS = 124
_MAX_PHONEMES = _MAX_INPUT_IDS - 2
_MAX_AUDIO_SAMPLES = 175_800
_SAFE_AUDIO_SAMPLES = int(6.8 * KOKORO_82M_SAMPLE_RATE)
_FADE_SAMPLES = int(0.005 * KOKORO_82M_SAMPLE_RATE)
_SPLIT_BOUNDARIES = (
    "!.?\u2026\u3002\uff01\uff1f;\uff1b:\n",
    ",\uff0c\u3001\u2014\u2013",
    " \t",
)


class CoreMlKokoro82mEngine:
    runtime_name = "coreml"

    def __init__(
        self,
        config: Kokoro82mInstanceConfig,
        resources: Kokoro82mResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None
        self._artifact: Path | None = None
        self._vocab: dict[str, int] = {}
        self._pipelines: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._session.device if self._session is not None else self._config.compute_units

    async def resolve(self) -> Path:
        return (await self._resources.resolve_coreml()).path

    async def load(self, artifact: Path) -> None:
        vocab_data = json.loads((artifact / "vocab_index.json").read_text(encoding="utf-8"))
        self._vocab = {str(key): int(value) for key, value in vocab_data["vocab"].items()}
        self._session = await CoreMLProvider().create_session(
            artifact / "kokoro_21_5s.mlmodelc",
            device=self._config.compute_units,
            options={},
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
        from kokoro import KPipeline  # type: ignore[import-untyped]

        if self._session is None or self._artifact is None:
            raise RuntimeError("Kokoro-82M Core ML engine is not loaded")
        if request.speed != 1.0:
            raise UnsupportedRuntimeError(
                "FluidInference Kokoro Core ML does not expose speed control"
            )
        language = request.language or self._config.default_language
        language_code = _LANGUAGE_CODES.get(language.lower(), language.lower())
        pipeline = self._pipelines.get(language_code)
        if pipeline is None:
            pipeline = KPipeline(
                lang_code=language_code,
                repo_id="hexgrad/Kokoro-82M",
                model=False,
            )
            self._pipelines[language_code] = pipeline

        voice = (request.voice or self._config.default_voice).strip().removesuffix(".json")
        if not voice or Path(voice).name != voice or "," in voice:
            raise ValueError("Kokoro voice must be a packaged voice name")
        voice_path = self._artifact / "voices" / f"{voice}.json"
        if not voice_path.is_file():
            raise FileNotFoundError(f"Kokoro Core ML voice is not downloaded: {voice}")
        embedding = np.asarray(
            json.loads(voice_path.read_text(encoding="utf-8"))["embedding"],
            dtype=np.float32,
        ).reshape(1, 256)

        waveforms: list[Any] = []
        generated_tokens = 0
        for result in pipeline(request.text, model=False):
            phonemes = result.phonemes or ""
            for chunk in self._fit_input_chunks(phonemes):
                if cancelled.is_set():
                    raise RuntimeError("Kokoro-82M synthesis was cancelled")
                chunk_waveforms, chunk_tokens = self._synthesize_phonemes(
                    chunk,
                    embedding=embedding,
                    cancelled=cancelled,
                )
                waveforms.extend(chunk_waveforms)
                generated_tokens += chunk_tokens
        if not waveforms:
            raise RuntimeError("Kokoro-82M Core ML produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(waveforms), dtype="<f4")
        return KokoroEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=KOKORO_82M_SAMPLE_RATE,
            duration_seconds=waveform.size / KOKORO_82M_SAMPLE_RATE,
            generated_tokens=generated_tokens,
        )

    def _synthesize_phonemes(
        self,
        phonemes: str,
        *,
        embedding: Any,
        cancelled: threading.Event,
    ) -> tuple[list[Any], int]:
        import numpy as np

        if cancelled.is_set():
            raise RuntimeError("Kokoro-82M synthesis was cancelled")
        ids = [1, *(self._vocab[symbol] for symbol in phonemes if symbol in self._vocab), 2]
        if len(ids) <= 2:
            return [], 0
        token_count = min(len(ids), _MAX_INPUT_IDS)
        padded = np.zeros((1, _MAX_INPUT_IDS), dtype=np.int32)
        padded[0, :token_count] = ids[:token_count]
        attention_mask = np.zeros_like(padded)
        attention_mask[0, :token_count] = 1
        if self._session is None:
            raise RuntimeError("Kokoro-82M Core ML engine is not loaded")
        output = self._session.run(
            {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "ref_s": embedding,
                "random_phases": np.random.random((1, 9)).astype(np.float32),
            }
        )
        audio = np.asarray(output["audio"], dtype=np.float32).reshape(-1)
        length = int(np.asarray(output["audio_length_samples"]).reshape(-1)[0])
        if length > _SAFE_AUDIO_SAMPLES:
            split = self._split_phonemes(phonemes)
            if split is not None:
                left, right = split
                left_audio, left_tokens = self._synthesize_phonemes(
                    left, embedding=embedding, cancelled=cancelled
                )
                right_audio, right_tokens = self._synthesize_phonemes(
                    right, embedding=embedding, cancelled=cancelled
                )
                return [*left_audio, *right_audio], left_tokens + right_tokens
        return [self._prepare_audio(audio, length)], token_count

    @classmethod
    def _fit_input_chunks(cls, phonemes: str) -> list[str]:
        if len(phonemes) <= _MAX_PHONEMES:
            return [phonemes] if phonemes.strip() else []
        split = cls._split_phonemes(phonemes)
        if split is None:
            return []
        left, right = split
        return [*cls._fit_input_chunks(left), *cls._fit_input_chunks(right)]

    @staticmethod
    def _split_phonemes(phonemes: str) -> tuple[str, str] | None:
        if len(phonemes) < 2:
            return None
        midpoint = len(phonemes) // 2
        for boundaries in _SPLIT_BOUNDARIES:
            candidates = [
                index + 1
                for index, symbol in enumerate(phonemes)
                if symbol in boundaries and 0 < index + 1 < len(phonemes)
            ]
            for split_at in sorted(candidates, key=lambda value: abs(value - midpoint)):
                left, right = phonemes[:split_at].strip(), phonemes[split_at:].strip()
                if left and right:
                    return left, right
        left, right = phonemes[:midpoint].strip(), phonemes[midpoint:].strip()
        return (left, right) if left and right else None

    @staticmethod
    def _prepare_audio(audio: Any, length: int) -> Any:
        import numpy as np

        safe_length = min(audio.size, max(0, length), _MAX_AUDIO_SAMPLES)
        trimmed = np.array(audio[:safe_length], copy=True)
        fade = min(trimmed.size, _FADE_SAMPLES)
        if fade >= 2:
            trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return trimmed

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
        self._artifact = None
        self._vocab = {}
        self._pipelines = {}
        await asyncio.to_thread(gc.collect)
