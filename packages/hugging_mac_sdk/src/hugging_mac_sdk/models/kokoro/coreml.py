"""Runtime for aufklarer's precompiled end-to-end Kokoro Core ML model."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import threading
from pathlib import Path
from typing import Any

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
_MAX_INPUT_IDS = 128
_MAX_PHONEMES = _MAX_INPUT_IDS - 2


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
            artifact / "kokoro_5s.mlmodelc",
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
            for start in range(0, len(phonemes), _MAX_PHONEMES):
                if cancelled.is_set():
                    raise RuntimeError("Kokoro-82M synthesis was cancelled")
                chunk = phonemes[start : start + _MAX_PHONEMES]
                ids = [1, *(self._vocab[symbol] for symbol in chunk if symbol in self._vocab), 2]
                if len(ids) <= 2:
                    continue
                generated_tokens += len(ids)
                token_count = min(len(ids), _MAX_INPUT_IDS)
                padded = np.zeros((1, _MAX_INPUT_IDS), dtype=np.int32)
                padded[0, :token_count] = ids[:token_count]
                attention_mask = np.zeros_like(padded)
                attention_mask[0, :token_count] = 1
                output = self._session.run(
                    {
                        "input_ids": padded,
                        "attention_mask": attention_mask,
                        "ref_s": embedding,
                        "speed": np.asarray([request.speed], dtype=np.float32),
                        "random_phases": np.random.random((1, 9)).astype(np.float32),
                    }
                )
                audio = np.asarray(output["audio"], dtype=np.float32).reshape(-1)
                length = int(np.asarray(output["audio_length_samples"]).reshape(-1)[0])
                waveforms.append(self._trim_trailing_artifacts(audio[: max(0, length)]))
        if not waveforms:
            raise RuntimeError("Kokoro-82M Core ML produced no audio")
        waveform = np.ascontiguousarray(np.concatenate(waveforms), dtype="<f4")
        return KokoroEngineOutput(
            audio=waveform.tobytes(),
            sample_rate=KOKORO_82M_SAMPLE_RATE,
            duration_seconds=waveform.size / KOKORO_82M_SAMPLE_RATE,
            generated_tokens=generated_tokens,
        )

    @staticmethod
    def _trim_trailing_artifacts(audio: Any) -> Any:
        import numpy as np

        if audio.size == 0:
            return audio
        window = max(1, int(0.05 * KOKORO_82M_SAMPLE_RATE))
        speech_end = audio.size
        for start in range(max(0, audio.size - window), 0, -(window // 2)):
            if float(np.sqrt(np.mean(np.square(audio[start : start + window])))) > 0.03:
                speech_end = min(audio.size, start + window)
                break
        trimmed = np.array(audio[:speech_end], copy=True)
        fade = min(trimmed.size, int(0.01 * KOKORO_82M_SAMPLE_RATE))
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
