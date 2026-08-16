"""Audio decoding and encoding helpers for DeepFilterNet3."""

from __future__ import annotations

import importlib
import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hugging_mac_sdk.schemas.transcription import AudioInput

from .dsp import NATIVE_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class PreparedEnhancementAudio:
    samples: NDArray[np.float32]
    source_sample_rate: int
    duration_seconds: float


def prepare_audio(audio: AudioInput) -> PreparedEnhancementAudio:
    soundfile = importlib.import_module("soundfile")
    source = str(audio.path) if audio.path is not None else io.BytesIO(audio.data or b"")
    samples, source_rate = soundfile.read(source, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("Audio input is empty")
    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    if not bool(np.isfinite(mono).all()):
        raise ValueError("Audio input contains non-finite samples")
    source_rate = int(source_rate)
    duration = mono.size / source_rate
    if source_rate != NATIVE_SAMPLE_RATE:
        signal = importlib.import_module("scipy.signal")
        divisor = __import__("math").gcd(source_rate, NATIVE_SAMPLE_RATE)
        mono = signal.resample_poly(
            mono,
            NATIVE_SAMPLE_RATE // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
    return PreparedEnhancementAudio(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        source_sample_rate=source_rate,
        duration_seconds=duration,
    )


def encode_wav(
    samples: NDArray[np.float32], *, output_sample_rate: int
) -> bytes:
    soundfile = importlib.import_module("soundfile")
    if output_sample_rate != NATIVE_SAMPLE_RATE:
        signal = importlib.import_module("scipy.signal")
        divisor = __import__("math").gcd(NATIVE_SAMPLE_RATE, output_sample_rate)
        samples = signal.resample_poly(
            samples,
            output_sample_rate // divisor,
            NATIVE_SAMPLE_RATE // divisor,
        ).astype(np.float32, copy=False)
    output = io.BytesIO()
    soundfile.write(
        output,
        np.clip(samples, -1.0, 1.0),
        output_sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()
