"""Audio decoding local to the Silero model package."""

from __future__ import annotations

import importlib
import io

from hugging_mac_sdk.schemas.transcription import AudioInput

from .types import PreparedAudio


def prepare_audio(audio: AudioInput, *, sample_rate: int) -> PreparedAudio:
    numpy = importlib.import_module("numpy")
    soundfile = importlib.import_module("soundfile")
    source = str(audio.path) if audio.path is not None else io.BytesIO(audio.data or b"")
    samples, source_rate = soundfile.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if int(source_rate) != sample_rate:
        scipy_signal = importlib.import_module("scipy.signal")
        divisor = __import__("math").gcd(int(source_rate), sample_rate)
        samples = scipy_signal.resample_poly(
            samples, sample_rate // divisor, int(source_rate) // divisor
        )
    samples = numpy.ascontiguousarray(samples, dtype=numpy.float32)
    return PreparedAudio(samples, sample_rate, len(samples) / sample_rate)
