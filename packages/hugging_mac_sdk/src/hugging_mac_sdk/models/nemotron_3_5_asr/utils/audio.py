"""Decode Nemotron input to bounded mono 16 kHz PCM."""

import importlib
from io import BytesIO
from math import gcd
from pathlib import Path
from typing import BinaryIO

from hugging_mac_sdk.schemas.transcription import AudioInput

from .types import PreparedAudio


def prepare_audio(source: AudioInput, *, sample_rate: int, max_seconds: float) -> PreparedAudio:
    np = importlib.import_module("numpy")
    soundfile = importlib.import_module("soundfile")
    stream: str | BinaryIO
    if source.path is not None:
        path = Path(source.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        stream = str(path)
    else:
        assert source.data is not None
        stream = BytesIO(source.data)
    samples, original_rate = soundfile.read(stream, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("Audio input is empty")
    samples = np.asarray(samples, dtype=np.float32).mean(axis=1)
    if not bool(np.isfinite(samples).all()):
        raise ValueError("Audio input contains non-finite samples")
    if int(original_rate) != sample_rate:
        signal = importlib.import_module("scipy.signal")
        divisor = gcd(int(original_rate), sample_rate)
        samples = signal.resample_poly(
            samples, sample_rate // divisor, int(original_rate) // divisor
        )
    samples = np.ascontiguousarray(samples[: round(max_seconds * sample_rate)], dtype=np.float32)
    return PreparedAudio(samples, sample_rate, float(samples.shape[0]) / sample_rate)
