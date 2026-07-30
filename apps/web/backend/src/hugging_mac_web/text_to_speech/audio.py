"""PCM waveform encoding for browser playback."""

from __future__ import annotations

import io
import wave

import numpy as np


def float32le_to_wav(audio: bytes, sample_rate: int) -> bytes:
    samples = np.frombuffer(audio, dtype="<f4")
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()
