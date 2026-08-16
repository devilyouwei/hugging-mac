"""Audio preparation helpers for the live ASR pipeline."""

from __future__ import annotations

import importlib
import io
from dataclasses import dataclass

import numpy as np
from hugging_mac_sdk.schemas.voice_activity import SpeechSegment


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    wav: bytes
    sample_rate: int
    source_duration_seconds: float
    speech_duration_seconds: float
    segment_count: int


def prepare_detected_speech(
    encoded_audio: bytes,
    segments: tuple[SpeechSegment, ...],
    *,
    vad_sample_rate: int,
    output_sample_rate: int = 16000,
    merge_gap_ms: int = 180,
    speech_pad_ms: int = 120,
) -> PreparedSpeech:
    """Decode, normalize and retain only VAD-confirmed speech intervals."""

    soundfile = importlib.import_module("soundfile")
    samples, source_rate = soundfile.read(
        io.BytesIO(encoded_audio), dtype="float32", always_2d=True
    )
    if samples.size == 0:
        raise ValueError("Audio input is empty")
    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    if not bool(np.isfinite(mono).all()):
        raise ValueError("Audio input contains non-finite samples")
    source_rate = int(source_rate)
    source_duration = len(mono) / source_rate
    if source_rate != output_sample_rate:
        signal = importlib.import_module("scipy.signal")
        divisor = __import__("math").gcd(source_rate, output_sample_rate)
        mono = signal.resample_poly(
            mono,
            output_sample_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)

    scale = output_sample_rate / vad_sample_rate
    padding = round(speech_pad_ms * output_sample_rate / 1000)
    merge_gap = round(merge_gap_ms * output_sample_rate / 1000)
    intervals = [
        (
            max(0, round(segment.start_sample * scale) - padding),
            min(len(mono), round(segment.end_sample * scale) + padding),
        )
        for segment in segments
    ]
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    speech = np.concatenate([mono[start:end] for start, end in merged])
    # Remove recorder DC bias. Deliberately avoid peak normalization because it
    # also amplifies quiet background noise and can reduce ASR accuracy.
    speech = np.ascontiguousarray(speech - np.mean(speech, dtype=np.float64), dtype=np.float32)
    output = io.BytesIO()
    soundfile.write(output, speech, output_sample_rate, format="WAV", subtype="PCM_16")
    return PreparedSpeech(
        wav=output.getvalue(),
        sample_rate=output_sample_rate,
        source_duration_seconds=source_duration,
        speech_duration_seconds=len(speech) / output_sample_rate,
        segment_count=len(merged),
    )
