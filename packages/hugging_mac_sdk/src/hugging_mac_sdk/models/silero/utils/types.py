from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    samples: object
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SileroInferenceOptions:
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    max_speech_seconds: float | None


@dataclass(frozen=True, slots=True)
class SampleInterval:
    start: int
    end: int
