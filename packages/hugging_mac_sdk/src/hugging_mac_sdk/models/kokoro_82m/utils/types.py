"""Internal Kokoro runtime values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KokoroEngineOutput:
    audio: bytes
    sample_rate: int
    duration_seconds: float
    generated_tokens: int
