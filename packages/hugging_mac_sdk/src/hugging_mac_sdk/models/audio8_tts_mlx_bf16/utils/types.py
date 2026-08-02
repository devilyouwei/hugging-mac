"""Private Audio8-TTS MLX runtime values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TtsEngineOutput:
    audio: bytes
    sample_rate: int
    duration_seconds: float
    generated_tokens: int = 0
