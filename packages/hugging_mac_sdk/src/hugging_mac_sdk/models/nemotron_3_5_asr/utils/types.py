"""Private Nemotron runtime values."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    samples: Any
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class AsrEngineOutput:
    text: str
    generated_tokens: int
    detected_language: str | None = None


@dataclass(frozen=True, slots=True)
class StreamingAsrEngineOutput:
    text: str
    delta: str
    generated_tokens: int
    detected_language: str | None = None
