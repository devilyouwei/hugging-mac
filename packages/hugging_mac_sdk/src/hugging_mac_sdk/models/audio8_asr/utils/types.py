"""Private values exchanged inside the Audio8-ASR model pack."""

from __future__ import annotations

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
    prompt_tokens: int
    generated_tokens: int
