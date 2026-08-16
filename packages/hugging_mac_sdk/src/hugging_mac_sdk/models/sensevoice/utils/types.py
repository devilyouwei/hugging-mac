"""Private values exchanged inside the SenseVoiceSmall model pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    samples: Any
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SenseVoiceInferenceOptions:
    language: str = "auto"
    use_itn: bool = True
    ban_unknown_emotion: bool = False


@dataclass(frozen=True, slots=True)
class SenseVoiceEngineOutput:
    text: str
    raw_text: str
    languages: tuple[str, ...]
    emotion: str | None
    events: tuple[str, ...]
    token_count: int
