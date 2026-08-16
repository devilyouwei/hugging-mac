from __future__ import annotations

from typing import Any, NamedTuple


class PreparedAudio(NamedTuple):
    samples: Any
    sample_rate: int
    duration_seconds: float


class AsrEngineOutput(NamedTuple):
    text: str
    generated_tokens: int
    prompt_tokens: int
