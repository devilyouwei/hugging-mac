"""Private MOSS-TTS-Nano runtime values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MossTtsNanoEngineOutput:
    audio: bytes
    sample_rate: int
    duration_seconds: float
    generated_tokens: int
