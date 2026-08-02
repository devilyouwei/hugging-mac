from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Qwen3TtsEngineOutput:
    audio: bytes
    sample_rate: int
    duration_seconds: float
    generated_tokens: int = 0
