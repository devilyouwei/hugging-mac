"""Internal Qwen3.5 generation values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
