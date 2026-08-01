"""Private runtime result types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
