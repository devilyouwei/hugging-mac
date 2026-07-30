"""Runtime-neutral automatic speech recognition schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioInput(BaseModel):
    """Encoded audio supplied as a local file or in-memory bytes."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path | None = None
    data: bytes | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> AudioInput:
        if (self.path is None) == (self.data is None):
            raise ValueError("Exactly one of path or data must be provided")
        return self


class TranscriptionRequest(BaseModel):
    """One short-form speech transcription request."""

    model_config = ConfigDict(frozen=True)

    audio: AudioInput
    prompt: str = Field(default="Please transcribe this audio.", min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class TranscriptionTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float | None = Field(default=None, ge=0.0)
    inference_ms: float | None = Field(default=None, ge=0.0)
    postprocess_ms: float | None = Field(default=None, ge=0.0)


class TranscriptionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    text: str
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    generated_tokens: int = Field(default=0, ge=0)
    timings: TranscriptionTimings = TranscriptionTimings()
