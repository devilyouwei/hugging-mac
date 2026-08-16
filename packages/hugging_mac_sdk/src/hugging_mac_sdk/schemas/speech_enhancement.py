"""Runtime-neutral speech enhancement schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.transcription import AudioInput


class SpeechEnhancementRequest(BaseModel):
    """Remove background noise from one encoded mono or multi-channel input."""

    model_config = ConfigDict(frozen=True)

    audio: AudioInput
    output_sample_rate: int | None = Field(default=None, ge=8000, le=192000)


class SpeechEnhancementTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float = Field(default=0.0, ge=0.0)
    inference_ms: float = Field(default=0.0, ge=0.0)
    postprocess_ms: float = Field(default=0.0, ge=0.0)


class SpeechEnhancementResponse(BaseModel):
    """Enhanced PCM WAV audio and stable runtime metadata."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    audio: bytes = Field(repr=False)
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    timings: SpeechEnhancementTimings = SpeechEnhancementTimings()
