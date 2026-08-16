"""Runtime-neutral voice activity detection schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.schemas.transcription import AudioInput


class VoiceActivityRequest(BaseModel):
    """Detect speech regions in one encoded audio input."""

    model_config = ConfigDict(frozen=True)

    audio: AudioInput
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, ge=0)
    min_silence_ms: int = Field(default=100, ge=0)
    speech_pad_ms: int = Field(default=30, ge=0)
    max_speech_seconds: float | None = Field(default=None, gt=0.0)


class SpeechSegment(BaseModel):
    """One detected half-open speech interval."""

    model_config = ConfigDict(frozen=True)

    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> SpeechSegment:
        if self.end_sample <= self.start_sample or self.end_seconds <= self.start_seconds:
            raise ValueError("Speech segment end must be after start")
        return self


class VoiceActivityTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float = Field(default=0.0, ge=0.0)
    inference_ms: float = Field(default=0.0, ge=0.0)
    postprocess_ms: float = Field(default=0.0, ge=0.0)


class VoiceActivityResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    speech_seconds: float = Field(ge=0.0)
    segments: tuple[SpeechSegment, ...]
    timings: VoiceActivityTimings = VoiceActivityTimings()
