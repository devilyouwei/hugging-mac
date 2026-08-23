"""Runtime-neutral text-to-speech schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.schemas.transcription import AudioInput


class SpeechSynthesisRequest(BaseModel):
    """Generate one speech waveform, optionally cloning a reference voice."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    voice: str | None = Field(default=None, min_length=1)
    language: str | None = Field(default=None, min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    reference_audio: AudioInput | None = None
    reference_text: str | None = None
    max_new_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    do_sample: bool = True

    @model_validator(mode="after")
    def validate_reference(self) -> SpeechSynthesisRequest:
        if (self.reference_audio is None) != (self.reference_text is None):
            raise ValueError("reference_audio and reference_text must be provided together")
        if self.reference_text is not None and not self.reference_text.strip():
            raise ValueError("reference_text must not be blank")
        return self


class SpeechSynthesisTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    preprocess_ms: float | None = Field(default=None, ge=0.0)
    inference_ms: float | None = Field(default=None, ge=0.0)
    postprocess_ms: float | None = Field(default=None, ge=0.0)


class SpeechSynthesisResponse(BaseModel):
    """A mono float32 waveform encoded as little-endian bytes."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    audio: bytes = Field(repr=False)
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    generated_tokens: int = Field(default=0, ge=0)
    audio_format: str = "f32le"
    timings: SpeechSynthesisTimings = SpeechSynthesisTimings()
