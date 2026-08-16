"""Stateful streaming speech-transcription schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.transcription import AudioInput, TranscriptionTimings


class StreamingTranscriptionSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    model_id: str
    instance_id: str
    runtime: str
    device: str
    sample_rate: int = Field(gt=0)


class StreamingTranscriptionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    audio: AudioInput


class StreamingTranscriptionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    model_id: str
    instance_id: str
    runtime: str
    device: str
    text: str
    delta: str
    sample_rate: int = Field(gt=0)
    audio_seconds: float = Field(ge=0.0)
    generated_tokens: int = Field(default=0, ge=0)
    detected_language: str | None = None
    is_final: bool = False
    timings: TranscriptionTimings = TranscriptionTimings()
