"""Runtime-neutral rich speech-understanding schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.transcription import AudioInput, TranscriptionTimings

SpeechLanguage = Literal["auto", "zh", "en", "yue", "ja", "ko", "nospeech"]


class SpeechUnderstandingRequest(BaseModel):
    """Recognize speech together with language, emotion, and acoustic events."""

    model_config = ConfigDict(frozen=True)

    audio: AudioInput
    language: SpeechLanguage = "auto"
    use_itn: bool = True
    ban_unknown_emotion: bool = False


class SpeechUnderstandingResponse(BaseModel):
    """Rich non-tensor output produced by a speech-understanding model."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    text: str
    raw_text: str
    languages: tuple[str, ...] = ()
    emotion: str | None = None
    events: tuple[str, ...] = ()
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    token_count: int = Field(default=0, ge=0)
    timings: TranscriptionTimings = TranscriptionTimings()
