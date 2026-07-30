"""Typed live transcription settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveTranscriptionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_LIVE_TRANSCRIPTION_",
        env_file=".env",
        extra="ignore",
    )

    model_id: str = "audio8/audio8-asr-0.1b"
    model_variant: str = "base"
    runtime: str = "coreml"
    prompt: str = "Transcribe the speech accurately in its original language."
    max_new_tokens: int = Field(default=128, ge=1, le=512)
