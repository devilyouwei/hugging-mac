"""Configuration for the Digital Human game."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DigitalHumanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GAME_DIGITAL_HUMAN_",
        env_file=".env",
        extra="ignore",
    )

    max_audio_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_image_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_history_messages: int = Field(default=15, ge=1, le=40)
    max_reply_tokens: int = Field(default=96, ge=16, le=512)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_vad: bool = True
    enable_speech_enhancement: bool = True
    system_prompt: str = (
        "Act as the user's digital human conversation partner. Respond naturally using the "
        "conversation and the current camera image. Keep replies conversational and concise, "
        "usually one or two sentences. Do not use Markdown, lists, emoji, stage directions, "
        "special symbols, or reveal your reasoning."
    )
