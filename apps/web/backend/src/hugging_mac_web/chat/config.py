"""Typed settings and supported model profiles for local chat."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CHAT_MODEL_ID = "mlx-community/qwen3.5-mlx"
DEFAULT_CHAT_VARIANT = "9b-4bit"


@dataclass(frozen=True)
class ChatModelProfile:
    profile_id: str
    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    disk_size_bytes: int
    supports_images: bool


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_CHAT_",
        env_file=".env",
        extra="ignore",
    )

    model_variant: str = DEFAULT_CHAT_VARIANT
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_images: int = Field(default=4, ge=1, le=16)
    max_prompt_characters: int = Field(default=32_000, ge=1)
