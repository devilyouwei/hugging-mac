"""Chat application settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_CHAT_",
        env_file=".env",
        extra="ignore",
    )

    model_id: str = "mlx-community/qwen3.5-9b-mlx-4bit"
    variant: str = "4bit"
    runtime: str = "mlx"
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_images: int = Field(default=4, ge=1, le=16)
    max_prompt_characters: int = Field(default=32_000, ge=1)
