"""Typed settings and supported model profiles for local chat."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ChatModelProfile:
    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    disk_size_bytes: int
    supports_images: bool


QWEN_9B_PROFILE = ChatModelProfile(
    model_id="mlx-community/qwen3.5-9b-mlx-4bit",
    display_name="Qwen3.5 9B MLX 4-bit",
    short_name="Qwen 3.5 9B",
    description="支持文本、图片与多轮上下文的本地多模态模型。",
    variant="4bit",
    runtime="mlx",
    required_artifact_id="model",
    disk_size_bytes=5_950_000_000,
    supports_images=True,
)

QWEN_4B_OPTIQ_PROFILE = ChatModelProfile(
    model_id="mlx-community/qwen3.5-4b-optiq-4bit",
    display_name="Qwen3.5 4B OptiQ 4-bit",
    short_name="Qwen 3.5 4B",
    description="更轻量的 OptiQ 混合精度多模态对话模型。",
    variant="4bit",
    runtime="mlx",
    required_artifact_id="model",
    disk_size_bytes=3_000_000_000,
    supports_images=True,
)

CHAT_MODEL_PROFILES = {
    profile.model_id: profile for profile in (QWEN_9B_PROFILE, QWEN_4B_OPTIQ_PROFILE)
}
DEFAULT_CHAT_MODEL_ID = QWEN_9B_PROFILE.model_id


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_CHAT_",
        env_file=".env",
        extra="ignore",
    )

    model_id: str = DEFAULT_CHAT_MODEL_ID
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_images: int = Field(default=4, ge=1, le=16)
    max_prompt_characters: int = Field(default=32_000, ge=1)
