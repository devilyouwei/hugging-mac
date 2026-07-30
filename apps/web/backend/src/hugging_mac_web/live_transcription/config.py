# ruff: noqa: RUF001
"""Typed live transcription settings and supported ASR profiles."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class AsrModelProfile:
    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    supports_coreml_conversion: bool = False
    rich_understanding: bool = False


AUDIO8_PROFILE = AsrModelProfile(
    model_id="audio8/audio8-asr-0.1b",
    display_name="Audio8-ASR 0.1B",
    short_name="Audio8-ASR",
    description="轻量自回归中英多语言转写，使用 Core ML 在 Apple Silicon 上运行。",
    variant="base",
    runtime="coreml",
    required_artifact_id="coreml",
    supports_coreml_conversion=True,
)

SENSEVOICE_PROFILE = AsrModelProfile(
    model_id="funaudiollm/sensevoice-small",
    display_name="SenseVoiceSmall",
    short_name="SenseVoice",
    description="非自回归多语言语音理解，同时识别文本、语种、情绪与音频事件。",
    variant="small",
    runtime="pytorch-mps",
    required_artifact_id="source",
    rich_understanding=True,
)

ASR_MODEL_PROFILES = {profile.model_id: profile for profile in (AUDIO8_PROFILE, SENSEVOICE_PROFILE)}


class LiveTranscriptionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUGGING_MAC_APP_LIVE_TRANSCRIPTION_",
        env_file=".env",
        extra="ignore",
    )

    prompt: str = "Transcribe the speech accurately in its original language."
    max_new_tokens: int = Field(default=128, ge=1, le=512)
