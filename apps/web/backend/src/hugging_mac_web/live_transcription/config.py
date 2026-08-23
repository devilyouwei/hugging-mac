"""Typed live transcription settings and supported ASR profiles."""

from dataclasses import dataclass
from typing import Literal

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
    streaming: bool = False
    streaming_chunk_seconds: float | None = None


AUDIO8_PROFILE = AsrModelProfile(
    model_id="audio8/audio8-asr",
    display_name="Audio8-ASR 0.1B",
    short_name="Audio8-ASR",
    description="Compact bilingual transcription optimized for Apple Silicon.",
    variant="0.1b",
    runtime="coreml",
    required_artifact_id="coreml",
    supports_coreml_conversion=True,
)

SENSEVOICE_PROFILE = AsrModelProfile(
    model_id="funaudiollm/sensevoice",
    display_name="SenseVoiceSmall",
    short_name="SenseVoice",
    description="Multilingual speech understanding with language, emotion, and event tags.",
    variant="small",
    runtime="coreml",
    required_artifact_id="coreml",
    supports_coreml_conversion=True,
    rich_understanding=True,
)

QWEN3_ASR_PROFILE = AsrModelProfile(
    model_id="qwen/qwen3-asr",
    display_name="Qwen3-ASR 0.6B",
    short_name="Qwen3-ASR",
    description="Multilingual ASR with a precompiled INT8 Core ML decoding pipeline.",
    variant="0.6b",
    runtime="coreml",
    required_artifact_id="coreml-int8",
)

NEMOTRON_3_5_ASR_PROFILE = AsrModelProfile(
    model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
    display_name="Nemotron 3.5 ASR 0.6B",
    short_name="Nemotron 3.5",
    description="Stateful multilingual RNN-T streaming transcription on Core ML.",
    variant="multilingual-2240ms",
    runtime="coreml",
    required_artifact_id="coreml-mixed",
    streaming=True,
    streaming_chunk_seconds=2.24,
)

ASR_MODEL_PROFILES = {
    profile.model_id: profile
    for profile in (
        AUDIO8_PROFILE,
        SENSEVOICE_PROFILE,
        QWEN3_ASR_PROFILE,
        NEMOTRON_3_5_ASR_PROFILE,
    )
}


class LiveTranscriptionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_LIVE_TRANSCRIPTION_",
        env_file=".env",
        extra="ignore",
    )

    prompt: str = "Transcribe the speech accurately in its original language."
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    # SenseVoice and Qwen3-ASR use this stability override. Audio8 deliberately
    # retains its own CPU_AND_NE default because its hybrid pipeline depends on
    # that tested compute configuration.
    coreml_compute_units: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "all"
