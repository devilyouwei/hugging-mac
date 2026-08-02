"""Configuration for Qwen3-TTS 0.6B Base MLX 4-bit."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_TTS_MODEL_ID: Final = "mlx-community/qwen3-tts-12hz-0.6b-base-4bit"
QWEN3_TTS_REPO_ID: Final = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
QWEN3_TTS_REVISION: Final = "0d6bb6f"
QWEN3_TTS_VARIANT: Final = "4bit"
QWEN3_TTS_SAMPLE_RATE: Final = 24_000
QWEN3_TTS_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)


class Qwen3TtsInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["4bit"] = "4bit"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
