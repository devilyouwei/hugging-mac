"""Configuration for the pinned Audio8-ASR integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

AUDIO8_ASR_MODEL_ID: Final = "audio8/audio8-asr-0.1b"
AUDIO8_ASR_REPO_ID: Final = "Audio8/Audio8-ASR-0.1B"
AUDIO8_ASR_REVISION: Final = "8487da63d581fa4fc9b5c60444cb57c3a523d7aa"
AUDIO8_ASR_WEIGHT_SHA256: Final = "971d17f64ef5f193fca567fa3e9dc063c4eede97faabb11c0c6abf0b92b23ca4"
AUDIO8_ASR_VARIANT: Final = "base"
AUDIO8_ASR_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

Audio8AsrDType = Literal["auto", "float16", "bfloat16", "float32"]


class Audio8AsrInstanceConfig(BaseModel):
    """Runtime and resource options for one Audio8-ASR instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml"] = "pytorch-mps"
    variant: Literal["base"] = "base"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["mps", "cpu"] | None = None
    decoder_device: Literal["mps", "cpu"] | None = None
    dtype: Audio8AsrDType = "auto"
    allow_cpu_fallback: bool = True
    compute_units: Literal[
        "all",
        "cpu-only",
        "cpu-and-gpu",
        "cpu-and-neural-engine",
    ] = "cpu-and-neural-engine"
    sample_rate: Literal[16000] = 16000
    max_audio_seconds: float = Field(default=30.0, gt=0.0, le=30.0)
