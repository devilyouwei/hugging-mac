"""Configuration for Qwen3-ASR 0.6B Core ML."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class Qwen3AsrCoreMlInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    runtime: Literal["coreml"] = "coreml"
    variant: str = "0.6b"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    tokenizer_path: Path | None = None
    sample_rate: int = Field(default=16000, gt=0)
    max_audio_seconds: float = Field(default=30.0, gt=0, le=30.0)
    device: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] | None = (
        "cpu-and-neural-engine"
    )
