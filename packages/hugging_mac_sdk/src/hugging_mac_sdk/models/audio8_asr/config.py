"""Instance configuration for the Audio8-ASR integration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

Audio8AsrDType = Literal["auto", "float16", "bfloat16", "float32"]


class Audio8AsrInstanceConfig(BaseModel):
    """Runtime and resource options for one Audio8-ASR instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml"] = "pytorch-mps"
    variant: str = "0.1b"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    tokenizer_path: Path | None = None
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
