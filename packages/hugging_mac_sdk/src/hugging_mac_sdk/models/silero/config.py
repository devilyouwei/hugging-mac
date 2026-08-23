"""Instance configuration for the Silero VAD integration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class SileroInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["onnx", "coreml"] = "coreml"
    variant: str = "v6.2.1"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    device: (
        Literal["auto", "coreml", "cpu", "all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"]
        | None
    ) = "auto"
    sample_rate: Literal[16000] = 16000
    chunk_samples: Literal[512] = 512
