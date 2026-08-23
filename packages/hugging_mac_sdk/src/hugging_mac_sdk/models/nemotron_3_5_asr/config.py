"""Configuration for Nemotron 3.5 ASR Core ML."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class NemotronCoreMlInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    runtime: Literal["coreml"] = "coreml"
    variant: str = "multilingual-2240ms"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    sample_rate: int = Field(default=16000, gt=0)
    max_audio_seconds: float = Field(default=300.0, gt=0)
    device: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] | None = (
        "cpu-and-neural-engine"
    )
