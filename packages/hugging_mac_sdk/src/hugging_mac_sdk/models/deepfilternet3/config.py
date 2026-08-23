"""Instance configuration for the DeepFilterNet3 Core ML artifact."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class DeepFilterNet3InstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["coreml"] = "coreml"
    variant: str = "default"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    device: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] | None = "all"
