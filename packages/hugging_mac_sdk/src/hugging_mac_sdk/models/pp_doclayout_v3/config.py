"""Instance configuration for PP-DocLayoutV3."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class PPDocLayoutV3InstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml"] = "coreml"
    variant: str = "base"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: str | None = None
    allow_cpu_fallback: bool = True
    compute_units: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "all"
