"""Laya instance configuration; input capacities remain owned by the original model."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class LayaInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    variant: str = "multilingual-legacy-s512"
    runtime: Literal["pytorch", "coreai"] = "coreai"
    max_length: int | None = Field(default=None, ge=1, le=8192)
    device: Literal["cpu", "auto"] = "cpu"
