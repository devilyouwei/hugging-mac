"""Vela instance configuration; input capacities remain owned by the original model."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class VelaInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    entity_route: str | None = None
    artifact_path: Path | None = None
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    variant: str = "3-fast"
    runtime: Literal["pytorch", "coreai"] = "pytorch"
    device: Literal["cpu", "auto"] = "cpu"
