"""Instance configuration for GOT-OCR2.0."""

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class GotOcr2MlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["mlx"] = "mlx"
    variant: str = "8bit"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
