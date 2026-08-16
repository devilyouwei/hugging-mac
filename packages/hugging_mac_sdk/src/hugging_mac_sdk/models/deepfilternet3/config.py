"""Configuration for the pinned DeepFilterNet3 Core ML artifact."""

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

DEEPFILTERNET3_MODEL_ID: Final = "deepfilternet/deepfilternet3"
DEEPFILTERNET3_REVISION: Final = "937bad9811f1ffc1a06ea0d676461b080b2bdc93"
DEEPFILTERNET3_VARIANT: Final = "default"


class DeepFilterNet3InstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["coreml"] = "coreml"
    variant: Literal["default"] = "default"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    device: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] | None = "all"
