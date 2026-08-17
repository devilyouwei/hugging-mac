"""Pinned py-feat RetinaFace configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

RETINAFACE_MODEL_ID: Final = "py-feat/retinaface"
RETINAFACE_REVISION: Final = "31702389094fccc7060c15299e6ad712ee880de6"
RETINAFACE_CONFIG_SHA256: Final = "22cfa2fdb1708298e7dc9590d32d3def33a39fcc511d91d85eb26ea46f9a90f7"
RETINAFACE_WEIGHTS_SHA256: Final = (
    "5f88f7957d9eaef82b49b9e0202bc56e97292e2aede2999209accd34442cc4df"
)


class RetinaFaceInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml"] = "coreml"
    variant: Literal["mobilenet0.25"] = "mobilenet0.25"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    input_size: int = Field(default=640, ge=320, le=1280, multiple_of=32)
    device: str | None = None
    allow_cpu_fallback: bool = True
    compute_units: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "all"
