"""Instance and conversion options for YOLOv8 variants."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field


class YoloV8CoreMlConfig(BaseModel):
    """Reproducible default optimized for the first Apple Silicon baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str = "n"
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=1, ge=1)
    dynamic: bool = False
    half: bool = True
    quantize: Literal[8, 16] = 16
    # Keep outputs raw; the SDK performs task-aware NMS for every runtime.
    nms: bool = False
    device: str = "cpu"
    compute_units: str = "all"


class YoloV8InstanceConfig(BaseModel):
    """Configuration shared by the PyTorch MPS and Core ML instances."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml", "onnx"] = "coreml"
    variant: str = "n"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    imgsz: int = Field(default=640, gt=0)
    device: str | None = None
    allow_cpu_fallback: bool = False
    compute_units: Literal[
        "all",
        "cpu-only",
        "cpu-and-gpu",
        "cpu-and-neural-engine",
    ] = "all"
