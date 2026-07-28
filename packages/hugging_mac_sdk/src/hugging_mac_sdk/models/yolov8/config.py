"""Pinned source and preferred conversion configuration for YOLOv8n."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

YOLOV8_REPO_ID: Final = "Ultralytics/YOLOv8"
YOLOV8_REPO_REVISION: Final = "8a9e1a5"
YOLOV8N_FILENAME: Final = "yolov8n.pt"
YOLOV8N_SHA256: Final = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
YOLOV8_MODEL_ID: Final = "ultralytics/yolov8n"
YOLOV8_MODEL_REVISION: Final = (
    f"{YOLOV8_REPO_REVISION}-{YOLOV8N_FILENAME.removesuffix('.pt')}"
)


class YoloV8CoreMlConfig(BaseModel):
    """Reproducible default optimized for the first Apple Silicon baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str = "n"
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=1, ge=1)
    dynamic: bool = False
    half: bool = True
    quantize: Literal[8, 16] = 16
    nms: bool = True
    device: str = "cpu"
    compute_units: str = "all"


class YoloV8InstanceConfig(BaseModel):
    """Configuration shared by the PyTorch MPS and Core ML instances."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml"] = "coreml"
    model_home: Path = Field(
        default_factory=lambda: user_cache_path("hugging-mac") / "models"
    )
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
