"""Pinned source and preferred conversion configuration for YOLOv8 variants."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

YOLOV8_REPO_ID: Final = "Ultralytics/YOLOv8"
YOLOV8_REPO_REVISION: Final = "8a9e1a5"
YoloV8Variant = Literal["n", "s", "m"]
YOLOV8_VARIANTS: Final[tuple[YoloV8Variant, ...]] = ("n", "s", "m")
YOLOV8_FILENAMES: Final[dict[YoloV8Variant, str]] = {
    variant: f"yolov8{variant}.pt" for variant in YOLOV8_VARIANTS
}
YOLOV8_SHA256: Final[dict[YoloV8Variant, str]] = {
    "n": "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95",
    "s": "268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d",
    "m": "6c25b0b63b1a433843f06d821a9ac1deb8d5805f74f0f38772c7308c5adc55a5",
}
YOLOV8_MODEL_ID: Final = "ultralytics/yolov8"
YOLOV8_MODEL_REVISION: Final = YOLOV8_REPO_REVISION


class YoloV8CoreMlConfig(BaseModel):
    """Reproducible default optimized for the first Apple Silicon baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: YoloV8Variant = "n"
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
    variant: YoloV8Variant = "n"
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
