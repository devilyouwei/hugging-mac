"""Configuration for the independently registered YOLOv8 Pose package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

YoloV8PoseVariant = Literal["n", "s", "m"]
YOLOV8_POSE_MODEL_ID = "ultralytics/yolov8-pose"
YOLOV8_POSE_REVISION = "v8.2.0"


class YoloV8PoseCoreMlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: YoloV8PoseVariant = "n"
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=1, ge=1)
    dynamic: bool = False
    half: bool = True
    quantize: Literal[8, 16] = 16
    # Ultralytics can only build the Core ML NMS pipeline for detect models.
    # Pose keeps the raw prediction tensor so Ultralytics can decode keypoints
    # and apply task-aware post-processing at inference time.
    nms: bool = False
    device: str = "cpu"
    compute_units: str = "all"


class YoloV8PoseInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps", "coreml", "onnx"] = "coreml"
    variant: YoloV8PoseVariant = "n"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    imgsz: int = Field(default=640, gt=0)
    device: str | None = None
    allow_cpu_fallback: bool = False
    compute_units: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "all"
