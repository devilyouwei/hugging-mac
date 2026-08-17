"""Configuration for Qualcomm MediaPipe Hand Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

MEDIAPIPE_HAND_DETECTION_MODEL_ID: Final = "qualcomm/mediapipe-hand-detection"
MEDIAPIPE_HAND_DETECTION_REVISION: Final = "013e27b599e37c3b4c69439de15de53cc5b5708e"
MEDIAPIPE_HAND_DETECTION_RELEASE: Final = "0.60.0"


class MediaPipeHandDetectionCoreMlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: Literal["float32"] = "float32"
    minimum_deployment_target: Literal["macos13"] = "macos13"


class MediaPipeHandDetectionInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["onnx", "coreml"] = "coreml"
    variant: Literal["float"] = "float"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    device: str | None = None
    input_size: Literal[256] = 256
    compute_units: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "all"
    coreml_precision: Literal["float32"] = "float32"
