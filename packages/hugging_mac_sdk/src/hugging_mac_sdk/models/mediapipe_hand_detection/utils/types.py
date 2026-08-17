from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedImage:
    tensor: Any
    pixels: Any
    original_width: int
    original_height: int
    scale: float
    pad_x: float
    pad_y: float
    preprocess_ms: float


@dataclass(frozen=True, slots=True)
class PalmCandidate:
    box: tuple[float, float, float, float]
    keypoints: Any
    confidence: float


@dataclass(frozen=True, slots=True)
class PreparedLandmarkCrop:
    tensor: Any
    center_x: float
    center_y: float
    size: float
    rotation: float
    preprocess_ms: float
