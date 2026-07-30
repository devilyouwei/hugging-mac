"""Private intermediate types for the YOLOv8 detection model pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedImage:
    image: Any
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: float
    pad_y: float
    preprocess_ms: float


@dataclass(frozen=True, slots=True)
class DecodedPredictions:
    boxes: Any
    confidences: Any
    classes: Any
    extras: Any
