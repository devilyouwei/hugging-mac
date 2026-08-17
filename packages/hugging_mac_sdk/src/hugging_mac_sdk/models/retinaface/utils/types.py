from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedImage:
    image: Any
    tensor: Any
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_x: float
    pad_y: float
    preprocess_ms: float
