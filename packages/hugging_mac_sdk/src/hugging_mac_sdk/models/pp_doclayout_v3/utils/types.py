from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedImage:
    tensor: Any
    original_width: int
    original_height: int
    input_size: int
    preprocess_ms: float
