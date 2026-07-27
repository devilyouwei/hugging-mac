"""Safe, lightweight media inspection helpers."""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_VIDEO_PREFIX = "video/"
_AUDIO_PREFIX = "audio/"


def classify_media(filename: str, content_type: str | None) -> str:
    detected = content_type or mimetypes.guess_type(filename)[0] or ""
    if detected in _IMAGE_TYPES:
        return "image"
    if detected.startswith(_VIDEO_PREFIX):
        return "video"
    if detected.startswith(_AUDIO_PREFIX):
        return "audio"
    return "unknown"


def safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if len(suffix) > 12 or not suffix.startswith(".") or not suffix[1:].isalnum():
        return ".bin"
    return suffix


def inspect_image(
    data: bytes,
    *,
    max_pixels: int,
    verify: bool = True,
) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError(f"Image exceeds pixel limit: {width}x{height}")
            if verify:
                image.verify()
            return {
                "width": width,
                "height": height,
                "format": image.format,
                "mode": image.mode,
            }
    except UnidentifiedImageError as error:
        raise ValueError("Uploaded file is not a supported image") from error
