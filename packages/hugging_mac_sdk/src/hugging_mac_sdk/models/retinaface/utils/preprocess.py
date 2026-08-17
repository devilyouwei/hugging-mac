"""Shared py-feat RetinaFace image preprocessing."""

from __future__ import annotations

import importlib
import io
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.detection import DetectionRequest

from .types import PreparedImage


def prepare_image(request: DetectionRequest, input_size: int) -> PreparedImage:
    started = perf_counter()
    image_module = _module("PIL.Image", "RetinaFace image decoding requires Pillow")
    if request.image.path is not None:
        path = request.image.path.expanduser()
        if not path.is_file():
            raise InferenceError(f"Image file does not exist: {path}")
        try:
            image = image_module.open(path)
        except Exception as error:
            raise InferenceError(f"Image file could not be decoded: {path}", cause=error) from error
    else:
        assert request.image.data is not None
        try:
            image = image_module.open(io.BytesIO(request.image.data))
        except Exception as error:
            raise InferenceError("Image bytes could not be decoded", cause=error) from error
    image.load()
    image = image.convert("RGB")
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), image_module.Resampling.BILINEAR)
    pad_x = round((input_size - resized_width) / 2)
    pad_y = round((input_size - resized_height) / 2)
    canvas = image_module.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))
    np = _module("numpy", "RetinaFace preprocessing requires NumPy")
    value = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)
    value -= np.asarray((123.0, 117.0, 104.0), dtype=np.float32)[:, None, None]
    tensor = np.ascontiguousarray(value[None])
    return PreparedImage(
        image=canvas,
        tensor=tensor,
        original_width=width,
        original_height=height,
        input_width=input_size,
        input_height=input_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        preprocess_ms=(perf_counter() - started) * 1000,
    )


def torch_input(prepared: PreparedImage, torch: Any) -> Any:
    return torch.from_numpy(prepared.tensor)


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error
