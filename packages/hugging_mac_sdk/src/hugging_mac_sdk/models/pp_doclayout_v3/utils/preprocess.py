"""The exact fixed-size preprocessing declared by the upstream checkpoint."""

from __future__ import annotations

import importlib
import io
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRequest

from .types import PreparedImage


def prepare_image(request: DocumentLayoutRequest, input_size: int = 800) -> PreparedImage:
    started = perf_counter()
    image_module = _module("PIL.Image", "PP-DocLayoutV3 image decoding requires Pillow")
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
    image = image.resize((input_size, input_size), image_module.Resampling.BICUBIC)
    np = _module("numpy", "PP-DocLayoutV3 preprocessing requires NumPy")
    value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(value.transpose(2, 0, 1)[None])
    return PreparedImage(
        tensor=tensor,
        original_width=width,
        original_height=height,
        input_size=input_size,
        preprocess_ms=(perf_counter() - started) * 1000,
    )


def torch_input(prepared: PreparedImage, torch: Any) -> Any:
    return torch.from_numpy(prepared.tensor)


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error
