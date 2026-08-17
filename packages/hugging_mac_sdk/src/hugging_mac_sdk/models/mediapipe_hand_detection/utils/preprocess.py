"""RGB letterbox preprocessing for the 256x256 palm detector."""

from __future__ import annotations

import importlib
import io
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.hand import HandDetectionRequest

from .types import PalmCandidate, PreparedImage, PreparedLandmarkCrop


def prepare_image(request: HandDetectionRequest, input_size: int) -> PreparedImage:
    started = perf_counter()
    image_module = _module("PIL.Image", "MediaPipe Hand Detection requires Pillow")
    try:
        if request.image.path is not None:
            path = request.image.path.expanduser()
            if not path.is_file():
                raise InferenceError(f"Image file does not exist: {path}")
            image = image_module.open(path)
        else:
            assert request.image.data is not None
            image = image_module.open(io.BytesIO(request.image.data))
        image.load()
        image = image.convert("RGB")
    except InferenceError:
        raise
    except Exception as error:
        raise InferenceError("Image could not be decoded", cause=error) from error
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), image_module.Resampling.BILINEAR)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = image_module.new("RGB", (input_size, input_size))
    canvas.paste(resized, (pad_x, pad_y))
    numpy = _module("numpy", "MediaPipe Hand Detection requires NumPy")
    pixels = numpy.asarray(image, dtype=numpy.uint8)
    value = numpy.asarray(canvas, dtype=numpy.float32) / 255.0
    tensor = numpy.ascontiguousarray(value.transpose(2, 0, 1))[None]
    return PreparedImage(
        tensor=tensor,
        pixels=pixels,
        original_width=width,
        original_height=height,
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        preprocess_ms=(perf_counter() - started) * 1000,
    )


def prepare_landmark_crop(
    prepared: PreparedImage,
    palm: PalmCandidate,
    input_size: int,
) -> PreparedLandmarkCrop:
    """Create MediaPipe's rotated 2.6x palm ROI without a framework dependency."""

    started = perf_counter()
    math = _module("math", "Python math module is required")
    image_module = _module("PIL.Image", "MediaPipe Hand Detection requires Pillow")
    numpy = _module("numpy", "MediaPipe Hand Detection requires NumPy")
    x1, _, x2, _ = palm.box
    box_size = max(x2 - x1, 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (palm.box[1] + palm.box[3]) / 2.0 - 0.5 * box_size
    keypoint0 = palm.keypoints[0]
    keypoint2 = palm.keypoints[2]
    rotation = (
        math.atan2(
            float(keypoint0[1] - keypoint2[1]),
            float(keypoint0[0] - keypoint2[0]),
        )
        - math.pi / 2.0
    )
    size = box_size * 2.6
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    step = size / (input_size - 1)
    affine = (
        cosine * step,
        -sine * step,
        center_x + size * (sine - cosine) / 2.0,
        sine * step,
        cosine * step,
        center_y - size * (sine + cosine) / 2.0,
    )
    source = image_module.fromarray(prepared.pixels, mode="RGB")
    crop = source.transform(
        (input_size, input_size),
        image_module.Transform.AFFINE,
        affine,
        resample=image_module.Resampling.BILINEAR,
    )
    value = numpy.asarray(crop, dtype=numpy.float32) / 255.0
    tensor = numpy.ascontiguousarray(value.transpose(2, 0, 1))[None]
    return PreparedLandmarkCrop(
        tensor=tensor,
        center_x=center_x,
        center_y=center_y,
        size=size,
        rotation=rotation,
        preprocess_ms=(perf_counter() - started) * 1000,
    )


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error
