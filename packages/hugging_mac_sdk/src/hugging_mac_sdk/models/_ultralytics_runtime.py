"""Small runtime-neutral helpers shared by independent Ultralytics task packages."""

from __future__ import annotations

import importlib
import io
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.detection import DetectionRequest, DetectionTimings


def decode_image(request: DetectionRequest) -> Any:
    if request.image.path is not None:
        path = request.image.path.expanduser()
        if not path.is_file():
            raise InferenceError(f"Image file does not exist: {path}")
        return str(path)
    assert request.image.data is not None
    try:
        image = importlib.import_module("PIL.Image").open(io.BytesIO(request.image.data))
        image.load()
        return image.convert("RGB")
    except ImportError as error:
        raise UnsupportedRuntimeError("Image decoding requires Pillow", cause=error) from error
    except Exception as error:
        raise InferenceError("Image bytes could not be decoded", cause=error) from error


def prediction_arguments(request: DetectionRequest, image_size: int, *, device: str | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "conf": request.confidence,
        "iou": request.iou_threshold,
        "max_det": request.max_detections,
        "classes": list(request.classes) if request.classes is not None else None,
        "imgsz": image_size,
        "verbose": False,
    }
    if device is not None:
        arguments["device"] = device
    return arguments


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def timings(result: Any) -> DetectionTimings:
    speed = getattr(result, "speed", {}) or {}
    return DetectionTimings(
        preprocess_ms=_optional_float(speed.get("preprocess")),
        inference_ms=_optional_float(speed.get("inference")),
        postprocess_ms=_optional_float(speed.get("postprocess")),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
