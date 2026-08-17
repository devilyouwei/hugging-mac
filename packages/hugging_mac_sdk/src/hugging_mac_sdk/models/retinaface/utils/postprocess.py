"""Decode MobileNet0.25 RetinaFace outputs and restore source coordinates."""

from __future__ import annotations

import importlib
from math import ceil
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    DetectionRequest,
    DetectionTimings,
    FaceDetection,
    FaceLandmarks5,
    Point2D,
)

from .types import PreparedImage

_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_STEPS = (8, 16, 32)
_VARIANCE = (0.1, 0.2)


def postprocess(
    outputs: dict[str, Any],
    request: DetectionRequest,
    prepared: PreparedImage,
    inference_ms: float,
) -> tuple[tuple[FaceDetection, ...], DetectionTimings]:
    started = perf_counter()
    np = _module("numpy", "RetinaFace postprocessing requires NumPy")
    try:
        locations = _array(outputs["locations"], np)[0]
        scores = _array(outputs["scores"], np)[0, :, 1]
        landmark_deltas = _array(outputs["landmarks"], np)[0].reshape(-1, 5, 2)
    except (KeyError, IndexError, ValueError) as error:
        raise InferenceError("Unexpected RetinaFace output contract", cause=error) from error
    priors = _priors(prepared.input_height, prepared.input_width, np)
    if locations.shape[0] != priors.shape[0]:
        raise InferenceError(
            "RetinaFace output/prior count mismatch",
            details={"outputs": locations.shape[0], "priors": priors.shape[0]},
        )
    centers = priors[:, :2] + locations[:, :2] * _VARIANCE[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(locations[:, 2:] * _VARIANCE[1])
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    landmarks = priors[:, None, :2] + landmark_deltas * _VARIANCE[0] * priors[:, None, 2:]
    scale = np.asarray(
        (prepared.input_width, prepared.input_height, prepared.input_width, prepared.input_height),
        dtype=np.float32,
    )
    boxes *= scale
    landmarks *= np.asarray((prepared.input_width, prepared.input_height), dtype=np.float32)
    indices = np.flatnonzero(scores >= request.confidence)
    indices = indices[np.argsort(scores[indices])[::-1]]
    kept: list[int] = []
    for index in indices:
        if all(_iou(boxes[index], boxes[prior]) <= request.iou_threshold for prior in kept):
            kept.append(int(index))
            if len(kept) >= request.max_detections:
                break
    faces = tuple(
        _face(float(scores[index]), boxes[index], landmarks[index], prepared) for index in kept
    )
    return faces, DetectionTimings(
        preprocess_ms=prepared.preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=(perf_counter() - started) * 1000,
    )


def _priors(height: int, width: int, np: Any) -> Any:
    anchors: list[tuple[float, float, float, float]] = []
    for sizes, step in zip(_MIN_SIZES, _STEPS, strict=True):
        for row in range(ceil(height / step)):
            for column in range(ceil(width / step)):
                for size in sizes:
                    anchors.append(
                        (
                            (column + 0.5) * step / width,
                            (row + 0.5) * step / height,
                            size / width,
                            size / height,
                        )
                    )
    return np.asarray(anchors, dtype=np.float32)


def _array(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _iou(left: Any, right: Any) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def _face(score: float, box: Any, points: Any, prepared: PreparedImage) -> FaceDetection:
    def restore(x: float, y: float) -> Point2D:
        return Point2D(
            x=float(min(max((x - prepared.pad_x) / prepared.scale, 0), prepared.original_width)),
            y=float(min(max((y - prepared.pad_y) / prepared.scale, 0), prepared.original_height)),
        )

    first, second = restore(float(box[0]), float(box[1])), restore(float(box[2]), float(box[3]))
    restored = tuple(restore(float(point[0]), float(point[1])) for point in points)
    return FaceDetection(
        box=BoundingBox(x1=first.x, y1=first.y, x2=second.x, y2=second.y),
        confidence=score,
        landmarks=FaceLandmarks5(
            left_eye=restored[0],
            right_eye=restored[1],
            nose=restored[2],
            left_mouth=restored[3],
            right_mouth=restored[4],
        ),
    )


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error
