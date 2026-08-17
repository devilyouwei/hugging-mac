"""MediaPipe palm decoding, weighted NMS, and landmark projection."""

from __future__ import annotations

import importlib
import math
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.hand import HandDetectionRequest, HandLandmark

from .types import PalmCandidate, PreparedImage, PreparedLandmarkCrop

_MEDIAPIPE_MAX_NMS_IOU = 0.3


def decode_palms(
    outputs: dict[str, Any],
    request: HandDetectionRequest,
    prepared: PreparedImage,
) -> tuple[tuple[PalmCandidate, ...], float]:
    started = perf_counter()
    numpy = importlib.import_module("numpy")
    try:
        coords = numpy.asarray(outputs["box_coords"], dtype=numpy.float32)[0]
        logits = numpy.asarray(outputs["box_scores"], dtype=numpy.float32)[0, :, 0]
    except (KeyError, IndexError, ValueError) as error:
        raise InferenceError("Unexpected MediaPipe hand detector outputs", cause=error) from error
    if coords.shape != (2944, 18) or logits.shape != (2944,):
        raise InferenceError(
            "Unexpected MediaPipe hand detector output shapes",
            details={"coords": coords.shape, "scores": logits.shape},
        )
    anchors = _anchors()
    scores = 1.0 / (1.0 + numpy.exp(-numpy.clip(logits, -80.0, 80.0)))
    centers = coords[:, :2] / 256.0 + anchors
    sizes = coords[:, 2:4] / 256.0
    decoded = numpy.empty_like(coords)
    decoded[:, :4] = numpy.column_stack(
        (
            centers[:, 0] - sizes[:, 0] / 2,
            centers[:, 1] - sizes[:, 1] / 2,
            centers[:, 0] + sizes[:, 0] / 2,
            centers[:, 1] + sizes[:, 1] / 2,
        )
    )
    decoded[:, 4:] = (coords[:, 4:].reshape(-1, 7, 2) / 256.0 + anchors[:, None, :]).reshape(-1, 14)
    decoded *= 256.0
    candidates = numpy.flatnonzero(scores >= request.confidence)
    selected = _weighted_nms(
        decoded,
        scores,
        candidates,
        min(request.iou_threshold, _MEDIAPIPE_MAX_NMS_IOU),
        request.max_detections,
    )
    palms: list[PalmCandidate] = []
    for values, score in selected:
        source = values.copy()
        source[0::2] = (source[0::2] - prepared.pad_x) / prepared.scale
        source[1::2] = (source[1::2] - prepared.pad_y) / prepared.scale
        x1, y1, x2, y2 = (float(value) for value in source[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        palms.append(
            PalmCandidate(
                box=(x1, y1, x2, y2),
                keypoints=source[4:].reshape(7, 2),
                confidence=float(score),
            )
        )
    return tuple(palms), (perf_counter() - started) * 1000


def decode_landmarks(
    outputs: dict[str, Any],
    crop: PreparedLandmarkCrop,
    *,
    minimum_confidence: float,
    image_width: int,
    image_height: int,
    input_mirrored: bool = False,
) -> tuple[
    tuple[HandLandmark, ...],
    float,
    Literal["left", "right"] | None,
    float | None,
    float,
]:
    started = perf_counter()
    numpy = importlib.import_module("numpy")
    try:
        confidence = float(numpy.asarray(outputs["scores"], dtype=numpy.float32).reshape(-1)[0])
        handedness_score = float(numpy.asarray(outputs["lr"], dtype=numpy.float32).reshape(-1)[0])
        raw = numpy.asarray(outputs["landmarks"], dtype=numpy.float32).reshape(21, 3)
    except (KeyError, IndexError, ValueError) as error:
        raise InferenceError("Unexpected MediaPipe hand landmark outputs", cause=error) from error
    if confidence < minimum_confidence:
        return (), confidence, None, None, (perf_counter() - started) * 1000
    cosine = math.cos(crop.rotation)
    sine = math.sin(crop.rotation)
    points: list[HandLandmark] = []
    for x_value, y_value, z_value in raw:
        local_x = (float(x_value) - 0.5) * crop.size
        local_y = (float(y_value) - 0.5) * crop.size
        x = crop.center_x + cosine * local_x - sine * local_y
        y = crop.center_y + sine * local_x + cosine * local_y
        points.append(
            HandLandmark(
                x=min(max(x, 0.0), float(image_width)),
                y=min(max(y, 0.0), float(image_height)),
                z=float(z_value),
            )
        )
    handedness: Literal["left", "right"] = "right" if handedness_score >= 0.5 else "left"
    if input_mirrored:
        handedness = "left" if handedness == "right" else "right"
    handedness_confidence = max(handedness_score, 1.0 - handedness_score)
    return (
        tuple(points),
        confidence,
        handedness,
        handedness_confidence,
        (perf_counter() - started) * 1000,
    )


@lru_cache(maxsize=1)
def _anchors() -> Any:
    numpy = importlib.import_module("numpy")
    anchors: list[tuple[float, float]] = []
    strides = (8, 16, 32, 32, 32)
    layer = 0
    while layer < len(strides):
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
        anchors_per_cell = 2 * (last - layer)
        grid = 256 // strides[layer]
        for y in range(grid):
            for x in range(grid):
                anchors.extend([((x + 0.5) / grid, (y + 0.5) / grid)] * anchors_per_cell)
        layer = last
    result = numpy.asarray(anchors, dtype=numpy.float32)
    if result.shape != (2944, 2):
        raise RuntimeError(f"Generated invalid MediaPipe anchor shape: {result.shape}")
    return result


def _weighted_nms(
    decoded: Any,
    scores: Any,
    candidates: Any,
    threshold: float,
    limit: int,
) -> list[tuple[Any, float]]:
    """Merge boxes and all seven palm keypoints with MediaPipe's weighted NMS."""

    numpy = importlib.import_module("numpy")
    boxes = decoded[:, :4]
    order = candidates[numpy.argsort(scores[candidates])[::-1]]
    selected: list[tuple[Any, float]] = []
    while order.size and len(selected) < limit:
        current = int(order[0])
        xx1 = numpy.maximum(boxes[current, 0], boxes[order, 0])
        yy1 = numpy.maximum(boxes[current, 1], boxes[order, 1])
        xx2 = numpy.minimum(boxes[current, 2], boxes[order, 2])
        yy2 = numpy.minimum(boxes[current, 3], boxes[order, 3])
        intersection = numpy.maximum(0.0, xx2 - xx1) * numpy.maximum(0.0, yy2 - yy1)
        area_a = numpy.maximum(0.0, boxes[current, 2] - boxes[current, 0]) * numpy.maximum(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        area_b = numpy.maximum(0.0, boxes[order, 2] - boxes[order, 0]) * numpy.maximum(
            0.0, boxes[order, 3] - boxes[order, 1]
        )
        union = area_a + area_b - intersection
        overlap = (
            numpy.divide(intersection, union, out=numpy.zeros_like(intersection), where=union > 0)
            > threshold
        )
        cluster = order[overlap]
        cluster_scores = scores[cluster]
        total_score = float(cluster_scores.sum())
        merged = (
            numpy.average(decoded[cluster], axis=0, weights=cluster_scores)
            if total_score > 0
            else decoded[current].copy()
        )
        selected.append((merged, float(cluster_scores.mean())))
        order = order[~overlap]
    return selected
