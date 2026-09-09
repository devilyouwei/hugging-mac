"""Decode raw PP-DocLayoutV3 tensors in one runtime-independent implementation."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.detection import BoundingBox, DetectionTimings, Point2D
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRegion, DocumentLayoutRequest

from .types import PreparedImage


def postprocess(
    outputs: dict[str, Any],
    request: DocumentLayoutRequest,
    prepared: PreparedImage,
    inference_ms: float,
    labels: tuple[str, ...],
) -> tuple[tuple[DocumentLayoutRegion, ...], DetectionTimings]:
    started = perf_counter()
    np = _module("numpy", "PP-DocLayoutV3 postprocessing requires NumPy")
    try:
        logits = _array(outputs["logits"], np)[0]
        boxes = _array(outputs["pred_boxes"], np)[0]
        order_logits = _array(outputs["order_logits"], np)[0]
        masks = _array(outputs["out_masks"], np)[0]
    except (KeyError, IndexError, ValueError) as error:
        raise InferenceError("Unexpected PP-DocLayoutV3 output contract", cause=error) from error
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise InferenceError("Invalid PP-DocLayoutV3 logits or box shape")
    if order_logits.shape != (logits.shape[0], logits.shape[0]):
        raise InferenceError("Invalid PP-DocLayoutV3 reading-order shape")

    scores = _sigmoid(logits, np)
    # Each decoder query describes one region. Selecting globally from the
    # flattened query-by-class matrix can return the same physical box multiple
    # times with different labels and omit other valid queries entirely.
    query_indices = np.arange(logits.shape[0])
    class_ids = np.argmax(scores, axis=1)
    selected_scores = scores[query_indices, class_ids]

    order_scores = _sigmoid(order_logits, np)
    votes = np.triu(order_scores, 1).sum(axis=0) + np.tril(1.0 - order_scores.T, -1).sum(axis=0)
    pointers = np.argsort(votes, kind="stable")
    ranks = np.empty_like(pointers)
    ranks[pointers] = np.arange(pointers.size)
    selected_order = ranks[query_indices]

    keep = selected_scores >= request.confidence
    selected_scores = selected_scores[keep]
    query_indices = query_indices[keep]
    class_ids = class_ids[keep]
    selected_order = selected_order[keep]
    nms_candidates = np.argsort(selected_scores, kind="stable")[::-1]
    nms_kept: list[int] = []
    for item in nms_candidates:
        if any(
            int(class_ids[item]) == int(class_ids[prior])
            and _normalized_box_iou(
                boxes[int(query_indices[item])], boxes[int(query_indices[prior])], np
            )
            >= 0.5
            for prior in nms_kept
        ):
            continue
        nms_kept.append(int(item))
    sequence = sorted(nms_kept, key=lambda item: int(selected_order[item]))[
        : request.max_regions
    ]

    regions: list[DocumentLayoutRegion] = []
    for output_order, item in enumerate(sequence):
        query = int(query_indices[item])
        box = _source_box(boxes[query], prepared, np)
        polygon = (
            _polygon(masks[query], box, request.confidence, prepared, np)
            if request.include_polygons
            else ()
        )
        class_id = int(class_ids[item])
        regions.append(
            DocumentLayoutRegion(
                box=box,
                polygon=polygon,
                confidence=float(selected_scores[item]),
                class_id=class_id,
                label=labels[class_id] if class_id < len(labels) else f"class_{class_id}",
                order=output_order,
            )
        )
    return tuple(regions), DetectionTimings(
        preprocess_ms=prepared.preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=(perf_counter() - started) * 1000,
    )


def _source_box(value: Any, prepared: PreparedImage, np: Any) -> BoundingBox:
    center_x, center_y, width, height = np.asarray(value, dtype=np.float32)
    x1 = float((center_x - width / 2) * prepared.original_width)
    y1 = float((center_y - height / 2) * prepared.original_height)
    x2 = float((center_x + width / 2) * prepared.original_width)
    y2 = float((center_y + height / 2) * prepared.original_height)
    return BoundingBox(
        x1=min(max(x1, 0.0), float(prepared.original_width)),
        y1=min(max(y1, 0.0), float(prepared.original_height)),
        x2=min(max(x2, 0.0), float(prepared.original_width)),
        y2=min(max(y2, 0.0), float(prepared.original_height)),
    )


def _normalized_box_iou(left: Any, right: Any, np: Any) -> float:
    def corners(value: Any) -> tuple[float, float, float, float]:
        center_x, center_y, width, height = np.asarray(value, dtype=np.float32)
        return (
            float(center_x - width / 2),
            float(center_y - height / 2),
            float(center_x + width / 2),
            float(center_y + height / 2),
        )

    lx1, ly1, lx2, ly2 = corners(left)
    rx1, ry1, rx2, ry2 = corners(right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = max(0.0, (lx2 - lx1) * (ly2 - ly1)) + max(
        0.0, (rx2 - rx1) * (ry2 - ry1)
    ) - intersection
    return intersection / union if union else 0.0


def _polygon(
    mask_logits: Any,
    box: BoundingBox,
    threshold: float,
    prepared: PreparedImage,
    np: Any,
) -> tuple[Point2D, ...]:
    fallback = _rectangle(box)
    cv2 = _module("cv2", "PP-DocLayoutV3 polygon extraction requires OpenCV")
    mask = _sigmoid(np.asarray(mask_logits), np) > threshold
    mask_height, mask_width = mask.shape[-2:]
    x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
    width, height = max(0, x2 - x1), max(0, y2 - y1)
    if width == 0 or height == 0:
        return fallback
    mx1 = int(np.clip(round(box.x1 * mask_width / prepared.original_width), 0, mask_width))
    mx2 = int(np.clip(round(box.x2 * mask_width / prepared.original_width), 0, mask_width))
    my1 = int(np.clip(round(box.y1 * mask_height / prepared.original_height), 0, mask_height))
    my2 = int(np.clip(round(box.y2 * mask_height / prepared.original_height), 0, mask_height))
    crop = mask[my1:my2, mx1:mx2]
    if crop.size == 0 or not crop.any():
        return fallback
    resized = cv2.resize(crop.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(polygon) < 4:
        return fallback
    return tuple(Point2D(x=float(x + x1), y=float(y + y1)) for x, y in polygon)


def _rectangle(box: BoundingBox) -> tuple[Point2D, ...]:
    return (
        Point2D(x=box.x1, y=box.y1),
        Point2D(x=box.x2, y=box.y1),
        Point2D(x=box.x2, y=box.y2),
        Point2D(x=box.x1, y=box.y2),
    )


def _sigmoid(value: Any, np: Any) -> Any:
    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _array(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error
