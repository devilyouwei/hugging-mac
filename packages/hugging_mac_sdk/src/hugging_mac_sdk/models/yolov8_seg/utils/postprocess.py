"""YOLOv8 segmentation decoding, NMS, mask handling, and contour extraction."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    DetectionRequest,
    DetectionTimings,
)
from hugging_mac_sdk.schemas.segmentation import PolygonPoint, Segmentation

from .types import DecodedPredictions, PreparedImage

COCO80_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _module("numpy", "YOLO postprocessing requires NumPy").asarray(value)


def raw_output(
    outputs: dict[str, Any] | Any,
    *,
    minimum_channels: int,
    exclude: tuple[str, ...] = (),
) -> Any:
    values = outputs.items() if isinstance(outputs, dict) else (("output", outputs),)
    candidates: list[Any] = []
    for name, value in values:
        if name in exclude:
            continue
        array = as_numpy(value)
        if array.ndim == 3 and minimum_channels <= min(array.shape[1], array.shape[2]):
            candidates.append(array)
    if not candidates:
        raise InferenceError("Runtime output does not contain a YOLO prediction tensor")
    prediction = max(candidates, key=lambda item: item.size)
    if prediction.shape[1] > prediction.shape[2]:
        prediction = prediction.transpose(0, 2, 1)
    return prediction[0]


def decode_predictions(
    prediction: Any,
    request: DetectionRequest,
    prepared: PreparedImage,
    *,
    class_count: int,
) -> DecodedPredictions:
    numpy = _module("numpy", "YOLO postprocessing requires NumPy")
    boxes_xywh = prediction[:4].transpose()
    class_scores = prediction[4 : 4 + class_count].transpose()
    extras = prediction[4 + class_count :].transpose()
    classes = class_scores.argmax(axis=1)
    confidences = class_scores[numpy.arange(len(classes)), classes]
    keep = confidences >= request.confidence
    if request.classes is not None:
        keep &= numpy.isin(classes, request.classes)
    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    classes = classes[keep]
    extras = extras[keep]
    if not len(boxes_xywh):
        return DecodedPredictions(
            boxes=numpy.empty((0, 4), dtype=numpy.float32),
            confidences=confidences,
            classes=classes,
            extras=extras,
        )
    boxes = xywh_to_xyxy(boxes_xywh)
    selected = class_aware_nms(
        boxes,
        confidences,
        classes,
        request.iou_threshold,
        request.max_detections,
    )
    return DecodedPredictions(
        boxes=inverse_letterbox(boxes[selected], prepared),
        confidences=confidences[selected],
        classes=classes[selected],
        extras=extras[selected],
    )


def decode_coreml_nms(
    outputs: dict[str, Any],
    request: DetectionRequest,
    prepared: PreparedImage,
) -> DecodedPredictions | None:
    numpy = _module("numpy", "YOLO postprocessing requires NumPy")
    if "coordinates" not in outputs or "confidence" not in outputs:
        return None
    coordinates = as_numpy(outputs["coordinates"])
    scores = as_numpy(outputs["confidence"])
    classes = scores.argmax(axis=1)
    confidences = scores[numpy.arange(len(classes)), classes]
    keep = confidences >= request.confidence
    if request.classes is not None:
        keep &= numpy.isin(classes, request.classes)
    coordinates = coordinates[keep]
    confidences = confidences[keep]
    classes = classes[keep]
    if len(coordinates) and float(numpy.nanmax(numpy.abs(coordinates))) <= 2.0:
        coordinates = coordinates * prepared.input_size
    boxes = xywh_to_xyxy(coordinates)
    boxes = inverse_letterbox(boxes, prepared)
    order = confidences.argsort()[::-1][: request.max_detections]
    return DecodedPredictions(
        boxes=boxes[order],
        confidences=confidences[order],
        classes=classes[order],
        extras=numpy.empty((len(order), 0), dtype=numpy.float32),
    )


def xywh_to_xyxy(boxes: Any) -> Any:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def inverse_letterbox(boxes: Any, prepared: PreparedImage) -> Any:
    boxes = boxes.copy()
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - prepared.pad_x) / prepared.scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - prepared.pad_y) / prepared.scale
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, prepared.original_width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, prepared.original_height)
    return boxes


def class_aware_nms(
    boxes: Any,
    scores: Any,
    classes: Any,
    threshold: float,
    limit: int,
) -> Any:
    numpy = _module("numpy", "YOLO postprocessing requires NumPy")
    selected: list[int] = []
    for class_id in numpy.unique(classes):
        indices = numpy.flatnonzero(classes == class_id)
        order = indices[numpy.argsort(scores[indices])[::-1]]
        while len(order):
            current = int(order[0])
            selected.append(current)
            if len(order) == 1:
                break
            overlaps = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][overlaps <= threshold]
    selected.sort(key=lambda index: float(scores[index]), reverse=True)
    return numpy.asarray(selected[:limit], dtype=numpy.int64)


def box_iou(box: Any, boxes: Any) -> Any:
    numpy = _module("numpy", "YOLO postprocessing requires NumPy")
    upper_left = numpy.maximum(box[:2], boxes[:, :2])
    lower_right = numpy.minimum(box[2:], boxes[:, 2:])
    intersection = numpy.prod(numpy.clip(lower_right - upper_left, 0, None), axis=1)
    area = numpy.prod(numpy.clip(box[2:] - box[:2], 0, None))
    other_area = numpy.prod(numpy.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)
    return intersection / numpy.maximum(area + other_area - intersection, 1e-7)


def timings(
    prepared: PreparedImage,
    inference_ms: float,
    postprocess_started: float,
) -> DetectionTimings:
    return DetectionTimings(
        preprocess_ms=prepared.preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=(perf_counter() - postprocess_started) * 1000,
    )


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error


def postprocess(
    outputs: dict[str, Any],
    request: DetectionRequest,
    prepared: PreparedImage,
    inference_ms: float,
) -> tuple[tuple[Segmentation, ...], DetectionTimings]:
    started = perf_counter()
    prediction = raw_output(outputs, minimum_channels=116, exclude=("prototypes",))
    decoded = decode_predictions(
        prediction,
        request,
        prepared,
        class_count=len(COCO80_NAMES),
    )
    prototypes = _prototype_output(outputs)
    polygons = masks_to_polygons(
        decoded.extras,
        prototypes,
        decoded.boxes,
        prepared,
    )
    segments = tuple(
        Segmentation(
            box=BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            ),
            confidence=float(confidence),
            class_id=int(class_id),
            label=COCO80_NAMES[int(class_id)],
            polygons=contours,
        )
        for box, confidence, class_id, contours in zip(
            decoded.boxes,
            decoded.confidences,
            decoded.classes,
            polygons,
            strict=True,
        )
    )
    return segments, timings(prepared, inference_ms, started)


def masks_to_polygons(
    coefficients: Any,
    prototypes: Any,
    boxes: Any,
    prepared: PreparedImage,
) -> tuple[tuple[tuple[PolygonPoint, ...], ...], ...]:
    if not len(coefficients):
        return ()
    try:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise UnsupportedRuntimeError(
            "YOLO segmentation postprocessing requires OpenCV and NumPy",
            cause=error,
        ) from error

    channels, proto_height, proto_width = prototypes.shape
    logits = coefficients[:, :channels] @ prototypes.reshape(channels, -1)
    masks = 1.0 / (1.0 + numpy.exp(-logits))
    masks = masks.reshape(-1, proto_height, proto_width)
    left = max(0, round(prepared.pad_x))
    top = max(0, round(prepared.pad_y))
    right = min(prepared.input_size, round(prepared.input_size - prepared.pad_x))
    bottom = min(prepared.input_size, round(prepared.input_size - prepared.pad_y))
    results: list[tuple[tuple[PolygonPoint, ...], ...]] = []
    for mask, box in zip(masks, boxes, strict=True):
        mask = cv2.resize(
            mask,
            (prepared.input_size, prepared.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = mask[top:bottom, left:right]
        mask = cv2.resize(
            mask,
            (prepared.original_width, prepared.original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        binary = (mask > 0.5).astype(numpy.uint8)
        crop = numpy.zeros_like(binary)
        x1, y1, x2, y2 = (
            max(0, int(box[0])),
            max(0, int(box[1])),
            min(prepared.original_width, int(box[2]) + 1),
            min(prepared.original_height, int(box[3]) + 1),
        )
        crop[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = tuple(
            tuple(PolygonPoint(x=float(point[0][0]), y=float(point[0][1])) for point in contour)
            for contour in contours
            if len(contour) >= 3
        )
        results.append(polygons)
    return tuple(results)


def _prototype_output(outputs: dict[str, Any]) -> Any:
    if "prototypes" in outputs:
        value = as_numpy(outputs["prototypes"])
        return value[0] if value.ndim == 4 else value
    candidates = []
    for value in outputs.values():
        array = as_numpy(value)
        if array.ndim == 4:
            candidates.append(array)
    if not candidates:
        raise InferenceError("Runtime output does not contain YOLO mask prototypes")
    return max(candidates, key=lambda item: item.size)[0]
