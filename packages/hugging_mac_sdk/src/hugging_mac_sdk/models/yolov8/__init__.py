"""Ultralytics YOLOv8 source model and conversion registration."""

from hugging_mac_sdk.models.yolov8.catalog import (
    YOLOV8_DEFINITION,
    YOLOV8_MANIFEST,
    YOLOV8N_DEFINITION,
    YOLOV8N_MANIFEST,
    register_yolov8,
)
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.models.yolov8.instance import (
    CoreMlYoloV8Instance,
    PyTorchMpsYoloV8Instance,
)

__all__ = [
    "CoreMlYoloV8Instance",
    "PyTorchMpsYoloV8Instance",
    "YOLOV8N_DEFINITION",
    "YOLOV8N_MANIFEST",
    "YOLOV8_DEFINITION",
    "YOLOV8_MANIFEST",
    "YoloV8Converter",
    "register_yolov8",
]
