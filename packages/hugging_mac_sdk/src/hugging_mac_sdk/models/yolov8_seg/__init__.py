"""Ultralytics YOLOv8 Seg model package."""

from hugging_mac_sdk.models.yolov8_seg.catalog import (
    YOLOV8_SEG_DEFINITION,
    YOLOV8_SEG_MANIFEST,
    register_yolov8_seg,
)
from hugging_mac_sdk.models.yolov8_seg.converter import YoloV8SegConverter
from hugging_mac_sdk.models.yolov8_seg.instance import (
    CoreMlYoloV8SegInstance,
    PyTorchMpsYoloV8SegInstance,
)

__all__ = [
    "YOLOV8_SEG_DEFINITION",
    "YOLOV8_SEG_MANIFEST",
    "CoreMlYoloV8SegInstance",
    "PyTorchMpsYoloV8SegInstance",
    "YoloV8SegConverter",
    "register_yolov8_seg",
]
