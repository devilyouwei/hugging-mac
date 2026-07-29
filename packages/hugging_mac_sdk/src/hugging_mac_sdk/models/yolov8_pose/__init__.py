"""Ultralytics YOLOv8 Pose model package."""

from hugging_mac_sdk.models.yolov8_pose.catalog import (
    YOLOV8_POSE_DEFINITION,
    YOLOV8_POSE_MANIFEST,
    register_yolov8_pose,
)
from hugging_mac_sdk.models.yolov8_pose.converter import YoloV8PoseConverter
from hugging_mac_sdk.models.yolov8_pose.instance import (
    CoreMlYoloV8PoseInstance,
    PyTorchMpsYoloV8PoseInstance,
)

__all__ = [
    "YOLOV8_POSE_DEFINITION",
    "YOLOV8_POSE_MANIFEST",
    "CoreMlYoloV8PoseInstance",
    "PyTorchMpsYoloV8PoseInstance",
    "YoloV8PoseConverter",
    "register_yolov8_pose",
]
