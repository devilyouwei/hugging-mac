"""Static Object Detection App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

OBJECT_DETECTION_MANIFEST = AppManifest(
    app_id="object-detection",
    name="Object Detection",
    description="在本机使用 YOLOv8 和 Apple Silicon 加速检测图片中的目标。",
    tags=frozenset({"vision", "detection", "yolo", "demo"}),
    frontend_route="/apps/object-detection",
    api_prefix="/api/v1/apps/object-detection",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8",
            capabilities=("object-detection",),
            required=True,
        ),
    ),
)
