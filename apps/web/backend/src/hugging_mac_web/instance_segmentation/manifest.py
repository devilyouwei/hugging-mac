"""Static YOLOv8 Seg App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

INSTANCE_SEGMENTATION_MANIFEST = AppManifest(
    app_id="instance-segmentation",
    name="Instance Segmentation",
    description="使用 YOLOv8 Seg 分割图片、视频与摄像头中的主体 / 叠加彩色 Mask。",
    tags=frozenset({"vision", "segmentation", "yolo", "demo"}),
    frontend_route="/apps/instance-segmentation",
    api_prefix="/api/v1/apps/instance-segmentation",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8-seg",
            capabilities=("instance-segmentation",),
            required=True,
        ),
    ),
)
