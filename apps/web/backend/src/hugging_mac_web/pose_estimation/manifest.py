"""Static YOLOv8 Pose App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

POSE_ESTIMATION_MANIFEST = AppManifest(
    app_id="pose-estimation",
    name="Pose Estimation",
    description="使用 YOLOv8 Pose 在图片、视频与摄像头画面中识别人类姿态和骨架。",
    tags=frozenset({"vision", "pose", "yolo", "demo"}),
    frontend_route="/apps/pose-estimation",
    api_prefix="/api/v1/apps/pose-estimation",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8-pose",
            capabilities=("pose-estimation",),
            required=True,
        ),
    ),
)
