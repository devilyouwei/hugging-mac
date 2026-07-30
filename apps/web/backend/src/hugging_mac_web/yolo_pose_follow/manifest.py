# ruff: noqa: RUF001
"""Static metadata for Yolo Pose Follow."""

from hugging_mac_web.app_registry import (
    AppCategory,
    AppManifest,
    AppModelRequirement,
)

YOLO_POSE_FOLLOW_MANIFEST = AppManifest(
    app_id="yolo-pose-follow",
    name="Yolo Pose Follow",
    description="在镜头前模仿不断加速的姿态，用本地 YOLO Pose 骨架匹配挑战连击。",
    category=AppCategory.GAME,
    tags=frozenset({"game", "pose", "yolo", "camera"}),
    frontend_route="/games/yolo-pose-follow",
    api_prefix="/api/v1/games/yolo-pose-follow",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8-pose",
            capabilities=("pose-estimation",),
            required=True,
        ),
    ),
)
