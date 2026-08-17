# ruff: noqa: RUF001
"""Static metadata for YOLO Fruit Slice."""

from hugging_mac_web.app_registry import AppCategory, AppManifest, AppModelRequirement

YOLO_FRUIT_SLICE_MANIFEST = AppManifest(
    app_id="yolo-fruit-slice",
    name="Yolo Fruit Slice",
    description="用本地 YOLO Pose 将双侧前臂变成刀刃，在摄像头画面上切水果、躲炸弹并挑战连击。",
    category=AppCategory.GAME,
    tags=frozenset({"game", "pose", "yolo", "camera", "motion"}),
    frontend_route="/games/yolo-fruit-slice",
    api_prefix="/api/v1/games/yolo-fruit-slice",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8-pose",
            capabilities=("pose-estimation",),
            required=True,
        ),
    ),
)
