"""Static YOLOv8 Pose App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

POSE_ESTIMATION_MANIFEST = AppManifest(
    app_id="pose-estimation",
    name="Pose Estimation",
    description="并行运行 YOLOv8 Pose、RetinaFace 与 MediaPipe Hand Detection。",
    tags=frozenset({"vision", "pose", "face", "hand", "yolo", "retinaface", "mediapipe", "demo"}),
    frontend_route="/apps/pose-estimation",
    api_prefix="/api/v1/apps/pose-estimation",
    required_models=(
        AppModelRequirement(
            model_id="ultralytics/yolov8-pose",
            capabilities=("pose-estimation",),
            required=True,
        ),
        AppModelRequirement(
            model_id="py-feat/retinaface",
            capabilities=("face-detection",),
            required=False,
        ),
        AppModelRequirement(
            model_id="qualcomm/mediapipe-hand-detection",
            capabilities=("hand-detection",),
            required=False,
        ),
    ),
)
