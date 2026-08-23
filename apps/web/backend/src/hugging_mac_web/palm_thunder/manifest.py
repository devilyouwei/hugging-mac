"""Static metadata for Palm Thunder."""

from hugging_mac_web.app_registry import AppCategory, AppManifest, AppModelRequirement

PALM_THUNDER_MANIFEST = AppManifest(
    app_id="palm-thunder",
    name="Palm Thunder",
    description=(
        "Control a live fighter with local MediaPipe hand tracking, clench your fist "
        "to bomb the screen, and use the Love gesture to revive your wingmate."
    ),
    category=AppCategory.GAME,
    tags=frozenset({"game", "hand", "mediapipe", "camera", "shooter", "co-op"}),
    frontend_route="/games/palm-thunder",
    api_prefix="/api/v1/games/palm-thunder",
    required_models=(
        AppModelRequirement(
            model_id="qualcomm/mediapipe-hand-detection",
            capabilities=("hand-detection",),
            required=True,
        ),
    ),
)
