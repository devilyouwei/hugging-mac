"""Static metadata for Palm Trace."""

from hugging_mac_web.app_registry import AppCategory, AppManifest, AppModelRequirement

PALM_TRACE_MANIFEST = AppManifest(
    app_id="palm-trace",
    name="Palm Trace",
    description=(
        "Testing preview: wipe away geometric trails with one or two palms before "
        "the clock runs out, powered by private on-device hand detection."
    ),
    category=AppCategory.GAME,
    tags=frozenset(
        {"game", "testing", "hand", "mediapipe", "camera", "arcade", "coordination"}
    ),
    frontend_route="/games/palm-trace",
    api_prefix="/api/v1/games/palm-trace",
    required_models=(
        AppModelRequirement(
            model_id="qualcomm/mediapipe-hand-detection",
            capabilities=("hand-detection",),
            required=True,
        ),
    ),
)
