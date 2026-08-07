"""Static metadata for the Digital Human game."""

from hugging_mac_web.app_registry import AppCategory, AppManifest, AppModelRequirement

DIGITAL_HUMAN_MANIFEST = AppManifest(
    app_id="digital-human",
    name="Digital Human",
    description="Have a live camera-aware conversation using fast local Kokoro speech.",
    category=AppCategory.GAME,
    tags=frozenset({"game", "asr", "vlm", "tts", "camera", "kokoro"}),
    frontend_route="/games/digital-human",
    api_prefix="/api/v1/games/digital-human",
    required_models=(
        AppModelRequirement(
            model_id="audio8/audio8-asr-0.1b",
            capabilities=("speech-transcription",),
            preferred_runtime="coreml",
        ),
        AppModelRequirement(
            model_id="mlx-community/qwen3.5-mlx",
            capabilities=("chat", "vision-language-generation"),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="mlx-community/kokoro-82m-bf16",
            capabilities=("speech-synthesis",),
            preferred_runtime="mlx",
        ),
    ),
)
