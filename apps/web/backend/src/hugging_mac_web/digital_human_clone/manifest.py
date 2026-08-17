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
            model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
            capabilities=("streaming-speech-transcription",),
            preferred_runtime="coreml",
        ),
        AppModelRequirement(
            model_id="audio8/audio8-asr",
            capabilities=("speech-transcription",),
            preferred_runtime="coreml",
        ),
        AppModelRequirement(
            model_id="qwen/qwen3.5",
            capabilities=("chat", "vision-language-generation"),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="google/gemma-4",
            capabilities=("chat",),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="hexgrad/kokoro",
            capabilities=("speech-synthesis",),
            preferred_runtime="coreml",
        ),
    ),
)
