# ruff: noqa: RUF001
"""Static text-to-speech App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

TEXT_TO_SPEECH_MANIFEST = AppManifest(
    app_id="text-to-speech",
    name="Text to Speech",
    description="在 Audio8、Kokoro 与 Qwen3-TTS 之间切换，将文字转换为本地语音。",
    tags=frozenset({"audio", "tts", "local"}),
    frontend_route="/apps/text-to-speech",
    api_prefix="/api/v1/apps/text-to-speech",
    required_models=(
        AppModelRequirement(
            model_id="audio8/audio8-tts-preview-0.6b",
            capabilities=("speech-synthesis",),
            preferred_runtime="pytorch",
        ),
        AppModelRequirement(
            model_id="mlx-community/audio8-tts-preview-0.6b-bf16",
            capabilities=("speech-synthesis",),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="mlx-community/kokoro-82m-bf16",
            capabilities=("speech-synthesis",),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="mlx-community/qwen3-tts-12hz-0.6b-base-4bit",
            capabilities=("speech-synthesis",),
            preferred_runtime="mlx",
        ),
    ),
)
