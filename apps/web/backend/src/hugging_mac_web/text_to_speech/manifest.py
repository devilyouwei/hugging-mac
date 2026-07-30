# ruff: noqa: RUF001
"""Static text-to-speech App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

TEXT_TO_SPEECH_MANIFEST = AppManifest(
    app_id="text-to-speech",
    name="Text to Speech",
    description="在 Audio8 TTS 与 Kokoro 之间切换，将文字转换为完全本地生成的语音。",
    tags=frozenset({"audio", "tts", "coreml", "local"}),
    frontend_route="/apps/text-to-speech",
    api_prefix="/api/v1/apps/text-to-speech",
    required_models=(
        AppModelRequirement(
            model_id="audio8/audio8-tts-preview-0.6b",
            capabilities=("speech-synthesis",),
            preferred_runtime="pytorch-mps",
        ),
        AppModelRequirement(
            model_id="hexgrad/kokoro-82m",
            capabilities=("speech-synthesis",),
            preferred_runtime="coreml",
        ),
    ),
)
