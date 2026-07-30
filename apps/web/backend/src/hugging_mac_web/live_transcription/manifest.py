# ruff: noqa: RUF001
"""Static live transcription App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

LIVE_TRANSCRIPTION_MANIFEST = AppManifest(
    app_id="live-transcription",
    name="Live Transcription",
    description="选择 Audio8-ASR 或 SenseVoice，使用本地 VAD 边录音边断句并实时转写。",
    tags=frozenset({"audio", "asr", "vad", "live"}),
    frontend_route="/apps/live-transcription",
    api_prefix="/api/v1/apps/live-transcription",
    required_models=(
        AppModelRequirement(
            model_id="audio8/audio8-asr-0.1b",
            capabilities=("speech-transcription",),
            preferred_runtime="coreml",
            required=True,
        ),
        AppModelRequirement(
            model_id="funaudiollm/sensevoice-small",
            capabilities=("speech-transcription", "speech-understanding"),
            preferred_runtime="pytorch-mps",
            required=True,
        ),
    ),
)
