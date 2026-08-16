"""Static live transcription App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

LIVE_TRANSCRIPTION_MANIFEST = AppManifest(
    app_id="live-transcription",
    name="Live Transcription",
    description=(
        "Live multi-model transcription with optional Silero VAD and DeepFilterNet3."
    ),
    tags=frozenset({"audio", "asr", "vad", "live"}),
    frontend_route="/apps/live-transcription",
    api_prefix="/api/v1/apps/live-transcription",
    required_models=(
        AppModelRequirement(
            model_id="audio8/audio8-asr",
            capabilities=("speech-transcription",),
            preferred_runtime="coreml",
            required=True,
        ),
        AppModelRequirement(
            model_id="funaudiollm/sensevoice",
            capabilities=("speech-transcription", "speech-understanding"),
            preferred_runtime="coreml",
            required=True,
        ),
        AppModelRequirement(
            model_id="qwen/qwen3-asr",
            capabilities=("speech-transcription",),
            preferred_runtime="coreml",
            required=True,
        ),
        AppModelRequirement(
            model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
            capabilities=("speech-transcription", "streaming-speech-transcription"),
            preferred_runtime="coreml",
            required=True,
        ),
        AppModelRequirement(
            model_id="snakers4/silero-vad",
            capabilities=("voice-activity-detection",),
            preferred_runtime="coreml",
            required=False,
        ),
        AppModelRequirement(
            model_id="deepfilternet/deepfilternet3",
            capabilities=("speech-enhancement",),
            preferred_runtime="coreml",
            required=False,
        ),
    ),
)
