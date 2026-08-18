"""Static local chat App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

CHAT_MANIFEST = AppManifest(
    app_id="chat",
    name="Local Chat",
    description="选择 Qwen3.5 或 Gemma 4 模型 / 在本机进行私密的多轮对话。",
    tags=frozenset({"chat", "llm", "vlm", "mlx", "local"}),
    frontend_route="/apps/chat",
    api_prefix="/api/v1/apps/chat",
    required_models=(
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
            model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
            capabilities=("streaming-speech-transcription",),
            preferred_runtime="coreml",
            required=False,
        ),
        AppModelRequirement(
            model_id="hexgrad/kokoro",
            capabilities=("speech-synthesis",),
            preferred_runtime="coreml",
            required=False,
        ),
    ),
)
