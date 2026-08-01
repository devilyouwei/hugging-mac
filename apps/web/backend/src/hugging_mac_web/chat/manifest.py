"""Static local chat App metadata."""

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

CHAT_MANIFEST = AppManifest(
    app_id="chat",
    name="Local Chat",
    description="使用 Qwen3.5 9B 在本机进行文本与图片多轮对话。",
    tags=frozenset({"chat", "llm", "vlm", "mlx", "local"}),
    frontend_route="/apps/chat",
    api_prefix="/api/v1/apps/chat",
    required_models=(
        AppModelRequirement(
            model_id="mlx-community/qwen3.5-9b-mlx-4bit",
            capabilities=("chat", "vision-language-generation"),
            preferred_runtime="mlx",
        ),
    ),
)
