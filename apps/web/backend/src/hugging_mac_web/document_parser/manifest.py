"""Document Parser application metadata."""
# ruff: noqa: RUF001

from hugging_mac_web.app_registry import AppManifest, AppModelRequirement

DOCUMENT_PARSER_MANIFEST = AppManifest(
    app_id="document-parser",
    name="Document Parser",
    description="在本机将 PDF 或有序图片转换为 Markdown，并提取结构化文档信息。",
    tags=frozenset({"ocr", "pdf", "document", "mlx", "local"}),
    frontend_route="/apps/document-parser",
    api_prefix="/api/v1/apps/document-parser",
    required_models=(
        AppModelRequirement(
            model_id="baidu/unlimited-ocr",
            capabilities=("document-parsing",),
            preferred_runtime="mlx",
        ),
        AppModelRequirement(
            model_id="zai-org/glm-ocr",
            capabilities=("document-parsing",),
            preferred_runtime="mlx",
            required=False,
        ),
        AppModelRequirement(
            model_id="stepfun-ai/got-ocr2.0",
            capabilities=("document-parsing",),
            preferred_runtime="mlx",
            required=False,
        ),
    ),
)
