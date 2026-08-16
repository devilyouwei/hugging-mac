"""Configuration for the Qwen3.5 MLX variant family."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_5_MLX_MODEL_ID: Final = "qwen/qwen3.5"
QWEN3_5_MLX_VARIANTS: Final = ("9b", "4b", "2b")
QWEN3_5_TOKENIZER_REVISION: Final = "67741b04f23bfdb46501f748ce27865ec82eccfb"
QWEN3_5_TOKENIZER_REQUIRED_FILES: Final = ("tokenizer.json", "tokenizer_config.json")
QWEN3_5_MLX_REQUIRED_FILES: Final[dict[str, tuple[str, ...]]] = {
    "9b": (
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "optiq/mtp.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "chat_template.jinja",
    ),
    "4b": (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optiq/mtp.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "chat_template.jinja",
    ),
    "2b": (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optiq/mtp.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "chat_template.jinja",
    ),
}


class Qwen35MlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["9b", "4b", "2b"] = "9b"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
