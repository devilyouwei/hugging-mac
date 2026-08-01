"""Configuration for the pinned Qwen3.5 9B MLX 4-bit integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_5_9B_MLX_4BIT_MODEL_ID: Final = "mlx-community/qwen3.5-9b-mlx-4bit"
QWEN3_5_9B_MLX_4BIT_REPO_ID: Final = "mlx-community/Qwen3.5-9B-MLX-4bit"
QWEN3_5_9B_MLX_4BIT_REVISION: Final = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
QWEN3_5_9B_MLX_4BIT_VARIANT: Final = "4bit"
QWEN3_5_9B_MLX_4BIT_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Qwen35MlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["4bit"] = "4bit"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
