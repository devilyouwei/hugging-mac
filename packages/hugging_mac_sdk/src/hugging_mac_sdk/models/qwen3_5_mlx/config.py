"""Configuration for the Qwen3.5 MLX variant family."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_5_MLX_MODEL_ID: Final = "mlx-community/qwen3.5-mlx"
QWEN3_5_MLX_VARIANTS: Final = ("9b-4bit", "4b-optiq-4bit", "2b-optiq-4bit")
QWEN3_5_MLX_REQUIRED_FILES: Final[dict[str, tuple[str, ...]]] = {
    "9b-4bit": (
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "4b-optiq-4bit": (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optiq/mtp.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "2b-optiq-4bit": (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optiq/mtp.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


class Qwen35MlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["9b-4bit", "4b-optiq-4bit", "2b-optiq-4bit"] = "9b-4bit"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
