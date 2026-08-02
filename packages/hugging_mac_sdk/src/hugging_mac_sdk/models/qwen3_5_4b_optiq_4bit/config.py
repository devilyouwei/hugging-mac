"""Configuration for the pinned Qwen3.5 4B OptiQ 4-bit integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_5_4B_OPTIQ_4BIT_MODEL_ID: Final = "mlx-community/qwen3.5-4b-optiq-4bit"
QWEN3_5_4B_OPTIQ_4BIT_REPO_ID: Final = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
QWEN3_5_4B_OPTIQ_4BIT_REVISION: Final = "6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9"
QWEN3_5_4B_OPTIQ_4BIT_VARIANT: Final = "4bit"
QWEN3_5_4B_OPTIQ_4BIT_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "optiq/mtp.safetensors",
    "optiq/optiq_vision.safetensors",
    "optiq_metadata.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Qwen35OptiQMlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["4bit"] = "4bit"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
