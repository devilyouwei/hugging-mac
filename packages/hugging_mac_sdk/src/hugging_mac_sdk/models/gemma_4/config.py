"""Configuration for the Gemma 4 MLX variant family."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

GEMMA_4_MODEL_ID: Final = "google/gemma-4"
GEMMA_4_MLX_VARIANTS: Final = ("e4b", "e2b")
GEMMA_4_TOKENIZER_REVISION: Final = "b5f52d33150fd5856c9de2d0b8c54438c266d28a"
GEMMA_4_TOKENIZER_REQUIRED_FILES: Final = ("tokenizer.json", "tokenizer_config.json")
GEMMA_4_MLX_REQUIRED_FILES: Final[dict[str, tuple[str, ...]]] = {
    "e4b": (
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "chat_template.jinja",
    ),
    "e2b": (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "chat_template.jinja",
    ),
}


class Gemma4MlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["e4b", "e2b"] = "e4b"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
