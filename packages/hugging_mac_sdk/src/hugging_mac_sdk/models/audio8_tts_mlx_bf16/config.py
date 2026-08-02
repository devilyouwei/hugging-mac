"""Configuration for the pinned Audio8-TTS MLX BF16 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path, user_data_path
from pydantic import BaseModel, ConfigDict, Field

AUDIO8_TTS_MLX_BF16_MODEL_ID: Final = "mlx-community/audio8-tts-preview-0.6b-bf16"
AUDIO8_TTS_MLX_BF16_REPO_ID: Final = "mlx-community/Audio8-TTS-Preview-0.6b-bf16"
AUDIO8_TTS_MLX_BF16_REVISION: Final = "f7be312aaaed724b6ecb8e916b21c9fd0842db02"
AUDIO8_TTS_MLX_BF16_VARIANT: Final = "bf16"
AUDIO8_TTS_MLX_BF16_SAMPLE_RATE: Final = 44_100
AUDIO8_TTS_MLX_BF16_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "codec.safetensors",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Audio8TtsMlxBf16InstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["bf16"] = "bf16"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    voice_home: Path = Field(
        default_factory=lambda: user_data_path("hugging-mac") / "voices" / "audio8-tts-mlx-bf16"
    )
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
