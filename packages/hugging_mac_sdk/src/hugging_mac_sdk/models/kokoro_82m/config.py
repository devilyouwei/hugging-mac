"""Configuration for the pinned Kokoro-82M MLX BF16 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

KOKORO_82M_MODEL_ID: Final = "mlx-community/kokoro-82m-bf16"
KOKORO_82M_REPO_ID: Final = "mlx-community/Kokoro-82M-bf16"
KOKORO_82M_REVISION: Final = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
KOKORO_82M_VARIANT: Final = "bf16"
KOKORO_82M_SAMPLE_RATE: Final = 24000
KOKORO_82M_DEFAULT_VOICE: Final = "af_heart"
KOKORO_82M_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "kokoro-v1_0.safetensors",
    "voices/af_heart.safetensors",
)


class Kokoro82mInstanceConfig(BaseModel):
    """Runtime and resource options for one Kokoro-82M instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx"] = "mlx"
    variant: Literal["bf16"] = "bf16"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
    default_voice: str = KOKORO_82M_DEFAULT_VOICE
    default_language: str = "a"
