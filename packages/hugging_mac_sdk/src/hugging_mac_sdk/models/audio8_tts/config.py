"""Instance configuration for the Audio8-TTS Preview integration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path, user_data_path
from pydantic import BaseModel, ConfigDict, Field

Audio8TtsDType = Literal["auto", "float32"]


class Audio8TtsInstanceConfig(BaseModel):
    """Runtime and resource options for one Audio8-TTS instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch"] = "pytorch"
    variant: str = "0.6b-preview"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["cpu"] = "cpu"
    dtype: Audio8TtsDType = "auto"


class Audio8TtsMlxInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    runtime: Literal["mlx"] = "mlx"
    variant: str = "0.6b-preview"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    voice_home: Path = Field(
        default_factory=lambda: user_data_path("hugging-mac") / "voices" / "audio8-tts"
    )
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu"] = "gpu"
