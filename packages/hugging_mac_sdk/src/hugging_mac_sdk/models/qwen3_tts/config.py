"""Configuration for Qwen3-TTS 0.6B Base runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_TTS_SAMPLE_RATE: Final = 24_000
QWEN3_TTS_COREML_GRAPHS: Final[tuple[str, ...]] = (
    "TextProjector",
    "CodeEmbedder",
    "MultiCodeEmbedder",
    "CodeDecoder",
    "MultiCodeDecoder",
    "SpeechDecoder",
)


class Qwen3TtsInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx", "coreml"] = "mlx"
    variant: str = "0.6b-base"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    coreml_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu", "all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "gpu"
