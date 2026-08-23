"""Instance configuration for the upstream Kokoro-82M integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

KOKORO_82M_SAMPLE_RATE: Final = 24000
KOKORO_82M_DEFAULT_VOICE: Final = "af_heart"


class Kokoro82mInstanceConfig(BaseModel):
    """Runtime and resource options for one Kokoro-82M instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["coreml", "pytorch-mps"] = "coreml"
    variant: str = "v1.0"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["mps", "cpu"] | None = None
    compute_units: Literal["all", "cpu-and-neural-engine", "cpu-and-gpu", "cpu-only"] = "all"
    allow_cpu_fallback: bool = True
    default_voice: str = KOKORO_82M_DEFAULT_VOICE
    default_language: str = "a"
