"""Configuration for the pinned upstream Kokoro-82M integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

KOKORO_82M_MODEL_ID: Final = "hexgrad/kokoro"
KOKORO_82M_REPO_ID: Final = "hexgrad/Kokoro-82M"
KOKORO_82M_REVISION: Final = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_82M_COREML_REPO_ID: Final = "aufklarer/Kokoro-82M-CoreML"
KOKORO_82M_COREML_REVISION: Final = "f8ff771e4cab0bb3368e8af3a090a7e847485401"
KOKORO_82M_WEIGHT_SHA256: Final = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
KOKORO_82M_VARIANT: Final = "v1.0"
KOKORO_82M_SAMPLE_RATE: Final = 24000
KOKORO_82M_DEFAULT_VOICE: Final = "af_heart"
KOKORO_82M_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "kokoro-v1_0.pth",
    "voices/af_heart.pt",
)


class Kokoro82mInstanceConfig(BaseModel):
    """Runtime and resource options for one Kokoro-82M instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["coreml", "pytorch-mps"] = "coreml"
    variant: Literal["v1.0"] = "v1.0"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    artifact_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["mps", "cpu"] | None = None
    compute_units: Literal[
        "all", "cpu-and-neural-engine", "cpu-and-gpu", "cpu-only"
    ] = "all"
    allow_cpu_fallback: bool = True
    default_voice: str = KOKORO_82M_DEFAULT_VOICE
    default_language: str = "a"
