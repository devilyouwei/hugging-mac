"""Configuration for the pinned SenseVoiceSmall integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

SENSEVOICE_SMALL_MODEL_ID: Final = "funaudiollm/sensevoice-small"
SENSEVOICE_SMALL_REPO_ID: Final = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_SMALL_REVISION: Final = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
SENSEVOICE_SMALL_WEIGHT_SHA256: Final = (
    "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
)
SENSEVOICE_SMALL_TOKENIZER_SHA256: Final = (
    "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
)
SENSEVOICE_SMALL_VARIANT: Final = "small"
SENSEVOICE_SMALL_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "model.pt",
    "config.yaml",
    "configuration.json",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)

SenseVoiceDType = Literal["auto", "float16", "float32"]


class SenseVoiceSmallInstanceConfig(BaseModel):
    """Runtime and resource options for one SenseVoiceSmall instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps"] = "pytorch-mps"
    variant: Literal["small"] = "small"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["mps", "cpu"] | None = None
    dtype: SenseVoiceDType = "auto"
    allow_cpu_fallback: bool = True
    sample_rate: Literal[16000] = 16000
    max_audio_seconds: float = Field(default=30.0, gt=0.0, le=30.0)
    fbank_dither: float = Field(default=0.0, ge=0.0)
