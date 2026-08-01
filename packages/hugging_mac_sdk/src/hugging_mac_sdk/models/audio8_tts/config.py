"""Configuration for the pinned Audio8-TTS Preview integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

AUDIO8_TTS_MODEL_ID: Final = "audio8/audio8-tts-preview-0.6b"
AUDIO8_TTS_REPO_ID: Final = "Audio8/Audio8-TTS-Preview-0.6b"
AUDIO8_TTS_REVISION: Final = "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17"
AUDIO8_TTS_WEIGHT_SHA256: Final = (
    "62dcff0adf6c2535b3260467a7c1d482b556da57266c96a444518b76e140d2c3"
)
AUDIO8_TTS_CODEC_SHA256: Final = (
    "c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048"
)
AUDIO8_TTS_VARIANT: Final = "preview"
AUDIO8_TTS_SAMPLE_RATE: Final = 44100
AUDIO8_TTS_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "codec.pth",
    "config.json",
    "configuration_arktts.py",
    "generation_config.json",
    "model.safetensors",
    "modeling_arktts.py",
    "modeling_arktts_codec.py",
    "preprocessor_config.json",
    "processing_arktts.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

Audio8TtsDType = Literal["auto", "float16", "bfloat16", "float32"]


class Audio8TtsInstanceConfig(BaseModel):
    """Runtime and resource options for one Audio8-TTS instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["pytorch-mps"] = "pytorch-mps"
    variant: Literal["preview"] = "preview"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["mps", "cpu"] | None = None
    dtype: Audio8TtsDType = "auto"
    allow_cpu_fallback: bool = True
