"""Configuration for Qwen3-TTS 0.6B Base MLX 4-bit."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

QWEN3_TTS_MODEL_ID: Final = "qwen/qwen3-tts-12hz"
QWEN3_TTS_REPO_ID: Final = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
QWEN3_TTS_REVISION: Final = "0d6bb6f"
QWEN3_TTS_COREML_REPO_ID: Final = "FluidInference/qwen3-tts-coreml"
QWEN3_TTS_COREML_REVISION: Final = "7bb6c4e5c425ddecc0aa2339f125398623d2da36"
QWEN3_TTS_VARIANT: Final = "0.6b-base"
QWEN3_TTS_SAMPLE_RATE: Final = 24_000
QWEN3_TTS_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
)
QWEN3_TTS_TOKENIZER_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)
QWEN3_TTS_COREML_GRAPHS: Final[tuple[str, ...]] = (
    "TextProjector",
    "CodeEmbedder",
    "MultiCodeEmbedder",
    "CodeDecoder",
    "MultiCodeDecoder",
    "SpeechDecoder",
)
QWEN3_TTS_COREML_REQUIRED_FILES: Final[tuple[str, ...]] = (
    *tuple(
        path
        for graph in QWEN3_TTS_COREML_GRAPHS
        for path in (
            f"{graph}.mlmodelc/model.mil",
            f"{graph}.mlmodelc/coremldata.bin",
            f"{graph}.mlmodelc/weights/weight.bin",
        )
    ),
    "speaker_embedding_official.npy",
)


class Qwen3TtsInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["mlx", "coreml"] = "mlx"
    variant: Literal["0.6b-base"] = "0.6b-base"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    coreml_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["gpu", "all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] = "gpu"
