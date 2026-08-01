"""Configuration for the pinned Audio8-TTS ONNX INT4 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path, user_data_path
from pydantic import BaseModel, ConfigDict, Field

AUDIO8_TTS_ONNX_INT4_MODEL_ID: Final = "audio8/audio8-tts-preview-0.6b-onnx-int4"
AUDIO8_TTS_ONNX_INT4_REPO_ID: Final = "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4"
AUDIO8_TTS_ONNX_INT4_REVISION: Final = "7af3196fed72de44708f5d095d42cbf8085f6123"
AUDIO8_TTS_ONNX_INT4_VARIANT: Final = "int4"
AUDIO8_TTS_ONNX_INT4_SAMPLE_RATE: Final = 44100
AUDIO8_TTS_ONNX_INT4_FINGERPRINT: Final = (
    "62dcff0adf6c2535b3260467a7c1d482b556da57266c96a444518b76e140d2c3"
)
AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "codec_decoder_fp16.onnx",
    "codec_decoder_fp16.onnx.data",
    "fast_ar_int4.onnx",
    "fast_ar_int4.onnx.data",
    "runtime_manifest.json",
    "slow_ar_int4.onnx",
    "slow_ar_int4.onnx.data",
    "tokenizer/tokenizer.json",
)
AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES: Final[tuple[str, ...]] = (
    "registration/codec_encoder_fp16.onnx",
    "registration/codec_encoder_fp16.onnx.data",
    "registration/registration_manifest.json",
)


class Audio8TtsOnnxInt4InstanceConfig(BaseModel):
    """Runtime and resource options for one ONNX INT4 instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["onnx"] = "onnx"
    variant: Literal["int4"] = "int4"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    voice_home: Path = Field(
        default_factory=lambda: user_data_path("hugging-mac") / "voices" / "audio8-tts-onnx-int4"
    )
    source_path: Path | None = None
    hf_token: str | None = Field(default=None, repr=False, exclude=True)
    device: Literal["cpu"] = "cpu"
    threads: int = Field(default=5, ge=1)
