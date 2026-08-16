"""Configuration for the pinned Silero VAD integration."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

SILERO_MODEL_ID: Final = "snakers4/silero-vad"
SILERO_REVISION: Final = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_COREML_REVISION: Final = "523876545a57961474fee9df913e833e130560b8"
SILERO_VARIANT: Final = "v6.2.1"
SILERO_FILENAME: Final = "silero_vad.onnx"
SILERO_SHA256: Final = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class SileroInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    runtime: Literal["onnx", "coreml"] = "coreml"
    variant: Literal["v6.2.1"] = "v6.2.1"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    device: Literal[
        "auto", "coreml", "cpu", "all", "cpu-only", "cpu-and-gpu",
        "cpu-and-neural-engine"
    ] | None = "auto"
    sample_rate: Literal[16000] = 16000
    chunk_samples: Literal[512] = 512
