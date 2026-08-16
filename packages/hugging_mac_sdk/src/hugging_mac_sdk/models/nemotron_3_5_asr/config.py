"""Configuration for Nemotron 3.5 ASR Core ML."""

from pathlib import Path
from typing import Final, Literal

from platformdirs import user_cache_path
from pydantic import BaseModel, ConfigDict, Field

NEMOTRON_3_5_ASR_MODEL_ID: Final = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_3_5_ASR_REVISION: Final = "1a41b75758b0337ff67db7d5408280aaaf23074e"
NemotronVariant = Literal[
    "latin-560ms",
    "latin-1120ms",
    "latin-2240ms",
    "latin-4480ms",
    "multilingual-560ms",
    "multilingual-1120ms",
    "multilingual-2240ms",
    "multilingual-4480ms",
]


def variant_source_path(variant: str) -> str:
    script, tier = variant.split("-", maxsplit=1)
    return f"{script}/{tier}"


class NemotronCoreMlInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    runtime: Literal["coreml"] = "coreml"
    variant: NemotronVariant = "multilingual-2240ms"
    model_home: Path = Field(default_factory=lambda: user_cache_path("hugging-mac") / "models")
    artifact_path: Path | None = None
    sample_rate: int = Field(default=16000, gt=0)
    max_audio_seconds: float = Field(default=300.0, gt=0)
    device: Literal["all", "cpu-only", "cpu-and-gpu", "cpu-and-neural-engine"] | None = (
        "cpu-and-neural-engine"
    )
