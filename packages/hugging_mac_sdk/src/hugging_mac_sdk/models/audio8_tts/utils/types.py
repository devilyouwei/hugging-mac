"""Internal Audio8-TTS runtime values."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class TtsEngineOutput:
    audio: bytes
    sample_rate: int
    duration_seconds: float
    generated_tokens: int


class CoreAILayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact: str
    target_artifact: str
    fast_graph: str
    codec_graph: str
    pytorch_directory: str
    codec_max_frames: int = Field(gt=0)
    stages: dict[str, str]
