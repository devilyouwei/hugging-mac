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
    slow_graph: str
    encoder_graph: str
    config_file: str
    codec_max_frames: int = Field(gt=0)
    stages: dict[str, str]


class CoreAINativeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: int = Field(gt=0)
    frame_length: int = Field(gt=0)
    num_codebooks: int = Field(gt=0)
    codebook_size: int = Field(gt=0)
    semantic_begin_id: int = Field(ge=0)
    max_seq_len: int = Field(gt=0)
    ras_window_size: int = Field(gt=0)
    ras_top_p: float = Field(gt=0, le=1)
    ras_temperature: float = Field(gt=0)
    slow_states: dict[str, tuple[int, ...]]
    fast_states: dict[str, tuple[int, ...]]
