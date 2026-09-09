"""Runtime-neutral schemas for document image parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentImage(BaseModel):
    """One document page supplied as a local image path."""

    model_config = ConfigDict(frozen=True)

    path: Path


class DocumentParsingRequest(BaseModel):
    """Parse one image or an ordered sequence of document pages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    images: tuple[DocumentImage, ...] = Field(min_length=1)
    prompt: str = Field(default="document parsing.", min_length=1)
    max_tokens: int = Field(default=8192, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0)


class DocumentParsingTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    inference_ms: float | None = Field(default=None, ge=0.0)


class DocumentParsingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    text: str
    page_count: int = Field(ge=1)
    timings: DocumentParsingTimings = DocumentParsingTimings()


class DocumentParsingStreamEvent(BaseModel):
    """One incremental text event from document generation."""

    model_config = ConfigDict(frozen=True)

    delta: str = ""
    finish_reason: Literal["stop", "length", "error"] | None = None
    generated_tokens: int | None = Field(default=None, ge=0)
