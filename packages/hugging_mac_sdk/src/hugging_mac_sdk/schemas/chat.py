"""Runtime-neutral schemas for text and vision-language chat."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatImage(BaseModel):
    """An image referenced by a local path or an HTTP(S) URL."""

    model_config = ConfigDict(frozen=True)

    path: Path | None = None
    url: str | None = Field(default=None, pattern=r"^https?://")

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> ChatImage:
        if (self.path is None) == (self.url is None):
            raise ValueError("Exactly one of path or url must be provided")
        return self

    def source(self) -> str:
        return str(self.path) if self.path is not None else str(self.url)


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)
    images: tuple[ChatImage, ...] = ()


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    enable_thinking: bool = False


class ChatTimings(BaseModel):
    model_config = ConfigDict(frozen=True)

    inference_ms: float | None = Field(default=None, ge=0.0)


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    runtime: str
    device: str
    message: ChatMessage
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    timings: ChatTimings = ChatTimings()


class ChatStreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: str = ""
    finish_reason: Literal["stop", "length", "error"] | None = None
    generated_tokens: int | None = Field(default=None, ge=0)
