"""HTTP schemas for local multimodal chat."""

from __future__ import annotations

from typing import Literal

from hugging_mac_sdk.core.instance import ModelInstanceInfo
from pydantic import BaseModel, ConfigDict, Field


class ChatHistoryMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class LoadedChatModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    variant: str
    runtime: str
    device: str
    state: str

    @classmethod
    def from_sdk(cls, info: ModelInstanceInfo) -> LoadedChatModelView:
        return cls(
            instance_id=info.instance_id,
            model_id=info.model_id or "",
            variant=info.variant or "",
            runtime=info.runtime or "",
            device=info.device or "",
            state=info.state.value,
        )


class ChatReplyView(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model_id: str
    instance_id: str
    runtime: str
    device: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    inference_ms: float | None = None


class ChatStreamEventView(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: str = ""
    finish_reason: Literal["stop", "length", "error"] | None = None
    model_id: str
    instance_id: str
    runtime: str
    device: str
    generated_tokens: int | None = None
    inference_ms: float | None = None
    error: str | None = None
