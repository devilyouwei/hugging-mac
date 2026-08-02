"""HTTP schemas for local multimodal chat."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from hugging_mac_sdk.core.instance import ModelInstanceInfo
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_web.chat.config import ChatModelProfile


class ChatHistoryMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatArtifactView(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    available: bool
    size_bytes: int | None


class ChatResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    artifacts: tuple[ChatArtifactView, ...]
    total_size_bytes: int

    @classmethod
    def from_sdk(cls, status: ModelResourceStatus) -> ChatResourceView:
        return cls(
            model_id=status.model_id,
            revision=status.revision,
            variant=status.variant,
            artifacts=tuple(
                ChatArtifactView(
                    artifact_id=artifact.artifact_id,
                    available=artifact.available,
                    size_bytes=artifact.size_bytes,
                )
                for artifact in status.artifacts
            ),
            total_size_bytes=status.total_size_bytes,
        )


class ChatModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    disk_size_bytes: int
    supports_images: bool
    resource: ChatResourceView
    ready_instance_id: str | None = None

    @classmethod
    def from_profile(
        cls,
        profile: ChatModelProfile,
        resource: ChatResourceView,
        *,
        ready_instance_id: str | None,
    ) -> ChatModelView:
        return cls(**asdict(profile), resource=resource, ready_instance_id=ready_instance_id)


class LoadChatModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str


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
