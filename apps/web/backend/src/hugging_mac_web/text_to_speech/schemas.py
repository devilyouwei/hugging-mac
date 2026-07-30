"""HTTP schemas for the text-to-speech application."""

from __future__ import annotations

from dataclasses import asdict

from hugging_mac_sdk.core.instance import ModelInstanceInfo
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_web.text_to_speech.config import TtsModelProfile


class TtsArtifactView(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    format: str
    runtime: str | None
    available: bool
    size_bytes: int | None


class TtsResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    artifacts: tuple[TtsArtifactView, ...]

    @classmethod
    def from_sdk(cls, status: ModelResourceStatus) -> TtsResourceView:
        return cls(
            model_id=status.model_id,
            revision=status.revision,
            variant=status.variant,
            artifacts=tuple(
                TtsArtifactView(
                    artifact_id=item.artifact_id,
                    format=item.format,
                    runtime=item.runtime,
                    available=item.available,
                    size_bytes=item.size_bytes,
                )
                for item in status.artifacts
            ),
        )


class TtsModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    supports_coreml_conversion: bool
    voices: tuple[str, ...]
    languages: tuple[str, ...]
    resource: TtsResourceView
    ready_instance_id: str | None = None

    @classmethod
    def from_profile(
        cls,
        profile: TtsModelProfile,
        resource: TtsResourceView,
        *,
        ready_instance_id: str | None,
    ) -> TtsModelView:
        return cls(
            **asdict(profile),
            resource=resource,
            ready_instance_id=ready_instance_id,
        )


class LoadTtsModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str


class LoadedTtsModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    variant: str
    runtime: str
    device: str
    state: str

    @classmethod
    def from_sdk(cls, info: ModelInstanceInfo) -> LoadedTtsModelView:
        return cls(
            instance_id=info.instance_id,
            model_id=info.model_id or "",
            variant=info.variant or "",
            runtime=info.runtime or "",
            device=info.device or "",
            state=info.state.value,
        )


class SynthesizeSpeechRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    instance_id: str
    text: str = Field(min_length=1, max_length=2000)
    voice: str | None = None
    language: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
