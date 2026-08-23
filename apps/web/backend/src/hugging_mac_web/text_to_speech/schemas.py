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


class TtsRuntimeResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    available: bool
    size_bytes: int
    artifact_ids: tuple[str, ...]


class TtsResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    artifacts: tuple[TtsArtifactView, ...]
    runtimes: tuple[TtsRuntimeResourceView, ...]

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
            runtimes=tuple(
                TtsRuntimeResourceView(
                    runtime=item.runtime,
                    available=item.available,
                    size_bytes=item.size_bytes,
                    artifact_ids=item.artifact_ids,
                )
                for item in status.runtimes
            ),
        )


class ReadyTtsInstanceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    variant: str
    runtime: str


class TtsVariantView(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    available: bool
    available_runtimes: tuple[str, ...] = ()


class TtsModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    voices: tuple[str, ...]
    languages: tuple[str, ...]
    requires_reference_voice: bool
    requires_reference_audio: bool
    max_new_tokens: int
    variants: tuple[TtsVariantView, ...] = ()
    resource: TtsResourceView
    ready_instance_id: str | None = None
    ready_instances: tuple[ReadyTtsInstanceView, ...] = ()

    @classmethod
    def from_profile(
        cls,
        profile: TtsModelProfile,
        resource: TtsResourceView,
        *,
        ready_instance_id: str | None,
        ready_instances: tuple[ReadyTtsInstanceView, ...] = (),
        variants: tuple[TtsVariantView, ...] = (),
    ) -> TtsModelView:
        return cls(
            **asdict(profile),
            resource=resource,
            variants=variants,
            ready_instance_id=ready_instance_id,
            ready_instances=ready_instances,
        )


class LoadTtsModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    variant: str | None = None
    runtime: str | None = None


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
