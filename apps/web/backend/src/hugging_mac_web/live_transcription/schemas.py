"""HTTP schemas for multi-model live transcription."""

from __future__ import annotations

from dataclasses import asdict

from hugging_mac_sdk.core.instance import ModelInstanceInfo
from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from hugging_mac_sdk.schemas.speech_understanding import SpeechUnderstandingResponse
from hugging_mac_sdk.schemas.transcription import TranscriptionResponse
from pydantic import BaseModel, ConfigDict

from hugging_mac_web.live_transcription.config import AsrModelProfile


class ArtifactResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    format: str
    runtime: str | None
    available: bool
    size_bytes: int | None


class ResourceStatusView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    artifacts: tuple[ArtifactResourceView, ...]

    @classmethod
    def from_sdk(cls, status: ModelResourceStatus) -> ResourceStatusView:
        return cls(
            model_id=status.model_id,
            revision=status.revision,
            variant=status.variant,
            artifacts=tuple(
                ArtifactResourceView(
                    artifact_id=artifact.artifact_id,
                    format=artifact.format,
                    runtime=artifact.runtime,
                    available=artifact.available,
                    size_bytes=artifact.size_bytes,
                )
                for artifact in status.artifacts
            ),
        )


class AsrModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    short_name: str
    description: str
    variant: str
    runtime: str
    required_artifact_id: str
    supports_coreml_conversion: bool
    rich_understanding: bool
    resource: ResourceStatusView
    ready_instance_id: str | None = None

    @classmethod
    def from_profile(
        cls,
        profile: AsrModelProfile,
        resource: ResourceStatusView,
        *,
        ready_instance_id: str | None,
    ) -> AsrModelView:
        return cls(
            **asdict(profile),
            resource=resource,
            ready_instance_id=ready_instance_id,
        )


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str


class LoadedModelView(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    variant: str
    runtime: str
    device: str
    state: str

    @classmethod
    def from_sdk(cls, info: ModelInstanceInfo) -> LoadedModelView:
        return cls(
            instance_id=info.instance_id,
            model_id=info.model_id or "",
            variant=info.variant or "",
            runtime=info.runtime or "",
            device=info.device or "",
            state=info.state.value,
        )


class TranscriptionResultView(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model_id: str
    instance_id: str
    runtime: str
    device: str
    sample_rate: int
    duration_seconds: float
    generated_tokens: int
    inference_ms: float | None
    languages: tuple[str, ...] = ()
    emotion: str | None = None
    events: tuple[str, ...] = ()

    @classmethod
    def from_sdk(cls, response: TranscriptionResponse) -> TranscriptionResultView:
        return cls(
            text=response.text,
            model_id=response.model_id,
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            sample_rate=response.sample_rate,
            duration_seconds=response.duration_seconds,
            generated_tokens=response.generated_tokens,
            inference_ms=response.timings.inference_ms,
        )

    @classmethod
    def from_understanding(
        cls,
        response: SpeechUnderstandingResponse,
    ) -> TranscriptionResultView:
        return cls(
            text=response.text,
            model_id=response.model_id,
            instance_id=response.instance_id,
            runtime=response.runtime,
            device=response.device,
            sample_rate=response.sample_rate,
            duration_seconds=response.duration_seconds,
            generated_tokens=response.token_count,
            inference_ms=response.timings.inference_ms,
            languages=response.languages,
            emotion=response.emotion,
            events=response.events,
        )
