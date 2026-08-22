"""HTTP schemas for multi-model live transcription."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

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


class RuntimeResourceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    available: bool
    size_bytes: int
    artifact_ids: tuple[str, ...]


class ResourceStatusView(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str
    artifacts: tuple[ArtifactResourceView, ...]
    runtimes: tuple[RuntimeResourceView, ...]

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
            runtimes=tuple(
                RuntimeResourceView(
                    runtime=runtime.runtime,
                    available=runtime.available,
                    size_bytes=runtime.size_bytes,
                    artifact_ids=runtime.artifact_ids,
                )
                for runtime in status.runtimes
            ),
        )


class ReadyAsrInstanceView(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    variant: str
    runtime: str


class AsrVariantView(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    available: bool
    available_runtimes: tuple[str, ...] = ()
    streaming_chunk_seconds: float | None = None


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
    streaming: bool
    streaming_chunk_seconds: float | None
    variants: tuple[AsrVariantView, ...] = ()
    resource: ResourceStatusView
    ready_instance_id: str | None = None
    ready_instances: tuple[ReadyAsrInstanceView, ...] = ()

    @classmethod
    def from_profile(
        cls,
        profile: AsrModelProfile,
        resource: ResourceStatusView,
        *,
        ready_instance_id: str | None,
        ready_instances: tuple[ReadyAsrInstanceView, ...] = (),
        variants: tuple[AsrVariantView, ...] = (),
    ) -> AsrModelView:
        return cls(
            **asdict(profile),
            resource=resource,
            variants=variants,
            ready_instance_id=ready_instance_id,
            ready_instances=ready_instances,
        )


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    variant: str | None = None
    runtime: str | None = None


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


class PipelineComponentView(BaseModel):
    """Availability and lifecycle state for an optional audio component."""

    model_config = ConfigDict(frozen=True)

    component_id: str
    model_id: str
    display_name: str
    description: str
    runtime: str
    downloaded: bool
    state: Literal["not-downloaded", "not-loaded", "loaded"]
    loaded_model: LoadedModelView | None = None


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
