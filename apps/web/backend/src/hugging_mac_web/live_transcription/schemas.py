"""HTTP schemas for live transcription."""

from __future__ import annotations

from hugging_mac_sdk.schemas.resources import ModelResourceStatus
from hugging_mac_sdk.schemas.transcription import TranscriptionResponse
from pydantic import BaseModel, ConfigDict


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
