from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[a-fA-F0-9]{64}$")]


class HuggingFaceSource(BaseModel):
    """A complete repository snapshot or one file from Hugging Face Hub."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["huggingface"] = "huggingface"
    repo_id: Annotated[str, Field(min_length=1)]
    revision: str = "main"
    filename: str | None = None
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    expected_sha256: Sha256 | None = None


class UrlFileSource(BaseModel):
    """One model artifact available at an HTTP(S) URL."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["url_file"] = "url_file"
    url: HttpUrl
    expected_sha256: Sha256 | None = None


class UrlArchiveSource(BaseModel):
    """A ZIP or TAR archive that represents a model directory."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["url_archive"] = "url_archive"
    url: HttpUrl
    format: Literal["auto", "zip", "tar"] = "auto"
    strip_components: int = Field(default=0, ge=0)
    expected_sha256: Sha256 | None = None


ResourceSource = Annotated[
    HuggingFaceSource | UrlFileSource | UrlArchiveSource,
    Field(discriminator="kind"),
]


class ResolvedResource(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    source: ResourceSource
    digest: str | None = None
    size_bytes: int


class ModelArtifactStatus(BaseModel):
    """Path-free artifact status safe to expose across the SDK boundary."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    format: str
    runtime: str | None = None
    provisioning: Literal["download", "convert"] = "download"
    available: bool
    size_bytes: int | None = Field(default=None, ge=0)


class ConversionTargetStatus(BaseModel):
    """A model conversion exposed to SDK consumers and platform UIs."""

    model_config = ConfigDict(frozen=True)

    target_format: str
    runtime: str | None = None
    artifact_id: str
    available: bool
    size_bytes: int | None = Field(default=None, ge=0)


class RuntimeResourceStatus(BaseModel):
    """Aggregate local footprint for all artifacts required by one runtime."""

    model_config = ConfigDict(frozen=True)

    runtime: str
    available: bool
    size_bytes: int = Field(ge=0)
    artifact_ids: tuple[str, ...]


class ModelResourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variant: str = "default"
    artifacts: tuple[ModelArtifactStatus, ...]
    runtimes: tuple[RuntimeResourceStatus, ...] = ()
    conversion_targets: tuple[ConversionTargetStatus, ...] = ()
    total_size_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def aggregate_sizes(self) -> ModelResourceStatus:
        grouped: dict[str, list[ModelArtifactStatus]] = {}
        for artifact in self.artifacts:
            if artifact.runtime is not None:
                grouped.setdefault(artifact.runtime, []).append(artifact)
        runtimes = tuple(
            RuntimeResourceStatus(
                runtime=runtime,
                available=all(artifact.available for artifact in artifacts),
                size_bytes=sum(artifact.size_bytes or 0 for artifact in artifacts),
                artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
            )
            for runtime, artifacts in grouped.items()
        )
        conversion_targets = tuple(
            ConversionTargetStatus(
                target_format=artifact.format,
                runtime=artifact.runtime,
                artifact_id=artifact.artifact_id,
                available=artifact.available,
                size_bytes=artifact.size_bytes,
            )
            for artifact in self.artifacts
            if artifact.provisioning == "convert"
        )
        object.__setattr__(self, "runtimes", runtimes)
        object.__setattr__(self, "conversion_targets", conversion_targets)
        object.__setattr__(
            self,
            "total_size_bytes",
            sum(artifact.size_bytes or 0 for artifact in self.artifacts),
        )
        return self
