from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

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
    available: bool
    size_bytes: int | None = Field(default=None, ge=0)


class ModelResourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    artifacts: tuple[ModelArtifactStatus, ...]
