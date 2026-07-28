"""Declarative model artifact metadata."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import ResourceSource, Sha256


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


class ModelArtifact(BaseModel):
    """A model file or directory required by one runtime implementation.

    ``path`` is normally relative to the registry's model storage root. Keeping
    this declaration separate from downloaded bytes makes definitions portable
    and safe to keep in Git.
    """

    model_config = ConfigDict(frozen=True)

    artifact_id: Annotated[str, Field(min_length=1)]
    runtime: Annotated[str, Field(min_length=1)]
    format: ArtifactFormat
    path: Path
    kind: ArtifactKind = ArtifactKind.FILE
    source: ResourceSource | None = None
    expected_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_path(self) -> ModelArtifact:
        if not self.path.parts or self.path == Path("."):
            raise ValueError("Artifact path must not be empty")
        if not self.path.is_absolute() and ".." in self.path.parts:
            raise ValueError("Relative artifact paths must not contain '..'")
        return self

    def resolve(self, storage_root: Path) -> Path:
        """Resolve the artifact path without touching the filesystem."""

        if self.path.is_absolute():
            return self.path.expanduser().resolve(strict=False)
        root = storage_root.expanduser().resolve(strict=False)
        resolved = (root / self.path).resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError(f"Artifact path escapes model storage: {self.path}")
        return resolved
