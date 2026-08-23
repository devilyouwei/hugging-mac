"""Declarative model artifact metadata."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import ResourceSource


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


class ModelArtifact(BaseModel):
    """A model file or directory required by one runtime implementation.

    ``path`` is normally relative to the registry's model storage root. Keeping
    this declaration separate from downloaded bytes makes definitions portable
    and safe to keep in Git.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: Annotated[str, Field(min_length=1)]
    runtime: Annotated[str, Field(min_length=1)] | None = None
    variant: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")] | None = None
    format: ArtifactFormat
    path: Path
    kind: ArtifactKind = ArtifactKind.FILE
    source: ResourceSource | None = None
    convert: bool = False
    shared: bool = False
    required_files: tuple[Path, ...] = ()
    required_shares: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    @model_validator(mode="after")
    def validate_path(self) -> ModelArtifact:
        if self.shared and (self.variant is not None or self.runtime is not None):
            raise ValueError("Shared artifacts must not declare variant or runtime ownership")
        if self.shared and self.required_shares:
            raise ValueError("Shared artifacts must not depend on other shared artifacts")
        if not self.shared and (self.variant is None or self.runtime is None):
            raise ValueError("Scoped artifacts must declare variant and runtime")
        if len(self.required_shares) != len(set(self.required_shares)):
            raise ValueError("required_shares must be unique")
        if not self.path.parts or self.path == Path("."):
            raise ValueError("Artifact path must not be empty")
        if not self.path.is_absolute() and ".." in self.path.parts:
            raise ValueError("Relative artifact paths must not contain '..'")
        for required_path in self.required_files:
            if required_path.is_absolute() or not required_path.parts or required_path == Path("."):
                raise ValueError("required_files must contain non-empty relative paths")
            if ".." in required_path.parts:
                raise ValueError("required_files must not contain '..'")
        if len(self.required_files) != len(set(self.required_files)):
            raise ValueError("required_files must be unique")
        if self.kind is ArtifactKind.FILE and self.required_files:
            raise ValueError("File artifacts must not declare required_files")
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

    def canonical_path(self, model_id: str) -> Path:
        """Return the compact, revision-independent path for this artifact."""

        model_root = Path(*model_id.split("/"))
        if self.shared:
            artifact_root = model_root / "_shared" / self.artifact_id
        else:
            assert self.variant is not None
            assert self.runtime is not None
            artifact_root = model_root / self.variant / self.runtime / self.artifact_id
        return artifact_root / self.path.name if self.kind is ArtifactKind.FILE else artifact_root

    def storage_path(self, storage_root: Path) -> Path:
        """Return the directory that owns the artifact's downloaded bytes."""

        resolved = self.resolve(storage_root)
        return resolved.parent if self.kind is ArtifactKind.FILE else resolved
