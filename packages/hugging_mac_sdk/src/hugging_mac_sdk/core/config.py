"""Load declarative model package metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from hugging_mac_sdk.errors import ManifestError, ResourceNotFoundError
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.manifest import ModelManifest


class ModelPackageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ModelManifest
    artifacts: tuple[ModelArtifact, ...] = ()
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_artifacts(self) -> ModelPackageConfig:
        """Keep every package on the same compact local-storage convention."""

        artifact_keys: set[tuple[str | None, str | None, str]] = set()
        shared_ids = {artifact.artifact_id for artifact in self.artifacts if artifact.shared}
        for artifact in self.artifacts:
            key = (artifact.variant, artifact.runtime, artifact.artifact_id)
            if key in artifact_keys:
                raise ValueError(f"Duplicate artifact declaration: {key}")
            artifact_keys.add(key)
            expected_path = artifact.canonical_path(self.manifest.model_id)
            if artifact.path != expected_path:
                raise ValueError(
                    f"Artifact {artifact.artifact_id!r} must use canonical path "
                    f"{expected_path}, got {artifact.path}"
                )
            missing_shares = set(artifact.required_shares) - shared_ids
            if missing_shares:
                raise ValueError(
                    f"Artifact {artifact.artifact_id!r} requires undeclared shares: "
                    f"{sorted(missing_shares)}"
                )
        return self

    def get_artifacts(
        self,
        *,
        variant: str | None = None,
        runtime: str | None = None,
        include_shared: bool = False,
    ) -> tuple[ModelArtifact, ...]:
        """Return declared artifacts without relying on YAML ordering."""

        selected_variant = self.manifest.get_variant(variant).name if variant else None
        scoped = tuple(
            artifact
            for artifact in self.artifacts
            if not artifact.shared
            and (selected_variant is None or artifact.variant == selected_variant)
            and (runtime is None or artifact.runtime == runtime)
        )
        required_shares = {share_id for artifact in scoped for share_id in artifact.required_shares}
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact in scoped
            or (include_shared and artifact.shared and artifact.artifact_id in required_shares)
        )

    def get_artifact(
        self,
        artifact_id: str | None = None,
        *,
        variant: str | None = None,
        runtime: str | None = None,
        shared: bool | None = None,
    ) -> ModelArtifact:
        """Return one artifact by semantic key and reject ambiguous declarations."""

        selected_variant = self.manifest.get_variant(variant).name if variant else None
        eligible_shared_ids = {
            share_id
            for artifact in self.artifacts
            if not artifact.shared
            and (selected_variant is None or artifact.variant == selected_variant)
            and (runtime is None or artifact.runtime == runtime)
            for share_id in artifact.required_shares
        }
        matches = tuple(
            artifact
            for artifact in self.artifacts
            if (artifact_id is None or artifact.artifact_id == artifact_id)
            and (shared is None or artifact.shared is shared)
            and (
                (
                    artifact.shared
                    and (
                        (variant is None and runtime is None)
                        or artifact.artifact_id in eligible_shared_ids
                    )
                )
                or (not artifact.shared and (variant is None or artifact.variant == variant))
            )
            and (runtime is None or artifact.shared or artifact.runtime == runtime)
        )
        if len(matches) != 1:
            raise ResourceNotFoundError(
                f"Expected one matching artifact, found {len(matches)}",
                details={
                    "model_id": self.manifest.model_id,
                    "artifact_id": artifact_id,
                    "variant": variant,
                    "runtime": runtime,
                },
            )
        return matches[0]


def load_model_config(path: Path) -> ModelPackageConfig:
    """Parse a YAML model declaration without importing runtime libraries."""

    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ModelPackageConfig.model_validate(data)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ManifestError(
            f"Could not load model config: {path}",
            details={"path": str(path)},
            cause=error,
        ) from error
