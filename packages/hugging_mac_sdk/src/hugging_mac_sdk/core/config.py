"""Load declarative model package metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from hugging_mac_sdk.errors import ManifestError
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.manifest import ModelManifest


class ModelPackageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: ModelManifest
    artifacts: tuple[ModelArtifact, ...] = ()


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
