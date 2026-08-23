from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_sdk.core.config import ModelPackageConfig
from hugging_mac_sdk.core.registry import ModelDefinition
from hugging_mac_sdk.errors import ManifestError
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.manifest import ModelManifest
from pydantic import ValidationError


def test_required_files_must_be_safe_relative_paths() -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        ModelArtifact.model_validate(
            {
                "artifact_id": "model",
                "runtime": "mlx",
                "variant": "default",
                "format": "mlx",
                "path": "model/mlx",
                "kind": "directory",
                "required_files": ["../weights.bin"],
            }
        )


def test_variant_rejects_resource_sources() -> None:
    with pytest.raises(ValidationError, match="resources"):
        ModelManifest.model_validate(
            {
                "model_id": "test/model",
                "display_name": "Test",
                "family": "test",
                "capabilities": ["chat"],
                "runtimes": [{"name": "mlx"}],
                "variants": [
                    {
                        "name": "default",
                        "display_name": "Default",
                        "resources": [{"kind": "huggingface", "repo_id": "test/model"}],
                    }
                ],
            }
        )


def test_artifact_lookup_is_semantic_and_rejects_ambiguity() -> None:
    manifest = ModelManifest.model_validate(
        {
            "schema_version": "1",
            "model_id": "test/model",
            "display_name": "Test",
            "family": "test",
            "capabilities": ["chat"],
            "runtimes": [{"name": "mlx"}],
            "variants": [
                {"name": "small", "display_name": "Small"},
                {"name": "large", "display_name": "Large"},
            ],
            "default_variant": "small",
            "default_runtime": "mlx",
        }
    )
    artifacts = tuple(
        ModelArtifact(
            artifact_id="weights",
            runtime="mlx",
            variant=variant,
            format="mlx",
            path=Path(f"test/model/{variant}/mlx/weights"),
            kind="directory",
        )
        for variant in ("small", "large")
    )
    package = ModelPackageConfig(manifest=manifest, artifacts=artifacts)

    assert package.get_artifact("weights", variant="large").path == Path(
        "test/model/large/mlx/weights"
    )
    with pytest.raises(Exception, match="found 2"):
        package.get_artifact("weights")


def test_shared_artifacts_form_many_to_many_artifact_dependencies() -> None:
    manifest = ModelManifest.model_validate(
        {
            "model_id": "test/shared-model",
            "display_name": "Shared model",
            "family": "test",
            "capabilities": ["chat"],
            "runtimes": [{"name": "mlx"}],
            "variants": [
                {
                    "name": "small",
                    "display_name": "Small",
                },
                {"name": "large", "display_name": "Large"},
            ],
            "default_variant": "small",
        }
    )
    shared = ModelArtifact(
        artifact_id="tokenizer",
        format="tokenizer",
        path=Path("test/shared-model/_shared/tokenizer"),
        kind="directory",
        shared=True,
        source={"kind": "huggingface", "repo_id": "test/model", "allow_patterns": ["*"]},
    )
    processor = ModelArtifact(
        artifact_id="processor",
        format="tokenizer",
        path=Path("test/shared-model/_shared/processor"),
        kind="directory",
        shared=True,
        source={"kind": "huggingface", "repo_id": "test/model", "allow_patterns": ["*"]},
    )
    small = ModelArtifact(
        artifact_id="weights",
        variant="small",
        runtime="mlx",
        format="mlx",
        path=Path("test/shared-model/small/mlx/weights"),
        kind="directory",
        required_shares=("tokenizer", "processor"),
    )
    large = ModelArtifact(
        artifact_id="weights",
        variant="large",
        runtime="mlx",
        format="mlx",
        path=Path("test/shared-model/large/mlx/weights"),
        kind="directory",
        required_shares=("tokenizer",),
    )
    artifacts = (small, large, shared, processor)
    definition = ModelDefinition(manifest=manifest, artifacts=artifacts)
    package = ModelPackageConfig(manifest=manifest, artifacts=artifacts)

    assert definition.required_shared_artifacts(variant="small") == (shared, processor)
    assert definition.required_shared_artifacts(variant="large") == (shared,)
    assert package.get_artifacts(variant="small", include_shared=True) == (
        small,
        shared,
        processor,
    )
    assert package.get_artifacts(variant="large", include_shared=True) == (large, shared)


def test_shared_artifact_rejects_variant_or_runtime_ownership() -> None:
    with pytest.raises(ValidationError, match="must not declare"):
        ModelArtifact(
            artifact_id="tokenizer",
            variant="small",
            runtime="mlx",
            format="tokenizer",
            path=Path("shared/tokenizer"),
            shared=True,
        )


def test_shared_artifact_must_be_referenced_by_an_artifact() -> None:
    manifest = ModelManifest.model_validate(
        {
            "model_id": "test/unowned-shared",
            "display_name": "Unowned shared",
            "family": "test",
            "capabilities": ["chat"],
            "runtimes": [{"name": "mlx"}],
        }
    )
    shared = ModelArtifact(
        artifact_id="tokenizer",
        format="tokenizer",
        path=Path("shared/tokenizer"),
        kind="directory",
        shared=True,
        source={"kind": "huggingface", "repo_id": "test/model", "allow_patterns": ["*"]},
    )

    with pytest.raises(ManifestError, match="referenced by at least one artifact"):
        ModelDefinition(manifest=manifest, artifacts=(shared,))
