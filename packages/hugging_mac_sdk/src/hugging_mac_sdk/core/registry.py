"""Thread-safe model definition registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING

from platformdirs import user_cache_path

from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.errors import (
    ManifestError,
    RegistrationConflictError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.manifest import ModelManifest

if TYPE_CHECKING:
    from hugging_mac_sdk.core.resources import ModelResourceProvider

InstanceFactory = Callable[[dict[str, object]], BaseModelInstance]


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Static model metadata paired with an instance factory."""

    manifest: ModelManifest
    factory: InstanceFactory | None = None
    converter_ids: tuple[str, ...] = ()
    resource_provider: ModelResourceProvider | None = None
    runtime_factories: Mapping[str, InstanceFactory] = field(default_factory=dict)
    artifacts: tuple[ModelArtifact, ...] = ()

    def __post_init__(self) -> None:
        factories = MappingProxyType(dict(self.runtime_factories))
        object.__setattr__(self, "runtime_factories", factories)
        declared = {runtime.name for runtime in self.manifest.runtimes}
        implemented = set(factories)
        unknown = implemented - declared
        if unknown:
            raise ManifestError(
                f"Runtime factories are not declared by {self.manifest.model_id}",
                details={"runtimes": sorted(unknown)},
            )
        if factories and implemented != declared:
            raise ManifestError(
                f"Every declared runtime needs an implementation for {self.manifest.model_id}",
                details={
                    "missing": sorted(declared - implemented),
                    "implemented": sorted(implemented),
                },
            )
        invalid_artifacts = sorted(
            {artifact.runtime for artifact in self.artifacts} - declared
        )
        if invalid_artifacts:
            raise ManifestError(
                f"Artifacts reference undeclared runtimes for {self.manifest.model_id}",
                details={"runtimes": invalid_artifacts},
            )
        artifact_keys = [
            (artifact.runtime, artifact.artifact_id) for artifact in self.artifacts
        ]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ManifestError(
                f"Artifact IDs must be unique within a runtime for {self.manifest.model_id}"
            )

    @property
    def supported_runtimes(self) -> tuple[str, ...]:
        if self.runtime_factories:
            return tuple(
                runtime.name
                for runtime in self.manifest.runtimes
                if runtime.name in self.runtime_factories
            )
        if self.factory is not None:
            # Backward-compatible factory definitions dispatch internally.
            return tuple(runtime.name for runtime in self.manifest.runtimes)
        return ()

    def create(
        self,
        options: dict[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> BaseModelInstance:
        """Create an instance, optionally selecting its runtime explicitly."""

        supported = self.supported_runtimes
        if not supported:
            raise UnsupportedRuntimeError(
                f"No runtime factory is registered for {self.manifest.model_id}"
            )
        merged_options = dict(options or {})
        option_runtime = merged_options.get("runtime")
        if runtime is not None and option_runtime not in {None, runtime}:
            raise UnsupportedRuntimeError(
                "Conflicting runtime values were provided",
                details={
                    "runtime": runtime,
                    "options_runtime": option_runtime,
                },
            )
        selected_runtime = (
            runtime
            or (str(option_runtime) if option_runtime is not None else None)
            or self.manifest.default_runtime
            or supported[0]
        )
        if selected_runtime == "auto":
            raise UnsupportedRuntimeError(
                "Runtime 'auto' must be resolved by ModelSdk or InstanceManager",
                details={"model_id": self.manifest.model_id},
            )
        if selected_runtime not in supported:
            raise UnsupportedRuntimeError(
                f"{self.manifest.model_id}@{self.manifest.revision} "
                f"does not support runtime {selected_runtime}",
                details={"runtime": selected_runtime, "supported_runtimes": supported},
            )
        merged_options["runtime"] = selected_runtime
        selected_factory = self.runtime_factories.get(selected_runtime, self.factory)
        assert selected_factory is not None
        return selected_factory(merged_options)

    def get_artifacts(self, runtime: str) -> tuple[ModelArtifact, ...]:
        artifacts = tuple(
            artifact for artifact in self.artifacts if artifact.runtime == runtime
        )
        if artifacts:
            return artifacts
        raise ResourceNotFoundError(
            f"No artifact is declared for {self.manifest.model_id} runtime {runtime}",
            details={"model_id": self.manifest.model_id, "runtime": runtime},
        )

    def get_artifact(
        self,
        runtime: str,
        artifact_id: str | None = None,
    ) -> ModelArtifact:
        artifacts = self.get_artifacts(runtime)
        if artifact_id is not None:
            for artifact in artifacts:
                if artifact.artifact_id == artifact_id:
                    return artifact
            raise ResourceNotFoundError(
                f"Artifact {artifact_id} is not declared for "
                f"{self.manifest.model_id} runtime {runtime}",
                details={
                    "model_id": self.manifest.model_id,
                    "runtime": runtime,
                    "artifact_id": artifact_id,
                },
            )
        if len(artifacts) != 1:
            raise ResourceNotFoundError(
                f"Runtime {runtime} declares multiple artifacts; specify artifact_id",
                details={
                    "model_id": self.manifest.model_id,
                    "runtime": runtime,
                    "artifact_ids": [artifact.artifact_id for artifact in artifacts],
                },
            )
        return artifacts[0]


class ModelRegistry:
    """Register and discover model definitions by stable ID and revision."""

    def __init__(self, *, storage_root: Path | None = None) -> None:
        self._definitions: dict[tuple[str, str], ModelDefinition] = {}
        self._storage_root = storage_root or user_cache_path("hugging-mac") / "models"
        self._lock = RLock()

    @property
    def storage_root(self) -> Path:
        return self._storage_root

    def register(self, definition: ModelDefinition, *, replace: bool = False) -> None:
        key = (definition.manifest.model_id, definition.manifest.revision)
        with self._lock:
            if key in self._definitions and not replace:
                raise RegistrationConflictError(
                    f"Model definition already registered: {key[0]}@{key[1]}"
                )
            self._definitions[key] = definition

    def unregister(self, model_id: str, revision: str) -> ModelDefinition:
        with self._lock:
            try:
                return self._definitions.pop((model_id, revision))
            except KeyError as error:
                raise ResourceNotFoundError(
                    f"Model definition not found: {model_id}@{revision}"
                ) from error

    def get(self, model_id: str, revision: str | None = None) -> ModelDefinition:
        with self._lock:
            if revision is not None:
                try:
                    return self._definitions[(model_id, revision)]
                except KeyError as error:
                    raise ResourceNotFoundError(
                        f"Model definition not found: {model_id}@{revision}"
                    ) from error

            candidates = [
                definition
                for (candidate_id, _), definition in self._definitions.items()
                if candidate_id == model_id
            ]
            if not candidates:
                raise ResourceNotFoundError(f"Model definition not found: {model_id}")
            if len(candidates) == 1:
                return candidates[0]
            main = [
                definition for definition in candidates if definition.manifest.revision == "main"
            ]
            if len(main) == 1:
                return main[0]
            raise ResourceNotFoundError(
                f"Multiple revisions are registered for {model_id}; specify revision",
                details={
                    "revisions": sorted(definition.manifest.revision for definition in candidates)
                },
            )

    def list(
        self,
        *,
        capability: str | None = None,
        runtime: str | None = None,
    ) -> tuple[ModelDefinition, ...]:
        with self._lock:
            definitions: Iterable[ModelDefinition] = tuple(self._definitions.values())

        if capability is not None:
            definitions = (item for item in definitions if capability in item.manifest.capabilities)
        if runtime is not None:
            definitions = (
                item
                for item in definitions
                if any(spec.name == runtime for spec in item.manifest.runtimes)
            )
        return tuple(
            sorted(
                definitions,
                key=lambda item: (item.manifest.model_id, item.manifest.revision),
            )
        )

    def list_models(
        self,
        *,
        capability: str | None = None,
        runtime: str | None = None,
    ) -> tuple[ModelDefinition, ...]:
        """Readable alias for :meth:`list`."""

        return self.list(capability=capability, runtime=runtime)

    def supported_runtimes(
        self,
        model_id: str,
        revision: str | None = None,
    ) -> tuple[str, ...]:
        return self.get(model_id, revision).supported_runtimes

    def get_artifact(
        self,
        model_id: str,
        runtime: str,
        *,
        revision: str | None = None,
        artifact_id: str | None = None,
    ) -> ModelArtifact:
        return self.get(model_id, revision).get_artifact(runtime, artifact_id)

    def get_artifacts(
        self,
        model_id: str,
        runtime: str,
        *,
        revision: str | None = None,
    ) -> tuple[ModelArtifact, ...]:
        return self.get(model_id, revision).get_artifacts(runtime)

    def resolve_artifact_path(
        self,
        model_id: str,
        runtime: str,
        *,
        revision: str | None = None,
        artifact_id: str | None = None,
        storage_root: Path | None = None,
    ) -> Path:
        artifact = self.get_artifact(
            model_id,
            runtime,
            revision=revision,
            artifact_id=artifact_id,
        )
        return artifact.resolve(storage_root or self._storage_root)

    def create_instance(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        runtime: str | None = None,
        options: dict[str, object] | None = None,
    ) -> BaseModelInstance:
        definition = self.get(model_id, revision)
        return definition.create(options, runtime=runtime)
