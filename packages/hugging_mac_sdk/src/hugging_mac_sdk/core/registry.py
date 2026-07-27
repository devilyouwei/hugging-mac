"""Thread-safe model definition registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.errors import (
    RegistrationConflictError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
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

    def create(
        self,
        options: dict[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> BaseModelInstance:
        """Create an instance, optionally selecting its runtime explicitly."""

        if self.factory is None:
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
        selected_runtime = runtime or (str(option_runtime) if option_runtime is not None else None)
        if selected_runtime is not None and not any(
            spec.name == selected_runtime for spec in self.manifest.runtimes
        ):
            raise UnsupportedRuntimeError(
                f"{self.manifest.model_id}@{self.manifest.revision} "
                f"does not support runtime {selected_runtime}",
                details={"runtime": selected_runtime},
            )
        if runtime is not None:
            merged_options["runtime"] = runtime
        return self.factory(merged_options)


class ModelRegistry:
    """Register and discover model definitions by stable ID and revision."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], ModelDefinition] = {}
        self._lock = RLock()

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
