"""Runtime backend and model adapter boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import ResolvedResource


class RuntimeSession(Protocol):
    """A loaded low-level runtime session without task semantics."""

    @property
    def device(self) -> str: ...

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class RuntimeBackend(ABC):
    """Create backend sessions without model-task semantics."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def available_devices(self) -> tuple[str, ...]:
        """Return provider-specific device names in preference order."""

        return ()

    @abstractmethod
    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> RuntimeSession: ...


class RuntimeAdapter(ABC):
    """Create model instances without exposing backend objects to callers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def create_instance(
        self,
        manifest: ModelManifest,
        resources: tuple[ResolvedResource, ...],
        options: Mapping[str, Any],
    ) -> BaseModelInstance: ...
