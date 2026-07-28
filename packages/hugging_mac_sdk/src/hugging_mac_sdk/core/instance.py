"""Composable model instance with a guarded lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from hugging_mac_sdk.errors import (
    HuggingMacSdkError,
    ModelLoadError,
    UnsupportedCapabilityError,
)
from hugging_mac_sdk.schemas.health import HealthReport, HealthStatus


class ModelState(StrEnum):
    CREATED = "created"
    RESOLVING = "resolving"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    FAILED = "failed"


class ModelInstanceInfo(BaseModel):
    """Runtime details for one local instance."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str | None = None
    revision: str | None = None
    runtime: str | None = None
    device: str | None = None
    artifact_path: Path | None = None
    state: ModelState
    capabilities: tuple[str, ...] = ()


CapabilityT = TypeVar("CapabilityT")


class BaseModelInstance(ABC):
    """Thin base class for lifecycle management and capability composition.

    Subclasses implement runtime-specific hooks and register task capabilities.
    They must not expose framework model objects through the public interface.
    """

    def __init__(
        self,
        *,
        instance_id: UUID | None = None,
        model_id: str | None = None,
        revision: str | None = None,
        runtime: str | None = None,
        device: str | None = None,
    ) -> None:
        self._instance_id = instance_id or uuid4()
        self._model_id = model_id
        self._revision = revision
        self._runtime = runtime
        self._device = device
        self._artifact_path: Path | None = None
        self._state = ModelState.CREATED
        self._state_lock = asyncio.Lock()
        self._capabilities: dict[type[Any], Any] = {}
        self._failure: BaseException | None = None

    @property
    def instance_id(self) -> UUID:
        return self._instance_id

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def capabilities(self) -> MappingProxyType[type[Any], Any]:
        return MappingProxyType(self._capabilities)

    def register_capability(
        self,
        capability: type[CapabilityT],
        implementation: CapabilityT,
    ) -> None:
        """Attach a capability implementation to this instance.

        Registration is only allowed before loading, so the manifest-visible
        capability set cannot change while the instance is serving requests.
        """

        if self._state is not ModelState.CREATED:
            raise RuntimeError("Capabilities can only be registered in CREATED state")
        if capability in self._capabilities:
            raise ValueError(f"Capability already registered: {capability.__name__}")
        self._capabilities[capability] = implementation

    def supports(self, capability: type[Any]) -> bool:
        return capability in self._capabilities

    def require(self, capability: type[CapabilityT]) -> CapabilityT:
        try:
            return cast(CapabilityT, self._capabilities[capability])
        except KeyError as error:
            raise UnsupportedCapabilityError(
                f"Instance {self.instance_id} does not support {capability.__name__}",
                details={"capability": capability.__name__},
            ) from error

    def info(self) -> ModelInstanceInfo:
        """Return stable metadata without exposing a framework session."""

        return ModelInstanceInfo(
            instance_id=str(self._instance_id),
            model_id=self._model_id,
            revision=self._revision,
            runtime=self._runtime,
            device=self._device,
            artifact_path=self._artifact_path,
            state=self._state,
            capabilities=tuple(
                sorted(capability.__name__ for capability in self._capabilities)
            ),
        )

    def _set_runtime_context(
        self,
        *,
        device: str | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        """Allow a runtime implementation to report resolved execution details."""

        if device is not None:
            self._device = device
        if artifact_path is not None:
            self._artifact_path = artifact_path

    async def load(self) -> None:
        """Load resources exactly once; concurrent lifecycle calls are serialized."""

        async with self._state_lock:
            if self._state is ModelState.READY:
                return
            if self._state in {ModelState.UNLOADING, ModelState.UNLOADED}:
                raise ModelLoadError(f"Cannot load instance in {self._state} state")

            try:
                self._state = ModelState.RESOLVING
                await self._resolve()
                self._state = ModelState.LOADING
                await self._load()
                self._state = ModelState.READY
                self._failure = None
            except asyncio.CancelledError:
                self._state = ModelState.FAILED
                await self._cleanup_after_failed_load()
                raise
            except HuggingMacSdkError as error:
                self._state = ModelState.FAILED
                self._failure = error
                await self._cleanup_after_failed_load()
                raise
            except Exception as error:
                self._state = ModelState.FAILED
                self._failure = error
                await self._cleanup_after_failed_load()
                raise ModelLoadError(
                    f"Failed to load instance {self.instance_id}",
                    cause=error,
                ) from error

    async def warmup(self) -> None:
        async with self._state_lock:
            if self._state is not ModelState.READY:
                raise ModelLoadError("Instance must be READY before warmup")
            await self._warmup()

    async def unload(self) -> None:
        """Release runtime resources. The operation is idempotent."""

        async with self._state_lock:
            if self._state is ModelState.UNLOADED:
                return
            self._state = ModelState.UNLOADING
            try:
                await self._unload()
            finally:
                self._state = ModelState.UNLOADED

    async def health(self) -> HealthReport:
        if self._state is ModelState.READY:
            return HealthReport(status=HealthStatus.HEALTHY, state=self._state)
        if self._state is ModelState.FAILED:
            return HealthReport(
                status=HealthStatus.UNHEALTHY,
                state=self._state,
                message=str(self._failure) if self._failure else None,
            )
        return HealthReport(status=HealthStatus.DEGRADED, state=self._state)

    async def _resolve(self) -> None:
        """Resolve model resources before runtime loading."""

        return None

    async def _warmup(self) -> None:
        """Optionally execute a representative warmup request."""

        return None

    async def _cleanup_after_failed_load(self) -> None:
        # The original load error is more useful to callers. Runtime adapters
        # should log cleanup failures with the instance ID.
        with contextlib.suppress(Exception):
            await self._unload()

    @abstractmethod
    async def _load(self) -> None:
        """Load framework/runtime resources."""

    @abstractmethod
    async def _unload(self) -> None:
        """Release framework/runtime resources."""
