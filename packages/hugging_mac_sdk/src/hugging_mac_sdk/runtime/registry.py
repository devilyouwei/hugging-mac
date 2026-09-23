"""Registry for reusable runtime backends."""

from __future__ import annotations

from threading import RLock

from hugging_mac_sdk.errors import (
    RegistrationConflictError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.runtime.base import RuntimeBackend


class RuntimeRegistry:
    """Register low-level runtime backends such as ONNX, Core ML, or MLX.

    Model packages remain responsible for task-specific preprocessing and
    postprocessing. A backend only owns runtime/session creation.
    """

    def __init__(self) -> None:
        self._backends: dict[str, RuntimeBackend] = {}
        self._lock = RLock()

    def register(self, backend: RuntimeBackend, *, replace: bool = False) -> None:
        with self._lock:
            if backend.name in self._backends and not replace:
                raise RegistrationConflictError(
                    f"Runtime backend already registered: {backend.name}"
                )
            self._backends[backend.name] = backend

    def get(self, name: str) -> RuntimeBackend:
        with self._lock:
            try:
                return self._backends[name]
            except KeyError as error:
                raise UnsupportedRuntimeError(
                    f"Runtime backend is not registered: {name}",
                    details={"runtime": name},
                ) from error

    def list(self) -> tuple[RuntimeBackend, ...]:
        with self._lock:
            return tuple(self._backends[name] for name in sorted(self._backends))


def create_default_runtime_registry() -> RuntimeRegistry:
    """Build the standard provider registry without importing heavy frameworks."""

    from hugging_mac_sdk.runtime.coreai import CoreAIProvider
    from hugging_mac_sdk.runtime.coreml import CoreMLProvider
    from hugging_mac_sdk.runtime.mlx import MlxProvider
    from hugging_mac_sdk.runtime.onnx import OnnxRuntimeProvider
    from hugging_mac_sdk.runtime.torch import TorchProvider

    registry = RuntimeRegistry()
    registry.register(CoreAIProvider())
    registry.register(CoreMLProvider())
    registry.register(MlxProvider())
    registry.register(OnnxRuntimeProvider())
    registry.register(TorchProvider())
    return registry
