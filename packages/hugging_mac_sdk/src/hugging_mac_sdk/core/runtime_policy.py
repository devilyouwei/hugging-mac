"""Machine-aware runtime preference without model loading side effects."""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.schemas.manifest import ModelManifest, RuntimeSpec


@dataclass(frozen=True, slots=True)
class RuntimeAvailability:
    available: bool
    reason: str | None = None


class RuntimePolicy:
    """Select a declared runtime using ordered machine preferences."""

    def __init__(self, preferences: tuple[str, ...] = ()) -> None:
        self._preferences = preferences

    def select(self, manifest: ModelManifest) -> str:
        declared = {runtime.name: runtime for runtime in manifest.runtimes}
        ordered = (
            self._preferences
            + ((manifest.default_runtime,) if manifest.default_runtime else ())
            + tuple(runtime.name for runtime in manifest.runtimes)
        )
        visited: set[str] = set()
        for name in ordered:
            if name in visited or name not in declared:
                continue
            visited.add(name)
            if self.availability(declared[name]).available:
                return name
        raise UnsupportedRuntimeError(
            f"No runtime is available for {manifest.model_id}",
            details={"declared": sorted(declared)},
        )

    def availability(self, runtime: RuntimeSpec) -> RuntimeAvailability:
        name = runtime.name
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if name == "coreml":
            if not is_apple_silicon:
                return RuntimeAvailability(False, "Core ML requires Apple Silicon macOS")
            return _module_availability("coremltools")
        if name == "pytorch-mps":
            if not is_apple_silicon:
                return RuntimeAvailability(False, "MPS requires Apple Silicon macOS")
            return _module_availability("torch")
        if name == "mlx":
            if not is_apple_silicon:
                return RuntimeAvailability(False, "MLX requires Apple Silicon macOS")
            return _module_availability("mlx")
        return RuntimeAvailability(True)


def _module_availability(module: str) -> RuntimeAvailability:
    if importlib.util.find_spec(module) is None:
        return RuntimeAvailability(False, f"Optional dependency is not installed: {module}")
    return RuntimeAvailability(True)
