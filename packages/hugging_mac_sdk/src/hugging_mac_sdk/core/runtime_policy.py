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

    def candidates(self, manifest: ModelManifest) -> tuple[str, ...]:
        """Return compatible runtimes in deterministic preference order."""

        declared = {runtime.name: runtime for runtime in manifest.runtimes}
        ordered = (
            self._preferences
            + ((manifest.default_runtime,) if manifest.default_runtime else ())
            + tuple(runtime.name for runtime in manifest.runtimes)
        )
        visited: set[str] = set()
        candidates: list[str] = []
        for name in ordered:
            if name in visited or name not in declared:
                continue
            visited.add(name)
            if self.availability(declared[name]).available:
                candidates.append(name)
        return tuple(candidates)

    def select(self, manifest: ModelManifest) -> str:
        candidates = self.candidates(manifest)
        if candidates:
            return candidates[0]
        raise UnsupportedRuntimeError(
            f"No runtime is available for {manifest.model_id}",
            details={"declared": sorted(runtime.name for runtime in manifest.runtimes)},
        )

    def resolve(self, manifest: ModelManifest, requested: str | None) -> str:
        """Resolve ``None``/``auto`` or validate an explicit runtime name."""

        if requested in {None, "auto"}:
            return self.select(manifest)
        assert requested is not None
        if not any(runtime.name == requested for runtime in manifest.runtimes):
            raise UnsupportedRuntimeError(
                f"{manifest.model_id}@{manifest.revision} does not support runtime {requested}",
                details={"runtime": requested},
            )
        return requested

    def select_device(self, runtime: RuntimeSpec, requested: str | None = None) -> str | None:
        """Select a declared device while preserving backend-specific names."""

        if not runtime.devices:
            if requested not in {None, "auto"}:
                raise UnsupportedRuntimeError(
                    f"Runtime {runtime.name} does not expose device selection",
                    details={"runtime": runtime.name, "device": requested},
                )
            return None
        if requested in {None, "auto"}:
            return runtime.devices[0]
        if requested not in runtime.devices:
            raise UnsupportedRuntimeError(
                f"Runtime {runtime.name} does not support device {requested}",
                details={
                    "runtime": runtime.name,
                    "device": requested,
                    "supported_devices": runtime.devices,
                },
            )
        return requested

    def availability(self, runtime: RuntimeSpec) -> RuntimeAvailability:
        name = runtime.name
        current_platform = sys.platform
        current_architecture = platform.machine().lower()
        if runtime.platforms and current_platform not in runtime.platforms:
            return RuntimeAvailability(
                False,
                f"{name} requires platform: {', '.join(runtime.platforms)}",
            )
        if runtime.architectures and current_architecture not in runtime.architectures:
            return RuntimeAvailability(
                False,
                f"{name} requires architecture: {', '.join(runtime.architectures)}",
            )
        required_modules = runtime.required_modules or _known_runtime_modules(name)
        for module in required_modules:
            availability = _module_availability(module)
            if not availability.available:
                return availability
        return RuntimeAvailability(True)


def _module_availability(module: str) -> RuntimeAvailability:
    try:
        installed = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        installed = module in sys.modules
    if not installed:
        return RuntimeAvailability(False, f"Optional dependency is not installed: {module}")
    return RuntimeAvailability(True)


def _known_runtime_modules(runtime: str) -> tuple[str, ...]:
    return {
        "coreml": ("coremltools",),
        "mlx": ("mlx",),
        "onnx": ("onnxruntime",),
        "onnxruntime": ("onnxruntime",),
        "openvino": ("openvino",),
        "pytorch": ("torch",),
        "pytorch-mps": ("torch",),
        "rknn": ("rknnlite",),
        "tflite": ("tflite_runtime",),
    }.get(runtime, ())
