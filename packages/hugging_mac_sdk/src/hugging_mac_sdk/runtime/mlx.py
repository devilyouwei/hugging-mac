"""Safe structure inspection for MLX and runtime-neutral SafeTensors archives."""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend
from hugging_mac_sdk.schemas.model_structure import ModelComponentStructure, TensorStructure

MlxLoader = Callable[[Path, Any], Any]
MlxRunner = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]


class MlxSession:
    """Own an MLX-loaded value and its framework-level lifecycle."""

    def __init__(self, mlx: Any, value: Any, runner: MlxRunner | None = None) -> None:
        self._mlx = mlx
        self._value = value
        self._runner = runner

    @property
    def device(self) -> str:
        return "gpu"

    @property
    def value(self) -> Any:
        if self._value is None:
            raise RuntimeError("MLX session is closed")
        return self._value

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._runner is None:
            raise UnsupportedRuntimeError(
                "This MLX session uses a model-specific inference interface"
            )
        return self._runner(self.value, inputs)

    async def close(self) -> None:
        had_value = self._value is not None
        self._value = None
        if not had_value:
            return
        await asyncio.to_thread(gc.collect)
        clear_cache = getattr(self._mlx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()


class MlxProvider(RuntimeBackend):
    """Load MLX artifacts while leaving task semantics inside model packages."""

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        if "mlx.core" in sys.modules:
            return True
        return importlib.util.find_spec("mlx") is not None

    def available_devices(self) -> tuple[str, ...]:
        return ("gpu",) if self.is_available() else ()

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> MlxSession:
        if not self.is_available():
            raise UnsupportedRuntimeError("MLX is not installed")
        requested = (device or "gpu").lower()
        if requested not in {"auto", "gpu", "metal"}:
            raise UnsupportedRuntimeError(
                f"MLX device is not available: {requested}",
                details={"available": self.available_devices()},
            )
        loader = options.get("model_loader")
        if not callable(loader):
            raise TypeError("MlxProvider requires a callable 'model_loader' option")
        runner = options.get("runner")
        if runner is not None and not callable(runner):
            raise TypeError("'runner' must be callable")
        mlx = importlib.import_module("mlx.core")
        value = await asyncio.to_thread(loader, artifact, mlx)
        return MlxSession(mlx, value, runner)


def materialize_mlx(value: Any, mlx: Any) -> None:
    """Materialize lazy MLX parameters before a temporary artifact view closes."""

    parameters = getattr(value, "parameters", None)
    mlx.eval(parameters() if callable(parameters) else value)


def inspect_mlx_artifact(path: Path) -> tuple[ModelComponentStructure, ...]:
    candidates = (
        (path,) if path.suffix == ".safetensors" else tuple(sorted(path.rglob("*.safetensors")))
    )
    if not candidates:
        raise UnsupportedRuntimeError("No SafeTensors weights were found in this artifact")
    components: list[ModelComponentStructure] = []
    for candidate in candidates:
        tensors: list[TensorStructure] = []
        parameter_count = 0
        header = _safetensors_header(candidate)
        for name, descriptor in header.items():
            if name == "__metadata__" or not isinstance(descriptor, dict):
                continue
            shape = tuple(int(value) for value in descriptor.get("shape", ()))
            parameter_count += _shape_size(shape)
            tensors.append(
                TensorStructure(
                    name=name,
                    dtype=str(descriptor.get("dtype", "unknown")).lower(),
                    shape=shape,
                )
            )
        components.append(
            ModelComponentStructure(
                name=candidate.name,
                model_type="safetensors",
                parameter_count=parameter_count,
                metadata={"tensor_count": len(tensors)},
                outputs=tuple(tensors),
            )
        )
    return tuple(components)


def _safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise UnsupportedRuntimeError("Invalid SafeTensors header")
        header_size = int.from_bytes(size_bytes, "little")
        if header_size <= 0 or header_size > 100_000_000:
            raise UnsupportedRuntimeError("Invalid SafeTensors header size")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise UnsupportedRuntimeError("Invalid SafeTensors metadata")
    return header


def _shape_size(shape: tuple[int, ...]) -> int:
    total = 1
    for value in shape:
        total *= value
    return total
