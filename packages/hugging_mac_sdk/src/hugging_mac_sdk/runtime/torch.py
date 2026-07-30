"""PyTorch provider with automatic Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend

ModelLoader = Callable[[Path, Any], Any]
OutputAdapter = Callable[[Any], Mapping[str, Any]]


class TorchSession:
    def __init__(self, torch: Any, model: Any, device: str, output_adapter: OutputAdapter) -> None:
        self._torch = torch
        self._model = model
        self._device = device
        self._output_adapter = output_adapter

    @property
    def device(self) -> str:
        return self._device

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("PyTorch session is closed")
        tensor = inputs.get("input")
        if tensor is None:
            raise ValueError("PyTorch session expects an 'input' tensor")
        if hasattr(tensor, "to"):
            tensor = tensor.to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        return self._output_adapter(output)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        if not had_model:
            return
        gc.collect()
        if self._device == "mps" and hasattr(self._torch, "mps"):
            empty_cache = getattr(self._torch.mps, "empty_cache", None)
            if empty_cache is not None:
                empty_cache()


class TorchProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "pytorch"

    def is_available(self) -> bool:
        return importlib.util.find_spec("torch") is not None

    def available_devices(self) -> tuple[str, ...]:
        if not self.is_available():
            return ()
        torch = importlib.import_module("torch")
        devices: list[str] = []
        if bool(torch.backends.mps.is_available()):
            devices.append("mps")
        devices.append("cpu")
        return tuple(devices)

    def resolve_device(self, requested: str | None, *, allow_cpu_fallback: bool) -> str:
        requested = requested or "mps"
        available = self.available_devices()
        if requested in available:
            return requested
        if allow_cpu_fallback and "cpu" in available:
            return "cpu"
        raise UnsupportedRuntimeError(
            f"PyTorch device is not available: {requested}",
            details={"available": available},
        )

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> TorchSession:
        if not self.is_available():
            raise UnsupportedRuntimeError("PyTorch is not installed; install the yolo extra")
        torch = importlib.import_module("torch")
        loader = options.get("model_loader")
        if not callable(loader):
            raise TypeError("TorchProvider requires a callable 'model_loader' option")
        output_adapter = options.get("output_adapter", _default_output_adapter)
        if not callable(output_adapter):
            raise TypeError("'output_adapter' must be callable")
        resolved = self.resolve_device(
            device,
            allow_cpu_fallback=bool(options.get("allow_cpu_fallback", True)),
        )
        model = await asyncio.to_thread(loader, artifact, torch)
        model = model.eval().to(resolved)
        return TorchSession(torch, model, resolved, output_adapter)


def _default_output_adapter(output: Any) -> Mapping[str, Any]:
    return {"output": output}
