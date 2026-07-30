"""Core ML provider and session.

This module owns Core ML model loading and Apple compute-unit selection. It
intentionally knows nothing about image preprocessing or model output schemas.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend

_COMPUTE_UNIT_ALIASES = {
    "auto": "ALL",
    "all": "ALL",
    "cpu": "CPU_ONLY",
    "cpu-only": "CPU_ONLY",
    "cpu_and_gpu": "CPU_AND_GPU",
    "cpu-and-gpu": "CPU_AND_GPU",
    "gpu": "CPU_AND_GPU",
    "cpu_and_ne": "CPU_AND_NE",
    "cpu-and-neural-engine": "CPU_AND_NE",
    "ane": "CPU_AND_NE",
    "npu": "CPU_AND_NE",
}


class CoreMLSession:
    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device
        self._input_names = frozenset(item.name for item in model.get_spec().description.input)

    @property
    def device(self) -> str:
        return self._device

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._model is None:
            raise RuntimeError("Core ML session is closed")
        selected = {name: value for name, value in inputs.items() if name in self._input_names}
        return dict(self._model.predict(selected))

    async def close(self) -> None:
        self._model = None


class CoreMLProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "coreml"

    def is_available(self) -> bool:
        return importlib.util.find_spec("coremltools") is not None

    def available_devices(self) -> tuple[str, ...]:
        return ("all", "cpu-and-neural-engine", "cpu-and-gpu", "cpu-only")

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> CoreMLSession:
        if not self.is_available():
            raise UnsupportedRuntimeError(
                "Core ML requires coremltools on Apple Silicon; install the yolo extra"
            )
        requested = (device or str(options.get("compute_units", "all"))).lower()
        try:
            unit_name = _COMPUTE_UNIT_ALIASES[requested]
        except KeyError as error:
            raise UnsupportedRuntimeError(
                f"Unsupported Core ML compute units: {requested}",
                details={"available": self.available_devices()},
            ) from error
        coremltools = importlib.import_module("coremltools")
        compute_units = getattr(coremltools.ComputeUnit, unit_name)
        model = await asyncio.to_thread(
            coremltools.models.MLModel,
            str(artifact),
            compute_units=compute_units,
        )
        return CoreMLSession(model, requested)
