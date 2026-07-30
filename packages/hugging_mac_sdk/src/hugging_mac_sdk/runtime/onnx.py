"""ONNX Runtime provider with Core ML EP preference on macOS."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend


class OnnxRuntimeSession:
    def __init__(self, session: Any, providers: tuple[str, ...]) -> None:
        self._session = session
        self.providers = providers

    @property
    def device(self) -> str:
        return self.providers[0] if self.providers else "unknown"

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is closed")
        output_names = [item.name for item in self._session.get_outputs()]
        values = self._session.run(output_names, dict(inputs))
        return dict(zip(output_names, values, strict=True))

    async def close(self) -> None:
        self._session = None


class OnnxRuntimeProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "onnx"

    def is_available(self) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def available_devices(self) -> tuple[str, ...]:
        if not self.is_available():
            return ()
        ort = importlib.import_module("onnxruntime")
        return tuple(str(item) for item in ort.get_available_providers())

    def select_execution_providers(
        self, requested: str | Sequence[str] | None = None
    ) -> tuple[str, ...]:
        available = self.available_devices()
        if requested is None or requested == "auto":
            preference: tuple[str, ...] = (
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            )
        elif isinstance(requested, str):
            aliases = {
                "coreml": "CoreMLExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            preference = (aliases.get(requested.lower(), requested), "CPUExecutionProvider")
        else:
            preference = tuple(requested)
        selected = tuple(dict.fromkeys(item for item in preference if item in available))
        if not selected:
            raise UnsupportedRuntimeError(
                "No requested ONNX Runtime execution provider is available",
                details={"requested": tuple(preference), "available": available},
            )
        return selected

    async def create_session(
        self,
        artifact: Path,
        *,
        device: str | None,
        options: Mapping[str, Any],
    ) -> OnnxRuntimeSession:
        if not self.is_available():
            raise UnsupportedRuntimeError("ONNX Runtime is not installed")
        ort = importlib.import_module("onnxruntime")
        requested = options.get("execution_providers", device or "auto")
        providers = self.select_execution_providers(requested)
        session_options = options.get("session_options")
        session = await asyncio.to_thread(
            ort.InferenceSession,
            str(artifact),
            sess_options=session_options,
            providers=list(providers),
        )
        return OnnxRuntimeSession(session, providers)
