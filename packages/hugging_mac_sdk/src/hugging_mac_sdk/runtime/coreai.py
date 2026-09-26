"""Apple Core AI runtime, isolated on one worker and one asyncio loop per session."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import platform
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.base import RuntimeBackend


def coreai_available() -> bool:
    """Probe installation without importing the native framework."""
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return False
    try:
        return importlib.util.find_spec("coreai") is not None
    except (ImportError, ValueError):
        return False


class CoreAISession:
    def __init__(self, device: str) -> None:
        self._device = device
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coreai")
        self._lock = Lock()
        self._closed = False
        self._runner: asyncio.Runner | None = None
        self._view: TemporaryDirectory[str] | None = None
        self._context: Any = None
        self._function: Any = None
        self._runtime: Any = None
        self._state: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._device

    def _open(self, artifact: Path, function_name: str) -> None:
        self._runner = asyncio.Runner()
        try:
            if artifact.suffix != ".aimodel":
                self._view = TemporaryDirectory(prefix="hugging-mac-coreai-", suffix=".aimodel")
                view = Path(self._view.name)
                for child in artifact.iterdir():
                    (view / child.name).symlink_to(
                        child.resolve(), target_is_directory=child.is_dir()
                    )
                artifact = view
            self._runner.run(self._load(artifact, function_name))
        except BaseException:
            self._runner.close()
            self._runner = None
            self._cleanup_view()
            raise

    async def _load(self, artifact: Path, function_name: str) -> None:
        authoring = importlib.import_module("coreai.authoring")
        self._runtime = importlib.import_module("coreai.runtime")
        options = None
        if self.device == "cpu":
            if not self._runtime.SpecializationOptions.is_supported():
                raise UnsupportedRuntimeError(
                    "Core AI CPU selection requires macOS with USE_OS_COREAI=1"
                )
            options = self._runtime.SpecializationOptions.cpu_only()
        asset = authoring.AIModelAsset.load(artifact)
        context = asset.executable(options)
        model = await context.__aenter__()
        try:
            self._function = model.load_function(function_name)
        except BaseException:
            await context.__aexit__(*sys.exc_info())
            raise
        self._context = context

    async def _infer(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        arrays = {name: self._runtime.NDArray(value) for name, value in inputs.items()}
        outputs = (
            await self._function(arrays, state=self._state)
            if self._state
            else await self._function(arrays)
        )
        return {name: value.numpy().copy() for name, value in outputs.items()}

    def _reset_state(self, values: Mapping[str, Any]) -> None:
        self._state = {name: self._runtime.NDArray(value.copy()) for name, value in values.items()}

    def reset_state(self, values: Mapping[str, Any]) -> None:
        """Replace persistent native state between requests on the session worker."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Core AI session is closed")
            future = self._executor.submit(self._reset_state, dict(values))
        future.result()

    def _run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        assert self._runner is not None
        return self._runner.run(self._infer(inputs))

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        # Submit under the lock so close is always queued after accepted inference.
        with self._lock:
            if self._closed:
                raise RuntimeError("Core AI session is closed")
            future = self._executor.submit(self._run, dict(inputs))
        return future.result()

    async def _release(self) -> None:
        try:
            if self._context is not None:
                await self._context.__aexit__(None, None, None)
        finally:
            self._context = None
            self._function = None
            self._state = {}
            self._runtime = None

    def _cleanup_view(self) -> None:
        if self._view is not None:
            self._view.cleanup()
            self._view = None

    def _shutdown(self) -> None:
        if self._runner is not None:
            try:
                self._runner.run(self._release())
            finally:
                self._runner.close()
                self._runner = None
                self._cleanup_view()

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._executor.submit(self._shutdown)
            self._executor.shutdown(wait=False)
        await asyncio.shield(asyncio.wrap_future(future))


class CoreAIProvider(RuntimeBackend):
    @property
    def name(self) -> str:
        return "coreai"

    def is_available(self) -> bool:
        return coreai_available()

    def available_devices(self) -> tuple[str, ...]:
        return ("auto", "cpu") if self.is_available() else ()

    async def create_session(
        self, artifact: Path, *, device: str | None, options: Mapping[str, Any]
    ) -> CoreAISession:
        if not self.is_available():
            raise UnsupportedRuntimeError(
                "Core AI requires Apple Silicon macOS and the SDK coreai extra"
            )
        requested = (device or "auto").lower()
        if requested not in {"auto", "cpu"}:
            raise UnsupportedRuntimeError(f"Unsupported Core AI device: {requested}")
        unknown = options.keys() - {"function_name"}
        if unknown:
            raise ValueError(f"Unknown Core AI options: {sorted(unknown)}")
        function_name = options.get("function_name", "main")
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("function_name must be a non-empty string")
        # SDK directory artifacts may have a canonical path without an extension.
        if not artifact.is_dir():
            raise ValueError(f"Core AI expects an .aimodel asset directory: {artifact}")
        session = CoreAISession(requested)
        future = asyncio.wrap_future(
            session._executor.submit(session._open, artifact, function_name)
        )
        try:
            await asyncio.shield(future)
        except BaseException:
            # Cleanup is queued after load, including when the caller cancels it.
            await session.close()
            if future.done() and not future.cancelled():
                future.exception()
            raise
        return session
