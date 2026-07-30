"""YOLOv8 engine for the generic ONNX Runtime provider."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.onnx import OnnxRuntimeProvider
from hugging_mac_sdk.schemas.detection import DetectionRequest

from .config import YoloV8InstanceConfig
from .resources import YoloV8ResourceResolver
from .utils.types import PreparedImage


class OnnxYoloV8Engine:
    runtime_name = "onnx"

    def __init__(
        self,
        config: YoloV8InstanceConfig,
        resources: YoloV8ResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        if self._session is not None:
            return self._session.device
        return self._config.device or "auto"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_onnx()).path

    async def load(self, artifact: Path) -> None:
        self._session = await OnnxRuntimeProvider().create_session(
            artifact,
            device=self._config.device or "auto",
            options={},
        )

    async def infer(
        self,
        prepared: PreparedImage,
        request: DetectionRequest,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("YOLOv8 ONNX engine is not loaded")
        numpy = importlib.import_module("numpy")
        value = numpy.asarray(prepared.image, dtype=numpy.float32)
        value = numpy.ascontiguousarray(value.transpose(2, 0, 1))[None] / 255.0
        return dict(await asyncio.to_thread(self._session.run, {"images": value}))

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
