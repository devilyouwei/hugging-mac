"""Core ML engine for locally converted py-feat RetinaFace."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider
from hugging_mac_sdk.schemas.detection import DetectionRequest

from .config import RetinaFaceInstanceConfig
from .resources import RetinaFaceResourceResolver
from .utils.types import PreparedImage


class CoreMlRetinaFaceEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: RetinaFaceInstanceConfig, resources: RetinaFaceResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        return self._session.device if self._session is not None else self._config.compute_units

    async def resolve(self) -> Path:
        return (await self._resources.resolve_coreml()).path

    async def load(self, artifact: Path) -> None:
        self._session = await CoreMLProvider().create_session(
            artifact, device=self._config.compute_units, options={}
        )

    async def infer(self, prepared: PreparedImage, request: DetectionRequest) -> dict[str, Any]:
        del request
        if self._session is None:
            raise RuntimeError("RetinaFace Core ML engine is not loaded")
        return dict(await asyncio.to_thread(self._session.run, {"image": prepared.tensor}))

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
