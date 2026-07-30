"""YOLOv8 Seg engine for the generic Core ML provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider
from hugging_mac_sdk.schemas.segmentation import SegmentationRequest

from .config import YoloV8SegInstanceConfig
from .resources import YoloV8SegResourceResolver
from .utils.types import PreparedImage


class CoreMlYoloV8SegEngine:
    runtime_name = "coreml"

    def __init__(
        self,
        config: YoloV8SegInstanceConfig,
        resources: YoloV8SegResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        return (
            self._session.device
            if self._session is not None
            else self._config.compute_units
        )

    async def resolve(self) -> Path:
        return (await self._resources.resolve_coreml()).path

    async def load(self, artifact: Path) -> None:
        self._session = await CoreMLProvider().create_session(
            artifact,
            device=self._config.compute_units,
            options={},
        )

    async def infer(
        self,
        prepared: PreparedImage,
        request: SegmentationRequest,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("YOLOv8 Seg Core ML engine is not loaded")
        return dict(await asyncio.to_thread(self._session.run, {"image": prepared.image}))

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
