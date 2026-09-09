"""Core ML engine for locally converted PP-DocLayoutV3."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider

from .config import PPDocLayoutV3InstanceConfig
from .resources import PPDocLayoutV3ResourceResolver
from .utils.types import PreparedImage


class CoreMlPPDocLayoutV3Engine:
    runtime_name = "coreml"

    def __init__(
        self, config: PPDocLayoutV3InstanceConfig, resources: PPDocLayoutV3ResourceResolver
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
            artifact / "model.mlpackage", device=self._config.compute_units, options={}
        )

    async def infer(self, prepared: PreparedImage) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("PP-DocLayoutV3 Core ML engine is not loaded")
        return dict(await asyncio.to_thread(self._session.run, {"pixel_values": prepared.tensor}))

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
