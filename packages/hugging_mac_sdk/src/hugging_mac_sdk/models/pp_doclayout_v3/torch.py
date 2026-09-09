"""PyTorch/MPS engine for PP-DocLayoutV3 source weights."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.torch import TorchProvider

from .config import PPDocLayoutV3InstanceConfig
from .resources import PPDocLayoutV3ResourceResolver
from .utils.modeling import load_pp_doclayout_v3, normalize_outputs
from .utils.preprocess import torch_input
from .utils.types import PreparedImage


class TorchPPDocLayoutV3Engine:
    runtime_name = "pytorch-mps"

    def __init__(
        self, config: PPDocLayoutV3InstanceConfig, resources: PPDocLayoutV3ResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        return self._session.device if self._session is not None else (self._config.device or "mps")

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        self._session = await TorchProvider().create_session(
            artifact,
            device=self._config.device or "mps",
            options={
                "allow_cpu_fallback": self._config.allow_cpu_fallback,
                "model_loader": load_pp_doclayout_v3,
                "output_adapter": normalize_outputs,
            },
        )

    async def infer(self, prepared: PreparedImage) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("PP-DocLayoutV3 PyTorch engine is not loaded")
        torch = importlib.import_module("torch")
        inputs = {"input": torch_input(prepared, torch)}
        return dict(await asyncio.to_thread(self._session.run, inputs))

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
