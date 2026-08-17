"""PyTorch MPS engine for py-feat RetinaFace."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.detection import DetectionRequest

from .config import RetinaFaceInstanceConfig
from .resources import RetinaFaceResourceResolver
from .utils.checkpoint import load_retinaface_checkpoint, normalize_retinaface_output
from .utils.preprocess import torch_input
from .utils.types import PreparedImage


class TorchRetinaFaceEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self, config: RetinaFaceInstanceConfig, resources: RetinaFaceResourceResolver
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
                "model_loader": load_retinaface_checkpoint,
                "output_adapter": normalize_retinaface_output,
            },
        )

    async def infer(self, prepared: PreparedImage, request: DetectionRequest) -> dict[str, Any]:
        del request
        if self._session is None:
            raise RuntimeError("RetinaFace PyTorch engine is not loaded")
        torch = importlib.import_module("torch")
        return dict(
            await asyncio.to_thread(self._session.run, {"input": torch_input(prepared, torch)})
        )

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
