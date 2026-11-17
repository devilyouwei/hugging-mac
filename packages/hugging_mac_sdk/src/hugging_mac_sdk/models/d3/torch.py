"""Original D3 PyTorch inference with unchanged upstream weights and processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .config import D3InstanceConfig
from .resources import D3ResourceResolver
from .utils.execution import run_blocking
from .utils.upstream import load_upstream


class TorchD3Engine:
    runtime_name = "pytorch"

    def __init__(
        self, config: D3InstanceConfig, resources: D3ResourceResolver, *, model_name: str
    ) -> None:
        self._config, self._resources = config, resources
        self._model_name = model_name
        self._model: Any = None
        self._views = ExitStack()

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else "cpu"

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    def _load_model(self, artifact: Path) -> None:
        upstream = load_upstream(artifact)
        self._model = upstream.D3(
            artifact,
            device="cpu",
            batch_size=1,
            model_name=self._model_name,
            permutation_average=self._config.permutation_average,
        )
        if self._model.image_unavailable or self._model.video_unavailable:
            raise RuntimeError("D3 source must support all declared text, image, and video inputs")

    async def load(self, artifact: Path) -> None:
        try:
            view = self._views.enter_context(self._resources.view())
            await run_blocking(self._load_model, view)
        except BaseException:
            await self.close()
            raise

    async def infer(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("D3 PyTorch engine is not loaded")
        return dict(
            await run_blocking(
                self._model.system_one,
                state=state,
                questions=questions,
                images=images,
                videos=videos,
            )
        )

    async def close(self) -> None:
        self._model = None
        self._views.close()
