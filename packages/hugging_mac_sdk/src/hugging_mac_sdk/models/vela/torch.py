"""Original Vela decision backbone; no optional task adapters are loaded."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .config import VelaInstanceConfig
from .resources import VelaResourceResolver
from .utils.execution import run_blocking
from .utils.upstream import load_upstream


class TorchVelaEngine:
    runtime_name = "pytorch"
    device = "cpu"

    def __init__(
        self, config: VelaInstanceConfig, resources: VelaResourceResolver, *, model_name: str
    ) -> None:
        self._resources, self._model_name = resources, model_name
        self._model: Any = None
        self._views = ExitStack()

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    def _load_model(self, root: Path) -> None:
        self._model = load_upstream(root).DecisionEncoder.load(root, device=self.device)

    async def load(self, artifact: Path) -> None:
        try:
            root = self._views.enter_context(self._resources.view())
            await run_blocking(self._load_model, root)
        except BaseException:
            await self.close()
            raise

    async def infer(self, *, state: Any, questions: Any, **kwargs: Any) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Vela PyTorch engine is not loaded")
        answers, tokens = await run_blocking(self._model.predict, state, questions)
        return {
            "model": self._model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    async def close(self) -> None:
        self._model = None
        self._views.close()
