"""Original Laya Agent loaded exclusively from prepared local resources."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .config import LayaInstanceConfig
from .resources import LayaResourceResolver
from .utils.execution import run_blocking


class TorchLayaEngine:
    runtime_name = "pytorch"
    device = "cpu"

    def __init__(self, config: LayaInstanceConfig, resources: LayaResourceResolver) -> None:
        self._config, self._resources = config, resources
        self._agent: Any = None
        self._views = ExitStack()

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    def _load(self, root: Path) -> None:
        from importlib import import_module

        self._agent = import_module("laya").Agent(str(root), device="cpu")

    async def load(self, artifact: Path) -> None:
        try:
            await run_blocking(self._load, self._views.enter_context(self._resources.view()))
        except BaseException:
            await self.close()
            raise

    async def infer(self, *, state: Any, questions: Any, **kwargs: Any) -> dict[str, Any]:
        if self._agent is None:
            raise RuntimeError("Laya is not loaded")
        # The typed SDK requires normalized probabilities, unlike upstream's four-digit display.
        result = await run_blocking(
            self._agent.predict, state, questions, max_len=self._config.max_length
        )
        if result["usage"].get("truncated"):
            raise ValueError(
                "Input exceeds the configured Laya context; no truncated answer returned"
            )
        result["usage"] = {key: result["usage"][key] for key in ("input_tokens", "output_tokens")}
        for answer in result["answers"].values():
            probabilities = answer.get("probabilities")
            if probabilities:
                total = sum(probabilities.values())
                answer["probabilities"] = {k: v / total for k, v in probabilities.items()}
                if answer["type"] == "score":
                    answer["score"] = sum(int(k) * v for k, v in answer["probabilities"].items())
        return dict(result)

    async def close(self) -> None:
        self._agent = None
        self._views.close()
