"""Native Core AI decision inference with NumPy/tokenizers host processing."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession

from .config import VelaInstanceConfig
from .resources import VelaResourceResolver
from .utils.execution import run_blocking
from .utils.types import CoreAILayout


class CoreAIVelaEngine:
    runtime_name = "coreai"

    def __init__(
        self,
        config: VelaInstanceConfig,
        resources: VelaResourceResolver,
        layout: CoreAILayout,
        *,
        model_name: str,
    ) -> None:
        self._config, self._resources, self._layout = config, resources, layout
        self._model_name = model_name
        self._session: CoreAISession | None = None
        self._packer: Any = None
        self._views = ExitStack()

    @property
    def device(self) -> str:
        return self._session.device if self._session else self._config.device

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    async def load(self, artifact: Path) -> None:
        from .utils.packing import DecisionPacker

        try:
            view = self._views.enter_context(self._resources.view())
            self._packer = DecisionPacker(view)
            self._session = await CoreAIProvider().create_session(
                artifact / self._layout.decision_asset,
                device=self._config.device,
                options={"function_name": self._packer.manifest["function"]},
            )
        except BaseException:
            await self.close()
            raise

    def _predict(self, state: Any, questions: Any) -> dict[str, Any]:
        import numpy as np

        if self._session is None or self._packer is None:
            raise RuntimeError("Vela Core AI engine is not loaded")
        packed = {key: self._packer.pack(state, q) for key, q in questions.items()}
        answers = {}
        tokens = 0
        for key, (arrays, labels) in packed.items():
            output = self._session.run(arrays)
            logits = np.asarray(output[self._packer.manifest["output"]], dtype=np.float64)
            if logits.shape != (1, len(labels)) or not np.isfinite(logits).all():
                raise FloatingPointError("Invalid Vela Core AI logits")
            p = np.exp(logits[0] - logits[0].max())
            p /= p.sum()
            probabilities = dict(zip(labels, p.tolist(), strict=True))
            answers[key] = (
                {"type": "noul", "noul": probabilities["yes"]}
                if questions[key]["type"] == "noul"
                else {
                    "type": "choice",
                    "choice": max(probabilities, key=probabilities.__getitem__),
                    "probabilities": probabilities,
                }
            )
            tokens += int(arrays["attention_mask"].sum())
        return {
            "model": self._model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    async def infer(self, *, state: Any, questions: Any, **kwargs: Any) -> dict[str, Any]:
        return await run_blocking(self._predict, state, questions)

    async def close(self) -> None:
        session, self._session = self._session, None
        self._packer = None
        try:
            if session is not None:
                await session.close()
        finally:
            self._views.close()
