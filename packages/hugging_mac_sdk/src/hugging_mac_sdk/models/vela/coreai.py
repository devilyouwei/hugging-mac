"""Native Core AI decision inference with NumPy/tokenizers host processing."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession
from hugging_mac_sdk.schemas.structured_prediction import StructuredPredictionRequest

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
        full_model: bool = False,
    ) -> None:
        self._config, self._resources, self._layout = config, resources, layout
        self._model_name = model_name
        self._full_model = full_model
        self._full: Any = None
        self._session: CoreAISession | None = None
        self._packer: Any = None
        self._native: Any = None
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
            if self._full_model:
                from .utils.full import VelaFullCoreAI

                await run_blocking(self._load_full, view, VelaFullCoreAI)
                return
            self._packer = DecisionPacker(view)
            if self._packer.component_manifest is not None:
                from .utils.native import NativeComponents

                self._native = NativeComponents(
                    artifact, self._packer.component_manifest, self._config.device
                )
                await run_blocking(self._native.load, "decision")
                self._session = self._native.session
                return
            self._session = await CoreAIProvider().create_session(
                artifact / self._layout.decision_asset,
                device=self._config.device,
                options={"function_name": self._packer.manifest["function"]},
            )
        except BaseException:
            await self.close()
            raise

    def _load_full(self, root: Path, factory: Any) -> None:
        self._full = factory(
            root, device=self._config.device, entity_route=self._config.entity_route
        )

    async def predict_structure(self, request: StructuredPredictionRequest) -> Any:
        if self._full is not None:
            method = getattr(self._full, request.method)
            if request.method == "system_one":
                return await run_blocking(method, request.state, request.questions, request.model)
            return await run_blocking(method, request.state, request.questions)
        result = await self.infer(state=request.state, questions=request.questions)
        if request.method == "predict":
            return [result["answers"], result["usage"]["input_tokens"]]
        if request.model is not None:
            result["model"] = request.model
        return result

    def _predict(self, state: Any, questions: Any) -> dict[str, Any]:
        import numpy as np

        if self._session is None or self._packer is None:
            raise RuntimeError("Vela Core AI engine is not loaded")
        packed = {key: self._packer.pack(state, q) for key, q in questions.items()}
        answers = {}
        tokens = 0
        for key, (arrays, labels) in packed.items():
            if self._native is None:
                output = self._session.run(arrays)
                logits = np.asarray(output[self._packer.manifest["output"]], dtype=np.float64)
            else:
                pool = np.flatnonzero(arrays["pool_mask"][0])
                positions = arrays["option_positions"][0]
                logits, _ = self._native.run(
                    "decision",
                    arrays["input_ids"],
                    arrays["attention_mask"],
                    [[0, 1, int(pool[0]), int(pool[-1]) + 1]],
                    [[0, int(position), 0] for position in positions],
                    [[0, 0]],
                    [[0, 0]],
                )
                logits = np.asarray(logits, dtype=np.float64)[None, :]
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
        if self._full is not None:
            return dict(await run_blocking(self._full.system_one, state, questions))
        return await run_blocking(self._predict, state, questions)

    async def close(self) -> None:
        full, self._full = self._full, None
        session, self._session = self._session, None
        self._packer = None
        native, self._native = self._native, None
        try:
            if full is not None:
                await run_blocking(full.close)
            if native is not None:
                await run_blocking(native.close)
            elif session is not None:
                await session.close()
        finally:
            self._views.close()
