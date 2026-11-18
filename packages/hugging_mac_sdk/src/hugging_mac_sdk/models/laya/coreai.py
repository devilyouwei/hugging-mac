"""Native Laya main/act execution; all host operations use NumPy/tokenizers."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.coreai import CoreAIProvider

from .config import LayaInstanceConfig
from .resources import LayaResourceResolver
from .utils.execution import run_blocking
from .utils.packing import text


class CoreAILayaEngine:
    runtime_name = "coreai"
    device = "cpu"

    def __init__(
        self, config: LayaInstanceConfig, resources: LayaResourceResolver, *, model_name: str
    ) -> None:
        self._config, self._resources = config, resources
        self._model_name = model_name
        self._main: Any = None
        self._act: Any = None
        self._packer: Any = None
        self._views = ExitStack()

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    async def load(self, artifact: Path) -> None:
        from .utils.packing import LayaPacker

        try:
            self._packer = LayaPacker(self._views.enter_context(self._resources.view()))
            asset = artifact / self._packer.meta["assets"]["main"]
            provider = CoreAIProvider()
            self._main = await provider.create_session(
                asset,
                device="cpu",
                options={"function_name": self._packer.layout["functions"]["main"]},
            )
            self._act = await provider.create_session(
                asset,
                device="cpu",
                options={"function_name": self._packer.layout["functions"]["act"]},
            )
        except BaseException:
            await self.close()
            raise

    def run_row(self, ids: list[int], markers: list[int], kind: str) -> tuple[Any, float]:
        import numpy as np

        if self._main is None or self._act is None:
            raise RuntimeError("Laya is not loaded")
        layout = self._packer.layout
        x = np.full((1, layout["window"]), layout["pad_token_id"], dtype=np.int32)
        x[0, : len(ids)] = ids
        mask = np.zeros_like(x)
        mask[0, : len(ids)] = 1
        qt = ("choice", "score", "noul").index(kind)
        output = self._main.run(
            {
                "input_ids": x,
                "attention_mask": mask,
                "qtype_onehot": np.eye(3, dtype=np.float32)[qt : qt + 1],
            }
        )
        logits = output["token_logits"][0, markers].astype(np.float64)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.sort(p)[::-1]
        k = max(len(p), 2)
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-30))) / np.log(k)
        features = np.array(
            [[top[0], top[0] - (top[1] if len(top) > 1 else 0), entropy, k / 255]], dtype=np.float32
        )
        act = self._act.run({"pooled_cls": output["pooled_cls"], "feats": features})["act_logits"][
            0
        ].astype(np.float64)
        act = np.exp(act - act.max())
        act /= act.sum()
        temperature = layout["source_temperature"][qt]
        p = np.exp(logits / temperature - (logits / temperature).max())
        p /= p.sum()
        if not np.isfinite(p).all() or not np.isfinite(act).all():
            raise FloatingPointError("Nonfinite Laya output")
        return p, float(act[0])

    def _predict(self, state: Any, questions: Any) -> dict[str, Any]:
        import numpy as np

        answers = {}
        tokens = 0
        for key, q in questions.items():
            ids, markers, labels = self._packer.pack(state, q)
            p, act = self.run_row(ids, markers, q["type"])
            tokens += len(ids)
            confidence = float(
                np.clip(1 + np.sum(p * np.log(np.maximum(p, 1e-30))) / np.log(max(len(p), 2)), 0, 1)
            )
            answer = {
                "type": q["type"],
                "confidence": confidence,
                "action": {"act_probability": act},
            }
            if q["type"] == "choice":
                answer.update(
                    choice=labels[int(p.argmax())],
                    probabilities=dict(zip(labels, p.tolist(), strict=True)),
                )
            elif q["type"] == "score":
                answer.update(
                    score=float(np.arange(len(p)) @ p),
                    probabilities=dict(zip(labels, p.tolist(), strict=True)),
                    legend={str(i): text(v) for i, v in enumerate(q["criteria"])},
                )
            else:
                answer.update(noul=float(p[1]), confidence=float(max(p)))
            answers[key] = answer
        return {
            "model": self._model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    async def infer(self, *, state: Any, questions: Any, **kwargs: Any) -> dict[str, Any]:
        return await run_blocking(self._predict, state, questions)

    async def close(self) -> None:
        main, act = self._main, self._act
        self._main = self._act = None
        try:
            if act is not None:
                await act.close()
        finally:
            try:
                if main is not None:
                    await main.close()
            finally:
                self._packer = None
                self._views.close()
