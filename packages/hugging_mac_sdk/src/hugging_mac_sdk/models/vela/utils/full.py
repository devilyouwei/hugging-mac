"""Full Vela Core AI inference with the original System One interface.

Only one adapter's native weights are resident at a time. Original shared FP32
weights are cached as immutable Core AI inputs; Torch performs no neural inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from types import MethodType
from typing import Any

import numpy as np

from .full_host import make_host
from .full_upstream import load_full_upstream
from .native import NativeComponents


def decision_host(root: Path, backend: NativeComponents) -> Any:
    import torch
    from transformers import AutoTokenizer

    upstream = load_full_upstream(root)
    model = upstream.DecisionEncoder.__new__(upstream.DecisionEncoder)
    torch.nn.Module.__init__(model)
    config = json.loads((root / "config.json").read_text())
    model.tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model.mk = config["decision"]["marker_ids"]
    model.max_len, model.budget, model.device = 32768, 65536, "cpu"

    def forward_recs(self: Any, records: Any) -> Any:
        result: list[Any] = [None] * len(records)
        order = sorted(range(len(records)), key=lambda i: len(records[i]["ids"]))
        start = 0
        while start < len(order):
            stop = start + 1
            while (
                stop < len(order)
                and len(records[order[stop]]["ids"]) * (stop - start + 1) <= self.budget
            ):
                stop += 1
            group = [records[index] for index in order[start:stop]]
            width = max(len(row["ids"]) for row in group)
            ids = np.full((len(group), width), self.tok.pad_token_id, np.int32)
            mask = np.zeros_like(ids)
            queries: list[tuple[int, ...]] = []
            options: list[tuple[int, ...]] = []
            for batch, row in enumerate(group):
                ids[batch, : len(row["ids"])] = row["ids"]
                mask[batch, : len(row["ids"])] = 1
                queries.append((batch, row["q_pos"], *row["pool"]))
                options.extend((batch, position, batch) for position in row["opt_pos"])
            logits, _ = backend.run("decision", ids, mask, queries, options, [(0, 0)], [(0, 0)])
            offset = 0
            for index, row in zip(order[start:stop], group, strict=True):
                count = len(row["opt_pos"])
                values = logits[offset : offset + count].astype(np.float64)
                probabilities = np.exp(values - values.max())
                result[index] = probabilities / probabilities.sum()
                offset += count
            start = stop
        return result

    model.forward_recs = MethodType(forward_recs, model)
    return model


class VelaFullCoreAI:
    """Synchronous upstream-compatible API; use a worker thread from async apps."""

    def __init__(
        self, artifact_path: Path | str, *, device: str = "cpu", entity_route: str | None = None
    ) -> None:
        self.root = Path(artifact_path).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != "vela-full-coreai/1":
            raise ValueError("A complete Vela Core AI bundle is required")
        self._lock = RLock()
        self._closed = False
        self._backend = NativeComponents(self.root, manifest, device)
        try:
            self._host = make_host(
                self.root,
                decision_host(self.root, self._backend),
                lambda name: lambda *args: self._backend.run(name, *args),
                entity_route=entity_route,
            )
        except BaseException:
            self._backend.close()
            raise

    @property
    def used_components(self) -> set[str]:
        return set(self._backend.used)

    def system_one(self, state: Any, questions: Any, model: Any = None) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Vela Core AI model is closed")
            return dict(self._host.system_one(state, questions, model=model))

    def predict(self, state: Any, questions: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("Vela Core AI model is closed")
            return self._host.predict(state, questions)

    def close(self) -> None:
        with self._lock:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> VelaFullCoreAI:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
