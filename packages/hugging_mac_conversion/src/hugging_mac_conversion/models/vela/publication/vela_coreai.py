"""Standalone Vela decision adapter; rendering adapted from Apache-2.0 upstream.

See NOTICE. Neural inference uses Core AI; no PyTorch or source weights are needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


class VelaCoreAI:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.config = json.loads((self.root / "decision.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self._model: Any = None
        self._function: Any = None
        self._busy = False

    @classmethod
    @asynccontextmanager
    async def open(cls, root: str | Path = ".") -> AsyncIterator[VelaCoreAI]:
        from coreai.runtime import AIModel, SpecializationOptions

        instance = cls(root)
        if not SpecializationOptions.is_supported():
            raise RuntimeError("CPU specialization requires the macOS Core AI runtime")
        instance._model = await AIModel.load(
            instance.root / "model.aimodel", SpecializationOptions.cpu_only()
        )
        try:
            instance._function = instance._model.load_function(instance.manifest["function"])
            yield instance
        finally:
            instance._function = instance._model = None

    @staticmethod
    def _text(value: Any) -> str:
        return (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )

    def pack(self, state: Any, question: Any) -> tuple[dict[str, np.ndarray[Any, Any]], list[str]]:
        if not isinstance(question, dict):
            raise ValueError("Provide a choice/noul question object")
        kind = question.get("type")
        options: dict[str, Any]
        if kind == "noul":
            options = {"no": "", "yes": ""}
        elif kind == "choice":
            criteria = question.get("criteria")
            if isinstance(criteria, (list, tuple)):
                criteria = {str(value): None for value in criteria}
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("choice requires a nonempty criteria mapping")
            options = {str(k): v for k, v in criteria.items()}
        else:
            raise ValueError("Only choice and noul are included in this experimental artifact")

        def encode(value: str) -> list[int]:
            return list(self.tokenizer.encode(value, add_special_tokens=False).ids)

        markers = self.config["marker_ids"]
        ids = [self.config["bos_token_id"], markers["[Q]"]]
        ids += encode(self._text(question.get("instructions", "")))
        option_positions = []
        for name, description in options.items():
            text = "" if description is None else self._text(description)
            option_positions.append(len(ids))
            ids += [markers["[O]"], *encode(f"{name}: {text}" if text else name)]
        ids.append(markers["[SEP_SCHEMA]"])
        start = len(ids)
        ids += [markers["[SEG_user]"], *encode(self._text(state if state is not None else ""))]
        end = len(ids)
        ids.append(self.config["eos_token_id"])
        capacity = self.manifest["sequence_length"]
        if len(ids) > capacity:
            raise ValueError(
                f"Prompt needs {len(ids)} tokens; capacity is {capacity}; no truncation"
            )
        width = len(ids) if self.manifest.get("dynamic_sequence_length") else capacity
        input_ids = np.full((1, width), self.config["pad_token_id"], dtype=np.int32)
        input_ids[0, : len(ids)] = ids
        attention_mask = np.zeros((1, width), dtype=np.int32)
        attention_mask[0, : len(ids)] = 1
        pool_mask = np.zeros((1, width), dtype=np.float32)
        pool_mask[0, start:end] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "option_positions": np.array([option_positions], dtype=np.int32),
            "pool_mask": pool_mask,
        }, list(options)

    async def probabilities(self, state: Any, question: Any) -> dict[str, float]:
        from coreai.runtime import NDArray

        if self._function is None:
            raise RuntimeError("Use async with VelaCoreAI.open(...)")
        if self._busy:
            raise RuntimeError("This model instance is busy; await the previous request")
        arrays, labels = self.pack(state, question)
        self._busy = True
        try:
            output = await self._function({k: NDArray(v) for k, v in arrays.items()})
            logits = np.asarray(output[self.manifest["output"]].numpy(), dtype=np.float64)
            if logits.shape != (1, len(labels)) or not np.isfinite(logits).all():
                raise FloatingPointError("Invalid Core AI logits")
            probabilities = np.exp(logits[0] - logits[0].max())
            probabilities /= probabilities.sum()
            return dict(zip(labels, probabilities.tolist(), strict=True))
        finally:
            self._busy = False

    async def predict(self, state: Any, question: Any) -> dict[str, Any]:
        probabilities = await self.probabilities(state, question)
        if question["type"] == "noul":
            return {"type": "noul", "noul": probabilities["yes"]}
        return {
            "type": "choice",
            "choice": max(probabilities, key=probabilities.__getitem__),
            "probabilities": probabilities,
        }

    async def predict_questions(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Preserve DecisionEncoder.predict's question mapping and token usage."""
        packed = {key: self.pack(state, question) for key, question in questions.items()}
        answers = {}
        for key, question in questions.items():
            answers[key] = await self.predict(state, question)
        tokens = sum(int(arrays["attention_mask"].sum()) for arrays, _ in packed.values())
        return answers, tokens
