"""NumPy/tokenizers decision rendering adapted from Apache-2.0 upstream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]


class DecisionPacker:
    noul_criteria = False

    def __init__(self, root: Path) -> None:
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.component_manifest = None
        if self.manifest.get("format") == "vela-full-coreai/1":
            self.component_manifest = self.manifest
            self.noul_criteria = True
            config = json.loads((root / "config.json").read_text())
            self.config = {
                key: config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id")
            }
            self.config["marker_ids"] = config["decision"]["marker_ids"]
            self.manifest = {
                "artifact_scope": "decision-only",
                "dynamic_sequence_length": True,
                "sequence_length": self.component_manifest["sequence_length"]["decision"],
                "function": "main",
                "output": "logits",
            }
        else:
            self.config = json.loads((root / "decision.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        if self.manifest.get("artifact_scope") != "decision-only":
            raise ValueError("Expected a Vela decision-only bundle")

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
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
            criteria = (question.get("criteria") or {}) if self.noul_criteria else {}
            if not isinstance(criteria, dict):
                raise ValueError("noul criteria must be a true/false mapping")
            options = {"no": criteria.get("false", ""), "yes": criteria.get("true", "")}
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
