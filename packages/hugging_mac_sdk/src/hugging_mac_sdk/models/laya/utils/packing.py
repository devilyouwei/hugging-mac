"""Laya 0.3.4 sequence layout adapted from Apache-2.0 upstream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class LayaPacker:
    def __init__(self, root: Path) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        self.meta = json.loads((root / "metadata.json").read_text())
        self.layout = self.meta["decision"]
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        mask = self.tokenizer.id_to_token(self.layout["mask_token_id"])
        if mask is None:
            raise ValueError("Laya tokenizer is missing the mask token")
        self.mask = mask

    def encode(self, value: str) -> list[int]:
        return list(
            self.tokenizer.encode(value.replace(self.mask, " "), add_special_tokens=False).ids
        )

    def pack(self, state: Any, question: dict[str, Any]) -> tuple[list[int], list[int], list[str]]:
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice":
            if isinstance(criteria, (list, tuple)):
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict):
                raise ValueError("choice requires a criteria mapping")
            labels = list(criteria)
            options = [
                str(k) if v is None or v == "" else f"{k}: {text(v)}" for k, v in criteria.items()
            ]
        elif kind == "score":
            if not isinstance(criteria, (list, tuple)):
                raise ValueError("score requires ordered criteria")
            labels = [str(i) for i in range(len(criteria))]
            options = [f"level {i}: {text(v)}" for i, v in enumerate(criteria)]
        elif kind == "noul":
            labels = ["false", "true"]
            criteria = criteria or {}
            defaults = ["no, the statement does not hold", "yes, the statement holds"]
            options = [
                f"{k}: {text(criteria[k]) if criteria.get(k) not in (None, '') else d}"
                for k, d in zip(labels, defaults, strict=True)
            ]
        else:
            raise ValueError("Unsupported Laya question type")
        if not options:
            raise ValueError("A question needs options")
        layout = self.layout
        head = self.encode(f"{kind} question: {text(question.get('instructions', ''))}")
        opts = [
            [layout["mask_token_id"], *self.encode(" " + opt)[: layout["option_text_tokens"]]]
            for opt in options
        ]
        budget = layout["head_max_len"] - sum(map(len, opts))
        if budget < 16:
            per = max(4, (layout["head_max_len"] - 16) // len(opts))
            opts = [opt[:per] for opt in opts]
            budget = layout["head_max_len"] - sum(map(len, opts))
        ids = [layout["cls_token_id"], *head[: max(8, budget)], layout["sep_token_id"]]
        markers = []
        for opt in opts:
            markers.append(len(ids))
            ids.extend(opt)
        ids.append(layout["sep_token_id"])
        state_ids = self.encode(text(state))
        # The upstream layout budgets option text; never silently discard user state.
        if len(ids) + len(state_ids) + 1 > layout["window"]:
            raise ValueError(
                f"Input exceeds the {layout['window']}-token Core AI window; no truncation"
            )
        return ids + state_ids + [layout["sep_token_id"]], markers, labels
