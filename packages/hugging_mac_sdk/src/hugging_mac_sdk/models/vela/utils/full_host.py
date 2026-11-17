"""Original Vela routing/decoding with injected neural backends and lazy weights."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .full_upstream import load_full_upstream


def adapter_files(root: Path) -> dict[str, str]:
    cfg = json.loads((root / "vela3_config.json").read_text())
    result = {"span": cfg["adapters"]["span"]["file"], "router": cfg["adapters"]["router"]["file"]}
    result["evidence"] = cfg["span"]["evidence"]["adapter"]
    result.update(cfg["span"]["task_adapters"])
    result["span_set"] = cfg["span"]["set_adapter"]["file"]
    for task, spec in cfg["router"]["task_adapters"].items():
        for index, filename in enumerate(spec.get("files") or [spec["file"]]):
            result[f"router_{task}_{index}"] = filename
    return result


def make_host(
    root: Path, decision: Any, backend: Callable[[str], Any], *, entity_route: str | None = None
) -> Any:
    upstream = load_full_upstream(root)
    vs = upstream.schema
    host = upstream.Vela3Encoder.__new__(upstream.Vela3Encoder)
    host.decision = decision
    host.cfg = json.loads((root / "vela3_config.json").read_text())
    host.router_q = {
        upstream._norm(text): spec
        for text, spec in json.loads((root / "router_questions.json").read_text())[
            "questions"
        ].items()
    }

    def engine(name: str, *, max_len: int = 8192, tokenizer: Any = None) -> Any:
        value = vs.SchemaEngine(
            root,
            None,
            "cpu",
            tokenizer=tokenizer,
            max_len=max_len,
            token_budget=max(max_len, 32768),
        )
        value.run = backend(name)
        return value

    host.span = engine("span")
    host.span_ev = engine(
        "evidence", max_len=host.cfg["span"]["evidence"]["max_len"], tokenizer=host.span.tok
    )
    host.task = {
        name: engine(name, tokenizer=host.span.tok) for name in host.cfg["span"]["task_adapters"]
    }
    host.entity_route = (
        entity_route
        or os.environ.get("VELA3_ENTITY_ROUTE")
        or host.cfg["span"].get("entity_route", "ner")
    )
    host.span_set = engine("span_set", tokenizer=host.span.tok)
    host.router = engine("router", tokenizer=host.span.tok)
    host.router_task = {}
    for task, spec in host.cfg["router"]["task_adapters"].items():
        engines = [
            engine(f"router_{task}_{index}", tokenizer=host.span.tok)
            for index, _ in enumerate(spec.get("files") or [spec["file"]])
        ]
        for key in spec["ids"]:
            host.router_task[key] = (task, engines)
    host._VS = vs
    host.name = host.cfg["name"]
    return host
