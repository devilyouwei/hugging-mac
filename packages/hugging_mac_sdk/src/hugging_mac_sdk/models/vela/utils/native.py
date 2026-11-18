"""Shared native component execution for full and decision-only Vela bundles."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from hugging_mac_sdk.runtime.coreai import CoreAIProvider


class NativeComponents:
    def __init__(self, root: Path, manifest: dict[str, Any], device: str) -> None:
        self.root, self.manifest, self.device = root, manifest, device
        self.name: str | None = None
        self.session: Any = None
        self.used: set[str] = set()

    def close(self) -> None:
        if self.session is not None:
            asyncio.run(self.session.close())
        self.name, self.session = None, None

    def load(self, name: str) -> None:
        if self.name != name:
            from safetensors import safe_open

            self.close()
            spec = self.manifest["components"][name]
            self.session = asyncio.run(
                CoreAIProvider().create_session(
                    self.root / spec["graph"] / "model.aimodel", device=self.device, options={}
                )
            )
            try:
                with ExitStack() as stack:
                    handles = {
                        filename: stack.enter_context(
                            safe_open(str(self.root / "weights" / filename), framework="numpy")
                        )
                        for filename in {row["file"] for row in spec["states"].values()}
                    }
                    values = {
                        key: handles[row["file"]].get_tensor(row["tensor"])
                        for key, row in spec["states"].items()
                    }
                    for key, value in values.items():
                        if (
                            list(value.shape) != spec["states"][key]["shape"]
                            or value.dtype != np.float32
                        ):
                            raise ValueError(f"Unexpected source weight format: {key}")
                    if self.manifest.get("weight_binding", "state") == "inputs":
                        self.session.set_constant_inputs(values)
                    else:
                        self.session.reset_state(values)
                    del values  # Release source mappings before native execution.
                del handles, value
                self.name = name
            except BaseException:
                self.close()
                raise

    def run(self, name: str, *args: Any) -> tuple[Any, Any]:
        self.load(name)
        names = ("input_ids", "attention_mask", "q_index", "opt_index", "unit_index", "ent_index")
        inputs = {
            key: np.asarray(value, dtype=np.int32) for key, value in zip(names, args, strict=True)
        }
        result = self.session.run(inputs)
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError(f"Nonfinite Core AI output from {name}")
        self.used.add(name)
        return result["logits"], result["span_logits"]
