"""Core AI export and execution, using conversion-owned authoring/runtime infrastructure."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .contracts import PreparedGraph


def export(graph: PreparedGraph, output: Path) -> Path:
    from hugging_mac_conversion.shared.coreai_export import convert_pytorch_to_coreai

    return convert_pytorch_to_coreai(
        graph.module,
        output / "model.aimodel",
        example_inputs=graph.args,
        dynamic_shapes=graph.dynamic_shapes,
        input_names=graph.input_names,
        output_names=[graph.output_name],
        optimize=False,
    )


async def predict(artifact: Path, inputs: dict[str, Any], device: str = "auto") -> dict[str, Any]:
    from hugging_mac_conversion.shared.coreai_runtime import CoreAIProvider

    session = await CoreAIProvider().create_session(artifact, device=device, options={})
    try:
        return dict(await asyncio.to_thread(session.run, inputs))
    finally:
        await session.close()
