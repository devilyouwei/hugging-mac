"""Load each converted session once, evaluate cases, and release it before the next backend."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


@asynccontextmanager
async def open_backend(
    runtime: str, artifact: Path, coreai_device: str = "auto"
) -> AsyncIterator[Callable[[dict[str, Any]], Any]]:
    if runtime == "coreml":
        ct = importlib.import_module("coremltools")
        model = ct.models.MLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_ONLY)

        async def predict(inputs: dict[str, Any]) -> dict[str, Any]:
            return dict(await asyncio.to_thread(model.predict, inputs))

        yield predict
    elif runtime == "coreai":
        from hugging_mac_conversion.shared.coreai_runtime import CoreAIProvider

        session = await CoreAIProvider().create_session(artifact, device=coreai_device, options={})
        try:

            async def run(inputs: dict[str, Any]) -> dict[str, Any]:
                return dict(await asyncio.to_thread(session.run, inputs))

            yield run
        finally:
            await session.close()
    else:
        raise ValueError(f"Unsupported evaluation runtime: {runtime}")
