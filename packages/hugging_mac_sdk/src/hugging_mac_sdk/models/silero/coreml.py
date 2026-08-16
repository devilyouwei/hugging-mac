"""Silero engine backed by a precompiled Core ML streaming model."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider

from .config import SileroInstanceConfig
from .resources import SileroResourceResolver
from .utils.types import PreparedAudio


class CoreMlSileroEngine:
    runtime_name = "coreml"

    def __init__(self, config: SileroInstanceConfig, resources: SileroResourceResolver) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        if self._session is not None:
            return self._session.device
        return self._config.device or "all"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_coreml()).path

    async def load(self, artifact: Path) -> None:
        self._session = await CoreMLProvider().create_session(
            artifact,
            device=self._config.device,
            options={},
        )

    async def probabilities(self, prepared: PreparedAudio) -> tuple[float, ...]:
        if self._session is None:
            raise RuntimeError("Silero Core ML engine is not loaded")
        return await asyncio.to_thread(self._run, prepared)

    def _run(self, prepared: PreparedAudio) -> tuple[float, ...]:
        assert self._session is not None
        numpy = importlib.import_module("numpy")
        samples = numpy.asarray(prepared.samples, dtype=numpy.float32)
        chunk_size = self._config.chunk_samples
        context_size = 64
        h = numpy.zeros((1, 1, 128), dtype=numpy.float32)
        c = numpy.zeros((1, 1, 128), dtype=numpy.float32)
        context = numpy.zeros(context_size, dtype=numpy.float32)
        probabilities: list[float] = []
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk = numpy.pad(chunk, (0, chunk_size - len(chunk)))
            audio = numpy.concatenate((context, chunk)).reshape(1, 1, -1)
            outputs = self._session.run({"audio": audio, "h": h, "c": c})
            probability = outputs.get("probability")
            h_out = outputs.get("h_out")
            c_out = outputs.get("c_out")
            if probability is None or h_out is None or c_out is None:
                raise RuntimeError("Unexpected Silero Core ML output names")
            probabilities.append(float(numpy.asarray(probability).reshape(-1)[0]))
            h = numpy.asarray(h_out, dtype=numpy.float32)
            c = numpy.asarray(c_out, dtype=numpy.float32)
            context = chunk[-context_size:]
        return tuple(probabilities)

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
