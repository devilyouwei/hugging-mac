"""Silero engine using the generic ONNX Runtime provider."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.onnx import OnnxRuntimeProvider

from .config import SileroInstanceConfig
from .resources import SileroResourceResolver
from .utils.types import PreparedAudio


class OnnxSileroEngine:
    runtime_name = "onnx"

    def __init__(self, config: SileroInstanceConfig, resources: SileroResourceResolver) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        if self._session is not None:
            return self._session.device
        return self._config.device or "auto"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_onnx()).path

    async def load(self, artifact: Path) -> None:
        self._session = await OnnxRuntimeProvider().create_session(
            artifact, device=self._config.device, options={}
        )

    async def probabilities(self, prepared: PreparedAudio) -> tuple[float, ...]:
        if self._session is None:
            raise RuntimeError("Silero ONNX engine is not loaded")
        return await asyncio.to_thread(self._run, prepared)

    def _run(self, prepared: PreparedAudio) -> tuple[float, ...]:
        assert self._session is not None
        numpy = importlib.import_module("numpy")
        samples = numpy.asarray(prepared.samples, dtype=numpy.float32)
        chunk_size = self._config.chunk_samples
        context_size = 64
        state = numpy.zeros((2, 1, 128), dtype=numpy.float32)
        context = numpy.zeros((1, context_size), dtype=numpy.float32)
        probabilities: list[float] = []
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk = numpy.pad(chunk, (0, chunk_size - len(chunk)))
            model_input = numpy.concatenate((context, chunk[None, :]), axis=1)
            outputs = self._session.run(
                {
                    "input": numpy.ascontiguousarray(model_input, dtype=numpy.float32),
                    "state": state,
                    "sr": numpy.asarray(prepared.sample_rate, dtype=numpy.int64),
                }
            )
            probability = outputs.get("output")
            next_state = outputs.get("stateN")
            if probability is None or next_state is None:
                raise RuntimeError("Unexpected Silero ONNX output names")
            probabilities.append(float(numpy.asarray(probability).reshape(-1)[0]))
            state = numpy.asarray(next_state, dtype=numpy.float32)
            context = model_input[:, -context_size:]
        return tuple(probabilities)

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()
