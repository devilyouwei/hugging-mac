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

    def create_stream(self) -> OnnxSileroStream:
        if self._session is None:
            raise RuntimeError("Silero ONNX engine is not loaded")
        return OnnxSileroStream(self._session, self._config.chunk_samples)

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


class OnnxSileroStream:
    def __init__(self, session: RuntimeSession, chunk_samples: int) -> None:
        numpy = importlib.import_module("numpy")
        self._session = session
        self._chunk_samples = chunk_samples
        self._state = numpy.zeros((2, 1, 128), dtype=numpy.float32)
        self._context = numpy.zeros((1, 64), dtype=numpy.float32)

    async def probability(self, samples: tuple[float, ...]) -> float:
        return await asyncio.to_thread(self._run, samples)

    def _run(self, samples: tuple[float, ...]) -> float:
        if len(samples) != self._chunk_samples:
            raise ValueError("Silero streaming frame has an invalid size")
        numpy = importlib.import_module("numpy")
        chunk = numpy.asarray(samples, dtype=numpy.float32).reshape(1, -1)
        model_input = numpy.concatenate((self._context, chunk), axis=1)
        outputs = self._session.run(
            {
                "input": numpy.ascontiguousarray(model_input, dtype=numpy.float32),
                "state": self._state,
                "sr": numpy.asarray(16000, dtype=numpy.int64),
            }
        )
        probability = outputs.get("output")
        next_state = outputs.get("stateN")
        if probability is None or next_state is None:
            raise RuntimeError("Unexpected Silero ONNX output names")
        self._state = numpy.asarray(next_state, dtype=numpy.float32)
        self._context = model_input[:, -64:]
        return float(numpy.asarray(probability).reshape(-1)[0])
