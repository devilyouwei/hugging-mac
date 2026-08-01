"""Runtime-independent Audio8-TTS ONNX INT4 synthesis instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.speech_synthesis import (
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
    SpeechSynthesisTimings,
)

from .config import (
    AUDIO8_TTS_ONNX_INT4_MODEL_ID,
    AUDIO8_TTS_ONNX_INT4_REVISION,
    Audio8TtsOnnxInt4InstanceConfig,
)
from .utils.types import TtsEngineOutput


class Audio8TtsOnnxInt4Engine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput: ...

    async def close(self) -> None: ...


class Audio8TtsOnnxInt4Instance(BaseModelInstance):
    def __init__(
        self,
        config: Audio8TtsOnnxInt4InstanceConfig,
        engine: Audio8TtsOnnxInt4Engine,
    ) -> None:
        super().__init__(
            model_id=AUDIO8_TTS_ONNX_INT4_MODEL_ID,
            revision=AUDIO8_TTS_ONNX_INT4_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            SpeechSynthesis,  # type: ignore[type-abstract]
            cast(SpeechSynthesis, self),
        )

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Audio8-TTS ONNX INT4 instance must be READY before synthesize")
        async with self._inference_lock:
            inference_started = perf_counter()
            try:
                output = await self._engine.infer(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Audio8-TTS ONNX INT4 synthesis failed",
                    details={"reason": _inference_reason(error)},
                    cause=error,
                ) from error
            inference_ms = (perf_counter() - inference_started) * 1000
        return SpeechSynthesisResponse(
            model_id=AUDIO8_TTS_ONNX_INT4_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            audio=output.audio,
            sample_rate=output.sample_rate,
            duration_seconds=output.duration_seconds,
            generated_tokens=output.generated_tokens,
            timings=SpeechSynthesisTimings(inference_ms=inference_ms),
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()


def _inference_reason(error: Exception) -> str:
    """Return a bounded local-runtime reason suitable for the API response."""

    message = str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)
    message = " ".join(message.strip().split())
    return (message or type(error).__name__)[:500]
