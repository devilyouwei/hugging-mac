"""Runtime-independent Audio8-TTS MLX BF16 synthesis instance."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
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
    AUDIO8_TTS_MLX_BF16_MODEL_ID,
    AUDIO8_TTS_MLX_BF16_REVISION,
    Audio8TtsMlxBf16InstanceConfig,
)
from .utils.types import TtsEngineOutput


class Audio8TtsMlxBf16Engine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...

    def resolve(self) -> Awaitable[Path]: ...
    def load(self, artifact: Path) -> Awaitable[None]: ...
    def infer(self, request: SpeechSynthesisRequest) -> Awaitable[TtsEngineOutput]: ...
    def close(self) -> Awaitable[None]: ...


class Audio8TtsMlxBf16Instance(BaseModelInstance):
    def __init__(
        self, config: Audio8TtsMlxBf16InstanceConfig, engine: Audio8TtsMlxBf16Engine
    ) -> None:
        super().__init__(
            model_id=AUDIO8_TTS_MLX_BF16_MODEL_ID,
            revision=AUDIO8_TTS_MLX_BF16_REVISION,
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
            raise InferenceError("Audio8-TTS MLX BF16 instance must be READY before synthesize")
        async with self._inference_lock:
            started = perf_counter()
            try:
                output = await self._engine.infer(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Audio8-TTS MLX BF16 synthesis failed",
                    details={"reason": _reason(error)},
                    cause=error,
                ) from error
            inference_ms = (perf_counter() - started) * 1000
        return SpeechSynthesisResponse(
            model_id=AUDIO8_TTS_MLX_BF16_MODEL_ID,
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


def _reason(error: Exception) -> str:
    return (" ".join(str(error).strip().split()) or type(error).__name__)[:500]
