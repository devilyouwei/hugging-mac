"""Runtime-independent Qwen3-TTS 0.6B Base instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.speech_synthesis import (
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
    SpeechSynthesisTimings,
)

from .config import Qwen3TtsInstanceConfig
from .utils.types import Qwen3TtsEngineOutput


class Qwen3TtsEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def infer(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput: ...

    async def close(self) -> None: ...


class Qwen3TtsInstance(BaseModelInstance):
    def __init__(
        self, config: Qwen3TtsInstanceConfig, engine: Qwen3TtsEngine, manifest: ModelManifest
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._manifest_model_id = manifest.model_id
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            SpeechSynthesis,  # type: ignore[type-abstract]
            cast(SpeechSynthesis, self),
        )

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Qwen3-TTS instance must be READY before synthesize")
        preprocess_started = perf_counter()
        preprocess_ms = (perf_counter() - preprocess_started) * 1000
        async with self._inference_lock:
            inference_started = perf_counter()
            try:
                output = await self._engine.infer(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("Qwen3-TTS synthesis failed", cause=error) from error
            inference_ms = (perf_counter() - inference_started) * 1000
        postprocess_started = perf_counter()
        return SpeechSynthesisResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            audio=output.audio,
            sample_rate=output.sample_rate,
            duration_seconds=output.duration_seconds,
            generated_tokens=output.generated_tokens,
            timings=SpeechSynthesisTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=(perf_counter() - postprocess_started) * 1000,
            ),
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
