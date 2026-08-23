"""Qwen3-ASR Core ML artifact lifecycle instance."""

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import SpeechTranscription
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionTimings,
)

from .config import Qwen3AsrCoreMlInstanceConfig
from .utils.audio import prepare_audio
from .utils.types import AsrEngineOutput, PreparedAudio


class Qwen3AsrCoreMlEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput: ...
    async def close(self) -> None: ...


class Qwen3AsrCoreMlInstance(BaseModelInstance):
    def __init__(
        self,
        config: Qwen3AsrCoreMlInstanceConfig,
        engine: Qwen3AsrCoreMlEngine,
        manifest: ModelManifest,
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._engine = engine
        self._manifest_model_id = manifest.model_id
        self._config = config
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            SpeechTranscription,  # type: ignore[type-abstract]
            cast(SpeechTranscription, self),
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Qwen3-ASR instance must be READY before transcribe")
        started = perf_counter()
        try:
            prepared = await asyncio.to_thread(
                prepare_audio,
                request.audio,
                sample_rate=self._config.sample_rate,
                max_seconds=self._config.max_audio_seconds,
            )
        except Exception as error:
            raise InferenceError("Qwen3-ASR audio preprocessing failed", cause=error) from error
        preprocess_ms = (perf_counter() - started) * 1000
        async with self._inference_lock:
            started = perf_counter()
            try:
                output = await self._engine.infer(prepared, request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("Qwen3-ASR transcription failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        text = output.text.strip()
        postprocess_ms = (perf_counter() - started) * 1000
        return TranscriptionResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=text,
            sample_rate=prepared.sample_rate,
            duration_seconds=prepared.duration_seconds,
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            timings=TranscriptionTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
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
