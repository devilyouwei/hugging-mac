"""DeepFilterNet3 lifecycle and speech-enhancement capability."""

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import SpeechEnhancement
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.speech_enhancement import (
    SpeechEnhancementRequest,
    SpeechEnhancementResponse,
    SpeechEnhancementTimings,
)

from .audio import PreparedEnhancementAudio, encode_wav, prepare_audio
from .config import DEEPFILTERNET3_MODEL_ID, DEEPFILTERNET3_REVISION, DeepFilterNet3InstanceConfig
from .dsp import EnhancementOutput


class DeepFilterNet3Engine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def enhance(self, prepared: PreparedEnhancementAudio) -> EnhancementOutput: ...
    async def close(self) -> None: ...


class DeepFilterNet3Instance(BaseModelInstance):
    def __init__(self, config: DeepFilterNet3InstanceConfig, engine: DeepFilterNet3Engine) -> None:
        super().__init__(
            model_id=DEEPFILTERNET3_MODEL_ID,
            revision=DEEPFILTERNET3_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._config = config
        self._engine = engine
        self.register_capability(
            SpeechEnhancement,  # type: ignore[type-abstract]
            cast(SpeechEnhancement, self),
        )

    async def enhance_speech(
        self, request: SpeechEnhancementRequest
    ) -> SpeechEnhancementResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("DeepFilterNet3 instance must be READY before inference")

        started = perf_counter()
        try:
            prepared = await asyncio.to_thread(prepare_audio, request.audio)
        except Exception as error:
            raise InferenceError(
                "DeepFilterNet3 audio preprocessing failed", cause=error
            ) from error
        preprocess_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        try:
            enhanced = await self._engine.enhance(prepared)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise InferenceError("DeepFilterNet3 inference failed", cause=error) from error
        inference_ms = (perf_counter() - started) * 1000

        output_sample_rate = request.output_sample_rate or prepared.source_sample_rate
        started = perf_counter()
        try:
            encoded = await asyncio.to_thread(
                encode_wav,
                enhanced.samples,
                output_sample_rate=output_sample_rate,
            )
        except Exception as error:
            raise InferenceError(
                "DeepFilterNet3 audio postprocessing failed", cause=error
            ) from error
        postprocess_ms = (perf_counter() - started) * 1000
        return SpeechEnhancementResponse(
            model_id=DEEPFILTERNET3_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            audio=encoded,
            sample_rate=output_sample_rate,
            duration_seconds=prepared.duration_seconds,
            timings=SpeechEnhancementTimings(
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
