"""Audio8-TTS engine for the official ONNX INT4 export."""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import (
    AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
    AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES,
    AUDIO8_TTS_ONNX_INT4_SAMPLE_RATE,
    Audio8TtsOnnxInt4InstanceConfig,
)
from .resources import Audio8TtsOnnxInt4ResourceResolver
from .utils.types import TtsEngineOutput


class OnnxAudio8TtsInt4Engine:
    runtime_name = "onnx"

    def __init__(
        self,
        config: Audio8TtsOnnxInt4InstanceConfig,
        resources: Audio8TtsOnnxInt4ResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._artifact: Path | None = None
        from .utils.voices import VoiceStore

        self._runtime: Any | None = None
        self._voices = VoiceStore(
            config.voice_home,
            num_codebooks=10,
            fingerprint=AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
        )

    @property
    def device(self) -> str:
        return "cpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        try:
            import onnxruntime  # type: ignore[import-untyped]  # noqa: F401
            import tokenizers  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Audio8-TTS ONNX dependencies are missing; install hugging-mac-sdk[tts]"
            ) from error
        self._artifact = artifact
        await asyncio.to_thread(self._load_runtime_sync)

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        return await asyncio.to_thread(self._infer_sync, request)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_runtime_sync)
        self._artifact = None

    def _load_runtime_sync(self) -> None:
        from .utils.runtime import Audio8OnnxRuntime

        if self._artifact is None:
            raise RuntimeError("Audio8-TTS ONNX artifact has not been resolved")
        self._runtime = Audio8OnnxRuntime(
            self._artifact,
            threads=self._config.threads,
        )

    def _close_runtime_sync(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()
            del runtime
            gc.collect()

    def _infer_sync(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        import numpy as np

        if self._runtime is None or self._artifact is None:
            raise RuntimeError("Audio8-TTS ONNX engine is not loaded")

        if request.reference_audio is not None:
            missing = [
                name
                for name in AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES
                if not (self._artifact / name).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "reference audio requires the optional ONNX registration files; "
                    f"missing: {', '.join(missing)}"
                )
            # The official runtime avoids holding the online sessions and the
            # registration encoder at the same time. This keeps peak memory near
            # the upstream measurements instead of summing both model footprints.
            self._close_runtime_sync()
            try:
                from .utils.registration import encode_reference_audio

                reference_codes = encode_reference_audio(self._artifact, request.reference_audio)
                assert request.reference_text is not None
                reference_text = request.reference_text
                if request.voice is not None:
                    self._voices.save(request.voice, reference_codes, reference_text)
            finally:
                gc.collect()
                self._load_runtime_sync()
        elif request.voice is not None:
            reference_codes, reference_text = self._voices.load(request.voice)
        else:
            raise ValueError(
                "Audio8-TTS ONNX requires either reference_audio + reference_text "
                "or the name of a previously saved voice profile"
            )

        assert self._runtime is not None
        waveform, codes = self._runtime.synthesize(
            text=request.text,
            reference_text=reference_text,
            reference_codes=reference_codes,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=max(1, request.top_k),
            do_sample=request.do_sample,
        )
        audio = np.ascontiguousarray(waveform, dtype="<f4")
        return TtsEngineOutput(
            audio=audio.tobytes(),
            sample_rate=AUDIO8_TTS_ONNX_INT4_SAMPLE_RATE,
            duration_seconds=audio.size / AUDIO8_TTS_ONNX_INT4_SAMPLE_RATE,
            generated_tokens=int(codes.shape[1]),
        )
