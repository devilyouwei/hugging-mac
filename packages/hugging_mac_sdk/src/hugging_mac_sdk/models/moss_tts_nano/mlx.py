"""MLX-Audio runtime for MOSS-TTS-Nano."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.mlx import MlxProvider, MlxSession
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput

from .config import MossTtsNanoInstanceConfig
from .resources import MossTtsNanoResourceResolver
from .utils.types import MossTtsNanoEngineOutput


class MlxMossTtsNanoEngine:
    runtime_name = "mlx"

    def __init__(
        self, config: MossTtsNanoInstanceConfig, resources: MossTtsNanoResourceResolver
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None
        self._session: MlxSession | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_audio_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        try:
            from mlx_audio.tts.utils import load  # type: ignore[import-untyped]
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "MOSS-TTS-Nano requires mlx-audio with moss_tts_nano support"
            ) from error

        def _load(source: Path, _mlx: Any) -> Any:
            return load(str(source), model_type="moss_tts_nano")

        try:
            self._session = await MlxProvider().create_session(
                artifact, device=None, options={"model_loader": _load}
            )
            self._model = self._session.value
        except BaseException:
            await self.close()
            raise

    async def infer(self, request: SpeechSynthesisRequest) -> MossTtsNanoEngineOutput:
        cancelled = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._infer_sync, request, cancelled))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            with contextlib.suppress(Exception):
                await asyncio.shield(worker)
            raise

    async def close(self) -> None:
        self._model = None
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    def _infer_sync(
        self, request: SpeechSynthesisRequest, cancelled: threading.Event
    ) -> MossTtsNanoEngineOutput:
        import numpy as np

        if self._model is None:
            raise RuntimeError("MOSS-TTS-Nano MLX engine is not loaded")
        reference_path, temporary = self._reference_path(request.reference_audio)
        try:
            generation_options: dict[str, object] = {
                "text": request.text,
                "mode": "voice_clone" if reference_path is not None else "continuation",
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "do_sample": request.do_sample,
                "audio_tokenizer_source": str(self._resources.audio_tokenizer_path),
                "audio_tokenizer_device": "cpu",
            }
            if reference_path is not None:
                generation_options["ref_audio"] = str(reference_path)
            results = self._model.generate(**generation_options)
            result = next(results)
            if cancelled.is_set():
                raise RuntimeError("MOSS-TTS-Nano synthesis was cancelled")
            waveform = np.asarray(result.audio, dtype=np.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1 if waveform.shape[1] <= 2 else 0)
            waveform = np.ascontiguousarray(waveform.reshape(-1), dtype="<f4")
            sample_rate = int(result.sample_rate)
            if not waveform.size:
                raise RuntimeError("MOSS-TTS-Nano produced no audio")
            return MossTtsNanoEngineOutput(
                audio=waveform.tobytes(),
                sample_rate=sample_rate,
                duration_seconds=waveform.size / sample_rate,
                generated_tokens=int(getattr(result, "token_count", 0)),
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _reference_path(source: AudioInput | None) -> tuple[Path | None, Path | None]:
        if source is None:
            return None, None
        if source.path is not None:
            return source.path, None
        assert source.data is not None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(source.data)
            path = Path(handle.name)
        return path, path
