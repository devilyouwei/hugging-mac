"""Audio8 neural inference entirely in Core AI, with NumPy host orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import math
from pathlib import Path
from typing import Any

from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Audio8TtsCoreAIInstanceConfig
from .utils.coreai_generation import prompt_segments, sample_token
from .utils.types import CoreAILayout, CoreAINativeMetadata, TtsEngineOutput


async def _settled_thread(function: Any, *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


class CoreAIAudio8TtsEngine:
    runtime_name: str = "coreai"

    def __init__(
        self,
        config: Audio8TtsCoreAIInstanceConfig,
        artifact: ModelArtifact,
        tokenizer: ModelArtifact,
        layout: CoreAILayout,
    ) -> None:
        self._config, self._artifact, self._tokenizer_artifact, self._layout = (
            config,
            artifact,
            tokenizer,
            layout,
        )
        self._slow: CoreAISession | None = None
        self._fast: CoreAISession | None = None
        self._codec: CoreAISession | None = None
        self._encoder: CoreAISession | None = None
        self._tokenizer: Any = None
        self._metadata: CoreAINativeMetadata | None = None
        self._path: Path | None = None
        self._lock = asyncio.Lock()

    @property
    def device(self) -> str:
        return self._config.device

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve(self) -> Path:
        path = self._config.source_path or self._artifact.resolve(self._config.model_home)
        if not artifact_available(self._artifact, path):
            raise ResourceNotFoundError(
                "Audio8 Core AI artifact is incomplete; download or convert it first"
            )
        if not artifact_available(self._tokenizer_artifact, self.tokenizer_path):
            raise ResourceNotFoundError("Audio8 shared tokenizer is incomplete")
        return path

    async def load(self, artifact: Path) -> None:
        async with self._lock:
            try:
                self._metadata = CoreAINativeMetadata.model_validate(
                    json.loads((artifact / self._layout.config_file).read_text())
                )
                self._tokenizer = importlib.import_module("tokenizers").Tokenizer.from_file(
                    str(self.tokenizer_path / "tokenizer.json")
                )
                provider = CoreAIProvider()
                for attr, graph in (
                    ("_slow", self._layout.slow_graph),
                    ("_fast", self._layout.fast_graph),
                    ("_codec", self._layout.codec_graph),
                ):
                    setattr(
                        self,
                        attr,
                        await provider.create_session(
                            artifact / graph, device=self.device, options={}
                        ),
                    )
                self._path = artifact
            except BaseException:
                await self._close()
                raise

    def _reference_codes(self, request: SpeechSynthesisRequest) -> Any:
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
        assert (
            request.reference_audio is not None
            and self._metadata is not None
            and self._encoder is not None
        )
        source = request.reference_audio.path or io.BytesIO(request.reference_audio.data or b"")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Reference audio must be nonempty and finite")
        target = self._metadata.sample_rate
        if rate != target:
            divisor = math.gcd(rate, target)
            audio = (
                importlib.import_module("scipy.signal")
                .resample_poly(audio, target // divisor, rate // divisor)
                .astype(np.float32)
            )
        frame = self._metadata.frame_length
        frames = (len(audio) + frame - 1) // frame
        if frames > self._layout.codec_max_frames:
            raise ValueError("Reference audio exceeds the converted encoder frame limit")
        audio = np.pad(audio, (0, frames * frame - len(audio)))
        return self._encoder.run({"audio": audio.reshape(1, 1, -1)})["codes"]

    def _infer_sync(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        np = importlib.import_module("numpy")
        meta = self._metadata
        if meta is None or self._slow is None or self._fast is None or self._codec is None:
            raise RuntimeError("Audio8 Core AI model is not loaded")
        prefix, suffix = prompt_segments(self._tokenizer, request.text, request.reference_text)
        reference = self._reference_codes(request) if request.reference_audio is not None else None
        count = 0 if reference is None else reference.shape[-1]
        width = len(prefix) + count + len(suffix)
        if width >= meta.max_seq_len:
            raise ValueError("Prompt exceeds the converted Slow AR context limit")
        prompt = np.zeros((1, meta.num_codebooks + 1, width), dtype=np.int32)
        prompt[0, 0, : len(prefix)] = prefix
        if suffix:
            prompt[0, 0, -len(suffix) :] = suffix
        if reference is not None:
            prompt[:, 1:, len(prefix) : len(prefix) + count] = reference
            prompt[:, 0, len(prefix) : len(prefix) + count] = (
                reference[:, 0] + meta.semantic_begin_id
            )
        self._slow.reset_state(
            {name: np.zeros(shape, np.float32) for name, shape in meta.slow_states.items()}
        )
        self._fast.reset_state(
            {name: np.zeros(shape, np.float32) for name, shape in meta.fast_states.items()}
        )
        output = None
        for position in range(width):
            output = self._slow.run(
                {
                    "ids": prompt[:, :, position : position + 1],
                    "position": np.asarray([position], np.int32),
                }
            )
        assert output is not None
        rng = np.random.default_rng()
        previous = None
        frames = []

        def sample(
            logits: Any, *, top_p: float = request.top_p, temperature: float = request.temperature
        ) -> int:
            return sample_token(
                logits,
                top_k=request.top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=request.do_sample,
                rng=rng,
            )

        for step in range(min(request.max_new_tokens, meta.max_seq_len - width)):
            semantic = sample(output["logits"])
            if request.do_sample:
                alternate = sample(
                    output["logits"], top_p=meta.ras_top_p, temperature=meta.ras_temperature
                )
                if previous is not None and semantic < meta.codebook_size and semantic in previous:
                    semantic = alternate
            if semantic == meta.codebook_size:
                break
            hidden = output["hidden"]
            current = semantic
            codes = [current]
            for position in range(meta.num_codebooks):
                result = self._fast.run(
                    {
                        "hidden": hidden,
                        "token": np.asarray([current], np.int32),
                        "position": np.asarray([position], np.int32),
                    }
                )
                if position:
                    current = sample(result["logits"])
                    codes.append(current)
            frames.append(codes)
            if previous is None:
                previous = np.zeros(meta.ras_window_size, np.int32)
            else:
                previous = np.roll(previous, -1)
                previous[-1] = semantic
            if step + 1 < min(request.max_new_tokens, meta.max_seq_len - width):
                ids = np.asarray([semantic + meta.semantic_begin_id, *codes], np.int32).reshape(
                    1, -1, 1
                )
                output = self._slow.run(
                    {"ids": ids, "position": np.asarray([width + step], np.int32)}
                )
        if frames:
            codes = np.asarray(frames, np.int32).T[None]
            audio = self._codec.run({"codes": codes})["waveform"].reshape(-1)
        else:
            audio = np.empty(0, np.float32)
        return TtsEngineOutput(
            audio=audio.astype("<f4", copy=False).tobytes(),
            sample_rate=meta.sample_rate,
            duration_seconds=len(audio) / meta.sample_rate,
            generated_tokens=len(frames),
        )

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        async with self._lock:
            try:
                if request.reference_audio is not None and self._encoder is None:
                    if self._path is None:
                        raise RuntimeError("Audio8 Core AI model is not loaded")
                    self._encoder = await CoreAIProvider().create_session(
                        self._path / self._layout.encoder_graph, device=self.device, options={}
                    )
                return await _settled_thread(self._infer_sync, request)  # type: ignore[no-any-return]
            finally:
                for session in (self._slow, self._fast):
                    if session is not None:
                        await _settled_thread(session.reset_state, {})

    async def _close(self) -> None:
        sessions = self._slow, self._fast, self._codec, self._encoder
        self._slow = self._fast = self._codec = self._encoder = None
        self._tokenizer = self._metadata = self._path = None
        results = await asyncio.gather(
            *(s.close() for s in sessions if s is not None), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def close(self) -> None:
        async with self._lock:
            await self._close()
