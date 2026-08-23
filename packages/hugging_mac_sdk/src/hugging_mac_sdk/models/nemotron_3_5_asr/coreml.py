"""Core ML streaming RNN-T runtime for Nemotron 3.5 ASR."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.coreml import CoreMLProvider, CoreMLSession
from hugging_mac_sdk.schemas.transcription import TranscriptionRequest

from .config import NemotronCoreMlInstanceConfig
from .resources import NemotronCoreMlResourceResolver
from .utils.types import AsrEngineOutput, PreparedAudio, StreamingAsrEngineOutput


def _output(outputs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in outputs:
            return outputs[name]
    raise RuntimeError(f"Core ML output missing one of {names}; got {tuple(outputs)}")


@dataclass(slots=True)
class _StreamState:
    cache_channel: Any
    cache_time: Any
    cache_len: Any
    mel_cache: Any
    h_state: Any
    c_state: Any
    pending_samples: Any
    blank: int
    token: int
    token_ids: list[int]
    text: str = ""


class CoreMlNemotronEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: NemotronCoreMlInstanceConfig, resources: NemotronCoreMlResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._provider = CoreMLProvider()
        self._sessions: dict[str, CoreMLSession] = {}
        self._metadata: dict[str, Any] = {}
        self._vocab: dict[int, str] = {}
        self._streams: dict[str, _StreamState] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron-coreml")

    @property
    def device(self) -> str:
        return self._config.device or "all"

    async def resolve(self) -> Path:
        return (await self._resources.resolve()).path

    async def load(self, artifact: Path) -> None:
        self._metadata = json.loads((artifact / "metadata.json").read_text())
        raw_vocab = json.loads((artifact / "tokenizer.json").read_text())
        self._vocab = {int(key): str(value) for key, value in raw_vocab.items()}
        try:
            for name in ("preprocessor", "encoder", "decoder_joint"):
                self._sessions[name] = await self._provider.create_session(
                    artifact / f"{name}.mlmodelc",
                    device=self._config.device,
                    options={},
                    executor=self._executor,
                )
        except BaseException:
            await self.close()
            raise

    async def infer(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput:
        del request
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._infer_sync, prepared)

    async def start_stream(self, session_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._start_stream_sync, session_id)

    async def infer_stream(
        self, session_id: str, prepared: PreparedAudio
    ) -> StreamingAsrEngineOutput:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._infer_stream_sync,
            session_id,
            prepared,
            False,
        )

    async def finish_stream(self, session_id: str) -> StreamingAsrEngineOutput:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._finish_stream_sync,
            session_id,
        )

    async def cancel_stream(self, session_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._streams.pop, session_id, None)

    def _infer_sync(self, prepared: PreparedAudio) -> AsrEngineOutput:
        state = self._new_stream_state()
        streamed = self._decode_samples(state, prepared.samples, final=True)
        return AsrEngineOutput(
            text=streamed.text,
            generated_tokens=streamed.generated_tokens,
            detected_language=streamed.detected_language,
        )

    def _start_stream_sync(self, session_id: str) -> None:
        if session_id in self._streams:
            raise RuntimeError(f"Nemotron stream already exists: {session_id}")
        self._streams[session_id] = self._new_stream_state()

    def _infer_stream_sync(
        self,
        session_id: str,
        prepared: PreparedAudio,
        final: bool,
    ) -> StreamingAsrEngineOutput:
        try:
            state = self._streams[session_id]
        except KeyError as error:
            raise RuntimeError(f"Unknown Nemotron stream: {session_id}") from error
        return self._decode_samples(state, prepared.samples, final=final)

    def _finish_stream_sync(self, session_id: str) -> StreamingAsrEngineOutput:
        try:
            state = self._streams.pop(session_id)
        except KeyError as error:
            raise RuntimeError(f"Unknown Nemotron stream: {session_id}") from error
        np = importlib.import_module("numpy")
        return self._decode_samples(
            state,
            np.empty((0,), dtype=np.float32),
            final=True,
        )

    def _new_stream_state(self) -> _StreamState:
        np = importlib.import_module("numpy")
        meta = self._metadata
        pre_cache = int(meta["pre_encode_cache"])
        blank = int(meta["blank_idx"])
        return _StreamState(
            cache_channel=np.zeros(meta["cache_channel_shape"], dtype=np.float32),
            cache_time=np.zeros(meta["cache_time_shape"], dtype=np.float32),
            cache_len=np.ones((1,), dtype=np.int32),
            mel_cache=np.zeros((1, int(meta["mel_features"]), pre_cache), dtype=np.float32),
            h_state=np.zeros(
                (int(meta["decoder_layers"]), 1, int(meta["decoder_hidden"])),
                dtype=np.float32,
            ),
            c_state=np.zeros(
                (int(meta["decoder_layers"]), 1, int(meta["decoder_hidden"])),
                dtype=np.float32,
            ),
            pending_samples=np.empty((0,), dtype=np.float32),
            blank=blank,
            token=blank,
            token_ids=[],
        )

    def _decode_samples(
        self,
        state: _StreamState,
        incoming: Any,
        *,
        final: bool,
    ) -> StreamingAsrEngineOutput:
        np = importlib.import_module("numpy")
        meta = self._metadata
        chunk_frames = int(meta["chunk_mel_frames"])
        chunk_samples = chunk_frames * 160
        samples = np.concatenate((state.pending_samples, np.asarray(incoming, dtype=np.float32)))
        complete_samples = samples.size - samples.size % chunk_samples
        valid_samples = samples.size
        if final and samples.size > complete_samples:
            process_samples = np.pad(samples, (0, chunk_samples - samples.size % chunk_samples))
            state.pending_samples = np.empty((0,), dtype=np.float32)
        else:
            process_samples = samples[:complete_samples]
            state.pending_samples = samples[complete_samples:].copy()
        for start in range(0, process_samples.size, chunk_samples):
            chunk = process_samples[start : start + chunk_samples][None, :]
            chunk_valid_samples = min(chunk_samples, max(0, valid_samples - start))
            prep = self._sessions["preprocessor"].run(
                {
                    "audio": chunk,
                    "audio_length": np.asarray([chunk_valid_samples], dtype=np.int32),
                }
            )
            mel = np.asarray(_output(prep, "mel", "features", "processed_signal"), dtype=np.float32)
            mel = mel[..., :chunk_frames]
            encoder_input = np.concatenate((state.mel_cache, mel), axis=-1)
            enc = self._sessions["encoder"].run(
                {
                    "mel": encoder_input,
                    "mel_length": np.asarray([encoder_input.shape[-1]], dtype=np.int32),
                    "cache_channel": state.cache_channel,
                    "cache_time": state.cache_time,
                    "cache_len": state.cache_len,
                    "prompt_id": np.asarray(
                        [int(meta.get("default_prompt_id", 101))], dtype=np.int32
                    ),
                }
            )
            encoded = np.asarray(_output(enc, "encoded"), dtype=np.float32)
            state.cache_channel = np.asarray(_output(enc, "cache_channel_out"), dtype=np.float32)
            state.cache_time = np.asarray(_output(enc, "cache_time_out"), dtype=np.float32)
            state.cache_len = np.asarray(_output(enc, "cache_len_out"), dtype=np.int32)
            state.mel_cache = mel[..., -int(meta["pre_encode_cache"]) :].copy()
            for frame in range(encoded.shape[-1]):
                encoder_step = encoded[:, :, frame : frame + 1]
                for _ in range(10):
                    result = self._sessions["decoder_joint"].run(
                        {
                            "token": np.asarray([[state.token]], dtype=np.int32),
                            "token_length": np.ones((1,), dtype=np.int32),
                            "h_in": state.h_state,
                            "c_in": state.c_state,
                            "encoder": encoder_step,
                        }
                    )
                    predicted = int(np.argmax(np.asarray(_output(result, "logits"))))
                    if predicted == state.blank:
                        break
                    state.token_ids.append(predicted)
                    state.token = predicted
                    state.h_state = np.asarray(_output(result, "h_out"), dtype=np.float32)
                    state.c_state = np.asarray(_output(result, "c_out"), dtype=np.float32)
        language_tags = set(map(int, meta.get("lang_tag_token_ids", ())))
        text = "".join(
            self._vocab.get(index, "") for index in state.token_ids if index not in language_tags
        )
        text = " ".join(text.replace("▁", " ").split())
        delta = text[len(state.text) :] if text.startswith(state.text) else text
        state.text = text
        detected = next(
            (
                self._vocab.get(index, "").strip("<>")
                for index in state.token_ids
                if index in language_tags
            ),
            None,
        )
        return StreamingAsrEngineOutput(
            text=text,
            delta=delta,
            generated_tokens=len(state.token_ids),
            detected_language=detected,
        )

    async def close(self) -> None:
        sessions, self._sessions = tuple(self._sessions.values()), {}
        await asyncio.gather(*(session.close() for session in sessions))
        self._metadata, self._vocab, self._streams = {}, {}, {}

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
