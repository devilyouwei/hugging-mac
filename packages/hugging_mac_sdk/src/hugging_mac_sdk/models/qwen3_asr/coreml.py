"""Full Qwen3-ASR Core ML encoder and stateful split-decoder pipeline."""

from __future__ import annotations

import asyncio
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider
from hugging_mac_sdk.schemas.transcription import TranscriptionRequest

from .config import Qwen3AsrCoreMlInstanceConfig
from .resources import Qwen3AsrCoreMlResourceResolver
from .utils.types import AsrEngineOutput, PreparedAudio

_GRAPHS = (
    "encoder.mlmodelc",
    "embedding.mlmodelc",
    "decoder_part1.mlmodelc",
    "decoder_part2.mlmodelc",
)
_IM_START = 151644
_IM_END = 151645
_AUDIO_START = 151669
_AUDIO_END = 151670
_ASR_TEXT = 151704
_NEWLINE = 198
_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091


class CoreMlQwen3AsrEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: Qwen3AsrCoreMlInstanceConfig, resources: Qwen3AsrCoreMlResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._sessions: list[RuntimeSession] = []
        self._tokenizer: Any = None
        self._feature_extractor: Any = None
        self._max_sequence = 1024
        self._hidden_size = 1024
        self._batch_size = 128
        self._position = 0
        self._part1_state: Any = None
        self._part2_state: Any = None
        # Stateful Core ML decoder state must never hop between arbitrary worker
        # threads across utterances.
        self._inference_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3-asr-coreml")

    @property
    def device(self) -> str:
        return (
            self._sessions[0].device
            if self._sessions
            else (self._config.device or "cpu-and-neural-engine")
        )

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve()
        await self._resources.resolve_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        provider = CoreMLProvider()
        self._sessions = list(
            await asyncio.gather(
                *(
                    provider.create_session(
                        artifact / graph,
                        device=self._config.device,
                        options={},
                        executor=self._executor,
                    )
                    for graph in _GRAPHS
                )
            )
        )
        config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
        self._max_sequence = int(config.get("max_seq_length", 1024))
        self._hidden_size = int(config.get("hidden_size", 1024))
        batches = config.get("enumerated_t", [128])
        self._batch_size = int(batches[0] if batches else 128)
        transformers = importlib.import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._resources.tokenizer_root(), local_files_only=True
        )
        self._feature_extractor = transformers.WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=self._config.sample_rate,
            hop_length=160,
            n_fft=400,
            chunk_length=30,
            return_attention_mask=True,
        )

    async def infer(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput:
        async with self._inference_lock:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._infer_sync, prepared, request
            )

    def _infer_sync(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput:
        if len(self._sessions) != 4 or self._tokenizer is None:
            raise RuntimeError("Qwen3-ASR Core ML engine is not loaded")
        np = importlib.import_module("numpy")
        features = self._feature_extractor(
            prepared.samples,
            sampling_rate=prepared.sample_rate,
            return_tensors="np",
            padding="max_length",
        ).input_features
        real_frames = min(3000, max(1, (int(prepared.samples.shape[0]) + 159) // 160))
        encoder = self._sessions[0].run(
            {
                "mel": np.asarray(features, dtype=np.float32),
                "mel_length": np.asarray([real_frames], dtype=np.int32),
            }
        )
        audio_embeddings = np.asarray(encoder["audio_embeddings"], dtype=np.float32)
        audio_tokens = int(np.asarray(encoder["output_length"]).reshape(-1)[0])
        audio_embeddings = audio_embeddings.reshape(-1, self._hidden_size)[:audio_tokens]

        self._reset_cache()
        prefix = [
            _IM_START,
            _SYSTEM,
            _NEWLINE,
            _IM_END,
            _NEWLINE,
            _IM_START,
            _USER,
            _NEWLINE,
            _AUDIO_START,
        ]
        suffix = [
            _AUDIO_END,
            _IM_END,
            _NEWLINE,
            _IM_START,
            _ASSISTANT,
            _NEWLINE,
        ]
        prompt = request.prompt.strip()
        default_prompts = {
            "Please transcribe this audio.",
            "Transcribe the speech accurately in its original language.",
        }
        if prompt and prompt not in default_prompts:
            suffix.extend(self._tokenizer.encode(prompt, add_special_tokens=False))
        suffix.append(_ASR_TEXT)

        logits = self._prefill_tokens(prefix)
        logits = self._prefill_embeddings(audio_embeddings)
        logits = self._prefill_tokens(suffix)
        generated: list[int] = []
        next_token = self._argmax(logits, skip=_IM_END)
        generated.append(next_token)
        available = self._scratch_start - self._position
        limit = max(1, min(request.max_new_tokens, available))
        for _ in range(1, limit):
            if next_token == _IM_END:
                break
            logits = self._prefill_tokens([next_token])
            next_token = self._argmax(logits)
            generated.append(next_token)
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1].strip()
        return AsrEngineOutput(
            text=text,
            generated_tokens=len(generated),
            prompt_tokens=len(prefix) + audio_tokens + len(suffix),
        )

    @property
    def _scratch_start(self) -> int:
        return self._max_sequence - (self._batch_size - 1)

    def _reset_cache(self) -> None:
        self._position = 0
        self._part1_state = self._sessions[2].make_state()  # type: ignore[attr-defined]
        self._part2_state = self._sessions[3].make_state()  # type: ignore[attr-defined]

    def _prefill_tokens(self, tokens: list[int]) -> Any:
        np = importlib.import_module("numpy")
        rows = []
        for token in tokens:
            output = self._sessions[1].run({"token_id": np.asarray([[token]], dtype=np.int32)})
            rows.append(np.asarray(output["embedding"], dtype=np.float32).reshape(-1))
        return self._prefill_embeddings(np.stack(rows))

    def _prefill_embeddings(self, embeddings: Any) -> Any:
        logits: Any = None
        for start in range(0, int(embeddings.shape[0]), self._batch_size):
            chunk = embeddings[start : start + self._batch_size]
            logits = self._run_chunk(chunk)
        return logits

    def _run_chunk(self, chunk: Any) -> Any:
        np = importlib.import_module("numpy")
        count = int(chunk.shape[0])
        if self._position + count > self._scratch_start:
            raise RuntimeError("Qwen3-ASR decoder KV cache capacity exceeded")
        first = self._batch_size - count
        embeds = np.zeros((1, self._batch_size, self._hidden_size), dtype=np.float32)
        embeds[0, first:] = np.asarray(chunk, dtype=np.float32)
        positions = np.empty((self._batch_size,), dtype=np.int32)
        positions[:first] = np.arange(self._scratch_start, self._scratch_start + first)
        positions[first:] = np.arange(self._position, self._position + count)
        mask = np.full((1, 1, self._batch_size, self._max_sequence), -1e4, dtype=np.float32)
        for slot in range(first, self._batch_size):
            absolute = self._position + slot - first
            mask[0, 0, slot, : absolute + 1] = 0.0
            mask[0, 0, slot, self._scratch_start :] = -1e4
        self._position += count
        common = {"input_embeds": embeds, "positions": positions, "attention_mask": mask}
        part1 = self._sessions[2].run_with_state(  # type: ignore[attr-defined]
            common, self._part1_state
        )
        part2 = self._sessions[3].run_with_state(  # type: ignore[attr-defined]
            {**common, "input_embeds": part1["hidden_state"]}, self._part2_state
        )
        return part2["logits"]

    @staticmethod
    def _argmax(logits: Any, *, skip: int | None = None) -> int:
        np = importlib.import_module("numpy")
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        if skip is not None and 0 <= skip < values.shape[0]:
            values = values.copy()
            values[skip] = -np.inf
        return int(np.nanargmax(values))

    async def close(self) -> None:
        sessions, self._sessions = self._sessions, []
        self._tokenizer = None
        self._feature_extractor = None
        self._part1_state = self._part2_state = None
        await asyncio.gather(*(session.close() for session in sessions))

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
