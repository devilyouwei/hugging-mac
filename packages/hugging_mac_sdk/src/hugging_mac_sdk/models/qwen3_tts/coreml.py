"""Six-graph Core ML runtime for Qwen3-TTS 0.6B Base."""

from __future__ import annotations

import asyncio
import gc
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.coreml import CoreMLProvider, CoreMLSession
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import QWEN3_TTS_COREML_GRAPHS, Qwen3TtsInstanceConfig
from .resources import Qwen3TtsCoreMlResourceResolver
from .utils.types import Qwen3TtsEngineOutput

_SAMPLE_RATE = 24_000
_EOS = 2150
_CODEC_VOCAB = 2048
_CODEC_PAD = 2148
_CODEC_BOS = 2149
_TTS_PAD = 151671
_TTS_BOS = 151672
_TTS_EOS = 151673
_IM_START = 151644
_IM_END = 151645
_LANGUAGE_IDS = {"english": 2050, "en": 2050, "chinese": 2055, "zh": 2055}


class CoreMlQwen3TtsEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: Qwen3TtsInstanceConfig, resources: Qwen3TtsCoreMlResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._provider = CoreMLProvider()
        self._sessions: dict[str, CoreMLSession] = {}
        self._tokenizer: Any | None = None
        self._speaker: Any | None = None

    @property
    def device(self) -> str:
        return self._config.device

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_tokenizers()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        np = importlib.import_module("numpy")
        transformers = importlib.import_module("transformers")
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._resources.tokenizer_path, local_files_only=True
            )
            self._speaker = np.load(artifact / "speaker_embedding_official.npy")
            for graph in QWEN3_TTS_COREML_GRAPHS:
                self._sessions[graph] = await self._provider.create_session(
                    artifact / f"{graph}.mlmodelc", device=self._config.device, options={}
                )
        except BaseException:
            await self.close()
            raise

    async def infer(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput:
        return await asyncio.to_thread(self._infer_sync, request)

    def _infer_sync(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput:
        np = importlib.import_module("numpy")
        if self._tokenizer is None or self._speaker is None:
            raise RuntimeError("Qwen3-TTS Core ML engine is not loaded")
        if request.reference_audio is not None:
            raise UnsupportedRuntimeError("Core ML Qwen3-TTS supports only its bundled speaker")
        if request.speed != 1.0:
            raise UnsupportedRuntimeError("Core ML Qwen3-TTS does not support speed control")
        language = (request.language or "english").lower()
        if language not in _LANGUAGE_IDS:
            raise UnsupportedRuntimeError(
                "Core ML Qwen3-TTS currently supports English and Chinese"
            )
        text_projector = self._sessions["TextProjector"]
        code_embedder = self._sessions["CodeEmbedder"]
        multi_embedder = self._sessions["MultiCodeEmbedder"]
        code_decoder = self._sessions["CodeDecoder"]
        multi_decoder = self._sessions["MultiCodeDecoder"]

        def text_embed(token: int) -> Any:
            return text_projector.run({"input_ids": np.asarray([token], np.int32)})["input_embeds"]

        def code_embed(token: int) -> Any:
            return code_embedder.run({"input_ids": np.asarray([token], np.int32)})["input_embeds"]

        assistant = self._tokenizer("assistant", add_special_tokens=False).input_ids
        newline = self._tokenizer("\n", add_special_tokens=False).input_ids
        text_ids = self._tokenizer(request.text, add_special_tokens=False).input_ids
        role_ids = [_IM_START, *assistant, *newline]
        tts_pad, codec_pad = text_embed(_TTS_PAD), code_embed(_CODEC_PAD)
        embeddings = [text_embed(token) for token in role_ids]
        controls = (2154, 2156, _LANGUAGE_IDS[language], 2157)
        embeddings.extend(tts_pad + code_embed(token) for token in controls)
        embeddings.append(tts_pad + self._speaker.reshape(1, 1024, 1, 1))
        embeddings.append(text_embed(_TTS_BOS) + codec_pad)
        embeddings.extend(text_embed(token) + codec_pad for token in text_ids)
        embeddings.append(text_embed(_TTS_EOS) + codec_pad)
        embeddings.append(tts_pad + code_embed(_CODEC_BOS))
        if len(embeddings) >= 255:
            raise ValueError("Qwen3-TTS Core ML text is too long for its 256-position cache")

        key = np.zeros((1, 28672, 1, 256), np.float16)
        value = np.zeros_like(key)
        output: dict[str, Any] = {}
        for position, embedding in enumerate(embeddings):
            output, key, value = self._decode_step(
                code_decoder, embedding, position, key, value, 256
            )
        hidden = output["hidden_states"]
        logits = np.asarray(output["logits"], np.float64).reshape(-1)
        logits[_CODEC_VOCAB:] = -np.inf
        cb0 = self._sample(logits, request, np)
        generated = [cb0]
        frames: list[list[int]] = []
        max_frames = min(request.max_new_tokens, 125, 255 - len(embeddings))
        rng_position = len(embeddings)
        for frame_index in range(max_frames):
            cb_tokens = self._predict_codebooks(
                cb0, hidden, code_embedder, multi_embedder, multi_decoder, request, np
            )
            frames.append([cb0, *cb_tokens])
            combined = code_embed(cb0).copy()
            for index, token in enumerate(cb_tokens):
                linear = index * _CODEC_VOCAB + token
                combined += multi_embedder.run({"input_ids": np.asarray([linear], np.int32)})[
                    "input_embeds"
                ]
            output, key, value = self._decode_step(
                code_decoder, combined + tts_pad, rng_position, key, value, 256
            )
            rng_position += 1
            hidden = output["hidden_states"]
            logits = np.asarray(output["logits"], np.float64).reshape(-1)
            eos_logit = logits[_EOS]
            logits[_CODEC_VOCAB:] = -np.inf
            if frame_index >= 1:
                logits[_EOS] = eos_logit
            for token in set(generated):
                logits[token] = logits[token] / 1.05 if logits[token] > 0 else logits[token] * 1.05
            cb0 = self._sample(logits, request, np)
            generated.append(cb0)
            if cb0 == _EOS:
                break
        if not frames:
            return Qwen3TtsEngineOutput(b"", _SAMPLE_RATE, 0.0, 0)
        padded = np.zeros((125, 16), np.int32)
        padded[: len(frames)] = np.asarray(frames, np.int32)
        audio = self._sessions["SpeechDecoder"].run({"audio_codes": padded.T.reshape(1, 16, 125)})[
            "audio"
        ]
        audio = np.asarray(audio, np.float32).reshape(-1)[: len(frames) * 1920]
        audio = self._trim_silence(audio, np)
        return Qwen3TtsEngineOutput(
            audio.astype("<f4", copy=False).tobytes(),
            _SAMPLE_RATE,
            float(audio.size) / _SAMPLE_RATE,
            len(frames),
        )

    @staticmethod
    def _decode_step(
        model: CoreMLSession, embedding: Any, position: int, key: Any, value: Any, length: int
    ) -> tuple[dict[str, Any], Any, Any]:
        np = importlib.import_module("numpy")
        mask = np.full((1, length), -1e4, np.float16)
        mask[0, : position + 1] = 0
        update = np.zeros((1, length), np.float16)
        update[0, position] = 1
        result = dict(
            model.run(
                {
                    "input_embeds": np.asarray(embedding, np.float16),
                    "cache_length": np.asarray([position], np.int32),
                    "key_cache": key,
                    "value_cache": value,
                    "key_padding_mask": mask,
                    "kv_cache_update_mask": update,
                }
            )
        )
        return result, result["new_key_cache"], result["new_value_cache"]

    def _predict_codebooks(
        self,
        cb0: int,
        hidden: Any,
        code_embedder: CoreMLSession,
        multi_embedder: CoreMLSession,
        decoder: CoreMLSession,
        request: SpeechSynthesisRequest,
        np: Any,
    ) -> list[int]:
        key = np.zeros((1, 5120, 1, 16), np.float16)
        value = np.zeros_like(key)
        output, key, value = self._decode_step(decoder, hidden, 0, key, value, 16)
        cb0_embed = code_embedder.run({"input_ids": np.asarray([cb0], np.int32)})["input_embeds"]
        output, key, value = self._decode_step(decoder, cb0_embed, 1, key, value, 16)
        tokens = [self._sample(np.asarray(output["all_logits"])[0, 0], request, np)]
        for step in range(1, 15):
            linear = (step - 1) * _CODEC_VOCAB + tokens[-1]
            embedding = multi_embedder.run({"input_ids": np.asarray([linear], np.int32)})[
                "input_embeds"
            ]
            output, key, value = self._decode_step(decoder, embedding, step + 1, key, value, 16)
            tokens.append(self._sample(np.asarray(output["all_logits"])[0, step], request, np))
        return tokens

    @staticmethod
    def _sample(logits: Any, request: SpeechSynthesisRequest, np: Any) -> int:
        values = np.asarray(logits, np.float64).reshape(-1)
        if not request.do_sample:
            return int(np.argmax(values))
        values = values / request.temperature
        if 0 < request.top_k < values.size:
            threshold = np.partition(values, -request.top_k)[-request.top_k]
            values[values < threshold] = -np.inf
        probabilities = np.exp(values - np.max(values))
        probabilities /= probabilities.sum()
        return int(np.random.choice(values.size, p=probabilities))

    @staticmethod
    def _trim_silence(audio: Any, np: Any) -> Any:
        indices = np.flatnonzero(np.abs(audio) > 1e-4)
        return audio[: indices[-1] + 1] if indices.size else audio

    async def close(self) -> None:
        sessions, self._sessions = tuple(self._sessions.values()), {}
        await asyncio.gather(*(session.close() for session in sessions))
        self._tokenizer = self._speaker = None
        await asyncio.to_thread(gc.collect)
