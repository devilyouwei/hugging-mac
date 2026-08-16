"""Core ML runtime for aufklarer's six-model Qwen3-TTS pipeline."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.coreml import CoreMLProvider, CoreMLSession
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import QWEN3_TTS_COREML_GRAPHS, Qwen3TtsInstanceConfig
from .resources import Qwen3TtsCoreMlResourceResolver
from .utils.tokenizer import Qwen3CoreMlTokenizer
from .utils.types import Qwen3TtsEngineOutput

_SAMPLE_RATE = 24_000
_SAMPLES_PER_FRAME = 1_920
_MAX_FRAMES = 125
_MAX_SEQUENCE = 256
_HIDDEN_SIZE = 1_024
_EOS = 2_150
_CODEC_VOCAB = 2_048
_CODEC_LOGITS = 3_072
_CODEC_PAD = 2_148
_CODEC_BOS = 2_149
_IM_START = 151_644
_ASSISTANT = 77_091
_NEWLINE = 198
_LANGUAGE_IDS = {
    "english": 2050,
    "german": 2053,
    "spanish": 2054,
    "chinese": 2055,
    "japanese": 2058,
    "french": 2061,
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
}


class CoreMlQwen3TtsEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: Qwen3TtsInstanceConfig, resources: Qwen3TtsCoreMlResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._provider = CoreMLProvider()
        self._sessions: dict[str, CoreMLSession] = {}
        self._tokenizer: Qwen3CoreMlTokenizer | None = None
        self._special_embeddings: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._config.device

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_tokenizers()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        np = importlib.import_module("numpy")
        try:
            self._tokenizer = Qwen3CoreMlTokenizer.from_files(
                self._resources.tokenizer_path / "vocab.json",
                self._resources.tokenizer_path / "merges.txt",
            )
            for name in ("speaker_embedding", "tts_pad_embed", "tts_bos_embed", "tts_eos_embed"):
                value = np.load(artifact / f"{name}.npy", allow_pickle=False)
                if value.size != _HIDDEN_SIZE:
                    raise ValueError(f"Qwen3-TTS {name}.npy must contain {_HIDDEN_SIZE} values")
                self._special_embeddings[name] = np.asarray(value, np.float32).reshape(
                    1, _HIDDEN_SIZE, 1, 1
                )
            for graph in QWEN3_TTS_COREML_GRAPHS:
                # The upstream runtime deliberately keeps the three tiny embedders on CPU
                # for stable FP32 accumulation and routes the decoder chain to ANE.
                is_embedder = graph.endswith("Embedder") or graph == "TextProjector"
                device = "cpu-only" if is_embedder else self._config.device
                self._sessions[graph] = await self._provider.create_session(
                    artifact / f"{graph}.mlmodelc",
                    device=device,
                    options={"retained_predictions": 1},
                )
        except BaseException:
            await self.close()
            raise

    async def infer(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput:
        return await asyncio.to_thread(self._infer_sync, request)

    def _infer_sync(self, request: SpeechSynthesisRequest) -> Qwen3TtsEngineOutput:
        np = importlib.import_module("numpy")
        if self._tokenizer is None or not self._special_embeddings:
            raise RuntimeError("Qwen3-TTS Core ML engine is not loaded")
        if request.reference_audio is not None:
            raise UnsupportedRuntimeError("Core ML Qwen3-TTS supports only its bundled speaker")
        if request.speed != 1.0:
            raise UnsupportedRuntimeError("Core ML Qwen3-TTS does not support speed control")
        language = (request.language or "english").lower()
        if language == "auto":
            language = "english"
        if language not in _LANGUAGE_IDS:
            raise UnsupportedRuntimeError(
                "Core ML Qwen3-TTS supports Chinese, English, German, Italian, Portuguese, "
                "Spanish, Japanese, Korean, French, and Russian"
            )

        text_projector = self._sessions["TextProjector"]
        code_embedder = self._sessions["CodeEmbedder"]
        multi_embedder = self._sessions["MultiCodeEmbedder"]
        code_decoder = self._sessions["CodeDecoder"]
        multi_decoder = self._sessions["MultiCodeDecoder"]
        special = self._special_embeddings

        def embed(session: CoreMLSession, token: int) -> Any:
            output = session.run({"input_ids": np.asarray([token], np.int32)})["input_embeds"]
            return np.asarray(output, np.float32).reshape(1, _HIDDEN_SIZE, 1, 1)

        codec_pad = embed(code_embedder, _CODEC_PAD)
        embeddings = [embed(text_projector, token) for token in (_IM_START, _ASSISTANT, _NEWLINE)]
        embeddings.extend(
            special["tts_pad_embed"] + embed(code_embedder, token)
            for token in (2154, 2156, _LANGUAGE_IDS[language], 2157)
        )
        embeddings.append(special["tts_pad_embed"] + special["speaker_embedding"])
        embeddings.append(special["tts_bos_embed"] + codec_pad)
        text_tokens = self._tokenizer.encode(request.text)
        embeddings.extend(embed(text_projector, token) + codec_pad for token in text_tokens)
        embeddings.append(special["tts_eos_embed"] + codec_pad)
        embeddings.append(special["tts_pad_embed"] + embed(code_embedder, _CODEC_BOS))
        if len(embeddings) >= _MAX_SEQUENCE:
            raise ValueError("Qwen3-TTS Core ML text is too long for its 256-position cache")

        code_state = code_decoder.make_state()
        key = value = None
        output: dict[str, Any] = {}
        for position, embedding in enumerate(embeddings):
            output, key, value = self._decode_step(
                code_decoder,
                embedding,
                position,
                key,
                value,
                _MAX_SEQUENCE,
                state=code_state,
            )

        hidden = output["hidden_states"]
        logits = np.asarray(output["logits"], np.float64).reshape(-1)
        if logits.size != _CODEC_LOGITS:
            raise ValueError(f"Qwen3-TTS CodeDecoder returned {logits.size} logits")
        logits[_CODEC_VOCAB:] = -np.inf
        cb0 = self._sample(logits, request, np)
        generated_cb0 = [cb0]
        frames: list[list[int]] = []
        effective_frames = min(
            request.max_new_tokens,
            _MAX_FRAMES,
            8 * len(embeddings),
            _MAX_SEQUENCE - len(embeddings),
        )

        for frame_index in range(effective_frames):
            codebook_tokens = self._predict_codebooks(
                cb0, hidden, code_embedder, multi_embedder, multi_decoder, request, np
            )
            frames.append([cb0, *codebook_tokens])
            if frame_index + 1 >= effective_frames:
                break

            # Accumulate all 16 codec embeddings in FP32 before the model-input cast.
            combined = embed(code_embedder, cb0)
            for index, token in enumerate(codebook_tokens):
                combined += embed(multi_embedder, index * _CODEC_VOCAB + token)
            output, key, value = self._decode_step(
                code_decoder,
                combined + special["tts_pad_embed"],
                len(embeddings) + frame_index,
                key,
                value,
                _MAX_SEQUENCE,
                state=code_state,
            )
            hidden = output["hidden_states"]
            logits = np.asarray(output["logits"], np.float64).reshape(-1)
            eos_logit = logits[_EOS]
            logits[_CODEC_VOCAB:] = -np.inf
            if frame_index >= 1:
                logits[_EOS] = eos_logit
            for token in set(generated_cb0):
                logits[token] = (
                    logits[token] / 1.05 if logits[token] > 0 else logits[token] * 1.05
                )
            cb0 = self._sample(logits, request, np)
            generated_cb0.append(cb0)
            if cb0 == _EOS:
                break

        if not frames:
            return Qwen3TtsEngineOutput(b"", _SAMPLE_RATE, 0.0, 0)
        padded = np.zeros((_MAX_FRAMES, 16), np.int32)
        padded[: len(frames)] = np.asarray(frames, np.int32)
        decoded = self._sessions["SpeechDecoder"].run(
            {"audio_codes": padded.T.reshape(1, 16, _MAX_FRAMES)}
        )["audio"]
        audio = np.asarray(decoded, np.float32).reshape(-1)[: len(frames) * _SAMPLES_PER_FRAME]
        peak = float(np.max(np.abs(audio), initial=0.0))
        if peak > 0.001:
            audio *= min(0.9 / peak, 10.0)
        return Qwen3TtsEngineOutput(
            audio.astype("<f4", copy=False).tobytes(),
            _SAMPLE_RATE,
            float(audio.size) / _SAMPLE_RATE,
            len(frames),
        )

    @staticmethod
    def _decode_step(
        model: CoreMLSession,
        embedding: Any,
        position: int,
        key: Any | None,
        value: Any | None,
        length: int,
        *,
        state: Any | None = None,
    ) -> tuple[dict[str, Any], Any, Any]:
        np = importlib.import_module("numpy")
        mask = np.full((1, length), -1e4, np.float16)
        mask[0, : position + 1] = 0
        update = np.zeros((1, length), np.float16)
        update[0, position] = 1
        input_embedding = np.asarray(embedding, np.float16)
        if input_embedding.size != _HIDDEN_SIZE:
            raise ValueError(
                f"Qwen3-TTS decoder embedding must contain {_HIDDEN_SIZE} values"
            )
        inputs = {
            "input_embeds": input_embedding.reshape(1, _HIDDEN_SIZE, 1, 1),
            "cache_length": np.asarray([position], np.int32),
            "key_padding_mask": mask,
            "kv_cache_update_mask": update,
        }
        if state is None:
            if key is None or value is None:
                raise ValueError("Stateless Qwen3-TTS decoder requires explicit KV caches")
            inputs.update({"key_cache": key, "value_cache": value})
            result = dict(model.run(inputs))
            return result, result["new_key_cache"], result["new_value_cache"]
        result = dict(model.run_with_state(inputs, state))
        return result, key, value

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
        decoder_state = decoder.make_state()
        key = value = None
        output, key, value = self._decode_step(
            decoder, hidden, 0, key, value, 16, state=decoder_state
        )
        cb0_embed = code_embedder.run({"input_ids": np.asarray([cb0], np.int32)})["input_embeds"]
        output, key, value = self._decode_step(
            decoder, cb0_embed, 1, key, value, 16, state=decoder_state
        )
        tokens = [self._sample(np.asarray(output["all_logits"])[0, 0], request, np)]
        for step in range(1, 15):
            linear = (step - 1) * _CODEC_VOCAB + tokens[-1]
            embedding = multi_embedder.run({"input_ids": np.asarray([linear], np.int32)})[
                "input_embeds"
            ]
            output, key, value = self._decode_step(
                decoder,
                embedding,
                step + 1,
                key,
                value,
                16,
                state=decoder_state,
            )
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
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Qwen3-TTS sampling produced no finite probabilities")
        probabilities /= total
        return int(np.random.choice(values.size, p=probabilities))

    async def close(self) -> None:
        sessions, self._sessions = tuple(self._sessions.values()), {}
        await asyncio.gather(*(session.close() for session in sessions))
        self._tokenizer = None
        self._special_embeddings.clear()
