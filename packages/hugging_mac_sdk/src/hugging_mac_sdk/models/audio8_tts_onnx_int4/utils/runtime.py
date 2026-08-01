"""Official Audio8 autoregressive ONNX inference flow, adapted for the SDK."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from .prompt import PromptBuilder

ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
    *,
    do_sample: bool,
) -> int:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not do_sample:
        return int(np.argmax(values))
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    probabilities = np.exp(sorted_values - np.max(sorted_values))
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    remove = (cumulative > float(top_p)) | (np.arange(probabilities.size) >= int(top_k))
    remove[0] = False
    masked = values.copy()
    masked[order[remove]] = -np.inf
    scaled = masked / max(float(temperature), 1e-5)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    noise = -np.log(np.clip(rng.random(probabilities.size), 1e-12, 1.0))
    return int(np.argmax(probabilities / noise))


class Audio8OnnxRuntime:
    """Own the three online ONNX sessions and generate codec frames."""

    def __init__(self, model_dir: Path, *, threads: int | None = None):
        import onnxruntime as ort  # type: ignore[import-untyped]

        self.model_dir = model_dir.resolve()
        self.manifest = json.loads(
            (self.model_dir / "runtime_manifest.json").read_text(encoding="utf-8")
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if threads is not None:
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = max(1, int(threads) // 2)
        session_args = {
            "sess_options": options,
            "providers": ["CPUExecutionProvider"],
        }
        self.slow: Any | None = ort.InferenceSession(
            str(self.model_dir / "slow_ar_int4.onnx"), **session_args
        )
        self.fast: Any | None = ort.InferenceSession(
            str(self.model_dir / "fast_ar_int4.onnx"), **session_args
        )
        self.decoder: Any | None = ort.InferenceSession(
            str(self.model_dir / "codec_decoder_fp16.onnx"), **session_args
        )
        self.prompt_builder = PromptBuilder(
            self.model_dir / "tokenizer",
            self.manifest["semantic_begin_id"],
            self.manifest["num_codebooks"],
        )
        self.slow_inputs = {item.name: item for item in self.slow.get_inputs()}
        self.fast_inputs = {item.name: item for item in self.fast.get_inputs()}

    def close(self) -> None:
        self.slow = None
        self.fast = None
        self.decoder = None

    def _empty_slow_caches(self) -> list[np.ndarray]:
        dtype = ORT_DTYPES[self.slow_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self.manifest["n_local_heads"]),
            int(self.manifest["max_seq_len"]),
            int(self.manifest["head_dim"]),
        )
        return [np.zeros(shape, dtype=dtype) for _ in range(2 * int(self.manifest["num_layers"]))]

    def _empty_fast_caches(self) -> list[np.ndarray]:
        dtype = ORT_DTYPES[self.fast_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self.manifest["fast_n_local_heads"]),
            int(self.manifest["num_codebooks"]),
            int(self.manifest["fast_head_dim"]),
        )
        return [
            np.zeros(shape, dtype=dtype) for _ in range(2 * int(self.manifest["num_fast_layers"]))
        ]

    @staticmethod
    def _update_caches(
        caches: list[np.ndarray], positions: np.ndarray, deltas: list[np.ndarray]
    ) -> None:
        for index, delta in enumerate(deltas):
            caches[index][:, :, positions, :] = delta

    def _slow_step(
        self,
        codes: np.ndarray,
        positions: np.ndarray,
        caches: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.slow is None:
            raise RuntimeError("Audio8 ONNX runtime is closed")
        feeds = {"codes": codes.astype(np.int64), "input_pos": positions.astype(np.int64)}
        for index in range(int(self.manifest["num_layers"])):
            feeds[f"cache_key_{index}"] = caches[2 * index]
            feeds[f"cache_value_{index}"] = caches[2 * index + 1]
        outputs = self.slow.run(None, feeds)
        self._update_caches(caches, positions, outputs[2:])
        return (
            cast(np.ndarray, np.asarray(outputs[0])[0, -1]),
            np.asarray(outputs[1])[:, -1:, :],
        )

    def _fast_step(
        self,
        hidden: np.ndarray,
        token_id: int,
        use_hidden: bool,
        position: int,
        caches: list[np.ndarray],
    ) -> np.ndarray:
        if self.fast is None:
            raise RuntimeError("Audio8 ONNX runtime is closed")
        hidden_dtype = ORT_DTYPES[self.fast_inputs["slow_hidden"].type]
        feeds = {
            "slow_hidden": np.asarray(hidden, dtype=hidden_dtype),
            "token_id": np.asarray([[token_id]], dtype=np.int64),
            "use_slow_hidden": np.asarray([use_hidden], dtype=np.bool_),
            "input_pos": np.asarray([position], dtype=np.int64),
        }
        for index in range(int(self.manifest["num_fast_layers"])):
            feeds[f"cache_key_{index}"] = caches[2 * index]
            feeds[f"cache_value_{index}"] = caches[2 * index + 1]
        outputs = self.fast.run(None, feeds)
        self._update_caches(caches, np.asarray([position]), outputs[1:])
        return cast(np.ndarray, np.asarray(outputs[0])[0, -1])

    def _sample_semantic(
        self,
        logits: np.ndarray,
        previous: list[int],
        temperature: float,
        top_p: float,
        top_k: int,
        rng: np.random.Generator,
        *,
        do_sample: bool,
    ) -> int:
        begin = int(self.manifest["semantic_begin_id"])
        end = int(self.manifest["semantic_end_id"])
        stop = int(self.manifest["im_end_id"])
        allowed_ids = np.concatenate([np.arange(begin, end + 1), np.asarray([stop])])
        values = np.asarray(logits).reshape(-1)
        allowed_logits = (
            values
            if self.manifest.get("slow_logits_layout") == "semantic_then_eos"
            else values[allowed_ids]
        )
        if allowed_logits.size != allowed_ids.size:
            raise ValueError(
                f"unexpected slow logits size: {allowed_logits.size}, expected {allowed_ids.size}"
            )
        normal = int(
            allowed_ids[
                _sample(
                    allowed_logits,
                    temperature,
                    top_p,
                    top_k,
                    rng,
                    do_sample=do_sample,
                )
            ]
        )
        if begin <= normal <= end and normal in previous and do_sample:
            return int(allowed_ids[_sample(allowed_logits, 1.0, 0.9, top_k, rng, do_sample=True)])
        return normal

    def iter_codes(
        self,
        *,
        text: str,
        reference_text: str,
        reference_codes: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        seed: int = 42,
    ) -> Iterator[np.ndarray]:
        prompt = self.prompt_builder.build(text, reference_text, reference_codes)
        prompt_len = int(prompt.shape[2])
        max_seq_len = int(self.manifest["max_seq_len"])
        if prompt_len >= max_seq_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds max sequence length {max_seq_len}"
            )
        token_limit = min(int(max_new_tokens), max_seq_len - prompt_len)
        rng = np.random.default_rng(int(seed))
        slow_caches = self._empty_slow_caches()
        positions = np.arange(prompt_len, dtype=np.int64)
        logits, hidden = self._slow_step(prompt, positions, slow_caches)
        previous: list[int] = []
        begin = int(self.manifest["semantic_begin_id"])
        stop = int(self.manifest["im_end_id"])
        codebook_size = int(self.manifest["codebook_size"])

        for step in range(token_limit):
            semantic = self._sample_semantic(
                logits,
                previous,
                temperature,
                top_p,
                top_k,
                rng,
                do_sample=do_sample,
            )
            if semantic == stop:
                return
            previous.append(semantic)
            previous = previous[-10:]
            fast_caches = self._empty_fast_caches()
            self._fast_step(hidden, 0, True, 0, fast_caches)
            token = min(max(semantic - begin, 0), codebook_size - 1)
            codebooks = [token]
            for fast_position in range(1, int(self.manifest["num_codebooks"])):
                fast_logits = self._fast_step(hidden, token, False, fast_position, fast_caches)
                token = _sample(
                    fast_logits,
                    temperature,
                    top_p,
                    top_k,
                    rng,
                    do_sample=do_sample,
                )
                codebooks.append(token)
            frame = np.asarray(codebooks, dtype=np.int64)
            yield frame
            if step + 1 >= token_limit:
                return
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            position = np.asarray([prompt_len + step], dtype=np.int64)
            logits, hidden = self._slow_step(column, position, slow_caches)

    def synthesize(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        frames = list(self.iter_codes(**kwargs))
        if not frames:
            raise RuntimeError("model produced no codec frames")
        codes = np.stack(frames, axis=1)
        if self.decoder is None:
            raise RuntimeError("Audio8 ONNX runtime is closed")
        audio = self.decoder.run(None, {"codes": codes[np.newaxis]})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1), codes
