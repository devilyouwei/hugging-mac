"""Audio8-ASR hybrid engine: Core ML audio tower + cached PyTorch decoder."""

from __future__ import annotations

import asyncio
import gc
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.transcription import TranscriptionRequest

from .config import Audio8AsrInstanceConfig
from .resources import Audio8AsrResourceResolver
from .utils.types import AsrEngineOutput, PreparedAudio


class CoreMlAudio8AsrEngine:
    """Run the audio tower on Core ML and generation with a PyTorch KV cache."""

    runtime_name = "coreml"

    def __init__(
        self,
        config: Audio8AsrInstanceConfig,
        resources: Audio8AsrResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._sessions: dict[int, RuntimeSession] = {}
        self._torch: Any | None = None
        self._language_model: Any | None = None
        self._tokenizer: Any | None = None
        self._feature_extractor: Any | None = None
        self._projector: dict[str, Any] = {}
        self._decoder_device: str = config.decoder_device or "mps"
        self._dtype: Any | None = None

    @property
    def device(self) -> str:
        return self._config.compute_units

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_coreml()
        await self._resources.resolve_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        provider = CoreMLProvider()
        tower = artifact / "audio_tower.mlpackage"
        sessions: dict[int, RuntimeSession] = {}
        try:
            for bucket in (500, 1000, 3000):
                sessions[bucket] = await provider.create_session(
                    tower,
                    device=self._config.compute_units,
                    options={"function_name": f"tower_{bucket // 100}s"},
                )
            await asyncio.to_thread(self._load_decoder_sync, artifact)
        except BaseException:
            for session in sessions.values():
                await session.close()
            raise
        self._sessions = sessions

    async def infer(
        self,
        prepared: PreparedAudio,
        request: TranscriptionRequest,
    ) -> AsrEngineOutput:
        if not self._sessions or self._language_model is None:
            raise RuntimeError("Audio8-ASR Core ML engine is not loaded")
        return await asyncio.to_thread(self._infer_sync, prepared, request)

    async def close(self) -> None:
        sessions, self._sessions = self._sessions, {}
        for session in sessions.values():
            await session.close()
        had_model = self._language_model is not None
        self._language_model = None
        self._tokenizer = None
        self._feature_extractor = None
        self._projector = {}
        if had_model:
            await asyncio.to_thread(gc.collect)
        if self._decoder_device == "mps" and self._torch is not None:
            empty_cache = getattr(getattr(self._torch, "mps", None), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()
        self._torch = None
        self._dtype = None

    def _load_decoder_sync(self, artifact: Path) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        safetensors = importlib.import_module("safetensors")
        from .utils.modeling import load_audio8_language_model

        self._decoder_device = TorchProvider().resolve_device(
            self._config.decoder_device,
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        dtype_name = self._config.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if self._decoder_device == "mps" else "float32"
        dtype = getattr(torch, dtype_name)
        language_model = load_audio8_language_model(
            artifact,
            device=self._decoder_device,
            dtype=dtype,
        )
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(
            self._resources.tokenizer_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
            artifact,
            local_files_only=True,
        )
        projector: dict[str, Any] = {}
        with safetensors.safe_open(
            artifact / "projector.safetensors",
            framework="pt",
            device="cpu",
        ) as checkpoint:
            for name in checkpoint.keys():  # noqa: SIM118 - safe_open is not iterable
                projector[name] = checkpoint.get_tensor(name).float().numpy()
        self._torch = torch
        self._dtype = dtype
        self._language_model = language_model
        self._tokenizer = tokenizer
        self._feature_extractor = feature_extractor
        self._projector = projector

    def _infer_sync(
        self,
        prepared: PreparedAudio,
        request: TranscriptionRequest,
    ) -> AsrEngineOutput:
        np = importlib.import_module("numpy")
        from .utils.coreml import (
            adaptive_pool_hidden,
            coreml_attention_mask,
            project_audio_hidden,
            select_audio_bucket,
        )
        from .utils.modeling import greedy_decode_embeddings

        torch = self._torch
        language_model = self._language_model
        tokenizer = self._tokenizer
        feature_extractor = self._feature_extractor
        dtype = self._dtype
        if torch is None:
            raise RuntimeError("Audio8-ASR Core ML decoder is not loaded")
        if language_model is None:
            raise RuntimeError("Audio8-ASR Core ML decoder is not loaded")
        if tokenizer is None:
            raise RuntimeError("Audio8-ASR Core ML tokenizer is not loaded")
        if feature_extractor is None or dtype is None:
            raise RuntimeError("Audio8-ASR Core ML decoder is not loaded")

        features = feature_extractor(
            prepared.samples,
            sampling_rate=prepared.sample_rate,
            return_tensors="np",
            return_attention_mask=False,
            padding="longest",
            max_length=int(self._config.max_audio_seconds * prepared.sample_rate),
        )["input_features"]
        features = np.asarray(features, dtype=np.float32)
        feature_frames = int(features.shape[-1])
        bucket = select_audio_bucket(feature_frames)
        padded = np.zeros((1, 128, bucket), dtype=np.float32)
        padded[..., :feature_frames] = features[..., :feature_frames]
        attention_mask, valid_tokens = coreml_attention_mask(
            bucket_frames=bucket,
            valid_feature_frames=feature_frames,
            np=np,
        )
        output = self._sessions[bucket].run({"audios": padded, "attn_mask": attention_mask})
        hidden = np.asarray(output["hidden"], dtype=np.float32)[:valid_tokens]
        hop_length = int(getattr(feature_extractor, "hop_length", 160))
        mel_frames = int(prepared.samples.shape[0]) // max(hop_length, 1)
        audio_tokens = max(((mel_frames + 1) // 2) // 4, 1)
        hidden = adaptive_pool_hidden(hidden, audio_tokens, np)
        projected = project_audio_hidden(
            hidden,
            norm_weight=self._projector["0.weight"],
            norm_bias=self._projector["0.bias"],
            linear_weight=self._projector["1.weight"],
            linear_bias=self._projector["1.bias"],
            np=np,
        )

        prompt = (
            "<|user|><|begin_of_audio|>"
            + "<|audio|>" * audio_tokens
            + "<|end_of_audio|>"
            + request.prompt
            + "<|assistant|>"
        )
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._decoder_device)
        text_attention_mask = encoded["attention_mask"].to(self._decoder_device)
        inputs_embeds = language_model.model.embed_tokens(input_ids).clone()
        positions = torch.nonzero(
            input_ids[0].eq(151646),
            as_tuple=False,
        ).flatten()
        if int(positions.numel()) != audio_tokens:
            raise ValueError("Audio token count does not match projected embeddings")
        projected_tensor = torch.from_numpy(projected).to(
            device=self._decoder_device,
            dtype=dtype,
        )
        inputs_embeds[0, positions, :] = projected_tensor
        with torch.inference_mode():
            generated = greedy_decode_embeddings(
                language_model,
                inputs_embeds=inputs_embeds,
                attention_mask=text_attention_mask,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=int(tokenizer.eos_token_id),
            )
        text = tokenizer.batch_decode(
            generated.detach().cpu(),
            skip_special_tokens=True,
        )[0]
        return AsrEngineOutput(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=int(generated.shape[1]),
        )
