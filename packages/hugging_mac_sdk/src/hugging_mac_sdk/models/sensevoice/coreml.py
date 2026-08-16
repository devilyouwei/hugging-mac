"""SenseVoiceSmall Core ML SANM/CTC runtime."""

from __future__ import annotations

import asyncio
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider

from .config import SenseVoiceSmallInstanceConfig
from .converter import FEATURE_BUCKETS
from .resources import SenseVoiceSmallResourceResolver
from .utils.types import PreparedAudio, SenseVoiceEngineOutput, SenseVoiceInferenceOptions


class CoreMlSenseVoiceSmallEngine:
    runtime_name = "coreml"

    def __init__(
        self, config: SenseVoiceSmallInstanceConfig, resources: SenseVoiceSmallResourceResolver
    ) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None
        self._session_bucket: int | None = None
        self._inference_lock = asyncio.Lock()
        # Core ML / Accelerate objects are native and are not reliably thread-affine
        # when successive calls are dispatched through asyncio's shared executor.
        # Keep all prediction work for this engine on one dedicated OS thread.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sensevoice-coreml"
        )
        self._tokenizer: Any | None = None
        self._artifact: Path | None = None

    @property
    def device(self) -> str:
        return self._config.compute_units

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_coreml()
        await self._resources.resolve_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        sentencepiece = importlib.import_module("sentencepiece")
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(
                self._resources.tokenizer_path
                / "chn_jpn_yue_eng_ko_spectok.bpe.model"
            )
        )
        self._artifact = artifact

    async def infer(
        self, prepared: PreparedAudio, options: SenseVoiceInferenceOptions
    ) -> SenseVoiceEngineOutput:
        if self._tokenizer is None or self._artifact is None:
            raise RuntimeError("SenseVoiceSmall Core ML engine is not loaded")
        async with self._inference_lock:
            inputs, valid, bucket = await asyncio.to_thread(
                self._prepare_inputs_sync, prepared, options
            )
            session = await self._resolve_session(bucket)
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._predict_sync, session, inputs, valid, options
            )

    async def close(self) -> None:
        session, self._session = self._session, None
        self._session_bucket = None
        if session is not None:
            await session.close()
        self._tokenizer = None
        self._artifact = None

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _resolve_session(self, bucket: int) -> RuntimeSession:
        if self._session is not None and self._session_bucket == bucket:
            return self._session
        previous, self._session = self._session, None
        self._session_bucket = None
        if previous is not None:
            await previous.close()
        artifact = self._artifact
        if artifact is None:
            raise RuntimeError("SenseVoiceSmall Core ML engine is not loaded")
        session = await CoreMLProvider().create_session(
            artifact / "sensevoice.mlpackage",
            device=self._config.compute_units,
            options={"function_name": f"encoder_{bucket}"},
            executor=self._executor,
        )
        self._session = session
        self._session_bucket = bucket
        return session

    def _prepare_inputs_sync(
        self, prepared: PreparedAudio, options: SenseVoiceInferenceOptions
    ) -> tuple[dict[str, Any], int, int]:
        np = importlib.import_module("numpy")
        from .utils.frontend import extract_features
        from .utils.modeling import SenseVoiceSmallModel

        artifact = self._artifact
        if artifact is None:
            raise RuntimeError("SenseVoiceSmall Core ML engine is not loaded")

        features, lengths = extract_features(
            prepared, cmvn_path=artifact / "am.mvn", dither=self._config.fbank_dither
        )
        valid = int(lengths[0].item())
        try:
            bucket = next(item for item in FEATURE_BUCKETS if valid <= item)
        except StopIteration as error:
            raise ValueError("SenseVoiceSmall audio exceeds the 30 second Core ML limit") from error
        padded = np.zeros((1, bucket, 560), dtype=np.float32)
        padded[:, :valid] = features.numpy()
        language_id = SenseVoiceSmallModel.LANGUAGE_IDS.get(options.language, 0)
        style_id = SenseVoiceSmallModel.TEXT_NORMALIZATION_IDS[
            "withitn" if options.use_itn else "woitn"
        ]
        return (
            {
                "features": padded,
                "lengths": np.asarray([valid], dtype=np.int32),
                "language_id": np.asarray([language_id], dtype=np.int32),
                "style_id": np.asarray([style_id], dtype=np.int32),
            },
            valid,
            bucket,
        )

    def _predict_sync(
        self,
        session: RuntimeSession,
        inputs: dict[str, Any],
        valid: int,
        options: SenseVoiceInferenceOptions,
    ) -> SenseVoiceEngineOutput:
        np = importlib.import_module("numpy")
        from .utils.modeling import SenseVoiceSmallModel
        from .utils.postprocess import parse_rich_transcript

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("SenseVoiceSmall Core ML engine is not loaded")
        output = session.run(inputs)
        logits = np.asarray(output["logits"])[0, : valid + 4]
        if options.ban_unknown_emotion:
            logits[:, SenseVoiceSmallModel.UNKNOWN_EMOTION_TOKEN_ID] = -np.inf
        predicted = logits.argmax(axis=-1)
        collapsed = predicted[np.concatenate(([True], predicted[1:] != predicted[:-1]))]
        token_ids = collapsed[collapsed != 0].tolist()
        raw_text = tokenizer.decode_ids(token_ids)
        rich = parse_rich_transcript(raw_text)
        return SenseVoiceEngineOutput(
            text=rich.text,
            raw_text=rich.raw_text,
            languages=rich.languages,
            emotion=rich.emotion,
            events=rich.events,
            token_count=len(token_ids),
        )
