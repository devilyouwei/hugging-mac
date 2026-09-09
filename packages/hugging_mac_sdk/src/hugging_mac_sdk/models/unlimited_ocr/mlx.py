"""MLX-VLM engine for the Unlimited-OCR four-bit checkpoint."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.runtime.mlx import MlxProvider, MlxSession, materialize_mlx
from hugging_mac_sdk.schemas.document_parsing import DocumentParsingRequest

from .config import UnlimitedOcrMlxInstanceConfig
from .resources import UnlimitedOcrResourceResolver

_BASE_SIZE = 1024
_GUNDAM_IMAGE_SIZE = 640
_BASE_IMAGE_SIZE = 1024
_NO_REPEAT_NGRAM_SIZE = 35
_SINGLE_IMAGE_NGRAM_WINDOW = 128
_MULTI_IMAGE_NGRAM_WINDOW = 1024


class MlxUnlimitedOcrEngine:
    runtime_name = "mlx"

    def __init__(
        self, config: UnlimitedOcrMlxInstanceConfig, resources: UnlimitedOcrResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._model: Any | None = None
        self._processor: Any | None = None
        self._session: MlxSession | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve()).path

    async def load(self, artifact: Path) -> None:
        def _load(source: Path, mx: Any) -> tuple[Any, Any]:
            from mlx_vlm import load

            model, processor = load(str(source), lazy=True)
            materialize_mlx(model, mx)
            return model, processor

        self._session = await MlxProvider().create_session(
            artifact, device=self._config.device, options={"model_loader": _load}
        )
        self._model, self._processor = self._session.value

    async def infer(self, request: DocumentParsingRequest) -> str:
        return await asyncio.to_thread(self._infer_sync, request)

    async def stream(self, request: DocumentParsingRequest) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        cancelled = threading.Event()

        def produce() -> None:
            try:
                for delta in self._stream_sync(request):
                    if cancelled.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except BaseException as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancelled.set()
            await worker

    def _infer_sync(self, request: DocumentParsingRequest) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Unlimited-OCR is not loaded")
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self._processor,
            self._model.config,
            request.prompt,
            num_images=len(request.images),
        )
        generation_options = self._generation_options(request)
        result = generate(
            self._model,
            self._processor,
            prompt,
            [str(image.path) for image in request.images],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **generation_options,
            verbose=False,
        )
        return str(getattr(result, "text", result)).strip()

    def _stream_sync(self, request: DocumentParsingRequest) -> Iterator[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Unlimited-OCR is not loaded")
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self._processor,
            self._model.config,
            request.prompt,
            num_images=len(request.images),
        )
        decoder = _IncrementalTokenDecoder(
            getattr(self._processor, "tokenizer", self._processor)
        )
        generation_options = self._generation_options(request)
        for result in stream_generate(
            self._model,
            self._processor,
            prompt,
            [str(image.path) for image in request.images],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **generation_options,
            verbose=False,
        ):
            token = getattr(result, "token", None)
            if token is not None:
                yield decoder.add(int(token))

    @classmethod
    def _generation_options(cls, request: DocumentParsingRequest) -> dict[str, Any]:
        multi_image = len(request.images) > 1
        window = (
            _MULTI_IMAGE_NGRAM_WINDOW if multi_image else _SINGLE_IMAGE_NGRAM_WINDOW
        )
        return {
            "base_size": _BASE_SIZE,
            "image_size": _BASE_IMAGE_SIZE if multi_image else _GUNDAM_IMAGE_SIZE,
            "cropping": not multi_image,
            "logits_processors": cls._logits_processors(window=window),
        }

    @staticmethod
    def _logits_processors(*, window: int) -> list[Any]:
        size = _NO_REPEAT_NGRAM_SIZE

        def prevent_repeated_ngram(tokens: Any, logits: Any) -> Any:
            import mlx.core as mx

            token_ids = [int(token) for token in tokens.tolist()]
            if len(token_ids) < size:
                return logits
            prefix = token_ids[-(size - 1) :] if size > 1 else []
            first = max(0, len(token_ids) - window)
            banned: set[int] = set()
            for index in range(first, len(token_ids) - size + 1):
                if size == 1 or token_ids[index : index + size - 1] == prefix:
                    banned.add(token_ids[index + size - 1])
            if banned:
                indices = mx.array(sorted(banned))
                logits[:, indices] = -float("inf")
            return logits

        return [prevent_repeated_ngram]

    async def close(self) -> None:
        self._model = self._processor = None
        session, self._session = self._session, None
        if session is not None:
            await session.close()


_REPLACEMENT_CHAR = "\ufffd"


class _IncrementalTokenDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._emitted = ""
        self._special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def add(self, token: int) -> str:
        if token in self._special_ids:
            return ""
        self._tokens.append(token)
        try:
            decoded = cast(
                str,
                self._tokenizer.decode(
                    self._tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            )
        except TypeError:
            decoded = cast(str, self._tokenizer.decode(self._tokens))
        printable = decoded.rstrip(_REPLACEMENT_CHAR)
        if not printable.startswith(self._emitted):
            return ""
        delta, self._emitted = printable[len(self._emitted) :], printable
        return delta
