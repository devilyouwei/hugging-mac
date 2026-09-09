"""MLX-VLM engine for the GOT-OCR2.0 eight-bit checkpoint."""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.runtime.mlx import MlxProvider, MlxSession, materialize_mlx
from hugging_mac_sdk.schemas.document_parsing import DocumentParsingRequest

from .config import GotOcr2MlxInstanceConfig
from .resources import GotOcr2ResourceResolver

_FORMATTED_OCR_PROMPT = "OCR with format: "


class MlxGotOcr2Engine:
    runtime_name = "mlx"

    def __init__(
        self, config: GotOcr2MlxInstanceConfig, resources: GotOcr2ResourceResolver
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
            from mlx_vlm.utils import load_model

            model = load_model(source, lazy=True)
            processor = _load_native_processor(source)
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
            raise RuntimeError("GOT-OCR2.0 is not loaded")
        from mlx_vlm.generate import generate

        pages: list[str] = []
        for index, image in enumerate(request.images, start=1):
            prompt = self._formatted_prompt()
            result = generate(
                self._model,
                self._processor,
                prompt,
                [str(image.path)],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                verbose=False,
            )
            text = str(getattr(result, "text", result)).strip()
            pages.append(self._page_output(index, text, page_count=len(request.images)))
        return "\n\n".join(pages)

    def _stream_sync(self, request: DocumentParsingRequest) -> Iterator[str]:
        if self._model is None or self._processor is None:
            raise RuntimeError("GOT-OCR2.0 is not loaded")
        from mlx_vlm.generate import stream_generate

        page_count = len(request.images)
        for index, image in enumerate(request.images, start=1):
            if page_count > 1:
                yield f"<!-- Page {index} -->\n\n"
            decoder = _IncrementalTokenDecoder(
                getattr(self._processor, "tokenizer", self._processor)
            )
            for result in stream_generate(
                self._model,
                self._processor,
                self._formatted_prompt(),
                [str(image.path)],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                verbose=False,
            ):
                token = getattr(result, "token", None)
                if token is not None:
                    yield decoder.add(int(token))
            if index < page_count:
                yield "\n\n"

    def _formatted_prompt(self) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("GOT-OCR2.0 is not loaded")
        from mlx_vlm.prompt_utils import apply_chat_template

        return cast(
            str,
            apply_chat_template(
                self._processor,
                self._model.config,
                _FORMATTED_OCR_PROMPT,
                num_images=1,
            ),
        )

    @staticmethod
    def _page_output(index: int, text: str, *, page_count: int) -> str:
        if page_count == 1:
            return text
        return f"<!-- Page {index} -->\n\n{text}"

    async def close(self) -> None:
        self._model = self._processor = None
        session, self._session = self._session, None
        if session is not None:
            await session.close()


def _load_native_processor(source: Path) -> Any:
    """Build GOT's MLX processor without importing its PyTorch model code."""
    tokenizer_path = source / "tokenization_qwen.py"
    spec = importlib.util.spec_from_file_location(
        "_hugging_mac_got_tokenizer", tokenizer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load GOT tokenizer from {tokenizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer_class = module.QWenTokenizer
    tokenizer = tokenizer_class.from_pretrained(str(source), local_files_only=True)

    from mlx_vlm.models.got.config import ENDOFTEXT_TOKEN, IM_END_TOKEN
    from mlx_vlm.models.got.processing_got import GotOcrProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    processor = GotOcrProcessor(tokenizer=tokenizer)
    detokenizer_class = load_tokenizer(source, return_tokenizer=False)
    processor.detokenizer = detokenizer_class(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(
        [ENDOFTEXT_TOKEN, IM_END_TOKEN], tokenizer
    )
    return processor


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
