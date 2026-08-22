"""MLX-VLM engine for Gemma 4 QAT OptiQ variants."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.runtime.mlx import MlxProvider, MlxSession, materialize_mlx
from hugging_mac_sdk.schemas.chat import ChatRequest

from .config import Gemma4MlxInstanceConfig
from .resources import Gemma4MlxResourceResolver
from .utils.types import GenerationOutput

_REPLACEMENT_CHAR = "\ufffd"


def _sanitize_vision_weights(model: Any, weights: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in model.sanitize(dict(weights)).items()
        if key.startswith("vision_tower.")
    }
    if not sanitized:
        raise RuntimeError("OptiQ vision sidecar does not contain Gemma 4 visual weights")
    return sanitized


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
                    self._tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
            )
        except TypeError:
            decoded = cast(str, self._tokenizer.decode(self._tokens))
        printable = decoded.rstrip(_REPLACEMENT_CHAR)
        if not printable.startswith(self._emitted):
            return ""
        delta, self._emitted = printable[len(self._emitted) :], printable
        return delta


class MlxGemma4Engine:
    runtime_name = "mlx"

    def __init__(
        self, config: Gemma4MlxInstanceConfig, resources: Gemma4MlxResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._model: Any | None = None
        self._processor: Any | None = None
        self._session: MlxSession | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        def _load(source: Path, mx: Any) -> tuple[Any, Any]:
            from mlx_vlm import load

            with merged_directory_view(source, (self._resources.tokenizer_path,)) as model_view:
                model, processor = load(str(model_view), lazy=True, strict=False)
                weights = _sanitize_vision_weights(
                    model, mx.load(str(source / "optiq" / "optiq_vision.safetensors"))
                )
                model.load_weights(list(weights.items()), strict=False)
                materialize_mlx(model, mx)
                return model, processor

        self._session = await MlxProvider().create_session(
            artifact, device=None, options={"model_loader": _load}
        )
        self._model, self._processor = self._session.value

    async def infer(self, request: ChatRequest) -> GenerationOutput:
        return await asyncio.to_thread(lambda: next(self._generate(request, final_only=True)))

    async def stream(self, request: ChatRequest) -> AsyncIterator[GenerationOutput]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[GenerationOutput | BaseException | None] = asyncio.Queue()
        cancelled = threading.Event()

        def produce() -> None:
            try:
                for output in self._generate(request):
                    if cancelled.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, output)
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

    async def close(self) -> None:
        self._model = self._processor = None
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    def _generate(
        self, request: ChatRequest, *, final_only: bool = False
    ) -> Iterator[GenerationOutput]:
        if self._model is None or self._processor is None:
            raise RuntimeError("MLX model is not loaded")
        from mlx_vlm import generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [{"role": item.role, "content": item.content} for item in request.messages]
        images = [image.source() for item in request.messages for image in item.images]
        prompt = apply_chat_template(
            self._processor,
            self._model.config,
            messages,
            num_images=len(images),
            enable_thinking=request.enable_thinking,
        )
        kwargs = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "verbose": False,
        }
        if final_only:
            result = generate(self._model, self._processor, prompt, images or None, **kwargs)
            yield GenerationOutput(
                text=getattr(result, "text", str(result)),
                prompt_tokens=getattr(result, "prompt_tokens", None),
                generated_tokens=getattr(result, "generation_tokens", None),
            )
            return
        decoder = _IncrementalTokenDecoder(getattr(self._processor, "tokenizer", self._processor))
        for result in stream_generate(
            self._model, self._processor, prompt, images or None, **kwargs
        ):
            final = getattr(result, "finish_reason", None) is not None
            token = getattr(result, "token", None)
            yield GenerationOutput(
                text=decoder.add(int(token)) if token is not None and not final else "",
                prompt_tokens=getattr(result, "prompt_tokens", None),
                generated_tokens=getattr(result, "generation_tokens", None),
            )
