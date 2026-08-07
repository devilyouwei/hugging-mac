"""MLX-VLM engine with a variant-specific OptiQ load strategy."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.schemas.chat import ChatRequest

from .config import Qwen35MlxInstanceConfig
from .resources import Qwen35MlxResourceResolver
from .utils.types import GenerationOutput

_REPLACEMENT_CHAR = "\ufffd"


def _sanitize_vision_weights(model: Any, weights: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in model.sanitize(dict(weights)).items()
        if key.startswith("vision_tower.")
    }
    if not sanitized:
        raise RuntimeError("OptiQ vision sidecar does not contain Qwen3.5 visual weights")
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


class MlxQwen35Engine:
    runtime_name = "mlx"

    def __init__(
        self, config: Qwen35MlxInstanceConfig, resources: Qwen35MlxResourceResolver
    ) -> None:
        self._config, self._resources = config, resources
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        def _load() -> tuple[Any, Any]:
            from mlx_vlm import load

            if self._config.variant == "9b-4bit":
                return cast(tuple[Any, Any], load(str(artifact)))
            import mlx.core as mx

            model, processor = load(str(artifact), lazy=True, strict=False)
            weights = _sanitize_vision_weights(
                model, mx.load(str(artifact / "optiq" / "optiq_vision.safetensors"))
            )
            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            return model, processor

        self._model, self._processor = await asyncio.to_thread(_load)

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
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

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
