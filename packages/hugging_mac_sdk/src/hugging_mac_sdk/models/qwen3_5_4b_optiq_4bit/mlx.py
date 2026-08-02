"""MLX-VLM runtime engine for Qwen3.5 4B OptiQ."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.schemas.chat import ChatRequest

from .config import Qwen35OptiQMlxInstanceConfig
from .resources import Qwen35OptiQMlxResourceResolver
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
    """Decode generated token IDs without waiting for word-boundary spaces.

    mlx-vlm's ByteLevel streaming detokenizer flushes buffered text when the
    next token starts with a space. That works well for English, but can buffer
    an entire Chinese sentence. Decoding the generated prefix directly lets us
    expose every complete UTF-8 character while still holding incomplete byte
    sequences.
    """

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

        # Byte-level tokens may end part-way through a multi-byte character.
        # Emit the valid prefix now and retain the replacement suffix until the
        # following token completes it.
        printable = decoded.rstrip(_REPLACEMENT_CHAR)
        if not printable.startswith(self._emitted):
            return ""
        delta = printable[len(self._emitted) :]
        self._emitted = printable
        return delta


class MlxQwen35OptiQEngine:
    runtime_name = "mlx"

    def __init__(
        self, config: Qwen35OptiQMlxInstanceConfig, resources: Qwen35OptiQMlxResourceResolver
    ) -> None:
        self._config = config
        self._resources = resources
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        def _load() -> tuple[Any, Any]:
            import mlx.core as mx  # type: ignore[import-not-found]
            from mlx_vlm import load

            # OptiQ publishes the language checkpoint in model.safetensors and
            # keeps the Qwen3.5 visual tower in a dedicated sidecar. Load the
            # base checkpoint lazily, then materialize the model only after the
            # sanitized visual weights have been attached.
            model, processor = load(str(artifact), lazy=True, strict=False)
            raw_vision_weights = mx.load(str(artifact / "optiq" / "optiq_vision.safetensors"))
            vision_weights = _sanitize_vision_weights(model, raw_vision_weights)
            model.load_weights(list(vision_weights.items()), strict=False)
            mx.eval(model.parameters())
            return model, processor

        self._model, self._processor = await asyncio.to_thread(_load)

    async def infer(self, request: ChatRequest) -> GenerationOutput:
        return await asyncio.to_thread(self._generate, request)

    async def stream(self, request: ChatRequest) -> AsyncIterator[GenerationOutput]:
        """Bridge mlx-vlm's synchronous token iterator into the async SDK boundary."""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[GenerationOutput | BaseException | None] = asyncio.Queue()
        cancelled = threading.Event()

        def produce() -> None:
            try:
                for output in self._stream_generate(request):
                    if cancelled.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, output)
            except BaseException as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancelled.set()
            await producer

    async def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    def _generate(self, request: ChatRequest) -> GenerationOutput:
        result = next(self._stream_generate(request, final_only=True))
        return result

    def _stream_generate(
        self,
        request: ChatRequest,
        *,
        final_only: bool = False,
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
                text=result.text if hasattr(result, "text") else str(result),
                prompt_tokens=getattr(result, "prompt_tokens", None),
                generated_tokens=getattr(result, "generation_tokens", None),
            )
            return

        results = stream_generate(self._model, self._processor, prompt, images or None, **kwargs)
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        decoder = _IncrementalTokenDecoder(tokenizer)
        for result in results:
            token = getattr(result, "token", None)
            # mlx-vlm emits a final bookkeeping result which repeats the last
            # token. Only ordinary generation results represent a new token.
            is_final = getattr(result, "finish_reason", None) is not None
            text = decoder.add(int(token)) if token is not None and not is_final else ""
            yield GenerationOutput(
                text=text,
                prompt_tokens=getattr(result, "prompt_tokens", None),
                generated_tokens=getattr(result, "generation_tokens", None),
            )
