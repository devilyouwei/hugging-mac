"""Runtime-neutral Unlimited-OCR instance."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from hugging_mac_sdk.capabilities import DocumentParsing
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.document_parsing import (
    DocumentParsingRequest,
    DocumentParsingResponse,
    DocumentParsingStreamEvent,
    DocumentParsingTimings,
)
from hugging_mac_sdk.schemas.manifest import ModelManifest

from .config import UnlimitedOcrMlxInstanceConfig


class UnlimitedOcrEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(self, request: DocumentParsingRequest) -> str: ...
    def stream(self, request: DocumentParsingRequest) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


class UnlimitedOcrInstance(BaseModelInstance):
    def __init__(
        self,
        config: UnlimitedOcrMlxInstanceConfig,
        engine: UnlimitedOcrEngine,
        manifest: ModelManifest,
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device,
        )
        self._engine, self._inference_lock = engine, asyncio.Lock()
        self._manifest_model_id = manifest.model_id
        self.register_capability(DocumentParsing, cast(DocumentParsing, self))  # type: ignore[type-abstract]

    async def parse_document(self, request: DocumentParsingRequest) -> DocumentParsingResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("Unlimited-OCR instance must be READY before parsing")
        started = perf_counter()
        async with self._inference_lock:
            try:
                text = await self._engine.infer(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Unlimited-OCR parsing failed",
                    details={"reason": str(error)[:500]},
                    cause=error,
                ) from error
        return DocumentParsingResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            text=text,
            page_count=len(request.images),
            timings=DocumentParsingTimings(inference_ms=(perf_counter() - started) * 1000),
        )

    async def stream_document(
        self, request: DocumentParsingRequest
    ) -> AsyncIterator[DocumentParsingStreamEvent]:
        if self.state is not ModelState.READY:
            raise InferenceError("Unlimited-OCR instance must be READY before streaming")
        async with self._inference_lock:
            try:
                async for delta in self._engine.stream(request):
                    if delta:
                        yield DocumentParsingStreamEvent(delta=delta)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Unlimited-OCR streaming failed",
                    details={"reason": str(error)[:500]},
                    cause=error,
                ) from error
        yield DocumentParsingStreamEvent(finish_reason="stop")

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
