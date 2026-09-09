"""Runtime-independent PP-DocLayoutV3 instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from hugging_mac_sdk.capabilities import DocumentLayoutAnalysis
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.detection import ImageSize
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRequest, DocumentLayoutResponse
from hugging_mac_sdk.schemas.manifest import ModelManifest

from .config import PPDocLayoutV3InstanceConfig
from .utils.postprocess import postprocess
from .utils.preprocess import prepare_image
from .utils.types import PreparedImage


class PPDocLayoutV3Engine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(self, prepared: PreparedImage) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class PPDocLayoutV3Instance(BaseModelInstance):
    def __init__(
        self,
        config: PPDocLayoutV3InstanceConfig,
        engine: PPDocLayoutV3Engine,
        manifest: ModelManifest,
        *,
        input_size: int,
        labels: tuple[str, ...],
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=config.compute_units if config.runtime == "coreml" else (config.device or "mps"),
        )
        self._manifest_model_id = manifest.model_id
        self._engine = engine
        self._input_size = input_size
        self._labels = labels
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            DocumentLayoutAnalysis,  # type: ignore[type-abstract]
            cast(DocumentLayoutAnalysis, self),
        )

    async def analyze_layout(self, request: DocumentLayoutRequest) -> DocumentLayoutResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("PP-DocLayoutV3 must be READY before analyze_layout")
        prepared = await asyncio.to_thread(prepare_image, request, self._input_size)
        async with self._inference_lock:
            started = perf_counter()
            try:
                outputs = await self._engine.infer(prepared)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("PP-DocLayoutV3 inference failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        regions, timings = await asyncio.to_thread(
            postprocess, outputs, request, prepared, inference_ms, self._labels
        )
        return DocumentLayoutResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            image_size=ImageSize(width=prepared.original_width, height=prepared.original_height),
            regions=regions,
            timings=timings,
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
