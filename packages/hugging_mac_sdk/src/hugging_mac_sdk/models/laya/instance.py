"""Runtime-neutral Laya lifecycle and decision capability mapping."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.core.decision import execute_decisions
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.decision import DecisionRequest, DecisionResponse, DecisionSupport
from hugging_mac_sdk.schemas.manifest import ModelManifest

from .config import LayaInstanceConfig
from .utils.preprocess import prepare_request


class LayaEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class LayaInstance(BaseModelInstance):
    def __init__(
        self,
        config: LayaInstanceConfig,
        engine: LayaEngine,
        manifest: ModelManifest,
        support: DecisionSupport,
    ) -> None:
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=engine.runtime_name,
            device=engine.device,
        )
        self._support = support
        self._response_model = str(manifest.get_variant(config.variant).metadata["response_model"])
        self._manifest_model_id = manifest.model_id
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(DecisionMaking, cast(DecisionMaking, self))  # type: ignore[type-abstract]

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        async with self._inference_lock:
            await self._engine.close()

    def decision_support(self) -> DecisionSupport:
        return self._support

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        async with self._inference_lock:
            if self.state is not ModelState.READY:
                raise InferenceError("Load the Laya instance before inference")
            try:
                return await execute_decisions(
                    request,
                    support=self._support,
                    model_id=self._manifest_model_id,
                    instance_id=str(self.instance_id),
                    runtime=self._engine.runtime_name,
                    device=self._engine.device,
                    model_name=self._response_model,
                    prepare=prepare_request,
                    infer=self._engine.infer,
                )
            except asyncio.CancelledError:
                raise
            except InferenceError:
                raise
            except Exception as error:
                raise InferenceError("Laya inference failed", cause=error) from error
