"""D3 Core AI sessions; Torch is used only for host tensor preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession

from .config import D3InstanceConfig
from .resources import D3ResourceResolver
from .utils.execution import run_blocking
from .utils.types import CoreAILayout


class CoreAID3Engine:
    runtime_name = "coreai"

    def __init__(
        self,
        config: D3InstanceConfig,
        resources: D3ResourceResolver,
        layout: CoreAILayout,
        *,
        model_name: str,
    ) -> None:
        self._config, self._resources = config, resources
        self._layout, self._model_name = layout, model_name
        self._decision: CoreAISession | None = None
        self._vision: CoreAISession | None = None
        self._adapter: Any = None
        self._views = ExitStack()

    @property
    def device(self) -> str:
        return self._decision.device if self._decision is not None else self._config.device

    async def resolve(self) -> Path:
        return await self._resources.resolve()

    async def load(self, artifact: Path) -> None:
        from .utils.adapter import create_runtime

        provider = CoreAIProvider()
        try:
            view = self._views.enter_context(self._resources.view())
            self._decision = await provider.create_session(
                artifact / self._layout.decision_asset, device=self._config.device, options={}
            )
            self._vision = await provider.create_session(
                artifact / self._layout.vision_asset, device=self._config.device, options={}
            )
            self._adapter = create_runtime(
                view,
                self._decision,
                self._vision,
                model_name=self._model_name,
                permutation_average=self._config.permutation_average,
            )
        except BaseException:
            await self.close()
            raise

    async def infer(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        if self._adapter is None:
            raise RuntimeError("D3 engine is not loaded")
        return dict(
            await run_blocking(
                self._adapter.system_one,
                state=state,
                questions=questions,
                images=images,
                videos=videos,
            )
        )

    async def close(self) -> None:
        vision, decision = self._vision, self._decision
        self._adapter = self._vision = self._decision = None
        try:
            if vision is not None:
                await vision.close()
        finally:
            try:
                if decision is not None:
                    await decision.close()
            finally:
                self._views.close()
