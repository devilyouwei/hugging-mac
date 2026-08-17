"""Core ML runtime engine for the two-stage MediaPipe Hand pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider

from .config import MediaPipeHandDetectionInstanceConfig
from .resources import MediaPipeHandDetectionResourceResolver
from .utils.types import PreparedImage, PreparedLandmarkCrop


class CoreMlMediaPipeHandDetectionEngine:
    runtime_name = "coreml"

    def __init__(
        self,
        config: MediaPipeHandDetectionInstanceConfig,
        resources: MediaPipeHandDetectionResourceResolver,
    ) -> None:
        self._config = config
        self._artifact_root: Path | None = None
        self._resources = resources
        self._detector_session: RuntimeSession | None = None
        self._landmark_session: RuntimeSession | None = None

    @property
    def device(self) -> str:
        return (
            self._detector_session.device
            if self._detector_session is not None
            else self._config.compute_units
        )

    async def resolve(self) -> Path:
        return (await self._resources.resolve_coreml()).path

    async def load(self, artifact: Path) -> None:
        self._artifact_root = artifact
        self._detector_session = await self._create_session(artifact / "hand_detector.mlpackage")

    async def infer_detector(self, prepared: PreparedImage) -> dict[str, Any]:
        if self._detector_session is None:
            raise RuntimeError("MediaPipe palm detector Core ML session is not loaded")
        return dict(await asyncio.to_thread(self._detector_session.run, {"image": prepared.tensor}))

    async def infer_landmarks(self, prepared: PreparedLandmarkCrop) -> dict[str, Any]:
        if self._landmark_session is None:
            if self._artifact_root is None:
                raise RuntimeError("MediaPipe Hand Core ML artifact is not resolved")
            self._landmark_session = await self._create_session(
                self._artifact_root / "hand_landmark_detector.mlpackage"
            )
        return dict(await asyncio.to_thread(self._landmark_session.run, {"image": prepared.tensor}))

    async def _create_session(self, path: Path) -> RuntimeSession:
        return await CoreMLProvider().create_session(
            path,
            device=self._config.compute_units,
            options={},
        )

    async def close(self) -> None:
        sessions = (self._landmark_session, self._detector_session)
        self._landmark_session = None
        self._detector_session = None
        self._artifact_root = None
        for session in sessions:
            if session is not None:
                await session.close()
