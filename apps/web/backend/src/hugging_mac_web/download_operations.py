"""In-process model artifact download jobs with resumable SSE subscriptions."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import datetime
from time import monotonic
from typing import Literal
from uuid import uuid4

from hugging_mac_sdk.resources.downloader import DownloadProgress
from pydantic import BaseModel, ConfigDict

from hugging_mac_web.shared.utils.time_util import utc_now


class ArtifactProgressView(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_key: str
    variant: str | None
    runtime: str | None
    artifact_id: str
    phase: str
    downloaded_bytes: int
    total_bytes: int | None


class DownloadOperationView(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: str
    model_id: str
    requested_artifact_key: str
    state: Literal["queued", "running", "completed", "error"]
    artifacts: tuple[ArtifactProgressView, ...]
    error: str | None
    created_at: datetime
    updated_at: datetime


ProgressReporter = Callable[
    [str, str | None, str | None, str, DownloadProgress],
    None,
]
DownloadRunner = Callable[[ProgressReporter], Awaitable[None]]


class _Operation:
    def __init__(self, operation_id: str, model_id: str, requested_artifact_key: str) -> None:
        now = utc_now()
        self.operation_id = operation_id
        self.model_id = model_id
        self.requested_artifact_key = requested_artifact_key
        self.state: Literal["queued", "running", "completed", "error"] = "queued"
        self.artifacts: dict[str, ArtifactProgressView] = {}
        self.error: str | None = None
        self.created_at = now
        self.updated_at = now
        self.subscribers: set[asyncio.Queue[DownloadOperationView]] = set()
        self.last_published: dict[str, tuple[str, float]] = {}
        self.task: asyncio.Task[None] | None = None

    def view(self) -> DownloadOperationView:
        return DownloadOperationView(
            operation_id=self.operation_id,
            model_id=self.model_id,
            requested_artifact_key=self.requested_artifact_key,
            state=self.state,
            artifacts=tuple(self.artifacts.values()),
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class DownloadOperationManager:
    def __init__(self) -> None:
        self._operations: dict[str, _Operation] = {}
        self._active_by_target: dict[tuple[str, str], str] = {}
        self._artifact_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def create(
        self,
        model_id: str,
        requested_artifact_key: str,
        runner: DownloadRunner,
    ) -> DownloadOperationView:
        target = (model_id, requested_artifact_key)
        existing_id = self._active_by_target.get(target)
        if existing_id is not None:
            return self._operations[existing_id].view()
        operation_id = uuid4().hex
        operation = _Operation(operation_id, model_id, requested_artifact_key)
        self._operations[operation_id] = operation
        self._active_by_target[target] = operation_id
        operation.task = asyncio.create_task(self._run(operation, runner))
        return operation.view()

    def list(self, *, active_only: bool = False) -> tuple[DownloadOperationView, ...]:
        operations: Iterable[_Operation] = self._operations.values()
        if active_only:
            operations = (
                operation
                for operation in operations
                if operation.state in {"queued", "running"}
            )
        return tuple(operation.view() for operation in operations)

    def get(self, operation_id: str) -> DownloadOperationView | None:
        operation = self._operations.get(operation_id)
        return operation.view() if operation is not None else None

    def artifact_lock(self, model_id: str, artifact_key: str) -> asyncio.Lock:
        return self._artifact_locks.setdefault((model_id, artifact_key), asyncio.Lock())

    async def subscribe(self, operation_id: str) -> AsyncIterator[DownloadOperationView]:
        operation = self._operations.get(operation_id)
        if operation is None:
            return
        queue: asyncio.Queue[DownloadOperationView] = asyncio.Queue(maxsize=8)
        operation.subscribers.add(queue)
        try:
            current = operation.view()
            yield current
            while current.state in {"queued", "running"}:
                try:
                    current = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    current = operation.view()
                yield current
        finally:
            operation.subscribers.discard(queue)

    async def close(self) -> None:
        tasks = [operation.task for operation in self._operations.values() if operation.task]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, operation: _Operation, runner: DownloadRunner) -> None:
        operation.state = "running"
        operation.updated_at = utc_now()
        self._publish(operation)
        loop = asyncio.get_running_loop()

        def report(
            artifact_key: str,
            variant: str | None,
            runtime: str | None,
            artifact_id: str,
            progress: DownloadProgress,
        ) -> None:
            args = (
                operation.operation_id,
                artifact_key,
                variant,
                runtime,
                artifact_id,
                progress,
            )
            try:
                same_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                same_loop = False
            if same_loop:
                self._record_progress(*args)
            else:
                loop.call_soon_threadsafe(self._record_progress, *args)

        try:
            await runner(report)
            operation.state = "completed"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            operation.state = "error"
            operation.error = str(error)[:500]
        finally:
            operation.updated_at = utc_now()
            self._active_by_target.pop(
                (operation.model_id, operation.requested_artifact_key), None
            )
            self._publish(operation)

    def _record_progress(
        self,
        operation_id: str,
        artifact_key: str,
        variant: str | None,
        runtime: str | None,
        artifact_id: str,
        progress: DownloadProgress,
    ) -> None:
        operation = self._operations.get(operation_id)
        if operation is None or operation.state not in {"queued", "running"}:
            return
        operation.artifacts[artifact_key] = ArtifactProgressView(
            artifact_key=artifact_key,
            variant=variant,
            runtime=runtime,
            artifact_id=artifact_id,
            phase=progress.phase,
            downloaded_bytes=progress.downloaded_bytes,
            total_bytes=progress.total_bytes,
        )
        operation.updated_at = utc_now()
        previous = operation.last_published.get(artifact_key)
        now = monotonic()
        if (
            previous is None
            or previous[0] != progress.phase
            or now - previous[1] >= 0.1
            or progress.phase == "completed"
        ):
            operation.last_published[artifact_key] = (progress.phase, now)
            self._publish(operation)

    @staticmethod
    def _publish(operation: _Operation) -> None:
        view = operation.view()
        for queue in tuple(operation.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(view)
