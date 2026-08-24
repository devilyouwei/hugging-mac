from __future__ import annotations

import asyncio

from hugging_mac_sdk.resources.downloader import DownloadProgress
from hugging_mac_web.download_operations import DownloadOperationManager


async def test_download_operation_survives_subscriber_reconnection() -> None:
    manager = DownloadOperationManager()
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(report):  # type: ignore[no-untyped-def]
        report(
            "shared:shared:tokenizer",
            None,
            None,
            "tokenizer",
            DownloadProgress(25, 100),
        )
        started.set()
        await release.wait()
        report(
            "shared:shared:tokenizer",
            None,
            None,
            "tokenizer",
            DownloadProgress(100, 100, "completed"),
        )

    created = manager.create("org/model", "shared:shared:tokenizer", runner)
    await started.wait()

    active = manager.list(active_only=True)
    assert active[0].operation_id == created.operation_id
    assert active[0].artifacts[0].downloaded_bytes == 25

    stream = manager.subscribe(created.operation_id)
    restored = await anext(stream)
    assert restored.artifacts[0].downloaded_bytes == 25

    release.set()
    terminal = await anext(stream)
    while terminal.state != "completed":
        terminal = await anext(stream)
    assert terminal.artifacts[0].downloaded_bytes == 100
    assert not manager.list(active_only=True)
    await stream.aclose()
    await manager.close()


async def test_duplicate_active_download_reuses_operation() -> None:
    manager = DownloadOperationManager()
    release = asyncio.Event()

    async def runner(_report):  # type: ignore[no-untyped-def]
        await release.wait()

    first = manager.create("org/model", "base:mlx:weights", runner)
    second = manager.create("org/model", "base:mlx:weights", runner)

    assert second.operation_id == first.operation_id
    release.set()
    await asyncio.sleep(0)
    await manager.close()
