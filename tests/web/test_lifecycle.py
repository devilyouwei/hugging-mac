from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from hugging_mac_web.lifecycle import close_context


@pytest.mark.asyncio
async def test_platform_shutdown_uses_dependency_order() -> None:
    order: list[str] = []

    async def stop(name: str) -> None:
        order.append(name)

    def stop_store() -> None:
        order.append("document_store")

    context: Any = SimpleNamespace(
        document_parser=SimpleNamespace(close=lambda: stop("document_parser")),
        downloads=SimpleNamespace(close=lambda: stop("downloads")),
        models=SimpleNamespace(
            instances=SimpleNamespace(
                unload_all=lambda **_kwargs: stop("model_instances")
            )
        ),
        documents=SimpleNamespace(close=stop_store),
    )

    await close_context(context)

    assert order == [
        "document_parser",
        "downloads",
        "model_instances",
        "document_store",
    ]
