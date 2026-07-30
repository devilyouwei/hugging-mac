"""Index catalog REST and SSE endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from hugging_mac_web.app_registry import AppCategory, AppSummary
from hugging_mac_web.context import PlatformContext
from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.sse_util import SseEvent
from hugging_mac_web.shared.utils.time_util import utc_now


def create_index_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])

    @router.get("/apps", response_model=ApiResponse[list[AppSummary]])
    async def list_apps(
        context: ContextDependency,
    ) -> ApiResponse[list[AppSummary]]:
        return ApiResponse(
            data=list(context.apps.list(AppCategory.APPLICATION)),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.get("/games", response_model=ApiResponse[list[AppSummary]])
    async def list_games(
        context: ContextDependency,
    ) -> ApiResponse[list[AppSummary]]:
        return ApiResponse(
            data=list(context.apps.list(AppCategory.GAME)),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.get("/events", response_class=StreamingResponse)
    async def catalog_events(
        request: Request,
        context: ContextDependency,
    ) -> StreamingResponse:
        return StreamingResponse(
            _catalog_event_stream(request, context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router


async def _catalog_event_stream(
    request: Request,
    context: PlatformContext,
) -> AsyncIterator[str]:
    snapshot = await context.models.catalog.snapshot()
    yield SseEvent(
        event="catalog",
        data={
            "models": [model.model_dump(mode="json") for model in snapshot.models],
            "apps": [
                app.model_dump(mode="json")
                for app in context.apps.list(AppCategory.APPLICATION)
            ],
            "games": [
                game.model_dump(mode="json")
                for game in context.apps.list(AppCategory.GAME)
            ],
            "generated_at": snapshot.generated_at.isoformat(),
        },
    ).encode()
    while not await request.is_disconnected():
        await asyncio.sleep(context.settings.sse_heartbeat_seconds)
        yield SseEvent(event="heartbeat", data={"time": utc_now().isoformat()}).encode()
