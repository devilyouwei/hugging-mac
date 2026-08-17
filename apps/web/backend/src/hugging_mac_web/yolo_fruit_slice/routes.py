"""Lightweight status route for the client-side fruit slicing game."""

from fastapi import APIRouter

from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/games/yolo-fruit-slice", tags=["yolo-fruit-slice"])

    @router.get("/status", response_model=ApiResponse[dict[str, str]])
    async def game_status() -> ApiResponse[dict[str, str]]:
        return ApiResponse(
            data={"status": "ready", "inference": "/api/v1/apps/pose-estimation/estimate/frame"},
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
