"""Lightweight status route for Palm Trace."""

from fastapi import APIRouter

from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/games/palm-trace", tags=["palm-trace"])

    @router.get("/status", response_model=ApiResponse[dict[str, str]])
    async def game_status() -> ApiResponse[dict[str, str]]:
        return ApiResponse(
            data={
                "status": "ready",
                "control": "mediapipe-palm-detection",
                "landmarks": "disabled",
            },
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
