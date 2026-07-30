"""Yolo Pose Follow game APIs."""

from fastapi import APIRouter, HTTPException, status

from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now
from hugging_mac_web.yolo_pose_follow.matching import match_pose
from hugging_mac_web.yolo_pose_follow.schemas import (
    MatchRequest,
    MatchResult,
    PointView,
    PoseTemplateView,
)
from hugging_mac_web.yolo_pose_follow.templates import TEMPLATE_BY_ID, TEMPLATES


def create_router() -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/games/yolo-pose-follow",
        tags=["yolo-pose-follow"],
    )

    @router.get("/templates", response_model=ApiResponse[tuple[PoseTemplateView, ...]])
    async def templates() -> ApiResponse[tuple[PoseTemplateView, ...]]:
        return ApiResponse(
            data=tuple(
                PoseTemplateView(
                    template_id=template.template_id,
                    name=template.name,
                    cue=template.cue,
                    difficulty=template.difficulty,
                    points=tuple(PointView(x=x, y=y) for x, y in template.points),
                    required_keypoints=template.required_keypoints,
                )
                for template in TEMPLATES
            ),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post("/match", response_model=ApiResponse[MatchResult])
    async def match(request: MatchRequest) -> ApiResponse[MatchResult]:
        template = TEMPLATE_BY_ID.get(request.template_id)
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown pose template: {request.template_id}",
            )
        return ApiResponse(
            data=match_pose(template, request),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
