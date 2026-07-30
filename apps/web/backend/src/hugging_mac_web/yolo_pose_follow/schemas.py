"""HTTP contracts for Yolo Pose Follow."""

from pydantic import BaseModel, ConfigDict, Field


class PointView(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class PoseTemplateView(BaseModel):
    model_config = ConfigDict(frozen=True)

    template_id: str
    name: str
    cue: str
    difficulty: int = Field(ge=1, le=4)
    points: tuple[PointView, ...]
    required_keypoints: tuple[int, ...]


class PlayerKeypoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    confidence: float | None = None


class MatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    template_id: str
    keypoints: tuple[PlayerKeypoint, ...] = Field(min_length=17, max_length=17)
    allow_mirror: bool = True


class MatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    template_id: str
    score: float
    matched: bool
    mirrored: bool
    visible_keypoints: int
    feedback: str
