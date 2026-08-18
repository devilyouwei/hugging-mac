"""HTTP contracts for Yolo Pose Follow."""

from typing import Literal

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


GestureName = Literal["open-palm", "fist", "victory", "point"]
HandSide = Literal["left", "right"]


class GestureLandmark(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float = 0.0


class GestureHand(BaseModel):
    model_config = ConfigDict(frozen=True)

    handedness: HandSide | None = None
    handedness_confidence: float | None = None
    landmark_confidence: float | None = None
    landmarks: tuple[GestureLandmark, ...] = Field(default=(), max_length=21)


class GestureMatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_hand: HandSide
    target_gesture: GestureName
    hands: tuple[GestureHand, ...] = Field(default=(), max_length=4)


class GestureMatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_hand: HandSide
    target_gesture: GestureName
    matched: bool
    detected_gesture: GestureName | None = None
    hand_found: bool
    feedback: str
