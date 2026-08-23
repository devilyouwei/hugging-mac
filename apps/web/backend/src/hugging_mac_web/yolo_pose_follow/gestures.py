"""Handedness-aware gesture matching from MediaPipe's 21 hand landmarks."""

from __future__ import annotations

from math import hypot

from hugging_mac_web.yolo_pose_follow.schemas import (
    GestureHand,
    GestureMatchRequest,
    GestureMatchResult,
    GestureName,
)

FINGER_JOINTS = (
    (5, 6, 8),  # index: MCP, PIP, tip
    (9, 10, 12),  # middle
    (13, 14, 16),
    (17, 18, 20),
)
GESTURE_MASKS: dict[GestureName, tuple[bool, bool, bool, bool]] = {
    "open-palm": (True, True, True, True),
    "fist": (False, False, False, False),
    "victory": (True, True, False, False),
    "point": (True, False, False, False),
}


def match_gesture(request: GestureMatchRequest) -> GestureMatchResult:
    hand = _target_hand(request)
    if hand is None:
        return GestureMatchResult(
            target_hand=request.target_hand,
            target_gesture=request.target_gesture,
            matched=False,
            hand_found=False,
            feedback=f"请将{_side_name(request.target_hand)}手完整放入画面",
        )
    detected = classify_gesture(hand)
    matched = detected == request.target_gesture
    return GestureMatchResult(
        target_hand=request.target_hand,
        target_gesture=request.target_gesture,
        matched=matched,
        detected_gesture=detected,
        hand_found=True,
        feedback=(
            "手势匹配成功" if matched else f"请用{_side_name(request.target_hand)}手做出目标手势"
        ),
    )


def classify_gesture(hand: GestureHand) -> GestureName | None:
    if len(hand.landmarks) != 21:
        return None
    extended = tuple(_finger_extended(hand, *joints) for joints in FINGER_JOINTS)
    for gesture, mask in GESTURE_MASKS.items():
        if extended == mask:
            return gesture
    return None


def _target_hand(request: GestureMatchRequest) -> GestureHand | None:
    candidates = [
        hand
        for hand in request.hands
        if hand.handedness == request.target_hand
        and (hand.handedness_confidence or 0.0) >= 0.65
        and (hand.landmark_confidence or 0.0) >= 0.55
        and len(hand.landmarks) == 21
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda hand: (
            hand.handedness_confidence or 0.0,
            hand.landmark_confidence or 0.0,
        ),
    )


def _finger_extended(hand: GestureHand, mcp: int, pip: int, tip: int) -> bool:
    wrist = hand.landmarks[0]
    a = hand.landmarks[mcp]
    b = hand.landmarks[pip]
    c = hand.landmarks[tip]
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    first_length = hypot(*first)
    second_length = hypot(*second)
    if first_length < 1e-6 or second_length < 1e-6:
        return False
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    tip_distance = hypot(c.x - wrist.x, c.y - wrist.y)
    pip_distance = hypot(b.x - wrist.x, b.y - wrist.y)
    return cosine <= -0.72 and tip_distance >= pip_distance * 1.04


def _side_name(side: str) -> str:
    return "左" if side == "left" else "右"
