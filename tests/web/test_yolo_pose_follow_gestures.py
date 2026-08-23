from __future__ import annotations

import pytest
from hugging_mac_web.yolo_pose_follow.gestures import classify_gesture, match_gesture
from hugging_mac_web.yolo_pose_follow.schemas import (
    GestureHand,
    GestureLandmark,
    GestureMatchRequest,
    GestureName,
)


def _hand(
    side: str,
    extended: tuple[bool, bool, bool, bool],
) -> GestureHand:
    points = [GestureLandmark(x=0, y=0) for _ in range(21)]
    for finger, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        x = float(finger + 1)
        points[mcp] = GestureLandmark(x=x, y=1)
        points[pip] = GestureLandmark(x=x, y=2)
        points[tip] = (
            GestureLandmark(x=x, y=4) if extended[finger] else GestureLandmark(x=x + 0.5, y=1.5)
        )
    return GestureHand(
        handedness=side,  # type: ignore[arg-type]
        handedness_confidence=0.95,
        landmark_confidence=0.9,
        landmarks=tuple(points),
    )


@pytest.mark.parametrize(
    ("gesture", "mask"),
    (
        ("open-palm", (True, True, True, True)),
        ("fist", (False, False, False, False)),
        ("victory", (True, True, False, False)),
        ("point", (True, False, False, False)),
    ),
)
def test_classifies_supported_hand_gestures(
    gesture: GestureName,
    mask: tuple[bool, bool, bool, bool],
) -> None:
    assert classify_gesture(_hand("right", mask)) == gesture


def test_gesture_match_strictly_requires_requested_physical_hand() -> None:
    request = GestureMatchRequest(
        target_hand="right",
        target_gesture="victory",
        hands=(_hand("left", (True, True, False, False)),),
    )

    result = match_gesture(request)

    assert not result.matched
    assert not result.hand_found
    assert result.detected_gesture is None


def test_gesture_match_rejects_uncertain_handedness() -> None:
    hand = _hand("right", (True, True, False, False)).model_copy(
        update={"handedness_confidence": 0.51}
    )

    result = match_gesture(
        GestureMatchRequest(
            target_hand="right",
            target_gesture="victory",
            hands=(hand,),
        )
    )

    assert not result.matched
    assert not result.hand_found


def test_gesture_match_accepts_matching_gesture_on_requested_hand() -> None:
    request = GestureMatchRequest(
        target_hand="left",
        target_gesture="point",
        hands=(
            _hand("right", (True, True, True, True)),
            _hand("left", (True, False, False, False)),
        ),
    )

    result = match_gesture(request)

    assert result.matched
    assert result.hand_found
    assert result.detected_gesture == "point"
