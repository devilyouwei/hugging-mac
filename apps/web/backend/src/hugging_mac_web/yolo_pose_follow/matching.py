# ruff: noqa: RUF001
"""Scale-independent pose matching."""

from math import hypot

from hugging_mac_web.yolo_pose_follow.schemas import MatchRequest, MatchResult
from hugging_mac_web.yolo_pose_follow.templates import PoseTemplate

CONFIDENCE_FLOOR = 0.22
MATCH_THRESHOLD = 0.72


def match_pose(template: PoseTemplate, request: MatchRequest) -> MatchResult:
    player = tuple((point.x, point.y, point.confidence) for point in request.keypoints)
    direct = _similarity(template, player, mirrored=False)
    mirrored = _similarity(template, player, mirrored=True) if request.allow_mirror else (-1.0, 0)
    is_mirrored = mirrored[0] > direct[0]
    score, visible = mirrored if is_mirrored else direct
    matched = score >= MATCH_THRESHOLD
    if visible < max(6, len(template.required_keypoints) // 2):
        feedback = "请让完整身体进入画面"
    elif matched:
        feedback = "姿态匹配成功"
    elif score >= 0.56:
        feedback = "很接近了，再调整手脚角度"
    else:
        feedback = "对照目标骨架调整姿势"
    return MatchResult(
        template_id=template.template_id,
        score=max(0.0, min(1.0, score)),
        matched=matched,
        mirrored=is_mirrored,
        visible_keypoints=visible,
        feedback=feedback,
    )


def _similarity(
    template: PoseTemplate,
    player: tuple[tuple[float, float, float | None], ...],
    *,
    mirrored: bool,
) -> tuple[float, int]:
    left_shoulder, right_shoulder = player[5], player[6]
    left_hip, right_hip = player[11], player[12]
    center_x = (left_hip[0] + right_hip[0]) / 2
    center_y = (left_hip[1] + right_hip[1]) / 2
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    scale = hypot(
        (left_shoulder[0] + right_shoulder[0]) / 2 - center_x,
        shoulder_y - center_y,
    )
    if scale < 1:
        return 0.0, 0

    normalized: list[tuple[float, float] | None] = []
    for x, y, confidence in player:
        if confidence is not None and confidence < CONFIDENCE_FLOOR:
            normalized.append(None)
            continue
        nx = (x - center_x) / scale
        normalized.append((-nx if mirrored else nx, (y - center_y) / scale))

    target_center_x = (template.points[11][0] + template.points[12][0]) / 2
    target_center_y = (template.points[11][1] + template.points[12][1]) / 2
    target_shoulder_y = (template.points[5][1] + template.points[6][1]) / 2
    target_scale = abs(target_center_y - target_shoulder_y)
    distances: list[float] = []
    for index in template.required_keypoints:
        point = normalized[index]
        if point is None:
            continue
        target = template.points[index]
        tx = (target[0] - target_center_x) / target_scale
        ty = (target[1] - target_center_y) / target_scale
        distances.append(hypot(point[0] - tx, point[1] - ty))
    if not distances:
        return 0.0, 0
    mean_distance = sum(min(distance, 2.0) for distance in distances) / len(distances)
    return max(0.0, 1.0 - mean_distance / 1.05), len(distances)
