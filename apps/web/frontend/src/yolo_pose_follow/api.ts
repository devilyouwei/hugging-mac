import { request } from "@/api/client"
import type { PoseResult } from "@/pose_estimation/types"

import type { GestureMatch, GestureTarget, PoseMatch, PoseTemplate } from "./types"

const PREFIX = "/api/v1/games/yolo-pose-follow"

export async function fetchPoseTemplates(): Promise<PoseTemplate[]> {
  return (await request<PoseTemplate[]>(`${PREFIX}/templates`)).data
}

export async function matchPose(
  templateId: string,
  pose: PoseResult["poses"][number],
  signal?: AbortSignal,
): Promise<PoseMatch> {
  return (
    await request<PoseMatch>(`${PREFIX}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        template_id: templateId,
        keypoints: pose.keypoints,
        allow_mirror: true,
      }),
      signal,
    })
  ).data
}

export async function matchGesture(
  target: GestureTarget,
  hands: PoseResult["hands"],
  signal?: AbortSignal,
): Promise<GestureMatch> {
  return (
    await request<GestureMatch>(`${PREFIX}/match-gesture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_hand: target.hand,
        target_gesture: target.gesture,
        hands,
      }),
      signal,
    })
  ).data
}
