import { request } from "@/api/client"
import type { PoseResult } from "@/pose_estimation/types"

import type { PoseMatch, PoseTemplate } from "./types"

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
