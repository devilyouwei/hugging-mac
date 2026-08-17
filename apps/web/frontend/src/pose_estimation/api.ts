import { createVisionApi } from "@/vision/api"
import { request } from "@/api/client"
import type { ResourceStatus } from "@/vision/types"

import type { PoseResult } from "./types"

export const {
  fetchResourceStatus,
  infer: estimatePoses,
} = createVisionApi<PoseResult>("/api/v1/apps/pose-estimation", "estimate")

export async function fetchFaceResourceStatus(): Promise<ResourceStatus> {
  return (await request<ResourceStatus>("/api/v1/apps/pose-estimation/resources/face")).data
}

export async function fetchHandResourceStatus(): Promise<ResourceStatus> {
  return (await request<ResourceStatus>("/api/v1/apps/pose-estimation/resources/hand")).data
}
