import { createVisionApi } from "@/vision/api"

import type { PoseResult } from "./types"

export const {
  fetchResourceStatus,
  downloadSource,
  convertCoreMl,
  infer: estimatePoses,
} = createVisionApi<PoseResult>("/api/v1/apps/pose-estimation", "estimate")
