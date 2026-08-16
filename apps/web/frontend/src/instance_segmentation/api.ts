import { createVisionApi } from "@/vision/api"

import type { SegmentationResult } from "./types"

export const {
  fetchResourceStatus,
  infer: segmentInstances,
} = createVisionApi<SegmentationResult>(
  "/api/v1/apps/instance-segmentation",
  "segment",
)
