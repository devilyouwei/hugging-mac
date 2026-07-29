import { createVisionApi } from "@/vision/api"

import type { SegmentationResult } from "./types"

export const {
  fetchResourceStatus,
  downloadSource,
  convertCoreMl,
  infer: segmentInstances,
} = createVisionApi<SegmentationResult>(
  "/api/v1/apps/instance-segmentation",
  "segment",
)
