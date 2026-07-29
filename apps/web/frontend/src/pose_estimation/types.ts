import type { VisionResultBase } from "@/vision/types"

export interface PoseResult extends VisionResultBase {
  poses: Array<{
    box: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    confidence: number
    class_id: number
    label: string
    keypoints: Array<{
      x: number
      y: number
      confidence: number | null
    }>
  }>
}
