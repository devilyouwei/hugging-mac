import type { VisionResultBase } from "@/vision/types"

export interface SegmentationResult extends VisionResultBase {
  segments: Array<{
    box: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    confidence: number
    class_id: number
    label: string
    polygons: Array<Array<{
      x: number
      y: number
    }>>
  }>
}
