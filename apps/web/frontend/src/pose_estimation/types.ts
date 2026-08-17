import type { VisionResultBase } from "@/vision/types"

export interface PoseResult extends VisionResultBase {
  parallel_timings: {
    pose_ms: number | null
    face_ms: number | null
    hand_ms: number | null
    sum_ms: number
    round_ms: number
  }
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
  faces: Array<{
    box: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    confidence: number
    landmarks: {
      left_eye: { x: number; y: number; confidence: null }
      right_eye: { x: number; y: number; confidence: null }
      nose: { x: number; y: number; confidence: null }
      left_mouth: { x: number; y: number; confidence: null }
      right_mouth: { x: number; y: number; confidence: null }
    }
  }>
  hands: Array<{
    box: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    confidence: number
    label: string
    handedness: string | null
    handedness_confidence: number | null
    landmark_confidence: number | null
    landmarks: Array<{
      x: number
      y: number
      z: number
    }>
  }>
}
