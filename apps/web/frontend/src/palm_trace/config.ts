import type { Difficulty } from "./types"
import type { ResourceStatus } from "@/vision/types"

export const HAND_MODEL_ID = "qualcomm/mediapipe-hand-detection"
export const LEVEL_COUNT = 12
export const STARTING_LIVES = 3
export const PALM_STALE_MS = 350
export const SYNC_WINDOW_MS = 700
export const COMPLETION_RATIO = 0.92

export const DIFFICULTIES: Record<Difficulty, { label: string; timeScale: number; toleranceScale: number; offPathMs: number }> = {
  calm: { label: "CALM", timeScale: 1.25, toleranceScale: 1.18, offPathMs: 380 },
  arcade: { label: "ARCADE", timeScale: 1, toleranceScale: 1, offPathMs: 280 },
  rush: { label: "RUSH", timeScale: .82, toleranceScale: .86, offPathMs: 210 },
}

export function selectPalmRuntime(resource: ResourceStatus | null): "coreml" | "onnx" | "" {
  if (resource?.artifacts.some((item) => item.runtime === "coreml" && item.available)) return "coreml"
  if (resource?.artifacts.some((item) => item.runtime === "onnx" && item.available)) return "onnx"
  return ""
}
