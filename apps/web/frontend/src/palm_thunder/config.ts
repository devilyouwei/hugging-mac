import type { Difficulty } from "./types"

export const HAND_MODEL_ID = "qualcomm/mediapipe-hand-detection"
export const GESTURE_HOLD_MS = 450
export const HAND_STALE_MS = 520
export const LEVEL_DURATION_MS = 150_000

export const DIFFICULTIES: Record<Difficulty, { label: string; enemySpeed: number; bulletSpeed: number; spawnMs: number; hp: number }> = {
  casual: { label: "CASUAL", enemySpeed: 0.78, bulletSpeed: 0.72, spawnMs: 980, hp: 0.8 },
  arcade: { label: "ARCADE", enemySpeed: 1, bulletSpeed: 1, spawnMs: 720, hp: 1 },
  hardcore: { label: "HARDCORE", enemySpeed: 1.22, bulletSpeed: 1.28, spawnMs: 520, hp: 1.28 },
}
