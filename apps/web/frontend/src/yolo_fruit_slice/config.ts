export const GAME_CONFIG = {
  initialLives: 3,
  gravityViewport: 0.88,
  initialSpawnMs: 1180,
  minimumSpawnMs: 470,
  spawnAccelerationMsPerSecond: 8,
  maximumBatch: 4,
  bombChanceStart: 0.055,
  bombChanceMaximum: 0.14,
  baseScore: 10,
  comboWindowMs: 260,
  minimumBladeSpeed: 0.38,
  criticalBladeSpeed: 1.05,
  criticalCenterRatio: 0.45,
  poseConfidence: 0.28,
  keypointConfidence: 0.32,
  poseStaleMs: 160,
  smoothingAlpha: 0.58,
} as const

export const COCO_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2],
  [1, 3], [2, 4],
]
