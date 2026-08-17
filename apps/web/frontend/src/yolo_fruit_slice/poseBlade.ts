import { COCO_CONNECTIONS, GAME_CONFIG } from "./config"
import { distance, mapCoverPoint } from "./geometry"
import type { BladeSweep, Point, PoseFrame, PosePerson } from "./types"

interface TrackedArm {
  elbow: Point
  wrist: Point
  updatedAt: number
}

const ARM_INDICES = {
  left: { elbow: 7, wrist: 9 },
  right: { elbow: 8, wrist: 10 },
} as const

export class PoseBladeTracker {
  private arms: Partial<Record<"left" | "right", TrackedArm>> = {}

  reset(): void {
    this.arms = {}
  }

  update(
    person: PosePerson | undefined,
    sourceWidth: number,
    sourceHeight: number,
    viewportWidth: number,
    viewportHeight: number,
    now: number,
    inferenceMs: number | null = null,
  ): PoseFrame {
    const mapped = (person?.keypoints ?? []).map((keypoint) => {
      if ((keypoint.confidence ?? 0) < GAME_CONFIG.keypointConfidence) return null
      return mapCoverPoint(keypoint, sourceWidth, sourceHeight, viewportWidth, viewportHeight, true)
    })
    const diagonal = Math.max(1, Math.hypot(viewportWidth, viewportHeight))
    const blades: BladeSweep[] = []

    for (const side of ["left", "right"] as const) {
      const indices = ARM_INDICES[side]
      const elbow = mapped[indices.elbow]
      const wrist = mapped[indices.wrist]
      const previous = this.arms[side]
      if (!elbow || !wrist) {
        if (previous && now - previous.updatedAt > GAME_CONFIG.poseStaleMs) delete this.arms[side]
        continue
      }

      const smoothedElbow = previous ? smooth(previous.elbow, elbow, GAME_CONFIG.smoothingAlpha) : elbow
      const smoothedWrist = previous ? smooth(previous.wrist, wrist, GAME_CONFIG.smoothingAlpha) : wrist
      const elapsedSeconds = previous ? Math.max(1 / 120, (now - previous.updatedAt) / 1000) : 1
      const speed = previous ? distance(previous.wrist, smoothedWrist) / diagonal / elapsedSeconds : 0
      blades.push({
        side,
        elbow: smoothedElbow,
        wrist: smoothedWrist,
        previousElbow: previous?.elbow ?? smoothedElbow,
        previousWrist: previous?.wrist ?? smoothedWrist,
        speed,
        active: Boolean(previous),
        updatedAt: now,
      })
      this.arms[side] = { elbow: smoothedElbow, wrist: smoothedWrist, updatedAt: now }
      mapped[indices.elbow] = smoothedElbow
      mapped[indices.wrist] = smoothedWrist
    }

    return { keypoints: mapped, blades, detected: Boolean(person), inferenceMs }
  }
}

function smooth(previous: Point, next: Point, alpha: number): Point {
  return {
    x: previous.x + (next.x - previous.x) * alpha,
    y: previous.y + (next.y - previous.y) * alpha,
  }
}

export function visibleSkeletonSegments(keypoints: Array<Point | null>): Array<[Point, Point]> {
  const segments: Array<[Point, Point]> = []
  for (const [fromIndex, toIndex] of COCO_CONNECTIONS) {
    const from = keypoints[fromIndex]
    const to = keypoints[toIndex]
    if (from && to) segments.push([from, to])
  }
  return segments
}
