import { distance } from "./geometry"
import type { BladeSweep, Point } from "./types"

const POINTER_STALE_MS = 90

export class PointerBladeTracker {
  private blade: BladeSweep | null = null
  private previous: { point: Point; updatedAt: number } | null = null

  update(point: Point, now: number, viewportWidth: number, viewportHeight: number): BladeSweep {
    const diagonal = Math.max(1, Math.hypot(viewportWidth, viewportHeight))
    const elapsedSeconds = this.previous ? Math.max(1 / 240, (now - this.previous.updatedAt) / 1000) : 1
    const speed = this.previous ? distance(this.previous.point, point) / diagonal / elapsedSeconds : 0
    const previousPoint = this.previous?.point ?? point
    this.blade = {
      side: "mouse",
      elbow: previousPoint,
      wrist: point,
      previousElbow: previousPoint,
      previousWrist: point,
      speed,
      active: Boolean(this.previous) && now - this.previous!.updatedAt <= POINTER_STALE_MS,
      updatedAt: now,
    }
    this.previous = { point, updatedAt: now }
    return this.blade
  }

  current(now: number): BladeSweep | null {
    if (!this.blade || now - this.blade.updatedAt > POINTER_STALE_MS) return null
    return this.blade
  }

  reset(): void {
    this.blade = null
    this.previous = null
  }
}
