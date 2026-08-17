import { describe, expect, it } from "vitest"

import { PointerBladeTracker } from "./pointerBlade"

describe("PointerBladeTracker", () => {
  it("turns consecutive mouse movement into a normalized blade sweep", () => {
    const tracker = new PointerBladeTracker()
    expect(tracker.update({ x: 10, y: 20 }, 0, 800, 600).active).toBe(false)
    const blade = tracker.update({ x: 110, y: 20 }, 50, 800, 600)
    expect(blade.side).toBe("mouse")
    expect(blade.active).toBe(true)
    expect(blade.elbow).toEqual({ x: 10, y: 20 })
    expect(blade.wrist).toEqual({ x: 110, y: 20 })
    expect(blade.speed).toBeCloseTo(2)
  })

  it("expires quickly when the mouse stops or leaves the stage", () => {
    const tracker = new PointerBladeTracker()
    tracker.update({ x: 0, y: 0 }, 0, 100, 100)
    tracker.update({ x: 20, y: 0 }, 20, 100, 100)
    expect(tracker.current(80)).not.toBeNull()
    expect(tracker.current(111)).toBeNull()
    tracker.reset()
    expect(tracker.current(20)).toBeNull()
  })
})
