import { describe, expect, it } from "vitest"

import type { PosePerson } from "./types"
import { PoseBladeTracker } from "./poseBlade"

function person(wristX: number, confidence = 0.9): PosePerson {
  const keypoints = Array.from({ length: 17 }, () => ({ x: 50, y: 50, confidence }))
  keypoints[7] = { x: 30, y: 50, confidence }
  keypoints[9] = { x: wristX, y: 50, confidence }
  keypoints[8] = { x: 70, y: 50, confidence }
  keypoints[10] = { x: 80, y: 50, confidence }
  return { keypoints } as PosePerson
}

describe("PoseBladeTracker", () => {
  it("builds independent elbow-to-wrist blades and smooths movement", () => {
    const tracker = new PoseBladeTracker()
    expect(tracker.update(person(20), 100, 100, 100, 100, 0).blades.every((blade) => !blade.active)).toBe(true)
    const frame = tracker.update(person(40), 100, 100, 100, 100, 100)
    const left = frame.blades.find((blade) => blade.side === "left")!
    expect(left.active).toBe(true)
    expect(left.wrist.x).toBeCloseTo(68.4)
    expect(left.speed).toBeGreaterThan(0)
  })

  it("rejects low-confidence elbows and wrists", () => {
    const tracker = new PoseBladeTracker()
    expect(tracker.update(person(20, 0.1), 100, 100, 100, 100, 0).blades).toHaveLength(0)
  })
})
