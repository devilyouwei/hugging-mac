import { describe, expect, it } from "vitest"
import { PalmTracker } from "./tracker"
import type { HandResult } from "./types"

const hand = (x: number, confidence = .9): HandResult => ({ box: { x1: x - 10, y1: 30, x2: x + 10, y2: 60 }, confidence, label: "hand", handedness: null, handedness_confidence: null, landmark_confidence: null, landmarks: [] })

describe("PalmTracker", () => {
  it("keeps stable ids and smooths movement", () => {
    const tracker = new PalmTracker(), first = tracker.update([hand(20), hand(80)], 100, 100, 0)
    const next = tracker.update([hand(85), hand(25)], 100, 100, 20)
    expect(next.map((palm) => palm.id)).toEqual(first.map((palm) => palm.id))
    expect(next[0]!.center.x).toBeGreaterThan(.2); expect(next[0]!.center.x).toBeLessThan(.25)
  })

  it("holds a missing palm briefly and then marks it inactive", () => {
    const tracker = new PalmTracker(); tracker.update([hand(30)], 100, 100, 0)
    expect(tracker.update([], 100, 100, 200)[0]!.active).toBe(true)
    expect(tracker.update([], 100, 100, 400)[0]!.active).toBe(false)
  })
})
