import { describe, expect, it } from "vitest"
import type { HandResult } from "./types"
import { classifyGesture, HandTracker } from "./gestures"

const base = [
  [50, 90], [42, 78], [34, 65], [25, 55], [16, 48],
  [40, 62], [39, 45], [39, 28], [39, 10],
  [50, 60], [50, 41], [50, 24], [50, 6],
  [60, 62], [61, 44], [61, 29], [61, 14],
  [70, 66], [72, 51], [74, 38], [76, 24],
]

function hand(overrides: Record<number, [number, number]> = {}): HandResult {
  const landmarks = base.map(([x, y], index) => ({ x: overrides[index]?.[0] ?? x!, y: overrides[index]?.[1] ?? y!, z: 0 }))
  return { box: { x1: 10, y1: 5, x2: 90, y2: 100 }, confidence: .9, label: "hand", handedness: "right", handedness_confidence: .9, landmark_confidence: .9, landmarks }
}

describe("Palm Thunder gestures", () => {
  it("recognizes an open palm", () => expect(classifyGesture(hand())).toBe("open"))
  it("recognizes a fist", () => expect(classifyGesture(hand({ 8: [48, 72], 12: [51, 73], 16: [58, 74], 20: [67, 76], 4: [42, 72] }))).toBe("fist"))
  it("recognizes I-love-you without confusing it with a fist", () => {
    const love = hand({ 12: [51, 73], 16: [58, 74] })
    expect(classifyGesture(love)).toBe("love")
  })
  it("requires 450ms and an open-hand rearm", () => {
    const tracker = new HandTracker(); const open = hand(); const fist = hand({ 8: [48, 72], 12: [51, 73], 16: [58, 74], 20: [67, 76], 4: [42, 72] })
    tracker.update([open], 100, 100, 0); tracker.update([fist], 100, 100, 100); tracker.update([fist], 100, 100, 549)
    expect(tracker.consume(1, "fist")).toBe(false)
    tracker.update([fist], 100, 100, 550); expect(tracker.consume(1, "fist")).toBe(true); expect(tracker.consume(1, "fist")).toBe(false)
    tracker.update([open], 100, 100, 600); tracker.update([fist], 100, 100, 700); tracker.update([fist], 100, 100, 1150); expect(tracker.consume(1, "fist")).toBe(true)
  })
  it("locks the configured number of hands for the whole game", () => {
    const tracker = new HandTracker()
    tracker.reset(1)
    expect(tracker.update([hand(), { ...hand(), handedness: "left", box: { x1: 110, y1: 5, x2: 190, y2: 100 } }], 200, 100, 0)).toHaveLength(1)
    tracker.lock()
    expect(tracker.update([], 200, 100, 600)).toHaveLength(1)
    expect(tracker.update([{ ...hand(), handedness: "left", box: { x1: 110, y1: 5, x2: 190, y2: 100 } }], 200, 100, 700)).toHaveLength(1)
  })
  it("allows one hand to claim a mouse-only locked solo slot", () => {
    const tracker = new HandTracker()
    tracker.reset(1)
    tracker.lock()
    expect(tracker.update([hand()], 100, 100, 500)).toHaveLength(1)
    expect(tracker.update([hand(), { ...hand(), handedness: "left" }], 100, 100, 600)).toHaveLength(1)
  })
})
