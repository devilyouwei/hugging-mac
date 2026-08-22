import { describe, expect, it } from "vitest"
import { PalmTraceEngine } from "./engine"
import type { TrackedPalm } from "./types"

function palm(x: number, y: number): TrackedPalm { return { id: 1, center: { x, y }, rawCenter: { x, y }, radius: .05, box: { x1: 0, y1: 0, x2: 1, y2: 1 }, confidence: .9, active: true, updatedAt: 0 } }

describe("PalmTraceEngine", () => {
  it("starts only near the glowing endpoint and advances continuously", () => {
    const engine = new PalmTraceEngine("arcade", 0); engine.setPhase("playing")
    engine.update(16, [palm(.05, .05)]); expect(engine.state.challenge.paths[0]!.started).toBe(false)
    const path = engine.state.challenge.paths[0]!, start = path.samples[0]!
    engine.update(16, [palm(start.x, start.y)]); expect(path.started).toBe(true)
    for (let index = 1; index < path.samples.length; index += 8) { const point = path.samples[index]!; engine.update(16, [palm(point.x, point.y)]) }
    expect(path.samples.filter((sample) => sample.erased).length).toBeGreaterThan(path.samples.length * .5)
  })

  it("charges one life after sustained movement far from an active trail", () => {
    const engine = new PalmTraceEngine("arcade", 0), path = engine.state.challenge.paths[0]!, start = path.samples[0]!
    engine.setPhase("playing"); engine.update(16, [palm(start.x, start.y)])
    engine.update(100, [palm(.02, .95)]); engine.update(100, [palm(.02, .95)]); engine.update(100, [palm(.02, .95)])
    expect(engine.state.lives).toBe(2); expect(engine.state.combo).toBe(0)
  })

  it("pauses the countdown while paused", () => {
    const engine = new PalmTraceEngine("arcade", 0); engine.setPhase("playing"); engine.update(100, [])
    const remaining = engine.state.remainingMs; engine.setPhase("paused"); engine.update(1000, [])
    expect(engine.state.remainingMs).toBe(remaining)
  })
})
