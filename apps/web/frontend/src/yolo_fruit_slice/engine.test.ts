import { describe, expect, it } from "vitest"

import { FruitSliceEngine } from "./engine"
import type { BladeSweep } from "./types"

const cuttingBlade: BladeSweep = {
  side: "left", elbow: { x: 0, y: 100 }, wrist: { x: 200, y: 100 },
  previousElbow: { x: 0, y: 90 }, previousWrist: { x: 200, y: 90 },
  speed: 1.2, active: true, updatedAt: 100,
}

describe("FruitSliceEngine", () => {
  it("spawns objects from the configured bottom and side zones", () => {
    const left = new FruitSliceEngine(1000, 700, () => 0).createObject("apple")
    const right = new FruitSliceEngine(1000, 700, () => 0.99).createObject("orange")
    expect(left.x).toBeLessThan(0)
    expect(left.vx).toBeGreaterThan(0)
    expect(right.x).toBeGreaterThan(1000)
    expect(right.vx).toBeLessThan(0)
  })

  it("scores, deduplicates a slice and awards critical center hits", () => {
    const engine = new FruitSliceEngine(800, 600, () => 0.5)
    engine.reset()
    const fruit = engine.createObject("apple")
    Object.assign(fruit, { x: 100, y: 100, previousX: 100, previousY: 100, vx: 0, vy: 0, radius: 30 })
    engine.state.objects = [fruit]
    engine.update(0, 100, [cuttingBlade, { ...cuttingBlade, side: "right" }])
    expect(engine.state.slicedCount).toBe(1)
    expect(engine.state.score).toBe(20)
    expect(engine.state.objects).toHaveLength(0)
  })

  it("removes a life for missed fruit but not bombs", () => {
    const engine = new FruitSliceEngine(800, 600, () => 0.5)
    engine.reset()
    const fruit = engine.createObject("kiwi")
    const bomb = engine.createObject("bomb")
    Object.assign(fruit, { y: 900, vy: 100 })
    Object.assign(bomb, { y: 900, vy: 100 })
    engine.state.objects = [fruit, bomb]
    engine.update(16, 100, [])
    expect(engine.state.lives).toBe(2)
    expect(engine.state.missedCount).toBe(1)
    expect(engine.state.objects).toHaveLength(0)
  })

  it("ends immediately when a blade hits a bomb", () => {
    const engine = new FruitSliceEngine(800, 600, () => 0.5)
    engine.reset()
    const bomb = engine.createObject("bomb")
    Object.assign(bomb, { x: 100, y: 100, previousX: 100, previousY: 100, vx: 0, vy: 0, radius: 30 })
    engine.state.objects = [bomb]
    engine.update(0, 100, [cuttingBlade])
    expect(engine.state.phase).toBe("finished")
    expect(engine.state.objects).toHaveLength(0)
    expect(engine.drainEvents()).toContainEqual({ type: "bomb" })
    const particle = engine.state.particles[0]!
    engine.update(34, 134, [])
    expect(particle.age).toBeGreaterThan(0)
    expect(engine.state.flash).toBeLessThan(1)
    expect(engine.state.shake).toBeLessThan(1)
  })

  it("caps spawn interval and batch size as difficulty rises", () => {
    const engine = new FruitSliceEngine(800, 600, () => 0)
    engine.reset()
    engine.state.elapsedMs = 1_000_000
    engine.state.score = 100_000
    engine.state.objects = []
    engine.spawnWave()
    expect(engine.state.objects.length).toBeLessThanOrEqual(4)
    const internals = engine as unknown as { currentSpawnInterval: () => number }
    expect(internals.currentSpawnInterval()).toBe(470)
  })
})
