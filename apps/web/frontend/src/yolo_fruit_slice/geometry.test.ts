import { describe, expect, it } from "vitest"

import { bladeHitsCircle, mapCoverPoint } from "./geometry"
import type { BladeSweep } from "./types"

describe("game geometry", () => {
  it("mirrors and cover-crops source coordinates", () => {
    expect(mapCoverPoint({ x: 0, y: 0 }, 640, 480, 1280, 720)).toEqual({ x: 1280, y: -120 })
    expect(mapCoverPoint({ x: 640, y: 480 }, 640, 480, 1280, 720)).toEqual({ x: 0, y: 840 })
  })

  it("detects a circle crossed between inference frames", () => {
    const blade: BladeSweep = {
      side: "left", elbow: { x: 20, y: 10 }, wrist: { x: 20, y: 30 },
      previousElbow: { x: 0, y: 10 }, previousWrist: { x: 0, y: 30 },
      speed: 0.8, active: true, updatedAt: 100,
    }
    expect(bladeHitsCircle(blade, { x: 10, y: 20 }, 3, 0.38).hit).toBe(true)
    expect(bladeHitsCircle({ ...blade, speed: 0.1 }, { x: 10, y: 20 }, 3, 0.38).hit).toBe(false)
  })
})
