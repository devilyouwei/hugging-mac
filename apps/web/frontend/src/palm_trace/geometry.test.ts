import { describe, expect, it } from "vitest"
import { createChallenge, createShape, pathCoverage, samplePolyline } from "./geometry"

describe("Palm Trace geometry", () => {
  it("samples closed geometry continuously", () => {
    const shape = createShape("square", .5, .5, .2)
    const samples = samplePolyline(shape.points, shape.closed)
    expect(samples.length).toBeGreaterThan(100)
    expect(samples[0]).toEqual(samples.at(-1))
  })

  it("keeps generated trails inside the safe play area", () => {
    for (let level = 1; level <= 12; level++) {
      const challenge = createChallenge(level, "arcade", .42)
      for (const path of challenge.paths) for (const point of path.samples) {
        expect(point.x).toBeGreaterThanOrEqual(.1); expect(point.x).toBeLessThanOrEqual(.9)
        expect(point.y).toBeGreaterThanOrEqual(.2); expect(point.y).toBeLessThanOrEqual(.82)
      }
    }
  })

  it("creates synchronized dual challenges in later levels", () => {
    const challenge = createChallenge(12, "arcade", .2)
    expect(challenge.dual).toBe(true); expect(challenge.paths.map((path) => path.side)).toEqual(["left", "right"])
  })

  it("reports erased coverage", () => {
    const path = createChallenge(1, "arcade", .1).paths[0]!
    path.samples.slice(0, Math.ceil(path.samples.length / 2)).forEach((sample) => { sample.erased = true })
    expect(pathCoverage(path)).toBeGreaterThanOrEqual(.5)
  })
})
