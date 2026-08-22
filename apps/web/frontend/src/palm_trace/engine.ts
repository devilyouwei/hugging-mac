import { COMPLETION_RATIO, DIFFICULTIES, LEVEL_COUNT, PALM_STALE_MS, STARTING_LIVES, SYNC_WINDOW_MS } from "./config"
import { createChallenge, distance, pathCoverage } from "./geometry"
import type { Challenge, Difficulty, GameEvent, GamePhase, GameSnapshot, Particle, TracePath, TrackedPalm } from "./types"

export class PalmTraceEngine {
  readonly events: GameEvent[] = []
  private phase: GamePhase = "countdown"
  private level = 1
  private lives = STARTING_LIVES
  private score = 0
  private combo = 0
  private remainingMs = 0
  private challenge: Challenge
  private particles: Particle[] = []
  private flash = 0
  private message = "FIND THE GLOWING START"
  private startTimes = new Map<string, number>()
  private clock = 0

  constructor(private difficulty: Difficulty, seed = Math.random()) {
    this.challenge = createChallenge(1, difficulty, seed)
    this.remainingMs = this.challenge.durationMs
  }

  setPhase(phase: GamePhase): void { this.phase = phase }
  get state(): GameSnapshot { return { phase: this.phase, level: this.level, lives: this.lives, score: this.score, combo: this.combo, remainingMs: this.remainingMs, challenge: this.challenge, particles: this.particles, flash: this.flash, message: this.message } }

  update(deltaMs: number, palms: TrackedPalm[]): void {
    if (this.phase !== "playing") return
    const dt = Math.min(deltaMs, 100), active = palms.filter((palm) => palm.active)
    this.clock += dt; this.remainingMs -= dt; this.flash = Math.max(0, this.flash - dt / 350)
    this.particles = this.particles.map((p) => ({ ...p, x: p.x + p.vx * dt, y: p.y + p.vy * dt, life: p.life - dt })).filter((p) => p.life > 0)
    if (this.remainingMs <= 0) { this.fail("TIME EXPIRED"); return }
    for (const path of this.challenge.paths) {
      const palm = this.resolvePalm(path, active)
      if (!palm) {
        if (path.started && !path.completed) { path.missingMs += dt; if (path.missingMs > PALM_STALE_MS) { this.fail("PALM TRACKING LOST"); return } }
        continue
      }
      path.missingMs = 0
      const outcome = this.trace(path, palm, dt)
      if (outcome === "failed") { this.fail("TOO FAR FROM THE TRAIL"); return }
    }
    if (this.challenge.dual && this.startTimes.size === 1) {
      const first = Math.min(...this.startTimes.values())
      if (this.clock - first > SYNC_WINDOW_MS) { this.fail("START BOTH TRAILS TOGETHER"); return }
    }
    if (this.challenge.paths.every((path) => path.completed)) this.completeLevel()
  }

  private resolvePalm(path: TracePath, palms: TrackedPalm[]): TrackedPalm | undefined {
    if (path.palmId != null) return palms.find((palm) => palm.id === path.palmId)
    const allowed = palms.filter((palm) => path.side === "single" || (path.side === "left" ? palm.center.x < .5 : palm.center.x >= .5))
    const start = path.samples[0]!
    const end = path.samples.at(-1)!
    return allowed.sort((a, b) => Math.min(distance(a.center, start), distance(a.center, end)) - Math.min(distance(b.center, start), distance(b.center, end)))[0]
  }

  private trace(path: TracePath, palm: TrackedPalm, dt: number): "ok" | "failed" {
    const tolerance = (.055 + palm.radius * .34) * DIFFICULTIES[this.difficulty].toleranceScale
    const outer = tolerance * 1.5
    if (!path.started) {
      const startDistance = distance(palm.center, path.samples[0]!), endDistance = distance(palm.center, path.samples.at(-1)!)
      if (Math.min(startDistance, endDistance) > tolerance) return "ok"
      path.started = true; path.palmId = palm.id; path.direction = endDistance < startDistance && !path.closed ? -1 : 1
      path.cursor = path.direction === 1 ? 0 : path.samples.length - 1; this.startTimes.set(path.id, this.clock); this.events.push("start")
      this.message = this.challenge.dual ? "KEEP BOTH PALMS MOVING" : "STAY ON THE TRAIL"
    }
    if (path.closed && path.samples.filter((sample) => sample.erased).length <= 4) {
      const lookahead = Math.min(10, path.samples.length - 2)
      const forward = distance(palm.center, path.samples[lookahead]!), backward = distance(palm.center, path.samples[path.samples.length - 1 - lookahead]!)
      if (Math.min(forward, backward) < tolerance) { path.direction = backward < forward ? -1 : 1; path.cursor = path.direction === 1 ? 0 : path.samples.length - 1 }
    }
    const nearest = Math.min(...path.samples.map((sample) => distance(palm.center, sample)))
    if (nearest > outer) path.offPathMs += dt; else path.offPathMs = Math.max(0, path.offPathMs - dt * 1.8)
    if (path.offPathMs >= DIFFICULTIES[this.difficulty].offPathMs) return "failed"
    const before = pathCoverage(path), length = path.samples.length
    for (let step = 0; step < 18; step++) {
      const index = path.cursor + step * path.direction
      if (index < 0 || index >= length) break
      if (distance(palm.center, path.samples[index]!) <= tolerance) {
        const through = index + path.direction * 3
        const lo = Math.max(0, Math.min(path.cursor, through)), hi = Math.min(length - 1, Math.max(path.cursor, through))
        for (let i = lo; i <= hi; i++) path.samples[i]!.erased = true
        path.cursor = Math.max(0, Math.min(length - 1, through))
      }
    }
    const after = pathCoverage(path)
    if (after > before) { this.events.push("erase"); this.spawn(palm.center.x, palm.center.y, path.side === "right" ? "#ff5cc8" : "#54f7e3") }
    if (after >= COMPLETION_RATIO) { path.completed = true; this.events.push("complete") }
    return "ok"
  }

  private completeLevel(): void {
    const accuracy = this.challenge.paths.reduce((sum, path) => sum + pathCoverage(path), 0) / this.challenge.paths.length
    this.combo++; this.score += Math.round((700 + this.level * 140 + this.remainingMs * .08) * this.challenge.complexity * (1 + this.combo * .08) * accuracy)
    if (this.level >= LEVEL_COUNT) { this.phase = "won"; this.message = "SCREEN PERFECT"; this.events.push("win"); return }
    this.level++; this.challenge = createChallenge(this.level, this.difficulty); this.remainingMs = this.challenge.durationMs; this.startTimes.clear(); this.message = "FIND THE GLOWING START"; this.events.push("complete")
  }

  private fail(message: string): void {
    this.lives--; this.combo = 0; this.flash = 1; this.message = message; this.events.push("mistake")
    if (this.lives <= 0) { this.phase = "lost"; this.events.push("lose"); return }
    this.challenge = createChallenge(this.level, this.difficulty); this.remainingMs = this.challenge.durationMs; this.startTimes.clear()
  }

  private spawn(x: number, y: number, color: string): void {
    for (let i = 0; i < 2; i++) this.particles.push({ x, y, vx: (Math.random() - .5) * .00025, vy: (Math.random() - .5) * .00025, life: 300 + Math.random() * 220, color })
  }
}
