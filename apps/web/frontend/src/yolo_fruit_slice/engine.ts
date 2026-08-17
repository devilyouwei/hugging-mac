import { PROGRAMMATIC_ASSETS } from "./assets"
import { GAME_CONFIG } from "./config"
import { bladeHitsCircle } from "./geometry"
import type {
  BladeSweep,
  FloatingText,
  FlyingObject,
  FruitKind,
  GameEvent,
  GamePhase,
  GameSnapshot,
  Particle,
  SlicePiece,
} from "./types"

type RandomSource = () => number

const FRUITS: Array<Exclude<FruitKind, "bomb">> = [
  "apple", "orange", "watermelon", "kiwi", "peach", "banana", "pear", "pineapple",
]

export class FruitSliceEngine {
  private random: RandomSource
  private nextId = 1
  private spawnAccumulatorMs = 0
  private lastSliceAt = -Infinity
  private events: GameEvent[] = []
  readonly state: GameSnapshot

  constructor(width: number, height: number, random: RandomSource = Math.random) {
    this.random = random
    this.state = this.emptyState(width, height)
  }

  private emptyState(width: number, height: number): GameSnapshot {
    return {
      width,
      height,
      phase: "lobby",
      score: 0,
      lives: GAME_CONFIG.initialLives,
      combo: 0,
      bestCombo: 0,
      slicedCount: 0,
      missedCount: 0,
      elapsedMs: 0,
      objects: [],
      pieces: [],
      particles: [],
      floatingTexts: [],
      flash: 0,
      shake: 0,
    }
  }

  reset(width = this.state.width, height = this.state.height): void {
    Object.assign(this.state, this.emptyState(width, height), { phase: "playing" as GamePhase })
    this.nextId = 1
    this.spawnAccumulatorMs = this.currentSpawnInterval() * 0.72
    this.lastSliceAt = -Infinity
    this.events = []
  }

  resize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return
    const scaleX = width / Math.max(1, this.state.width)
    const scaleY = height / Math.max(1, this.state.height)
    for (const object of this.state.objects) {
      object.x *= scaleX
      object.previousX *= scaleX
      object.y *= scaleY
      object.previousY *= scaleY
      object.vx *= scaleX
      object.vy *= scaleY
      object.radius *= Math.min(scaleX, scaleY)
    }
    this.state.width = width
    this.state.height = height
  }

  setPhase(phase: GamePhase): void {
    this.state.phase = phase
  }

  update(deltaMs: number, now: number, blades: BladeSweep[]): void {
    const safeDeltaMs = Math.min(34, Math.max(0, deltaMs))
    const dt = safeDeltaMs / 1000
    if (this.state.phase !== "playing") {
      if (this.state.phase === "finished") {
        this.updateEffects(dt)
        this.state.flash = Math.max(0, this.state.flash - dt * 1.25)
        this.state.shake = Math.max(0, this.state.shake - dt * 2.7)
      }
      return
    }
    this.state.elapsedMs += safeDeltaMs
    this.spawnAccumulatorMs += safeDeltaMs
    const spawnInterval = this.currentSpawnInterval()
    if (this.spawnAccumulatorMs >= spawnInterval) {
      this.spawnAccumulatorMs %= spawnInterval
      this.spawnWave()
    }

    const gravity = this.state.height * GAME_CONFIG.gravityViewport
    for (const object of this.state.objects) {
      object.previousX = object.x
      object.previousY = object.y
      object.x += object.vx * dt
      object.y += object.vy * dt
      object.vy += gravity * dt
      object.rotation += object.angularVelocity * dt
    }

    this.resolveSlices(blades, now)
    if (this.state.phase === "playing") this.resolveMisses()
    this.updateEffects(dt)
    this.state.flash = Math.max(0, this.state.flash - dt * 3.2)
    this.state.shake = Math.max(0, this.state.shake - dt * 3.5)
  }

  drainEvents(): GameEvent[] {
    return this.events.splice(0)
  }

  spawnWave(): void {
    const difficulty = this.difficulty()
    const batchRoll = this.random()
    const count = batchRoll < difficulty.batchThreeChance ? 3 : batchRoll < difficulty.batchTwoChance ? 2 : 1
    for (let index = 0; index < Math.min(count, GAME_CONFIG.maximumBatch); index += 1) {
      const isBomb = this.random() < difficulty.bombChance
      this.state.objects.push(this.createObject(isBomb ? "bomb" : FRUITS[Math.floor(this.random() * FRUITS.length)]!))
    }
  }

  createObject(kind: FruitKind): FlyingObject {
    const { width, height } = this.state
    const size = Math.min(width, height)
    const radiusScale = kind === "watermelon" ? 0.086
      : kind === "banana" || kind === "pineapple" ? 0.08
        : kind === "pear" ? 0.074
          : kind === "bomb" ? 0.052 : 0.071
    const radius = size * radiusScale * (0.9 + this.random() * 0.18)
    const sideRoll = this.random()
    let x: number
    let y: number
    let vx: number
    let vy: number
    if (sideRoll < 0.12) {
      x = -radius
      y = height * (0.72 + this.random() * 0.14)
      vx = width * (0.35 + this.random() * 0.18)
      vy = -height * (0.56 + this.random() * 0.18)
    } else if (sideRoll > 0.88) {
      x = width + radius
      y = height * (0.72 + this.random() * 0.14)
      vx = -width * (0.35 + this.random() * 0.18)
      vy = -height * (0.56 + this.random() * 0.18)
    } else {
      x = width * (0.14 + this.random() * 0.72)
      y = height + radius
      const targetX = width * (0.24 + this.random() * 0.52)
      vx = (targetX - x) * (0.38 + this.random() * 0.16)
      vy = -height * (0.96 + this.random() * 0.22)
    }
    return {
      id: this.nextId++, kind, x, y, previousX: x, previousY: y, vx, vy, radius,
      rotation: this.random() * Math.PI * 2,
      angularVelocity: (this.random() - 0.5) * 5,
      sliced: false,
    }
  }

  private currentSpawnInterval(): number {
    return Math.max(
      GAME_CONFIG.minimumSpawnMs,
      GAME_CONFIG.initialSpawnMs - (this.state.elapsedMs / 1000) * GAME_CONFIG.spawnAccelerationMsPerSecond - this.state.score * 0.45,
    )
  }

  private difficulty(): { bombChance: number; batchTwoChance: number; batchThreeChance: number } {
    const progress = Math.min(1, this.state.elapsedMs / 90_000 + this.state.score / 1800)
    return {
      bombChance: Math.min(GAME_CONFIG.bombChanceMaximum, GAME_CONFIG.bombChanceStart + progress * 0.075),
      batchTwoChance: Math.min(0.72, 0.22 + progress * 0.5),
      batchThreeChance: Math.min(0.32, 0.03 + progress * 0.29),
    }
  }

  private resolveSlices(blades: BladeSweep[], now: number): void {
    const removed = new Set<number>()
    for (const object of this.state.objects) {
      if (object.sliced) continue
      for (const blade of blades) {
        const collision = bladeHitsCircle(blade, object, object.radius * 1.05, GAME_CONFIG.minimumBladeSpeed)
        if (!collision.hit) continue
        object.sliced = true
        removed.add(object.id)
        if (object.kind === "bomb") {
          this.state.phase = "finished"
          this.state.flash = 1
          this.state.shake = 1
          this.events.push({ type: "bomb" })
          this.addExplosion(object.x, object.y)
          this.state.objects = this.state.objects.filter((item) => item.id !== object.id)
          return
        }
        this.scoreSlice(object, blade, collision.centerDistance, now)
        break
      }
    }
    if (removed.size) this.state.objects = this.state.objects.filter((object) => !removed.has(object.id))
  }

  private scoreSlice(object: FlyingObject, blade: BladeSweep, centerDistance: number, now: number): void {
    const inComboWindow = now - this.lastSliceAt <= GAME_CONFIG.comboWindowMs
    this.state.combo = inComboWindow ? this.state.combo + 1 : 1
    this.state.bestCombo = Math.max(this.state.bestCombo, this.state.combo)
    this.lastSliceAt = now
    const critical = blade.speed >= GAME_CONFIG.criticalBladeSpeed && centerDistance <= object.radius * GAME_CONFIG.criticalCenterRatio
    const comboBonus = this.state.combo >= 3 ? this.state.combo * 5 : 0
    const points = GAME_CONFIG.baseScore + comboBonus + (critical ? 10 : 0)
    this.state.score += points
    this.state.slicedCount += 1
    this.events.push({ type: "slice", critical, combo: this.state.combo })
    this.addSliceEffects(object, blade, points, critical)
  }

  private resolveMisses(): void {
    const remaining: FlyingObject[] = []
    for (const object of this.state.objects) {
      const outsideBottom = object.vy > 0 && object.y - object.radius > this.state.height + object.radius * 1.5
      const outsideSide = (object.x < -object.radius * 3 || object.x > this.state.width + object.radius * 3) && object.vy > 0
      if (!outsideBottom && !outsideSide) {
        remaining.push(object)
        continue
      }
      if (object.kind !== "bomb") {
        this.state.lives -= 1
        this.state.missedCount += 1
        this.state.combo = 0
        this.state.shake = Math.max(this.state.shake, 0.32)
        this.events.push({ type: "miss", lives: this.state.lives })
        if (this.state.lives <= 0) this.state.phase = "finished"
      }
    }
    this.state.objects = remaining
  }

  private addSliceEffects(object: FlyingObject, blade: BladeSweep, points: number, critical: boolean): void {
    if (object.kind === "bomb") return
    const colors = PROGRAMMATIC_ASSETS.fruitColors[object.kind]
    const normalX = blade.wrist.y - blade.elbow.y
    const normalY = -(blade.wrist.x - blade.elbow.x)
    const normalLength = Math.max(1, Math.hypot(normalX, normalY))
    for (const side of [-1, 1] as const) {
      this.state.pieces.push({
        id: this.nextId++, kind: object.kind, side, x: object.x, y: object.y,
        vx: object.vx + (normalX / normalLength) * side * object.radius * 4,
        vy: object.vy + (normalY / normalLength) * side * object.radius * 2,
        radius: object.radius, rotation: object.rotation,
        angularVelocity: object.angularVelocity + side * 4, age: 0, maxAge: 1.35,
      })
    }
    for (let index = 0; index < (critical ? 22 : 13); index += 1) {
      const angle = this.random() * Math.PI * 2
      const speed = object.radius * (2 + this.random() * 7)
      this.state.particles.push({
        id: this.nextId++, x: object.x, y: object.y,
        vx: Math.cos(angle) * speed + object.vx * 0.25,
        vy: Math.sin(angle) * speed + object.vy * 0.15,
        radius: 2 + this.random() * 5, color: colors.juice, age: 0, maxAge: 0.45 + this.random() * 0.55,
      })
    }
    this.state.floatingTexts.push({
      id: this.nextId++, x: object.x, y: object.y - object.radius,
      text: critical ? `CRITICAL +${points}` : this.state.combo >= 3 ? `${this.state.combo} COMBO +${points}` : `+${points}`,
      color: critical ? "#fff36b" : colors.juice, age: 0, maxAge: 0.9,
    })
  }

  private addExplosion(x: number, y: number): void {
    for (let index = 0; index < 42; index += 1) {
      const angle = this.random() * Math.PI * 2
      const speed = this.state.height * (0.15 + this.random() * 0.45)
      this.state.particles.push({
        id: this.nextId++, x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
        radius: 3 + this.random() * 8, color: index % 3 === 0 ? "#ffef72" : "#ff542f", age: 0, maxAge: 0.6 + this.random() * 0.5,
      })
    }
  }

  private updateEffects(dt: number): void {
    const gravity = this.state.height * GAME_CONFIG.gravityViewport
    for (const piece of this.state.pieces) {
      piece.age += dt
      piece.x += piece.vx * dt
      piece.y += piece.vy * dt
      piece.vy += gravity * dt
      piece.rotation += piece.angularVelocity * dt
    }
    for (const particle of this.state.particles) {
      particle.age += dt
      particle.x += particle.vx * dt
      particle.y += particle.vy * dt
      particle.vy += gravity * 0.45 * dt
    }
    for (const text of this.state.floatingTexts) {
      text.age += dt
      text.y -= this.state.height * 0.055 * dt
    }
    this.state.pieces = this.state.pieces.filter((piece) => piece.age < piece.maxAge)
    this.state.particles = this.state.particles.filter((particle) => particle.age < particle.maxAge)
    this.state.floatingTexts = this.state.floatingTexts.filter((text) => text.age < text.maxAge)
  }
}
