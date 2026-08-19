import { DIFFICULTIES, LEVEL_DURATION_MS } from "./config"
import type { Bullet, BulletStyle, Difficulty, Enemy, GameEvent, GamePhase, GameSnapshot, Particle, Pickup, Player, TrackedHand, WeaponMode } from "./types"

const hit = (a: { x: number; y: number; radius: number }, b: { x: number; y: number; radius: number }) => Math.hypot(a.x - b.x, a.y - b.y) < a.radius + b.radius
type RegularEnemyKind = Exclude<Enemy["kind"], "boss">

export class PalmThunderEngine {
  readonly events: GameEvent[] = []
  state: GameSnapshot
  private nextId = 1
  private spawnMs = 0
  private giftMs = 11_000
  private bossSpawned = false

  constructor(width: number, height: number, difficulty: Difficulty = "arcade") {
    this.state = this.fresh(width, height, difficulty)
  }

  private fresh(width: number, height: number, difficulty: Difficulty): GameSnapshot {
    return { width, height, phase: "lobby", elapsedMs: 0, difficulty, players: [this.player(1), this.player(2)], bullets: [], enemies: [], pickups: [], particles: [], bossHp: null, bossMaxHp: null, flash: 0, shake: 0, bombWave: 0 }
  }

  private player(id: 1 | 2): Player { return { id, x: id === 1 ? 0.4 : 0.6, y: 0.78, lives: 3, bombs: 3, score: 0, weapon: 1, weaponMode: "vulcan", joined: false, active: false, dead: false, invulnerableMs: 0, fireCooldownMs: 0 } }
  reset(difficulty = this.state.difficulty): void { const { width, height } = this.state; this.state = this.fresh(width, height, difficulty); this.nextId = 1; this.spawnMs = 0; this.giftMs = 11_000; this.bossSpawned = false; this.events.length = 0 }
  resize(width: number, height: number): void { this.state.width = width; this.state.height = height }
  setPhase(phase: GamePhase): void { this.state.phase = phase }
  configurePlayers(count: 1 | 2): void {
    for (const player of this.state.players) player.joined = player.id <= count
  }

  setHands(hands: TrackedHand[]): void {
    for (const player of this.state.players) {
      const hand = hands.find((item) => item.id === player.id)
      if (!hand) { player.active = false; continue }
      if (!player.joined) { player.active = false; continue }
      player.active = hand.active
      if (hand.active && !player.dead) { player.x = hand.center.x; player.y = hand.center.y }
    }
  }

  setPointer(point: { x: number; y: number } | null): void {
    const player = this.state.players[0]!
    if (!point || !player.joined || player.dead || player.active) return
    player.active = true
    player.x = Math.min(1, Math.max(0, point.x))
    player.y = Math.min(1, Math.max(0, point.y))
  }

  triggerBomb(playerId: 1 | 2): boolean {
    const player = this.state.players[playerId - 1]!
    if (player.dead || player.bombs < 1 || this.state.phase !== "playing") return false
    player.bombs--; this.state.bullets = this.state.bullets.filter((bullet) => bullet.owner !== 0)
    for (const enemy of this.state.enemies) enemy.hp -= enemy.kind === "boss" ? enemy.maxHp * 0.14 : enemy.maxHp
    this.state.flash = 1; this.state.shake = 1; this.state.bombWave = 1; this.events.push({ type: "bomb", player: playerId })
    return true
  }

  triggerLove(rescuerId: 1 | 2): boolean {
    if (this.state.phase !== "playing") return false
    const rescuer = this.state.players[rescuerId - 1]!
    const target = this.state.players.find((player) => player.id !== rescuerId && player.joined && player.dead)
    if (!rescuer.joined || rescuer.dead || !target) return false
    target.dead = false; target.lives = 1; target.active = true; target.invulnerableMs = 2600; target.x = Math.min(0.86, Math.max(0.14, rescuer.x + (rescuerId === 1 ? 0.12 : -0.12))); target.y = Math.min(0.88, rescuer.y + 0.06)
    this.burst(target.x, target.y, "#ff77d7", 44); this.events.push({ type: "revive", player: target.id }); return true
  }

  update(deltaMs: number): void {
    if (this.state.phase !== "playing") return
    const dt = Math.min(deltaMs, 40); const cfg = DIFFICULTIES[this.state.difficulty]; this.state.elapsedMs += dt
    this.state.flash = Math.max(0, this.state.flash - dt / 380); this.state.shake = Math.max(0, this.state.shake - dt / 620); this.state.bombWave = Math.max(0, this.state.bombWave - dt / 900)
    for (const p of this.state.players) { p.invulnerableMs = Math.max(0, p.invulnerableMs - dt); if (p.joined && p.active && !p.dead) this.firePlayer(p, dt) }
    if (!this.bossSpawned && this.state.elapsedMs >= LEVEL_DURATION_MS) this.spawnBoss()
    if (!this.bossSpawned) { this.spawnMs -= dt; if (this.spawnMs <= 0) { this.spawnEnemy(); this.spawnMs = cfg.spawnMs * (0.82 + Math.random() * 0.45) } }
    this.giftMs -= dt; if (this.giftMs <= 0) { this.spawnPickup(); this.giftMs = 13_000 + Math.random() * 10_000 }
    this.move(dt, cfg.enemySpeed, cfg.bulletSpeed); this.collisions(); this.cleanup(); this.finishCheck()
  }

  private firePlayer(p: Player, dt: number): void {
    p.fireCooldownMs -= dt; if (p.fireCooldownMs > 0) return
    if (p.weaponMode === "laser") {
      p.fireCooldownMs = 82
      const offsets = p.weapon >= 3 ? [-0.012, 0.012] : [0]
      for (const offset of offsets) this.addBullet(p.id, "laser", p.x + offset, p.y - 0.04, 0, -1.18, 0.006, p.weapon >= 4 ? 2.1 : 1.45)
    } else if (p.weaponMode === "spread") {
      p.fireCooldownMs = 210 - p.weapon * 12
      const count = Math.min(7, 2 + p.weapon)
      for (let i = 0; i < count; i++) {
        const vx = count === 1 ? 0 : -0.32 + 0.64 * i / (count - 1)
        this.addBullet(p.id, "spread", p.x, p.y - 0.03, vx, -0.72, 0.009, 0.85)
      }
    } else {
      p.fireCooldownMs = 138 - p.weapon * 13
      const offsets = p.weapon >= 3 ? [-0.015, 0, 0.015] : p.weapon >= 2 ? [-0.01, 0.01] : [0]
      for (const offset of offsets) this.addBullet(p.id, "vulcan", p.x + offset, p.y - 0.035, offset * 2.5, -0.9, 0.006, 1)
    }
    if (Math.random() < 0.28) this.events.push({ type: "shoot" })
  }

  private addBullet(owner: 0 | 1 | 2, style: BulletStyle, x: number, y: number, vx: number, vy: number, radius: number, damage: number): void {
    this.state.bullets.push({ id: this.nextId++, owner, style, x, y, vx, vy, radius, damage })
  }

  private spawnEnemy(): void {
    const progress = this.state.elapsedMs / LEVEL_DURATION_MS
    const pool: RegularEnemyKind[] = progress < .22 ? ["scout", "interceptor", "fighter"] : progress < .55 ? ["scout", "interceptor", "fighter", "bomber", "turret"] : ["interceptor", "fighter", "bomber", "turret", "elite"]
    const kind = pool[Math.floor(Math.random() * pool.length)]!
    const stats: Record<RegularEnemyKind, [number, number, number]> = { scout: [2, .15, .026], interceptor: [3, .23, .024], fighter: [5, .13, .032], bomber: [12, .075, .048], turret: [8, .055, .038], elite: [16, .09, .052] }
    const [baseHp, speed, radius] = stats[kind]; const hp = Math.ceil(baseHp * DIFFICULTIES[this.state.difficulty].hp)
    this.state.enemies.push({ id: this.nextId++, kind, x: 0.08 + Math.random() * 0.84, y: -0.08, vx: (Math.random() - 0.5) * (kind === "interceptor" ? .34 : .16), vy: speed, radius, hp, maxHp: hp, fireMs: 450 + Math.random() * 1200, phase: Math.random() * 6.28 })
  }

  private spawnBoss(): void { this.bossSpawned = true; const hp = Math.round(280 * DIFFICULTIES[this.state.difficulty].hp); this.state.enemies.push({ id: this.nextId++, kind: "boss", x: 0.5, y: -0.16, vx: 0.1, vy: 0.055, radius: 0.105, hp, maxHp: hp, fireMs: 900, phase: 0 }); this.state.bossHp = hp; this.state.bossMaxHp = hp; this.events.push({ type: "warning" }) }
  private spawnPickup(): void {
    const bomb = Math.random() < .55
    const modes: WeaponMode[] = ["vulcan", "spread", "laser"]
    this.state.pickups.push({ id: this.nextId++, kind: bomb ? "bomb" : "weapon", amount: bomb ? 1 + Math.floor(Math.random() * 3) : 1, weaponMode: bomb ? undefined : modes[Math.floor(Math.random() * modes.length)], x: 0.12 + Math.random() * 0.76, y: -0.05, vy: 0.09, radius: 0.03 })
  }

  private move(dtMs: number, enemySpeed: number, bulletSpeed: number): void {
    const dt = dtMs / 1000
    for (const bullet of this.state.bullets) { bullet.x += bullet.vx * dt * (bullet.owner === 0 ? bulletSpeed : 1); bullet.y += bullet.vy * dt * (bullet.owner === 0 ? bulletSpeed : 1) }
    for (const enemy of this.state.enemies) {
      if (enemy.kind === "boss" && enemy.y >= 0.17) { enemy.vy = 0; enemy.x += Math.sin(this.state.elapsedMs / 1200) * dt * 0.12 } else { enemy.x += enemy.vx * dt * enemySpeed; enemy.y += enemy.vy * dt * enemySpeed }
      enemy.fireMs -= dtMs; if (enemy.fireMs <= 0 && enemy.y > 0.02) { this.enemyFire(enemy); enemy.fireMs = enemy.kind === "boss" ? 330 : enemy.kind === "elite" ? 620 : enemy.kind === "bomber" ? 980 : enemy.kind === "turret" ? 760 : 1250 + Math.random() * 650 }
    }
    for (const pickup of this.state.pickups) pickup.y += pickup.vy * dt
    for (const particle of this.state.particles) { particle.x += particle.vx * dt; particle.y += particle.vy * dt; particle.life -= dtMs }
  }

  private enemyFire(e: Enemy): void {
    const count = e.kind === "boss" ? 11 : e.kind === "elite" ? 5 : e.kind === "bomber" ? 3 : e.kind === "turret" ? 2 : 1
    const style: BulletStyle = e.kind === "boss" ? "orb" : e.kind === "bomber" ? "missile" : e.kind === "elite" || e.kind === "turret" ? "plasma" : "orb"
    for (let i = 0; i < count; i++) { const angle = count === 1 ? Math.PI / 2 : Math.PI * (.2 + .6 * i / (count - 1)); const speed = style === "missile" ? .18 : e.kind === "boss" ? .31 : .25; this.addBullet(0, style, e.x, e.y + e.radius, Math.cos(angle) * speed, Math.sin(angle) * speed, style === "missile" ? .012 : e.kind === "boss" ? .01 : .007, 1) }
  }

  private collisions(): void {
    for (const bullet of this.state.bullets.filter((b) => b.owner !== 0)) for (const enemy of this.state.enemies) if (bullet.damage > 0 && enemy.hp > 0 && hit(bullet, enemy)) { enemy.hp -= bullet.damage; bullet.damage = 0; this.events.push({ type: "hit" }); const p = this.state.players[bullet.owner - 1]; if (p) p.score += 10 }
    for (const player of this.state.players.filter((p) => p.joined && p.active && !p.dead && p.invulnerableMs <= 0)) {
      const body = { x: player.x, y: player.y, radius: 0.018 }
      const danger = this.state.bullets.find((b) => b.owner === 0 && hit(body, b)) || this.state.enemies.find((e) => e.kind !== "boss" && hit(body, e))
      if (danger) { if ("owner" in danger) danger.damage = 0; player.lives--; player.invulnerableMs = 1800; this.burst(player.x, player.y, "#54dfff", 26); this.events.push({ type: "explosion", heavy: true }); if (player.lives <= 0) player.dead = true }
      for (const pickup of this.state.pickups) if (pickup.amount > 0 && hit(body, pickup)) { if (pickup.kind === "bomb") player.bombs = Math.min(3, player.bombs + pickup.amount); else { const nextMode = pickup.weaponMode ?? "vulcan"; player.weapon = nextMode === player.weaponMode ? Math.min(4, player.weapon + 1) : 1; player.weaponMode = nextMode } pickup.amount = 0; player.score += 250; this.events.push({ type: "pickup" }) }
    }
    for (const enemy of this.state.enemies.filter((e) => e.hp <= 0)) { this.burst(enemy.x, enemy.y, enemy.kind === "boss" ? "#ffcf57" : "#ff5b45", enemy.kind === "boss" ? 90 : 18); this.events.push({ type: "explosion", heavy: enemy.kind === "boss" }) }
  }

  private cleanup(): void { const boss = this.state.enemies.find((e) => e.kind === "boss"); if (boss) this.state.bossHp = Math.max(0, boss.hp); this.state.bullets = this.state.bullets.filter((b) => b.damage > 0 && b.y > -0.12 && b.y < 1.12 && b.x > -0.12 && b.x < 1.12); this.state.enemies = this.state.enemies.filter((e) => e.hp > 0 && e.y < 1.2); this.state.pickups = this.state.pickups.filter((p) => p.amount > 0 && p.y < 1.1); this.state.particles = this.state.particles.filter((p) => p.life > 0) }
  private finishCheck(): void { if (this.bossSpawned && !this.state.enemies.some((e) => e.kind === "boss")) { this.state.phase = "won"; this.events.push({ type: "win" }); return } const joined = this.state.players.filter((p) => p.joined); if (joined.length && joined.every((p) => p.dead)) this.state.phase = "lost" }
  private burst(x: number, y: number, color: string, count: number): void { for (let i = 0; i < count; i++) { const a = Math.random() * Math.PI * 2; const speed = 0.05 + Math.random() * 0.32; this.state.particles.push({ id: this.nextId++, x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, life: 450 + Math.random() * 700, maxLife: 1150, color, size: 2 + Math.random() * 5 }) } }
}
