import type { PoseResult } from "@/pose_estimation/types"

export type Point = { x: number; y: number }
export type Difficulty = "casual" | "arcade" | "hardcore"
export type GamePhase = "lobby" | "calibrating" | "countdown" | "playing" | "paused" | "won" | "lost"
export type Gesture = "open" | "fist" | "love" | "unknown"
export type WeaponMode = "vulcan" | "spread" | "laser"
export type BulletStyle = "vulcan" | "spread" | "laser" | "plasma" | "orb" | "missile"

export interface TrackedHand {
  id: 1 | 2
  center: Point
  box: { x1: number; y1: number; x2: number; y2: number }
  gesture: Gesture
  gestureProgress: number
  handedness: string | null
  active: boolean
  updatedAt: number
}

export interface Player {
  id: 1 | 2
  x: number
  y: number
  lives: number
  bombs: number
  score: number
  weapon: number
  weaponMode: WeaponMode
  joined: boolean
  active: boolean
  dead: boolean
  invulnerableMs: number
  fireCooldownMs: number
}

export interface Bullet { id: number; owner: 0 | 1 | 2; style: BulletStyle; x: number; y: number; vx: number; vy: number; radius: number; damage: number }
export interface Enemy { id: number; kind: "scout" | "interceptor" | "fighter" | "bomber" | "turret" | "elite" | "boss"; x: number; y: number; vx: number; vy: number; radius: number; hp: number; maxHp: number; fireMs: number; phase: number }
export interface Pickup { id: number; kind: "bomb" | "weapon"; amount: number; weaponMode?: WeaponMode; x: number; y: number; vy: number; radius: number }
export interface Particle { id: number; x: number; y: number; vx: number; vy: number; life: number; maxLife: number; color: string; size: number }

export type GameEvent =
  | { type: "shoot" }
  | { type: "hit" }
  | { type: "explosion"; heavy: boolean }
  | { type: "bomb"; player: 1 | 2 }
  | { type: "pickup" }
  | { type: "revive"; player: 1 | 2 }
  | { type: "warning" }
  | { type: "win" }

export interface GameSnapshot {
  width: number; height: number; phase: GamePhase; elapsedMs: number; difficulty: Difficulty
  players: Player[]; bullets: Bullet[]; enemies: Enemy[]; pickups: Pickup[]; particles: Particle[]
  bossHp: number | null; bossMaxHp: number | null; flash: number; shake: number; bombWave: number
}

export type HandResult = PoseResult["hands"][number]
