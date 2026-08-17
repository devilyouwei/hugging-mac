import type { PoseResult } from "@/pose_estimation/types"

export type GamePhase = "lobby" | "calibrating" | "countdown" | "playing" | "paused" | "finished"
export type FruitKind = "apple" | "orange" | "watermelon" | "kiwi" | "peach" | "banana" | "pear" | "pineapple" | "bomb"
export type BladeSource = "left" | "right" | "mouse"

export interface Point {
  x: number
  y: number
}

export interface BladeSweep {
  side: BladeSource
  elbow: Point
  wrist: Point
  previousElbow: Point
  previousWrist: Point
  speed: number
  active: boolean
  updatedAt: number
}

export interface PoseFrame {
  keypoints: Array<Point | null>
  blades: BladeSweep[]
  detected: boolean
  inferenceMs: number | null
}

export interface FlyingObject {
  id: number
  kind: FruitKind
  x: number
  y: number
  previousX: number
  previousY: number
  vx: number
  vy: number
  radius: number
  rotation: number
  angularVelocity: number
  sliced: boolean
}

export interface SlicePiece {
  id: number
  kind: Exclude<FruitKind, "bomb">
  side: -1 | 1
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  rotation: number
  angularVelocity: number
  age: number
  maxAge: number
}

export interface Particle {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  color: string
  age: number
  maxAge: number
}

export interface FloatingText {
  id: number
  x: number
  y: number
  text: string
  color: string
  age: number
  maxAge: number
}

export type GameEvent =
  | { type: "slice"; critical: boolean; combo: number }
  | { type: "miss"; lives: number }
  | { type: "bomb" }

export interface GameSnapshot {
  width: number
  height: number
  phase: GamePhase
  score: number
  lives: number
  combo: number
  bestCombo: number
  slicedCount: number
  missedCount: number
  elapsedMs: number
  objects: FlyingObject[]
  pieces: SlicePiece[]
  particles: Particle[]
  floatingTexts: FloatingText[]
  flash: number
  shake: number
}

export type PosePerson = PoseResult["poses"][number]

export interface GameAssets {
  fruitColors: Record<Exclude<FruitKind, "bomb">, { skin: string; flesh: string; juice: string }>
  fruitLabels: Record<Exclude<FruitKind, "bomb">, string>
}
