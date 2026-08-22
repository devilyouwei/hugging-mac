import type { PoseResult } from "@/pose_estimation/types"

export type Point = { x: number; y: number }
export type Difficulty = "calm" | "arcade" | "rush"
export type GamePhase = "lobby" | "calibrating" | "countdown" | "playing" | "paused" | "won" | "lost"
export type ShapeKind = "line" | "circle" | "arc" | "square" | "triangle" | "cross" | "parallel" | "zigzag" | "diamond" | "wave" | "spiral" | "eight" | "polygon"

export interface TraceSample extends Point { erased: boolean }
export interface TracePath {
  id: string
  side: "single" | "left" | "right"
  kind: ShapeKind
  closed: boolean
  samples: TraceSample[]
  started: boolean
  direction: 1 | -1
  cursor: number
  offPathMs: number
  missingMs: number
  completed: boolean
  palmId: number | null
}
export interface TraceShape { kind: ShapeKind; points: Point[]; closed: boolean }
export interface Challenge { level: number; dual: boolean; paths: TracePath[]; durationMs: number; complexity: number }
export interface TrackedPalm {
  id: number
  center: Point
  rawCenter: Point
  radius: number
  box: { x1: number; y1: number; x2: number; y2: number }
  confidence: number
  active: boolean
  updatedAt: number
}
export interface Particle { x: number; y: number; vx: number; vy: number; life: number; color: string }
export type GameEvent = "start" | "erase" | "complete" | "mistake" | "tick" | "win" | "lose"
export interface GameSnapshot {
  phase: GamePhase
  level: number
  lives: number
  score: number
  combo: number
  remainingMs: number
  challenge: Challenge
  particles: Particle[]
  flash: number
  message: string
}
export type HandResult = PoseResult["hands"][number]
