import type { Challenge, Difficulty, Point, ShapeKind, TracePath, TraceShape } from "./types"
import { DIFFICULTIES } from "./config"

const distance = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.y - b.y)

export function samplePolyline(points: Point[], closed: boolean, spacing = .012): Point[] {
  const source = closed ? [...points, points[0]!] : points
  const result: Point[] = []
  for (let segment = 0; segment < source.length - 1; segment++) {
    const a = source[segment]!, b = source[segment + 1]!, length = distance(a, b)
    const count = Math.max(1, Math.ceil(length / spacing))
    for (let index = 0; index < count; index++) {
      const t = index / count
      result.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
    }
  }
  result.push({ ...(closed ? points[0]! : points.at(-1)!) })
  return result
}

function parametric(kind: "circle" | "arc" | "wave" | "spiral" | "eight", cx: number, cy: number, scale: number): TraceShape {
  const points: Point[] = [], count = kind === "spiral" || kind === "eight" ? 150 : 110
  const end = kind === "arc" ? Math.PI * 1.45 : Math.PI * 2
  for (let index = 0; index <= count; index++) {
    const t = end * index / count
    if (kind === "circle" || kind === "arc") points.push({ x: cx + Math.cos(t) * scale, y: cy + Math.sin(t) * scale })
    else if (kind === "wave") points.push({ x: cx - scale + 2 * scale * index / count, y: cy + Math.sin(t * 2) * scale * .42 })
    else if (kind === "spiral") { const r = scale * (.14 + .86 * index / count); points.push({ x: cx + Math.cos(t * 2.4) * r, y: cy + Math.sin(t * 2.4) * r }) }
    else points.push({ x: cx + Math.sin(t) * scale, y: cy + Math.sin(t * 2) * scale * .64 })
  }
  return { kind, points, closed: kind === "circle" || kind === "eight" }
}

export function createShape(kind: ShapeKind, cx: number, cy: number, scale: number, variant = 0): TraceShape {
  if (["circle", "arc", "wave", "spiral", "eight"].includes(kind)) return parametric(kind as "circle" | "arc" | "wave" | "spiral" | "eight", cx, cy, scale)
  let points: Point[]
  if (kind === "line") points = variant % 3 === 0 ? [{ x: cx, y: cy - scale }, { x: cx, y: cy + scale }] : variant % 3 === 1 ? [{ x: cx - scale, y: cy }, { x: cx + scale, y: cy }] : [{ x: cx - scale, y: cy + scale }, { x: cx + scale, y: cy - scale }]
  else if (kind === "square") points = [{ x: cx - scale, y: cy - scale }, { x: cx + scale, y: cy - scale }, { x: cx + scale, y: cy + scale }, { x: cx - scale, y: cy + scale }]
  else if (kind === "triangle") points = [{ x: cx, y: cy - scale }, { x: cx + scale, y: cy + scale }, { x: cx - scale, y: cy + scale }]
  else if (kind === "diamond") points = [{ x: cx, y: cy - scale }, { x: cx + scale, y: cy }, { x: cx, y: cy + scale }, { x: cx - scale, y: cy }]
  else if (kind === "cross") points = [{ x: cx - scale, y: cy }, { x: cx + scale, y: cy }, { x: cx, y: cy }, { x: cx, y: cy - scale }, { x: cx, y: cy + scale }]
  else if (kind === "parallel") points = [{ x: cx - scale * .55, y: cy - scale }, { x: cx - scale * .55, y: cy + scale }, { x: cx + scale * .55, y: cy + scale }, { x: cx + scale * .55, y: cy - scale }]
  else if (kind === "zigzag") points = Array.from({ length: 6 }, (_, i) => ({ x: cx - scale + i * scale * .4, y: cy + (i % 2 ? scale * .65 : -scale * .65) }))
  else points = [{ x: cx - scale, y: cy + scale * .65 }, { x: cx - scale * .55, y: cy - scale }, { x: cx, y: cy - scale * .25 }, { x: cx + scale * .5, y: cy - scale }, { x: cx + scale, y: cy + scale * .65 }]
  return { kind, points, closed: ["square", "triangle", "diamond"].includes(kind) }
}

const pools: ShapeKind[][] = [
  ["line", "circle", "arc"], ["line", "circle", "arc"], ["line", "circle", "arc"],
  ["square", "triangle", "cross", "parallel"], ["square", "triangle", "cross", "parallel"], ["square", "triangle", "cross", "parallel"],
  ["zigzag", "diamond", "wave", "square"], ["zigzag", "diamond", "wave", "triangle"], ["zigzag", "diamond", "wave", "cross"],
  ["spiral", "eight", "polygon"], ["spiral", "eight", "polygon", "wave"], ["spiral", "eight", "polygon"],
]

function makePath(id: string, side: TracePath["side"], kind: ShapeKind, cx: number, scale: number, seed: number): TracePath {
  const shape = createShape(kind, cx, .54, scale, seed)
  return { id, side, kind, closed: shape.closed, samples: samplePolyline(shape.points, shape.closed).map((point) => ({ ...point, erased: false })), started: false, direction: 1, cursor: 0, offPathMs: 0, missingMs: 0, completed: false, palmId: null }
}

export function createChallenge(level: number, difficulty: Difficulty, seed = Math.random()): Challenge {
  const pool = pools[Math.min(level - 1, pools.length - 1)]!
  const pick = (salt: number): ShapeKind => pool[Math.floor(((seed * 997 + salt * .37) % 1) * pool.length)]!
  const dual = level >= 4 && (level % 3 === 0 || level >= 10)
  const complexity = dual ? 1.32 : 1
  const base = 9000 - (level - 1) * 500
  const durationMs = Math.round(Math.max(3500, base) * complexity * DIFFICULTIES[difficulty].timeScale)
  const paths = dual
    ? [makePath("left", "left", pick(1), .27, .17, level), makePath("right", "right", pick(2), .73, .17, level + 1)]
    : [makePath("single", "single", pick(0), .5, level > 9 ? .25 : .22, level)]
  return { level, dual, paths, durationMs, complexity }
}

export function pathCoverage(path: TracePath): number { return path.samples.filter((sample) => sample.erased).length / Math.max(1, path.samples.length) }
export { distance }
