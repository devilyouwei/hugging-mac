import type { BladeSweep, Point } from "./types"

export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

export function distanceToSegment(point: Point, start: Point, end: Point): number {
  const dx = end.x - start.x
  const dy = end.y - start.y
  const lengthSquared = dx * dx + dy * dy
  if (lengthSquared === 0) return distance(point, start)
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared))
  return distance(point, { x: start.x + t * dx, y: start.y + t * dy })
}

export function bladeHitsCircle(
  blade: BladeSweep,
  center: Point,
  radius: number,
  minimumSpeed: number,
): { hit: boolean; centerDistance: number } {
  if (!blade.active || blade.speed < minimumSpeed) return { hit: false, centerDistance: Infinity }
  const distances = [
    distanceToSegment(center, blade.elbow, blade.wrist),
    distanceToSegment(center, blade.previousElbow, blade.previousWrist),
    distanceToSegment(center, blade.previousElbow, blade.elbow),
    distanceToSegment(center, blade.previousWrist, blade.wrist),
  ]
  const insideSweep = pointInPolygon(center, [
    blade.previousElbow,
    blade.previousWrist,
    blade.wrist,
    blade.elbow,
  ])
  const centerDistance = insideSweep ? 0 : Math.min(...distances)
  return { hit: insideSweep || centerDistance <= radius, centerDistance }
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const currentPoint = polygon[index]!
    const previousPoint = polygon[previous]!
    const crosses = (currentPoint.y > point.y) !== (previousPoint.y > point.y)
      && point.x < (previousPoint.x - currentPoint.x) * (point.y - currentPoint.y)
        / (previousPoint.y - currentPoint.y || Number.EPSILON) + currentPoint.x
    if (crosses) inside = !inside
  }
  return inside
}

export function mapCoverPoint(
  point: Point,
  sourceWidth: number,
  sourceHeight: number,
  viewportWidth: number,
  viewportHeight: number,
  mirrored = true,
): Point {
  const scale = Math.max(viewportWidth / sourceWidth, viewportHeight / sourceHeight)
  const renderedWidth = sourceWidth * scale
  const renderedHeight = sourceHeight * scale
  const offsetX = (viewportWidth - renderedWidth) / 2
  const offsetY = (viewportHeight - renderedHeight) / 2
  const sourceX = mirrored ? sourceWidth - point.x : point.x
  return { x: sourceX * scale + offsetX, y: point.y * scale + offsetY }
}
