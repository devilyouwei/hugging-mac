import { PROGRAMMATIC_ASSETS } from "./assets"
import { visibleSkeletonSegments } from "./poseBlade"
import type { BladeSource, BladeSweep, FlyingObject, FruitKind, GameSnapshot, Point, PoseFrame, SlicePiece } from "./types"

interface TrailPoint extends Point {
  age: number
  side: BladeSource
}

export class GameRenderer {
  private context: CanvasRenderingContext2D
  private trails: TrailPoint[] = []

  constructor(private canvas: HTMLCanvasElement) {
    const context = canvas.getContext("2d", { alpha: true, desynchronized: true })
    if (!context) throw new Error("浏览器无法创建游戏画布")
    this.context = context
  }

  resize(cssWidth: number, cssHeight: number): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const width = Math.max(1, Math.round(cssWidth * dpr))
    const height = Math.max(1, Math.round(cssHeight * dpr))
    if (this.canvas.width !== width) this.canvas.width = width
    if (this.canvas.height !== height) this.canvas.height = height
    this.canvas.style.width = `${cssWidth}px`
    this.canvas.style.height = `${cssHeight}px`
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  render(state: GameSnapshot, pose: PoseFrame | null, deltaSeconds: number, blades: BladeSweep[] = pose?.blades ?? []): void {
    const context = this.context
    context.clearRect(0, 0, state.width, state.height)
    context.save()
    if (state.shake > 0) {
      const strength = state.shake * 18
      context.translate((Math.random() - 0.5) * strength, (Math.random() - 0.5) * strength)
    }
    if (pose) this.drawSkeleton(pose)
    for (const blade of blades) {
      if (blade.side === "mouse") this.drawBlade(blade)
    }
    this.updateAndDrawTrails(blades, deltaSeconds)
    for (const object of state.objects) this.drawObject(object)
    for (const piece of state.pieces) this.drawPiece(piece)
    for (const particle of state.particles) {
      context.globalAlpha = Math.max(0, 1 - particle.age / particle.maxAge)
      context.fillStyle = particle.color
      context.beginPath()
      context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
      context.fill()
    }
    context.globalAlpha = 1
    for (const text of state.floatingTexts) {
      context.globalAlpha = Math.max(0, 1 - text.age / text.maxAge)
      context.fillStyle = text.color
      context.font = `900 ${Math.max(16, state.height * 0.027)}px ui-monospace, monospace`
      context.textAlign = "center"
      context.shadowColor = "#000"
      context.shadowBlur = 8
      context.fillText(text.text, text.x, text.y)
      context.shadowBlur = 0
    }
    context.restore()
    context.globalAlpha = 1
    if (state.flash > 0) {
      context.fillStyle = `rgba(255, 241, 195, ${state.flash * 0.42})`
      context.fillRect(0, 0, state.width, state.height)
      this.drawExplosionBorder(state.width, state.height, state.flash)
    }
  }

  private drawSkeleton(pose: PoseFrame): void {
    const context = this.context
    context.save()
    context.lineCap = "round"
    context.lineJoin = "round"
    context.strokeStyle = "rgba(164, 246, 255, .3)"
    context.lineWidth = 2
    context.setLineDash([5, 8])
    for (const [from, to] of visibleSkeletonSegments(pose.keypoints)) {
      context.beginPath()
      context.moveTo(from.x, from.y)
      context.lineTo(to.x, to.y)
      context.stroke()
    }
    context.setLineDash([])
    context.fillStyle = "rgba(164, 246, 255, .5)"
    for (const point of pose.keypoints) {
      if (!point) continue
      context.beginPath()
      context.arc(point.x, point.y, 3, 0, Math.PI * 2)
      context.fill()
    }
    for (const blade of pose.blades) this.drawBlade(blade)
    context.restore()
  }

  private drawBlade(blade: BladeSweep): void {
    const context = this.context
    const color = blade.side === "left" ? "#58e8ff" : blade.side === "right" ? "#ffe66b" : "#f4fff6"
    context.shadowColor = color
    context.shadowBlur = blade.active ? 16 : 7
    context.strokeStyle = color
    context.lineWidth = blade.active ? 6 : 3
    context.beginPath()
    context.moveTo(blade.elbow.x, blade.elbow.y)
    context.lineTo(blade.wrist.x, blade.wrist.y)
    context.stroke()
    context.fillStyle = "#fff"
    for (const point of [blade.elbow, blade.wrist]) {
      context.beginPath()
      context.arc(point.x, point.y, 5, 0, Math.PI * 2)
      context.fill()
    }
    context.shadowBlur = 0
  }

  private updateAndDrawTrails(blades: BladeSweep[], deltaSeconds: number): void {
    for (const trail of this.trails) trail.age += deltaSeconds
    this.trails = this.trails.filter((trail) => trail.age < 0.24)
    for (const blade of blades) {
      if (blade.active && blade.speed > 0.2) this.trails.push({ ...blade.wrist, side: blade.side, age: 0 })
    }
    const context = this.context
    for (const side of ["left", "right", "mouse"] as const) {
      const points = this.trails.filter((trail) => trail.side === side)
      if (points.length < 2) continue
      context.save()
      context.lineCap = "round"
      context.lineJoin = "round"
      context.strokeStyle = side === "left" ? "rgba(88,232,255,.65)" : side === "right" ? "rgba(255,230,107,.65)" : "rgba(244,255,246,.78)"
      context.shadowColor = side === "left" ? "#58e8ff" : side === "right" ? "#ffe66b" : "#ffffff"
      context.shadowBlur = 18
      context.lineWidth = 9
      context.beginPath()
      context.moveTo(points[0]!.x, points[0]!.y)
      for (const point of points.slice(1)) context.lineTo(point.x, point.y)
      context.stroke()
      context.restore()
    }
  }

  private drawObject(object: FlyingObject): void {
    if (object.kind === "bomb") {
      this.drawBomb(object)
      return
    }
    this.drawFruit(object.kind, object.x, object.y, object.radius, object.rotation)
  }

  private drawFruit(kind: Exclude<FruitKind, "bomb">, x: number, y: number, radius: number, rotation: number): void {
    const context = this.context
    const colors = PROGRAMMATIC_ASSETS.fruitColors[kind]
    context.save()
    context.translate(x, y)
    context.rotate(rotation)
    context.shadowColor = "rgba(0,0,0,.45)"
    context.shadowBlur = 16
    if (kind === "banana") {
      this.drawBanana(radius)
      context.restore()
      return
    }
    if (kind === "pear") {
      this.drawPear(radius)
      context.restore()
      return
    }
    if (kind === "pineapple") {
      this.drawPineapple(radius)
      context.restore()
      return
    }
    const gradient = context.createRadialGradient(-radius * 0.35, -radius * 0.42, radius * 0.08, 0, 0, radius)
    gradient.addColorStop(0, "#fff7")
    gradient.addColorStop(0.16, colors.skin)
    gradient.addColorStop(1, shade(colors.skin, -28))
    context.fillStyle = gradient
    context.beginPath()
    context.arc(0, 0, radius, 0, Math.PI * 2)
    context.fill()
    context.shadowBlur = 0
    if (kind === "watermelon") {
      context.strokeStyle = "rgba(7,67,35,.5)"
      context.lineWidth = radius * 0.1
      for (const offset of [-0.5, 0, 0.5]) {
        context.beginPath()
        context.ellipse(offset * radius, 0, radius * 0.17, radius * 0.88, 0, 0, Math.PI * 2)
        context.stroke()
      }
    }
    context.strokeStyle = "#54351e"
    context.lineWidth = Math.max(3, radius * 0.09)
    context.beginPath()
    context.moveTo(0, -radius * 0.8)
    context.quadraticCurveTo(radius * 0.08, -radius * 1.16, radius * 0.2, -radius * 1.25)
    context.stroke()
    context.fillStyle = "#4eb94f"
    context.beginPath()
    context.ellipse(radius * 0.34, -radius * 1.05, radius * 0.3, radius * 0.13, -0.45, 0, Math.PI * 2)
    context.fill()
    context.restore()
  }

  private drawBanana(radius: number): void {
    const context = this.context
    const gradient = context.createLinearGradient(-radius, -radius, radius, radius)
    gradient.addColorStop(0, "#fff27c")
    gradient.addColorStop(0.42, "#f4cf32")
    gradient.addColorStop(1, "#c99516")
    context.fillStyle = gradient
    context.beginPath()
    context.moveTo(-radius * 0.95, -radius * 0.55)
    context.bezierCurveTo(-radius * 0.58, radius * 0.72, radius * 0.6, radius * 0.86, radius * 1.04, -radius * 0.28)
    context.bezierCurveTo(radius * 0.5, radius * 0.38, -radius * 0.32, radius * 0.26, -radius * 0.7, -radius * 0.72)
    context.closePath()
    context.fill()
    context.shadowBlur = 0
    context.strokeStyle = "#806018"
    context.lineWidth = radius * 0.09
    context.beginPath()
    context.moveTo(-radius * 0.88, -radius * 0.58)
    context.lineTo(-radius * 1.08, -radius * 0.72)
    context.moveTo(radius * 0.96, -radius * 0.3)
    context.lineTo(radius * 1.1, -radius * 0.5)
    context.stroke()
    context.strokeStyle = "rgba(255,255,255,.34)"
    context.lineWidth = radius * 0.05
    context.beginPath()
    context.moveTo(-radius * 0.55, -radius * 0.42)
    context.bezierCurveTo(-radius * 0.18, radius * 0.35, radius * 0.42, radius * 0.4, radius * 0.72, -radius * 0.06)
    context.stroke()
  }

  private drawPear(radius: number): void {
    const context = this.context
    const gradient = context.createRadialGradient(-radius * 0.28, -radius * 0.35, 1, 0, 0, radius)
    gradient.addColorStop(0, "#d9f16d")
    gradient.addColorStop(0.42, "#91bd36")
    gradient.addColorStop(1, "#557f20")
    context.fillStyle = gradient
    this.tracePear(radius)
    context.fill()
    context.shadowBlur = 0
    context.strokeStyle = "#563b1f"
    context.lineWidth = radius * 0.1
    context.beginPath()
    context.moveTo(0, -radius * 0.9)
    context.quadraticCurveTo(radius * 0.08, -radius * 1.25, radius * 0.2, -radius * 1.35)
    context.stroke()
    context.fillStyle = "#48a63e"
    context.beginPath()
    context.ellipse(radius * 0.37, -radius * 1.15, radius * 0.32, radius * 0.13, -0.42, 0, Math.PI * 2)
    context.fill()
  }

  private tracePear(radius: number): void {
    const context = this.context
    context.beginPath()
    context.moveTo(0, -radius)
    context.bezierCurveTo(-radius * 0.44, -radius * 0.76, -radius * 0.38, -radius * 0.3, -radius * 0.82, radius * 0.18)
    context.bezierCurveTo(-radius * 1.04, radius * 0.72, -radius * 0.58, radius, 0, radius)
    context.bezierCurveTo(radius * 0.58, radius, radius * 1.04, radius * 0.72, radius * 0.82, radius * 0.18)
    context.bezierCurveTo(radius * 0.38, -radius * 0.3, radius * 0.44, -radius * 0.76, 0, -radius)
    context.closePath()
  }

  private drawPineapple(radius: number): void {
    const context = this.context
    const gradient = context.createRadialGradient(-radius * 0.28, -radius * 0.35, 1, 0, 0, radius)
    gradient.addColorStop(0, "#ffe56e")
    gradient.addColorStop(0.35, "#d99125")
    gradient.addColorStop(1, "#9b5e19")
    context.fillStyle = gradient
    context.beginPath()
    context.ellipse(0, radius * 0.16, radius * 0.78, radius, 0, 0, Math.PI * 2)
    context.fill()
    context.shadowBlur = 0
    context.save()
    context.beginPath()
    context.ellipse(0, radius * 0.16, radius * 0.77, radius * 0.98, 0, 0, Math.PI * 2)
    context.clip()
    context.strokeStyle = "rgba(91,74,22,.48)"
    context.lineWidth = radius * 0.055
    for (let offset = -2; offset <= 2; offset += 1) {
      context.beginPath()
      context.moveTo(-radius * 1.2, offset * radius * 0.42 - radius)
      context.lineTo(radius * 1.2, offset * radius * 0.42 + radius)
      context.stroke()
      context.beginPath()
      context.moveTo(radius * 1.2, offset * radius * 0.42 - radius)
      context.lineTo(-radius * 1.2, offset * radius * 0.42 + radius)
      context.stroke()
    }
    context.restore()
    context.fillStyle = "#3f9d45"
    for (const [angle, scale] of [[-0.72, 0.86], [-0.34, 1], [0, 1.1], [0.34, 1], [0.72, 0.86]] as const) {
      context.save()
      context.rotate(angle)
      context.beginPath()
      context.moveTo(0, -radius * 0.68)
      context.lineTo(-radius * 0.18, -radius * (0.72 + scale))
      context.lineTo(radius * 0.18, -radius * (0.72 + scale))
      context.closePath()
      context.fill()
      context.restore()
    }
  }

  private drawPiece(piece: SlicePiece): void {
    const context = this.context
    const colors = PROGRAMMATIC_ASSETS.fruitColors[piece.kind]
    context.save()
    context.translate(piece.x, piece.y)
    context.rotate(piece.rotation)
    context.globalAlpha = Math.min(1, (piece.maxAge - piece.age) * 2.5)
    if (piece.kind === "banana") {
      this.drawBananaPiece(piece)
      context.restore()
      return
    }
    if (piece.kind === "pear") {
      this.drawPearPiece(piece)
      context.restore()
      return
    }
    context.fillStyle = colors.skin
    context.beginPath()
    context.arc(0, 0, piece.radius, piece.side === -1 ? Math.PI / 2 : -Math.PI / 2, piece.side === -1 ? Math.PI * 1.5 : Math.PI / 2)
    context.closePath()
    context.fill()
    context.fillStyle = colors.flesh
    context.beginPath()
    context.ellipse(0, 0, piece.radius * 0.18, piece.radius * 0.88, 0, 0, Math.PI * 2)
    context.fill()
    context.restore()
  }

  private drawBananaPiece(piece: SlicePiece): void {
    const context = this.context
    const radius = piece.radius
    context.save()
    context.beginPath()
    if (piece.side === -1) context.rect(-radius * 1.25, -radius * 1.25, radius * 1.25, radius * 2.5)
    else context.rect(0, -radius * 1.25, radius * 1.25, radius * 2.5)
    context.clip()
    this.drawBanana(radius)
    context.restore()
    context.strokeStyle = PROGRAMMATIC_ASSETS.fruitColors.banana.flesh
    context.lineWidth = radius * 0.11
    context.lineCap = "round"
    context.shadowColor = PROGRAMMATIC_ASSETS.fruitColors.banana.juice
    context.shadowBlur = 7
    context.beginPath()
    context.moveTo(piece.side * radius * 0.018, -radius * 0.22)
    context.lineTo(piece.side * radius * 0.018, radius * 0.47)
    context.stroke()
    context.shadowBlur = 0
  }

  private drawPearPiece(piece: SlicePiece): void {
    const context = this.context
    const radius = piece.radius
    context.save()
    context.beginPath()
    if (piece.side === -1) context.rect(-radius * 1.15, -radius * 1.45, radius * 1.15, radius * 2.55)
    else context.rect(0, -radius * 1.45, radius * 1.15, radius * 2.55)
    context.clip()
    this.drawPear(radius)
    context.shadowBlur = 0
    context.fillStyle = PROGRAMMATIC_ASSETS.fruitColors.pear.flesh
    this.tracePear(radius * 0.79)
    context.fill()
    context.fillStyle = "#604326"
    for (const seedX of [-0.13, 0.13]) {
      context.beginPath()
      context.ellipse(radius * seedX, radius * 0.2, radius * 0.055, radius * 0.11, seedX, 0, Math.PI * 2)
      context.fill()
    }
    context.restore()
    context.strokeStyle = PROGRAMMATIC_ASSETS.fruitColors.pear.juice
    context.lineWidth = radius * 0.055
    context.beginPath()
    context.moveTo(piece.side * radius * 0.012, -radius * 0.78)
    context.lineTo(piece.side * radius * 0.012, radius * 0.8)
    context.stroke()
  }

  private drawExplosionBorder(width: number, height: number, intensity: number): void {
    const context = this.context
    context.save()
    const outerWidth = 18 + intensity * 28
    context.strokeStyle = `rgba(255, 54, 24, ${0.4 + intensity * 0.55})`
    context.lineWidth = outerWidth
    context.shadowColor = "#ff7a18"
    context.shadowBlur = 45 + intensity * 45
    context.strokeRect(outerWidth / 2, outerWidth / 2, width - outerWidth, height - outerWidth)
    context.strokeStyle = `rgba(255, 210, 45, ${intensity * 0.82})`
    context.lineWidth = 5 + intensity * 8
    context.shadowColor = "#ffd52b"
    context.shadowBlur = 25
    const inset = outerWidth + 3
    context.strokeRect(inset, inset, width - inset * 2, height - inset * 2)
    context.restore()
  }

  private drawBomb(object: FlyingObject): void {
    const context = this.context
    context.save()
    context.translate(object.x, object.y)
    context.rotate(object.rotation)
    const gradient = context.createRadialGradient(-object.radius * 0.3, -object.radius * 0.4, 2, 0, 0, object.radius)
    gradient.addColorStop(0, "#656a70")
    gradient.addColorStop(0.38, "#25272a")
    gradient.addColorStop(1, "#070808")
    context.fillStyle = gradient
    context.shadowColor = "#000"
    context.shadowBlur = 18
    context.beginPath()
    context.arc(0, 0, object.radius, 0, Math.PI * 2)
    context.fill()
    context.shadowBlur = 0
    context.strokeStyle = "#b97839"
    context.lineWidth = object.radius * 0.13
    context.beginPath()
    context.moveTo(object.radius * 0.25, -object.radius * 0.85)
    context.quadraticCurveTo(object.radius * 0.65, -object.radius * 1.25, object.radius * 0.88, -object.radius * 1.48)
    context.stroke()
    context.fillStyle = "#ffd94e"
    context.shadowColor = "#ff4b22"
    context.shadowBlur = 15
    context.beginPath()
    context.arc(object.radius * 0.9, -object.radius * 1.5, object.radius * 0.13, 0, Math.PI * 2)
    context.fill()
    context.restore()
  }
}

function shade(hex: string, amount: number): string {
  const value = Number.parseInt(hex.slice(1), 16)
  const red = Math.max(0, Math.min(255, (value >> 16) + amount))
  const green = Math.max(0, Math.min(255, ((value >> 8) & 0xff) + amount))
  const blue = Math.max(0, Math.min(255, (value & 0xff) + amount))
  return `rgb(${red},${green},${blue})`
}
