import { pathCoverage } from "./geometry"
import type { GameSnapshot, TrackedPalm } from "./types"

export class TraceRenderer {
  private context: CanvasRenderingContext2D
  private dpr = 1
  constructor(private canvas: HTMLCanvasElement) { const context = canvas.getContext("2d"); if (!context) throw new Error("Canvas is not available"); this.context = context }
  resize(width: number, height: number): void { this.dpr = Math.min(2, devicePixelRatio || 1); this.canvas.width = Math.round(width * this.dpr); this.canvas.height = Math.round(height * this.dpr); this.canvas.style.width = `${width}px`; this.canvas.style.height = `${height}px` }
  render(state: GameSnapshot, palms: TrackedPalm[]): void {
    const ctx = this.context, width = this.canvas.width / this.dpr, height = this.canvas.height / this.dpr, short = Math.min(width, height)
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.clearRect(0, 0, width, height)
    const lineWidth = short * .06
    for (const path of state.challenge.paths) {
      const color = path.side === "right" ? "#ff5cc8" : "#54f7e3"
      ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = lineWidth
      for (let i = 1; i < path.samples.length; i++) {
        const a = path.samples[i - 1]!, b = path.samples[i]!
        if (a.erased && b.erased) continue
        ctx.strokeStyle = `${color}38`; ctx.shadowColor = color; ctx.shadowBlur = lineWidth * .45
        ctx.beginPath(); ctx.moveTo(a.x * width, a.y * height); ctx.lineTo(b.x * width, b.y * height); ctx.stroke()
      }
      ctx.shadowBlur = 0
      const start = path.direction === -1 ? path.samples.at(-1)! : path.samples[0]!
      if (!path.started) { const pulse = 1 + Math.sin(performance.now() / 180) * .18; ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(start.x * width, start.y * height, lineWidth * .7 * pulse, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = "#fff"; ctx.font = `700 ${Math.max(10, short * .018)}px ui-monospace`; ctx.textAlign = "center"; ctx.fillText("START", start.x * width, start.y * height - lineWidth) }
      const percent = Math.round(pathCoverage(path) * 100); ctx.fillStyle = color; ctx.font = `800 ${Math.max(11, short * .018)}px ui-monospace`; ctx.textAlign = "center"; ctx.fillText(`${percent}%`, (path.side === "left" ? .27 : path.side === "right" ? .73 : .5) * width, .88 * height)
    }
    for (const particle of state.particles) { ctx.globalAlpha = Math.max(0, particle.life / 500); ctx.fillStyle = particle.color; ctx.beginPath(); ctx.arc(particle.x * width, particle.y * height, 2 + particle.life / 100, 0, Math.PI * 2); ctx.fill() }
    ctx.globalAlpha = 1
    for (const palm of palms.filter((item) => item.active)) { const x = palm.center.x * width, y = palm.center.y * height, radius = palm.radius * short; ctx.strokeStyle = palm.center.x < .5 ? "#54f7e3" : "#ff5cc8"; ctx.lineWidth = 3; ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 18; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke(); ctx.shadowBlur = 0; ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill() }
    if (state.flash > 0) { ctx.fillStyle = `rgba(255,45,75,${state.flash * .28})`; ctx.fillRect(0, 0, width, height) }
  }
}
