import type { GameEvent } from "./types"

export class ThunderAudio {
  private context: AudioContext | null = null
  enabled = true
  volume = 0.7
  async unlock(): Promise<void> { this.context ??= new AudioContext(); if (this.context.state === "suspended") await this.context.resume() }
  play(event: GameEvent | "tick" | "go"): void {
    if (!this.context || !this.enabled) return
    const kind = typeof event === "string" ? event : event.type; if (kind === "hit" || kind === "shoot" && Math.random() > 0.35) return
    const map: Record<string, [number, number, OscillatorType]> = { shoot: [720, .045, "square"], hit: [160, .035, "square"], explosion: [82, .32, "sawtooth"], bomb: [48, .82, "sawtooth"], pickup: [920, .18, "sine"], revive: [520, .72, "triangle"], warning: [210, .55, "square"], win: [660, .8, "triangle"], tick: [360, .08, "sine"], go: [840, .24, "triangle"] }
    const [freq, duration, type] = map[kind] ?? map.hit!; const now = this.context.currentTime; const oscillator = this.context.createOscillator(); const gain = this.context.createGain(); oscillator.type = type; oscillator.frequency.setValueAtTime(freq, now); oscillator.frequency.exponentialRampToValueAtTime(kind === "bomb" ? 28 : Math.max(45, freq * .72), now + duration); gain.gain.setValueAtTime(.0001, now); gain.gain.exponentialRampToValueAtTime(this.volume * (kind === "bomb" ? .32 : .11), now + .008); gain.gain.exponentialRampToValueAtTime(.0001, now + duration); oscillator.connect(gain).connect(this.context.destination); oscillator.start(now); oscillator.stop(now + duration + .04)
  }
  async close(): Promise<void> { await this.context?.close(); this.context = null }
}
