import type { GameEvent } from "./types"

export class TraceAudio {
  enabled = true
  private context: AudioContext | null = null
  async unlock(): Promise<void> { this.context ??= new AudioContext(); if (this.context.state === "suspended") await this.context.resume() }
  play(event: GameEvent): void {
    if (!this.enabled || !this.context || event === "erase") return
    const frequencies: Partial<Record<GameEvent, number>> = { start: 520, complete: 820, mistake: 130, tick: 440, win: 1040, lose: 90 }
    const oscillator = this.context.createOscillator(), gain = this.context.createGain(), now = this.context.currentTime
    oscillator.type = event === "mistake" || event === "lose" ? "sawtooth" : "sine"; oscillator.frequency.value = frequencies[event] ?? 400
    gain.gain.setValueAtTime(.07, now); gain.gain.exponentialRampToValueAtTime(.001, now + (event === "win" ? .45 : .16))
    oscillator.connect(gain).connect(this.context.destination); oscillator.start(now); oscillator.stop(now + (event === "win" ? .46 : .17))
  }
  async close(): Promise<void> { await this.context?.close(); this.context = null }
}
