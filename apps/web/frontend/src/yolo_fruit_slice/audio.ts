export class GameAudio {
  private context: AudioContext | null = null

  async unlock(): Promise<void> {
    const AudioCtor = window.AudioContext
    this.context ??= new AudioCtor()
    if (this.context.state === "suspended") await this.context.resume()
  }

  play(kind: "tick" | "go" | "slice" | "critical" | "miss" | "bomb"): void {
    if (!this.context) return
    const now = this.context.currentTime
    const voices = kind === "slice" ? [[520, 0.055], [840, 0.075]]
      : kind === "critical" ? [[620, 0.06], [980, 0.09], [1320, 0.12]]
        : kind === "miss" ? [[190, 0.12], [125, 0.2]]
          : kind === "bomb" ? [[95, 0.42], [54, 0.62]]
            : kind === "go" ? [[620, 0.09], [880, 0.14]] : [[360, 0.08]]
    voices.forEach(([frequency, duration], index) => {
      const oscillator = this.context!.createOscillator()
      const gain = this.context!.createGain()
      oscillator.type = kind === "bomb" || kind === "miss" ? "sawtooth" : kind === "slice" ? "triangle" : "sine"
      oscillator.frequency.setValueAtTime(frequency!, now + index * 0.045)
      if (kind === "bomb") oscillator.frequency.exponentialRampToValueAtTime(32, now + duration!)
      gain.gain.setValueAtTime(0.0001, now)
      gain.gain.exponentialRampToValueAtTime(kind === "bomb" ? 0.22 : 0.1, now + 0.008)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + duration!)
      oscillator.connect(gain).connect(this.context!.destination)
      oscillator.start(now + index * 0.045)
      oscillator.stop(now + duration! + 0.06)
    })
  }

  async close(): Promise<void> {
    await this.context?.close()
    this.context = null
  }
}
