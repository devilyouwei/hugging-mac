import { PALM_STALE_MS } from "./config"
import { distance } from "./geometry"
import type { HandResult, Point, TrackedPalm } from "./types"

interface Slot { palm: TrackedPalm }

export class PalmTracker {
  private slots: Slot[] = []
  private nextId = 1

  reset(): void { this.slots = []; this.nextId = 1 }

  update(hands: HandResult[], imageWidth: number, imageHeight: number, now: number): TrackedPalm[] {
    const candidates = hands.filter((hand) => hand.confidence >= .5).slice(0, 2).map((hand) => {
      const center = { x: ((hand.box.x1 + hand.box.x2) / 2) / imageWidth, y: ((hand.box.y1 + hand.box.y2) / 2) / imageHeight }
      return { hand, center, radius: Math.min(.09, Math.max(.035, Math.min(hand.box.x2 - hand.box.x1, hand.box.y2 - hand.box.y1) / Math.min(imageWidth, imageHeight) * .34)) }
    })
    const unused = new Set(candidates)
    for (const slot of this.slots) {
      let best: typeof candidates[number] | undefined, bestDistance = Infinity
      for (const item of unused) { const value = distance(slot.palm.rawCenter, item.center); if (value < bestDistance) { best = item; bestDistance = value } }
      if (!best || bestDistance > .32) { slot.palm.active = now - slot.palm.updatedAt <= PALM_STALE_MS; continue }
      unused.delete(best)
      const alpha = .42
      const smoothed: Point = { x: slot.palm.center.x + (best.center.x - slot.palm.center.x) * alpha, y: slot.palm.center.y + (best.center.y - slot.palm.center.y) * alpha }
      slot.palm = { ...slot.palm, center: smoothed, rawCenter: best.center, radius: slot.palm.radius + (best.radius - slot.palm.radius) * alpha, box: best.hand.box, confidence: best.hand.confidence, active: true, updatedAt: now }
    }
    for (const item of unused) {
      if (this.slots.length >= 2) break
      this.slots.push({ palm: { id: this.nextId++, center: item.center, rawCenter: item.center, radius: item.radius, box: item.hand.box, confidence: item.hand.confidence, active: true, updatedAt: now } })
    }
    return this.slots.map(({ palm }) => ({ ...palm, center: { ...palm.center }, rawCenter: { ...palm.rawCenter }, box: { ...palm.box } }))
  }
}
