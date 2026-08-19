import { GESTURE_HOLD_MS, HAND_STALE_MS } from "./config"
import type { Gesture, HandResult, Point, TrackedHand } from "./types"

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

function extended(points: Point[], tip: number, pip: number, mcp: number): boolean {
  return distance(points[tip]!, points[0]!) > distance(points[pip]!, points[0]!) * 1.08
    && distance(points[tip]!, points[mcp]!) > distance(points[pip]!, points[mcp]!) * 0.72
}

export function classifyGesture(hand: HandResult): Gesture {
  if ((hand.landmark_confidence ?? 0) < 0.48 || hand.landmarks.length !== 21) return "unknown"
  const p = hand.landmarks
  const fingers = [extended(p, 8, 6, 5), extended(p, 12, 10, 9), extended(p, 16, 14, 13), extended(p, 20, 18, 17)]
  const palmWidth = Math.max(1, distance(p[5]!, p[17]!))
  const thumbOpen = distance(p[4]!, p[5]!) > palmWidth * 0.58
  const foldedTips = [8, 12, 16, 20].filter((index) => distance(p[index]!, p[0]!) < palmWidth * 1.7).length
  if (thumbOpen && fingers[0] && !fingers[1] && !fingers[2] && fingers[3]) return "love"
  if (!fingers.some(Boolean) && foldedTips >= 3) return "fist"
  if (fingers.filter(Boolean).length >= 3) return "open"
  return "unknown"
}

interface Slot { tracked: TrackedHand; candidate: Gesture; since: number; armed: boolean }

export class HandTracker {
  private slots: Slot[] = []
  private maximumHands: 1 | 2 = 2
  private locked = false

  reset(maximumHands: 1 | 2 = 2): void { this.slots = []; this.maximumHands = maximumHands; this.locked = false }
  lock(): void { this.locked = true }

  update(hands: HandResult[], imageWidth: number, imageHeight: number, now: number): TrackedHand[] {
    const candidates = hands.filter((hand) => hand.confidence >= 0.55).slice(0, 2).map((hand) => ({
      hand,
      center: { x: ((hand.box.x1 + hand.box.x2) / 2) / imageWidth, y: ((hand.box.y1 + hand.box.y2) / 2) / imageHeight },
      gesture: classifyGesture(hand),
    }))
    const unused = new Set(candidates)
    for (const slot of this.slots) {
      let best: typeof candidates[number] | undefined
      let bestDistance = Infinity
      for (const item of unused) {
        const d = distance(slot.tracked.center, item.center) - (slot.tracked.handedness === item.hand.handedness ? 0.12 : 0)
        if (d < bestDistance) { best = item; bestDistance = d }
      }
      if (!best || bestDistance > 0.55) { slot.tracked.active = now - slot.tracked.updatedAt < HAND_STALE_MS; continue }
      unused.delete(best)
      this.apply(slot, best.hand, best.center, best.gesture, now)
    }
    for (const item of unused) {
      // A mouse-only solo run may lock before any hand exists. Allow exactly the
      // first hand to claim P1 later, while still preventing a new P2 mid-game.
      if ((this.locked && this.slots.length > 0) || this.slots.length >= this.maximumHands) break
      const id = (this.slots.some((slot) => slot.tracked.id === 1) ? 2 : 1) as 1 | 2
      const tracked: TrackedHand = { id, center: item.center, box: item.hand.box, gesture: item.gesture, gestureProgress: 0, handedness: item.hand.handedness, active: true, updatedAt: now }
      this.slots.push({ tracked, candidate: item.gesture, since: now, armed: true })
    }
    return this.slots.map((slot) => ({ ...slot.tracked, center: { ...slot.tracked.center }, box: { ...slot.tracked.box } }))
  }

  private apply(slot: Slot, hand: HandResult, center: Point, gesture: Gesture, now: number): void {
    if (gesture !== slot.candidate) { slot.candidate = gesture; slot.since = now }
    if (gesture === "open") slot.armed = true
    slot.tracked = { ...slot.tracked, center, box: hand.box, handedness: hand.handedness, active: true, updatedAt: now, gesture, gestureProgress: gesture === "fist" || gesture === "love" ? Math.min(1, (now - slot.since) / GESTURE_HOLD_MS) : 0 }
  }

  consume(id: 1 | 2, gesture: "fist" | "love"): boolean {
    const slot = this.slots.find((item) => item.tracked.id === id)
    if (!slot || !slot.armed || slot.tracked.gesture !== gesture || slot.tracked.gestureProgress < 1) return false
    slot.armed = false
    return true
  }
}
