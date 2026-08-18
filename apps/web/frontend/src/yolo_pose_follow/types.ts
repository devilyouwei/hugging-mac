export interface PoseTemplate {
  template_id: string
  name: string
  cue: string
  difficulty: number
  points: Array<{ x: number; y: number }>
  required_keypoints: number[]
}

export interface PoseMatch {
  template_id: string
  score: number
  matched: boolean
  mirrored: boolean
  visible_keypoints: number
  feedback: string
}

export type GestureName = "open-palm" | "fist" | "victory" | "point"
export type HandSide = "left" | "right"

export interface GestureTarget {
  hand: HandSide
  gesture: GestureName
}

export interface GestureMatch {
  target_hand: HandSide
  target_gesture: GestureName
  matched: boolean
  detected_gesture: GestureName | null
  hand_found: boolean
  feedback: string
}
