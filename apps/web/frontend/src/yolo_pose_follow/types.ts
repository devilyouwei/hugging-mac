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
