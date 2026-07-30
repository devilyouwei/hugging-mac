export interface AsrResourceStatus {
  model_id: string
  revision: string
  variant: string
  artifacts: Array<{
    artifact_id: string
    format: string
    runtime: string | null
    available: boolean
    size_bytes: number | null
  }>
}

export interface TranscriptionResult {
  text: string
  model_id: string
  instance_id: string
  runtime: string
  device: string
  sample_rate: number
  duration_seconds: number
  generated_tokens: number
  inference_ms: number | null
}

export interface TranscriptSegment {
  id: number
  createdAt: Date
  durationSeconds: number
  status: "recognizing" | "complete" | "error"
  text: string
  inferenceMs: number | null
}
