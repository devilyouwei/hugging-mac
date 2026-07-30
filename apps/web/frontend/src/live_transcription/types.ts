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

export interface AsrModel {
  model_id: string
  display_name: string
  short_name: string
  description: string
  variant: string
  runtime: string
  required_artifact_id: string
  supports_coreml_conversion: boolean
  rich_understanding: boolean
  resource: AsrResourceStatus
  ready_instance_id: string | null
}

export interface LoadedAsrModel {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  device: string
  state: string
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
  languages: string[]
  emotion: string | null
  events: string[]
}

export interface TranscriptSegment {
  id: number
  createdAt: Date
  durationSeconds: number
  status: "recognizing" | "complete" | "error"
  text: string
  inferenceMs: number | null
  modelName: string
  languages: string[]
  emotion: string | null
  events: string[]
}
