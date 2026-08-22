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
  runtimes: Array<{
    runtime: string
    available: boolean
    size_bytes: number
    artifact_ids: string[]
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
  streaming: boolean
  streaming_chunk_seconds: number | null
  variants: Array<{
    name: string
    display_name: string
    available: boolean
    available_runtimes: string[]
    streaming_chunk_seconds: number | null
  }>
  resource: AsrResourceStatus
  ready_instance_id: string | null
  ready_instances: Array<{
    instance_id: string
    variant: string
    runtime: string
  }>
}

export interface LoadedAsrModel {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  device: string
  state: string
}

export interface PipelineComponent {
  component_id: "vad" | "enhancement"
  model_id: string
  display_name: string
  description: string
  runtime: string
  downloaded: boolean
  state: "not-downloaded" | "not-loaded" | "loaded"
  loaded_model: LoadedAsrModel | null
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
  speechDurationSeconds: number | null
  vadInferenceMs: number | null
  enhancementInferenceMs: number | null
  inputAudioUrl: string | null
}

export interface StreamingPlaybackItem {
  id: number
  text: string
  durationSeconds: number
  inferenceMs: number | null
  inputAudioUrl: string
}
