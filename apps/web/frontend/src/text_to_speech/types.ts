export interface TtsResource {
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
  runtimes?: Array<{
    runtime: string
    available: boolean
    size_bytes: number
    artifact_ids: string[]
  }>
}

export interface TtsModel {
  model_id: string
  display_name: string
  short_name: string
  description: string
  variant: string
  runtime: string
  required_artifact_id: string
  voices: string[]
  languages: string[]
  requires_reference_voice: boolean
  requires_reference_audio: boolean
  max_new_tokens: number
  resource: TtsResource
  ready_instance_id: string | null
  ready_instances?: Array<{
    instance_id: string
    runtime: string
  }>
}

export interface LoadedTtsModel {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  device: string
  state: string
}

export interface SynthesisOptions {
  model_id: string
  instance_id: string
  text: string
  voice: string | null
  language: string | null
  speed: number
  referenceAudio?: File | null
  referenceText?: string | null
}

export interface GeneratedSpeech {
  id: number
  modelId: string
  modelName: string
  text: string
  voice: string | null
  runtime: string
  device: string
  durationSeconds: number
  inferenceMs: number
  url: string
  createdAt: Date
}
