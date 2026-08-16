export interface ModelState {
  role: "asr" | "llm" | "tts"
  model_id: string
  runtime: string
  resources_ready: boolean
  ready_instance_id: string | null
}

export interface Setup {
  models: ModelState[]
  llm_variants: Array<{ name: string; display_name: string; description: string }>
  selected_llm_variant: string
  vad_instance_id: string | null
  enhancement_instance_id: string | null
}

export interface LoadedModels {
  asr_instance_id: string
  llm_instance_id: string
  tts_instance_id: string
  vad_instance_id: string | null
  enhancement_instance_id: string | null
}

export interface Transcript {
  text: string
  inference_ms: number | null
  source_duration_seconds: number | null
  speech_duration_seconds: number | null
  speech_segment_count: number
  vad_inference_ms: number | null
  enhancement_inference_ms: number | null
}

export interface ConversationMessage {
  role: "user" | "assistant"
  content: string
}

export interface ChatStreamEvent {
  delta: string
  finish_reason: "stop" | "length" | "error" | null
  generated_tokens: number | null
}
