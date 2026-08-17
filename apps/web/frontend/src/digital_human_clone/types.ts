export interface ModelState {
  role: "asr" | "llm" | "tts"
  model_id: string
  runtime: string
  resources_ready: boolean
  ready_instance_id: string | null
}

export interface Setup {
  models: ModelState[]
  asr_models: AsrModelOption[]
  selected_asr_model_id: string
  llm_variants: Array<{ model_id: string; name: string; display_name: string; description: string }>
  selected_llm_model_id: string
  selected_llm_variant: string
  vad_instance_id: string | null
  enhancement_instance_id: string | null
}

export interface AsrModelOption {
  model_id: string
  display_name: string
  description: string
  streaming: boolean
  streaming_chunk_seconds: number | null
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

export interface StreamingSession {
  session_id: string
  model_id: string
  instance_id: string
  runtime: string
  device: string
  sample_rate: number
}

export interface StreamingTranscript {
  model_id: string
  text: string
  delta: string
  audio_seconds: number
  generated_tokens: number
  detected_language: string | null
  inference_ms: number | null
  is_final: boolean
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
