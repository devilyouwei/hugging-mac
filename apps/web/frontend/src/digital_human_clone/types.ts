export interface ModelState {
  role: "asr" | "llm" | "tts"
  model_id: string
  runtime: string
  resources_ready: boolean
  ready_instance_id: string | null
}

export interface Setup {
  models: ModelState[]
}

export interface LoadedModels {
  asr_instance_id: string
  llm_instance_id: string
  tts_instance_id: string
}

export interface Transcript {
  text: string
  inference_ms: number | null
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
