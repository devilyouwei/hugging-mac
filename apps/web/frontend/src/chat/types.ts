export interface LoadedChatModel {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  device: string
  state: string
}

export interface ChatReply {
  content: string
  model_id: string
  instance_id: string
  runtime: string
  device: string
  prompt_tokens: number | null
  generated_tokens: number | null
  inference_ms: number | null
}

export interface ChatStreamEvent extends Omit<ChatReply, "content" | "prompt_tokens"> {
  delta: string
  finish_reason: "stop" | "length" | "error" | null
  error: string | null
}

export interface ConversationMessage {
  id: number
  role: "user" | "assistant"
  content: string
  images: Array<{ name: string; url: string }>
  meta?: string
}
