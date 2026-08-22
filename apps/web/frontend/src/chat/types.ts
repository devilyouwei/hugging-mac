export interface LoadedChatModel {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  device: string
  state: string
}

export interface ChatResource {
  model_id: string
  revision: string
  variant: string
  artifacts: Array<{ artifact_id: string; available: boolean; size_bytes: number | null }>
  total_size_bytes: number
}

export interface ChatModel {
  profile_id: string
  model_id: string
  display_name: string
  short_name: string
  description: string
  variant: string
  runtime: string
  required_artifact_id: string
  disk_size_bytes: number
  supports_images: boolean
  resource: ChatResource
  ready_instance_id: string | null
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
  audio?: {
    status: "generating" | "ready"
    url: string | null
    durationSeconds: number
  }
  status?: "complete" | "streaming" | "interrupted"
  source?: "text" | "asr"
  asrTiming?: {
    totalMs: number
    vadMs: number
    inferenceMs: number | null
  }
  llmTiming?: {
    runtime: string
    generatedTokens: number | null
    inferenceMs: number | null
    tokensPerSecond: number | null
  }
  ttsTiming?: {
    runtime: string
    segmentCount: number
    firstAudioMs: number
    slowestSegmentMs: number
  }
}
