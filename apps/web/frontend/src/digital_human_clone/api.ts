import { ApiError, request } from "@/api/client"

import type {
  ChatStreamEvent,
  ConversationMessage,
  LoadedModels,
  Setup,
  Transcript,
} from "./types"

const PREFIX = "/api/v1/games/digital-human"
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

export async function fetchSetup(llmVariant?: string): Promise<Setup> {
  const query = llmVariant ? `?llm_variant=${encodeURIComponent(llmVariant)}` : ""
  return (await request<Setup>(`${PREFIX}/setup${query}`)).data
}

export async function loadModels(llmVariant: string): Promise<LoadedModels> {
  return (
    await request<LoadedModels>(
      `${PREFIX}/setup/load?llm_variant=${encodeURIComponent(llmVariant)}`,
      { method: "POST" },
    )
  ).data
}

export async function loadModel(role: "asr" | "llm" | "tts", llmVariant: string): Promise<string> {
  return (
    await request<string>(
      `${PREFIX}/setup/models/${role}/load?llm_variant=${encodeURIComponent(llmVariant)}`,
      { method: "POST" },
    )
  ).data
}

export async function transcribe(
  audio: Blob,
  instanceId: string,
  vadInstanceId?: string | null,
  enhancementInstanceId?: string | null,
  signal?: AbortSignal,
): Promise<Transcript> {
  const form = new FormData()
  const extension = audio.type.includes("mp4") ? "m4a" : audio.type.includes("webm") ? "webm" : "wav"
  form.append("file", audio, `utterance.${extension}`)
  form.append("instance_id", instanceId)
  if (vadInstanceId) form.append("vad_instance_id", vadInstanceId)
  if (enhancementInstanceId) form.append("enhancement_instance_id", enhancementInstanceId)
  return (
    await request<Transcript>(`${PREFIX}/transcribe`, { method: "POST", body: form, signal })
  ).data
}

export async function streamReply(
  options: {
    image: Blob
    instanceId: string
    prompt: string
    history: ConversationMessage[]
  },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append("image", options.image, "webcam.jpg")
  form.append("instance_id", options.instanceId)
  form.append("prompt", options.prompt)
  form.append("history_json", JSON.stringify(options.history))
  const response = await fetch(`${API_BASE}${PREFIX}/chat/stream`, {
    method: "POST",
    body: form,
    signal,
  })
  if (!response.ok || !response.body) {
    throw new ApiError("Digital human generation failed", { status: response.status })
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
      if (data) onEvent(JSON.parse(data) as ChatStreamEvent)
    }
    if (done) break
  }
}

export async function synthesize(
  text: string,
  voice: string,
  language: string,
  instanceId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const form = new FormData()
  form.append("text", text)
  form.append("voice", voice)
  form.append("language", language)
  form.append("instance_id", instanceId)
  const response = await fetch(`${API_BASE}${PREFIX}/synthesize`, {
    method: "POST",
    body: form,
    signal,
  })
  if (!response.ok) throw new ApiError("Speech synthesis failed", { status: response.status })
  return response.blob()
}
