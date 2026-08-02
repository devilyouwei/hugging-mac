import { ApiError, request } from "@/api/client"

import type { ChatModel, ChatReply, ChatResource, ChatStreamEvent, LoadedChatModel } from "./types"

const PREFIX = "/api/v1/apps/chat"
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

export async function fetchChatModels(): Promise<ChatModel[]> {
  return (await request<ChatModel[]>(`${PREFIX}/models`)).data
}

export async function fetchLoadedChatModel(modelId: string): Promise<LoadedChatModel | null> {
  return (
    await request<LoadedChatModel | null>(
      `${PREFIX}/model?model_id=${encodeURIComponent(modelId)}`,
    )
  ).data
}

export async function downloadChatModel(modelId: string): Promise<ChatResource> {
  return (
    await request<ChatResource>(
      `${PREFIX}/resources/source/download?model_id=${encodeURIComponent(modelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function deleteChatModel(modelId: string): Promise<ChatResource> {
  return (
    await request<ChatResource>(
      `${PREFIX}/resources?model_id=${encodeURIComponent(modelId)}`,
      { method: "DELETE" },
    )
  ).data
}

export async function loadChatModel(modelId: string): Promise<LoadedChatModel> {
  return (
    await request<LoadedChatModel>(`${PREFIX}/model/load`, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
      headers: { "Content-Type": "application/json" },
    })
  ).data
}

export async function unloadChatModel(modelId: string): Promise<number> {
  return (
    await request<number>(`${PREFIX}/model?model_id=${encodeURIComponent(modelId)}`, {
      method: "DELETE",
    })
  ).data
}

export async function sendChatMessage(options: {
  instanceId: string
  prompt: string
  history: Array<{ role: "user" | "assistant"; content: string }>
  images: File[]
  maxTokens: number
  temperature: number
  enableThinking: boolean
}): Promise<ChatReply> {
  const form = new FormData()
  form.append("instance_id", options.instanceId)
  form.append("prompt", options.prompt)
  form.append("history_json", JSON.stringify(options.history))
  form.append("max_tokens", String(options.maxTokens))
  form.append("temperature", String(options.temperature))
  form.append("enable_thinking", String(options.enableThinking))
  options.images.forEach((image) => form.append("images", image))
  return (
    await request<ChatReply>(`${PREFIX}/messages`, {
      method: "POST",
      body: form,
    })
  ).data
}

export async function streamChatMessage(
  options: {
    instanceId: string
    prompt: string
    history: Array<{ role: "user" | "assistant"; content: string }>
    images: File[]
    maxTokens: number
    temperature: number
    enableThinking: boolean
  },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<ChatStreamEvent> {
  const form = new FormData()
  form.append("instance_id", options.instanceId)
  form.append("prompt", options.prompt)
  form.append("history_json", JSON.stringify(options.history))
  form.append("max_tokens", String(options.maxTokens))
  form.append("temperature", String(options.temperature))
  form.append("enable_thinking", String(options.enableThinking))
  options.images.forEach((image) => form.append("images", image))

  const response = await fetch(`${API_BASE}${PREFIX}/messages/stream`, {
    method: "POST",
    body: form,
    signal,
  })
  if (!response.ok || !response.body) {
    throw new ApiError(`Chat stream failed with status ${response.status}`, {
      status: response.status,
    })
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finalEvent: ChatStreamEvent | null = null
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const eventName = frame.match(/^event:\s*(.+)$/m)?.[1] ?? "message"
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
      if (!data) continue
      const payload = JSON.parse(data) as ChatStreamEvent | { message?: string }
      if (eventName === "error") {
        throw new Error("message" in payload ? payload.message : "Chat stream failed")
      }
      const event = payload as ChatStreamEvent
      onEvent(event)
      if (eventName === "done") finalEvent = event
    }
    if (done) break
  }
  if (!finalEvent) throw new Error("Chat stream ended before completion")
  return finalEvent
}
