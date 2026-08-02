import { ApiError, request } from "@/api/client"

import type {
  LoadedTtsModel,
  SynthesisOptions,
  TtsModel,
  TtsResource,
} from "./types"

const PREFIX = "/api/v1/apps/text-to-speech"
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

function modelQuery(modelId: string): string {
  return `?model_id=${encodeURIComponent(modelId)}`
}

export async function fetchTtsModels(): Promise<TtsModel[]> {
  return (await request<TtsModel[]>(`${PREFIX}/models`)).data
}

export async function downloadTtsWeights(modelId: string): Promise<TtsResource> {
  return (
    await request<TtsResource>(
      `${PREFIX}/resources/source/download${modelQuery(modelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function loadTtsModel(modelId: string): Promise<LoadedTtsModel> {
  return (
    await request<LoadedTtsModel>(`${PREFIX}/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    })
  ).data
}

export async function synthesizeSpeech(
  options: SynthesisOptions,
  signal?: AbortSignal,
): Promise<{ blob: Blob; headers: Headers }> {
  let endpoint = `${PREFIX}/synthesize`
  let body: BodyInit
  let headers: HeadersInit | undefined
  if (options.referenceAudio) {
    const form = new FormData()
    form.append("file", options.referenceAudio)
    form.append("model_id", options.model_id)
    form.append("instance_id", options.instance_id)
    form.append("text", options.text)
    if (options.voice) form.append("voice", options.voice)
    form.append("reference_text", options.referenceText ?? "")
    form.append("speed", String(options.speed))
    if (options.language) form.append("language", options.language)
    endpoint = `${PREFIX}/synthesize/reference`
    body = form
  } else {
    const { referenceAudio: _, referenceText: __, ...jsonOptions } = options
    headers = { "Content-Type": "application/json" }
    body = JSON.stringify(jsonOptions)
  }
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers,
    body,
    signal,
  })
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const body = (await response.json()) as {
        error?: { message?: string; details?: { reason?: string } }
      }
      message = body.error?.message ?? message
      if (body.error?.details?.reason) {
        message = `${message}: ${body.error.details.reason}`
      }
    } catch {
      // Keep the HTTP fallback message.
    }
    throw new ApiError(message, { status: response.status })
  }
  return { blob: await response.blob(), headers: response.headers }
}
