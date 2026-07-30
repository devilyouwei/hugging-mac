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

export async function convertTtsCoreMl(modelId: string): Promise<TtsResource> {
  return (
    await request<TtsResource>(
      `${PREFIX}/resources/coreml/convert${modelQuery(modelId)}`,
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
  const response = await fetch(`${API_BASE}${PREFIX}/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
    signal,
  })
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const body = (await response.json()) as { error?: { message?: string } }
      message = body.error?.message ?? message
    } catch {
      // Keep the HTTP fallback message.
    }
    throw new ApiError(message, { status: response.status })
  }
  return { blob: await response.blob(), headers: response.headers }
}
