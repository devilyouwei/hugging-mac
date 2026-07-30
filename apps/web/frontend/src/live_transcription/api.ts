import { request } from "@/api/client"

import type {
  AsrModel,
  AsrResourceStatus,
  LoadedAsrModel,
  TranscriptionResult,
} from "./types"

const PREFIX = "/api/v1/apps/live-transcription"

function modelQuery(modelId: string): string {
  return `?model_id=${encodeURIComponent(modelId)}`
}

export async function fetchAsrModels(): Promise<AsrModel[]> {
  return (await request<AsrModel[]>(`${PREFIX}/models`)).data
}

export async function downloadAsrWeights(modelId: string): Promise<AsrResourceStatus> {
  return (
    await request<AsrResourceStatus>(
      `${PREFIX}/resources/source/download${modelQuery(modelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function convertAsrCoreMl(modelId: string): Promise<AsrResourceStatus> {
  return (
    await request<AsrResourceStatus>(
      `${PREFIX}/resources/coreml/convert${modelQuery(modelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function loadAsrModel(modelId: string): Promise<LoadedAsrModel> {
  return (
    await request<LoadedAsrModel>(`${PREFIX}/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    })
  ).data
}

export async function transcribeUtterance(
  audio: Blob,
  modelId: string,
  instanceId: string,
  signal?: AbortSignal,
): Promise<TranscriptionResult> {
  const body = new FormData()
  body.append("file", audio, "vad-utterance.wav")
  body.append("model_id", modelId)
  body.append("instance_id", instanceId)
  return (
    await request<TranscriptionResult>(`${PREFIX}/transcribe`, {
      method: "POST",
      body,
      signal,
    })
  ).data
}
