import { request } from "@/api/client"

import type { AsrResourceStatus, TranscriptionResult } from "./types"

const PREFIX = "/api/v1/apps/live-transcription"

export async function fetchAsrResourceStatus(): Promise<AsrResourceStatus> {
  return (await request<AsrResourceStatus>(`${PREFIX}/resources`)).data
}

export async function downloadAsrWeights(): Promise<AsrResourceStatus> {
  return (
    await request<AsrResourceStatus>(`${PREFIX}/resources/source/download`, {
      method: "POST",
    })
  ).data
}

export async function convertAsrCoreMl(): Promise<AsrResourceStatus> {
  return (
    await request<AsrResourceStatus>(`${PREFIX}/resources/coreml/convert`, {
      method: "POST",
    })
  ).data
}

export async function transcribeUtterance(
  audio: Blob,
  signal?: AbortSignal,
): Promise<TranscriptionResult> {
  const body = new FormData()
  body.append("file", audio, "vad-utterance.wav")
  return (
    await request<TranscriptionResult>(`${PREFIX}/transcribe`, {
      method: "POST",
      body,
      signal,
    })
  ).data
}
