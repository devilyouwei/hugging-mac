import { request } from "@/api/client"

import type {
  AsrModel,
  LoadedAsrModel,
  PipelineComponent,
  StreamingSession,
  StreamingTranscriptionResult,
  TranscriptionResult,
  VadDetection,
} from "./types"

const PREFIX = "/api/v1/apps/live-transcription"

export async function fetchAsrModels(): Promise<AsrModel[]> {
  return (await request<AsrModel[]>(`${PREFIX}/models`)).data
}

export async function loadAsrModel(
  modelId: string,
  runtime: string,
): Promise<LoadedAsrModel> {
  return (await request<LoadedAsrModel>(`${PREFIX}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, runtime }),
  })).data
}

export async function fetchPipelineComponents(): Promise<PipelineComponent[]> {
  return (await request<PipelineComponent[]>(`${PREFIX}/pipeline/components`)).data
}

export async function loadVadModel(): Promise<LoadedAsrModel> {
  return (await request<LoadedAsrModel>(`${PREFIX}/vad/model/load`, {
    method: "POST",
  })).data
}

export async function loadEnhancementModel(): Promise<LoadedAsrModel> {
  return (await request<LoadedAsrModel>(`${PREFIX}/enhancement/model/load`, {
    method: "POST",
  })).data
}

export async function detectVoiceActivity(
  audio: Blob,
  instanceId: string,
  threshold: number,
  signal?: AbortSignal,
): Promise<VadDetection> {
  const body = new FormData()
  body.append("file", audio, "silero-window.wav")
  body.append("instance_id", instanceId)
  body.append("threshold", String(threshold))
  return (
    await request<VadDetection>(`${PREFIX}/vad/detect`, {
      method: "POST",
      body,
      signal,
    })
  ).data
}

export async function startTranscriptionStream(
  modelId: string,
  instanceId: string,
): Promise<StreamingSession> {
  return (await request<StreamingSession>(`${PREFIX}/stream/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, instance_id: instanceId }),
  })).data
}

export async function transcribeStreamChunk(
  audio: Blob,
  session: StreamingSession,
  signal?: AbortSignal,
): Promise<StreamingTranscriptionResult> {
  const body = new FormData()
  body.append("file", audio, "nemotron-stream-chunk.wav")
  body.append("model_id", session.model_id)
  body.append("instance_id", session.instance_id)
  body.append("session_id", session.session_id)
  return (await request<StreamingTranscriptionResult>(`${PREFIX}/stream/chunk`, {
    method: "POST",
    body,
    signal,
  })).data
}

export async function finishTranscriptionStream(
  session: StreamingSession,
): Promise<StreamingTranscriptionResult> {
  return (await request<StreamingTranscriptionResult>(`${PREFIX}/stream/finish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: session.model_id,
      instance_id: session.instance_id,
      session_id: session.session_id,
    }),
  })).data
}

export async function cancelTranscriptionStream(session: StreamingSession): Promise<void> {
  await request<void>(`${PREFIX}/stream/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: session.model_id,
      instance_id: session.instance_id,
      session_id: session.session_id,
    }),
  })
}

export async function transcribeUtterance(
  audio: Blob,
  modelId: string,
  instanceId: string,
  vadInstanceId: string,
  enhancementInstanceId: string,
  vadThreshold: number,
  signal?: AbortSignal,
): Promise<TranscriptionResult> {
  const body = new FormData()
  body.append("file", audio, "vad-utterance.wav")
  body.append("model_id", modelId)
  body.append("instance_id", instanceId)
  if (vadInstanceId) body.append("vad_instance_id", vadInstanceId)
  if (enhancementInstanceId) body.append("enhancement_instance_id", enhancementInstanceId)
  body.append("vad_threshold", String(vadThreshold))
  return (
    await request<TranscriptionResult>(`${PREFIX}/transcribe`, {
      method: "POST",
      body,
      signal,
    })
  ).data
}
