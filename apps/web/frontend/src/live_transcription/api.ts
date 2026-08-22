import { request } from "@/api/client"

import type {
  AsrModel,
  LoadedAsrModel,
  PipelineComponent,
} from "./types"

const PREFIX = "/api/v1/apps/live-transcription"

export async function fetchAsrModels(): Promise<AsrModel[]> {
  return (await request<AsrModel[]>(`${PREFIX}/models`)).data
}

export async function loadAsrModel(
  modelId: string,
  variant: string,
  runtime: string,
): Promise<LoadedAsrModel> {
  return (await request<LoadedAsrModel>(`${PREFIX}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, variant, runtime }),
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
