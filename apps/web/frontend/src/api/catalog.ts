import { request } from "./client"
import type { AppSummary, ModelSummary, SystemInfo } from "./types"

export async function fetchCatalog(): Promise<{
  models: ModelSummary[]
  apps: AppSummary[]
}> {
  const [models, apps] = await Promise.all([
    request<ModelSummary[]>("/api/v1/catalog/models"),
    request<AppSummary[]>("/api/v1/catalog/apps"),
  ])
  return {
    models: models.data,
    apps: apps.data,
  }
}

export async function fetchModels(): Promise<ModelSummary[]> {
  return (await request<ModelSummary[]>("/api/v1/catalog/models")).data
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  return (await request<SystemInfo>("/api/v1/system/info")).data
}
