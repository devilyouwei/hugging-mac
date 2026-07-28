import { request } from "./client"
import type {
  AppSummary,
  InstanceSummary,
  ModelResourceStatus,
  ModelSummary,
  SystemInfo,
  UnloadResult,
} from "./types"

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

export async function loadModel(
  modelId: string,
  runtime: string,
  options: { device?: string; warmup?: boolean } = {},
): Promise<InstanceSummary> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<InstanceSummary>(`/api/v1/catalog/models/${path}/instances`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runtime,
        device: options.device ?? null,
        warmup: options.warmup ?? false,
      }),
    })
  ).data
}

export async function unloadModel(instanceId: string): Promise<UnloadResult> {
  return (
    await request<UnloadResult>(
      `/api/v1/catalog/instances/${encodeURIComponent(instanceId)}`,
      { method: "DELETE" },
    )
  ).data
}

export async function fetchModelResources(
  modelId: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources`,
    )
  ).data
}

export async function downloadModelResources(
  modelId: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources/download`,
      { method: "POST" },
    )
  ).data
}

export async function convertModelResources(
  modelId: string,
  targetFormat: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources/convert`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_format: targetFormat }),
      },
    )
  ).data
}

export async function deleteModelResources(
  modelId: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources`,
      { method: "DELETE" },
    )
  ).data
}
