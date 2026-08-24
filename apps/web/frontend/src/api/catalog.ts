import { API_BASE, ApiError, request } from "./client"
import type {
  AppSummary,
  InstanceSummary,
  ModelResourceStatus,
  ModelSummary,
  ModelInventory,
  ModelStructure,
  ArtifactInventoryItem,
  SystemInfo,
  UnloadResult,
  DownloadOperation,
} from "./types"

export async function fetchCatalog(): Promise<{
  models: ModelSummary[]
  apps: AppSummary[]
  games: AppSummary[]
}> {
  const [models, apps, games] = await Promise.all([
    request<ModelSummary[]>("/api/v1/catalog/models"),
    request<AppSummary[]>("/api/v1/catalog/apps"),
    request<AppSummary[]>("/api/v1/catalog/games"),
  ])
  return {
    models: models.data,
    apps: apps.data,
    games: games.data,
  }
}

function modelPath(modelId: string): string {
  return modelId.split("/").map(encodeURIComponent).join("/")
}

export async function fetchModelInventory(modelId: string): Promise<ModelInventory> {
  return (await request<ModelInventory>(`/api/v1/catalog/models/${modelPath(modelId)}/inventory`)).data
}

export async function fetchModelStructure(
  modelId: string,
  artifact: ArtifactInventoryItem,
): Promise<ModelStructure> {
  if (artifact.variant === null || artifact.runtime === null) {
    throw new Error("Shared artifacts do not have an inspectable runtime structure")
  }
  const query = new URLSearchParams({
    variant: artifact.variant,
    runtime: artifact.runtime,
    artifact_id: artifact.artifact_id,
  })
  return (
    await request<ModelStructure>(
      `/api/v1/catalog/models/${modelPath(modelId)}/structure?${query}`,
    )
  ).data
}

export async function downloadModelArtifact(
  modelId: string,
  artifact: ArtifactInventoryItem,
  overwrite = false,
): Promise<ModelInventory> {
  const query = overwrite ? "?overwrite=true" : ""
  return (await request<ModelInventory>(`/api/v1/catalog/models/${modelPath(modelId)}/resources/download-one${query}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      variant: artifact.variant,
      runtime: artifact.runtime,
      artifact_id: artifact.artifact_id,
    }),
  })).data
}

export async function createArtifactDownload(
  modelId: string,
  artifact: ArtifactInventoryItem,
  overwrite = false,
): Promise<DownloadOperation> {
  const query = overwrite ? "?overwrite=true" : ""
  return (await request<DownloadOperation>(
    `/api/v1/catalog/models/${modelPath(modelId)}/download-operations${query}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variant: artifact.variant,
        runtime: artifact.runtime,
        artifact_id: artifact.artifact_id,
      }),
    },
  )).data
}

export async function fetchActiveArtifactDownloads(): Promise<DownloadOperation[]> {
  return (await request<DownloadOperation[]>("/api/v1/catalog/download-operations?active=true")).data
}

export async function streamArtifactDownload(
  operationId: string,
  onEvent: (operation: DownloadOperation) => void,
  signal?: AbortSignal,
): Promise<DownloadOperation> {
  const response = await fetch(
    `${API_BASE}/api/v1/catalog/download-operations/${encodeURIComponent(operationId)}/events`,
    { signal },
  )
  if (!response.ok || !response.body) {
    throw new ApiError(`Download progress stream failed with status ${response.status}`, {
      status: response.status,
    })
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let latest: DownloadOperation | null = null
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
      if (!data) continue
      latest = JSON.parse(data) as DownloadOperation
      onEvent(latest)
    }
    if (done) break
  }
  if (!latest) throw new Error("Download progress stream ended before receiving status")
  return latest
}

export async function deleteModelArtifact(modelId: string, artifact: ArtifactInventoryItem): Promise<ModelInventory> {
  return (await request<ModelInventory>(`/api/v1/catalog/models/${modelPath(modelId)}/artifacts`, {
    method: "DELETE", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(artifact),
  })).data
}

export async function fetchModels(): Promise<ModelSummary[]> {
  return (await request<ModelSummary[]>("/api/v1/catalog/models")).data
}

export async function fetchApps(): Promise<AppSummary[]> {
  return (await request<AppSummary[]>("/api/v1/catalog/apps")).data
}

export async function fetchGames(): Promise<AppSummary[]> {
  return (await request<AppSummary[]>("/api/v1/catalog/games")).data
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  return (await request<SystemInfo>("/api/v1/system/info")).data
}

export async function loadModel(
  modelId: string,
  runtime: string,
  options: { variant?: string; device?: string; warmup?: boolean } = {},
): Promise<InstanceSummary> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<InstanceSummary>(`/api/v1/catalog/models/${path}/instances`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runtime,
        variant: options.variant ?? null,
        device: options.device ?? null,
        warmup: options.warmup ?? false,
      }),
    })
  ).data
}

export async function unloadModel(instanceId: string, force = false): Promise<UnloadResult> {
  const query = force ? "?force=true" : ""
  return (
    await request<UnloadResult>(
      `/api/v1/catalog/instances/${encodeURIComponent(instanceId)}${query}`,
      { method: "DELETE" },
    )
  ).data
}

export async function fetchModelResources(
  modelId: string,
  variant?: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  const query = variant ? `?variant=${encodeURIComponent(variant)}` : ""
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources${query}`,
    )
  ).data
}

export async function downloadModelResources(
  modelId: string,
  variant?: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  const query = variant ? `?variant=${encodeURIComponent(variant)}` : ""
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources/download${query}`,
      { method: "POST" },
    )
  ).data
}

export async function convertModelResources(
  modelId: string,
  artifact: ArtifactInventoryItem,
  overwrite = false,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources/convert`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          variant: artifact.variant,
          runtime: artifact.runtime,
          artifact_id: artifact.artifact_id,
          overwrite,
        }),
      },
    )
  ).data
}

export async function deleteModelResources(
  modelId: string,
  variant?: string,
): Promise<ModelResourceStatus> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  const query = variant ? `?variant=${encodeURIComponent(variant)}` : ""
  return (
    await request<ModelResourceStatus>(
      `/api/v1/catalog/models/${path}/resources${query}`,
      { method: "DELETE" },
    )
  ).data
}
