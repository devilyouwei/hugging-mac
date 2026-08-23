import { request } from "@/api/client"

export interface LoadedModelInstance {
  instance_id: string
  model_id: string
  variant: string
  runtime: string
  state: string
}

export async function loadSharedModel(
  modelId: string,
  variant: string,
  runtime: string,
): Promise<LoadedModelInstance> {
  const path = modelId.split("/").map(encodeURIComponent).join("/")
  return (
    await request<LoadedModelInstance>(
      `/api/v1/catalog/models/${path}/instances`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variant, runtime }),
      },
    )
  ).data
}

export async function unloadModel(instanceId: string): Promise<void> {
  await request(`/api/v1/catalog/instances/${encodeURIComponent(instanceId)}`, {
    method: "DELETE",
  })
}

export function errorMessage(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback
}
