import type { ResourceStatus } from "@/vision/types"

export type GameRuntime = "coreml" | "pytorch-mps"

export function availableGameRuntimes(resource: ResourceStatus | null): GameRuntime[] {
  if (!resource) return []
  const available = new Set(
    resource.artifacts
      .filter((artifact) => artifact.available)
      .map((artifact) => artifact.runtime),
  )
  const runtimes: GameRuntime[] = []
  if (available.has("coreml")) runtimes.push("coreml")
  if (available.has("pytorch-mps")) runtimes.push("pytorch-mps")
  return runtimes
}
