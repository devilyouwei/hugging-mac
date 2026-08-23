import type { ResourceStatus } from "@/vision/types"

export type GameRuntime = "coreml" | "pytorch-mps" | "onnx"

export interface PoseGameModelSelection {
  variant: string
  runtime: GameRuntime
}

const SUPPORTED_RUNTIMES = new Set<GameRuntime>(["coreml", "pytorch-mps", "onnx"])

export function availableGameRuntimes(resource: ResourceStatus | null): GameRuntime[] {
  if (!resource) return []
  const runtimes: GameRuntime[] = []
  for (const artifact of resource.artifacts) {
    const runtime = artifact.runtime as GameRuntime | null
    if (
      artifact.available
      && runtime
      && SUPPORTED_RUNTIMES.has(runtime)
      && !runtimes.includes(runtime)
    ) {
      runtimes.push(runtime)
    }
  }
  return runtimes
}

export function firstAvailableModel(
  resources: ResourceStatus[],
): PoseGameModelSelection | null {
  for (const resource of resources) {
    const runtime = availableGameRuntimes(resource)[0]
    if (runtime) return { variant: resource.variant, runtime }
  }
  return null
}
