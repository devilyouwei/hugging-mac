import { request } from "@/api/client"

import type {
  ResourceStatus,
  VisionInference,
  VisionOptions,
  VisionResultBase,
} from "./types"

function variantQuery(variant: string): string {
  return `?variant=${encodeURIComponent(variant)}`
}

export function createVisionApi<Result extends VisionResultBase>(
  apiPrefix: string,
  action: string,
) {
  return {
    async fetchResourceStatus(variant?: string): Promise<ResourceStatus> {
      return (
        await request<ResourceStatus>(
          `${apiPrefix}/resources${variant ? variantQuery(variant) : ""}`,
        )
      ).data
    },

    infer: (async (
      file: File,
      options: VisionOptions,
      requestOptions: {
        signal?: AbortSignal
        cacheInput?: boolean
      } = {},
    ): Promise<Result> => {
      if (requestOptions.cacheInput === false) {
        const query = new URLSearchParams({
          runtime: options.runtime,
          variant: options.variant,
          confidence: String(options.confidence),
          iou_threshold: String(options.iouThreshold),
          max_detections: String(options.maxDetections),
        })
        return (
          await request<Result>(`${apiPrefix}/${action}/frame?${query}`, {
            method: "POST",
            body: file,
            headers: { "Content-Type": file.type || "image/jpeg" },
            signal: requestOptions.signal,
          })
        ).data
      }

      const body = new FormData()
      body.append("file", file)
      body.append("runtime", options.runtime)
      body.append("variant", options.variant)
      body.append("confidence", String(options.confidence))
      body.append("iou_threshold", String(options.iouThreshold))
      body.append("max_detections", String(options.maxDetections))
      body.append("cache_input", "true")

      return (
        await request<Result>(`${apiPrefix}/${action}`, {
          method: "POST",
          body,
          signal: requestOptions.signal,
        })
      ).data
    }) satisfies VisionInference,
  }
}
