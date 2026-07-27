import { request } from "@/api/client"

import type { DetectOptions, DetectionResult, ResourceStatus } from "./types"

export async function fetchResourceStatus(): Promise<ResourceStatus> {
  return (
    await request<ResourceStatus>("/api/v1/apps/object-detection/resources")
  ).data
}

export async function downloadSource(): Promise<ResourceStatus> {
  return (
    await request<ResourceStatus>(
      "/api/v1/apps/object-detection/resources/source/download",
      { method: "POST" },
    )
  ).data
}

export async function convertCoreMl(): Promise<ResourceStatus> {
  return (
    await request<ResourceStatus>(
      "/api/v1/apps/object-detection/resources/coreml/convert",
      { method: "POST" },
    )
  ).data
}

export async function detectObjects(
  file: File,
  options: DetectOptions,
  requestOptions: {
    signal?: AbortSignal
    cacheInput?: boolean
  } = {},
): Promise<DetectionResult> {
  if (requestOptions.cacheInput === false) {
    const query = new URLSearchParams({
      runtime: options.runtime,
      confidence: String(options.confidence),
      iou_threshold: String(options.iouThreshold),
      max_detections: String(options.maxDetections),
    })
    return (
      await request<DetectionResult>(
        `/api/v1/apps/object-detection/detect/frame?${query}`,
        {
          method: "POST",
          body: file,
          headers: { "Content-Type": file.type || "image/jpeg" },
          signal: requestOptions.signal,
        },
      )
    ).data
  }

  const body = new FormData()
  body.append("file", file)
  body.append("runtime", options.runtime)
  body.append("confidence", String(options.confidence))
  body.append("iou_threshold", String(options.iouThreshold))
  body.append("max_detections", String(options.maxDetections))
  body.append("cache_input", "true")

  return (
    await request<DetectionResult>("/api/v1/apps/object-detection/detect", {
      method: "POST",
      body,
      signal: requestOptions.signal,
    })
  ).data
}
