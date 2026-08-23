export type RuntimeChoice = "auto" | "coreml" | "pytorch-mps" | "onnx"

export interface VisionOptions {
  runtime: RuntimeChoice
  variant: string
  confidence: number
  iouThreshold: number
  maxDetections: number
  poseEnabled?: boolean
  faceEnabled?: boolean
  faceConfidence?: number
  handEnabled?: boolean
  handConfidence?: number
  handLandmarksEnabled?: boolean
  handLandmarkConfidence?: number
  handInputMirrored?: boolean
}

export interface VisionResultBase {
  input_cache_id: string | null
  model_id: string
  variant: string
  instance_id: string
  runtime: string
  device: string
  image_size: {
    width: number
    height: number
  }
  timings: {
    preprocess_ms: number | null
    inference_ms: number | null
    postprocess_ms: number | null
  }
}

export interface ResourceStatus {
  model_id: string
  revision: string
  variant: string
  default_runtime: string | null
  variants: Array<{
    name: string
    display_name: string
    description: string
    default: boolean
  }>
  artifacts: Array<{
    artifact_id: string
    format: string
    runtime: string | null
    available: boolean
    size_bytes: number | null
  }>
}

export interface InferenceRequestOptions {
  signal?: AbortSignal
  cacheInput?: boolean
}

export type VisionInference = (
  file: File,
  options: VisionOptions,
  requestOptions?: InferenceRequestOptions,
) => Promise<VisionResultBase>
