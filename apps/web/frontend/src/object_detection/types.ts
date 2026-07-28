export type RuntimeChoice = "auto" | "coreml" | "pytorch-mps"

export interface DetectOptions {
  runtime: RuntimeChoice
  variant: string
  confidence: number
  iouThreshold: number
  maxDetections: number
}

export interface DetectionResult {
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
  detections: Array<{
    box: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    confidence: number
    class_id: number
    label: string
  }>
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
