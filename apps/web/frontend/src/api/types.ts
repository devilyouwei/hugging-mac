export interface ResponseMeta {
  generated_at: string
  schema_version: string
}

export interface ApiResponse<T> {
  data: T
  meta: ResponseMeta
}

export interface RuntimeSummary {
  name: string
  available: boolean
  default: boolean
  devices: string[]
  dtypes: string[]
  quantizations: string[]
  unavailable_reason: string | null
  instance_count: number
  ready_count: number
}

export interface InstanceSummary {
  instance_id: string
  model_id: string
  revision: string
  runtime: string
  state: string
  created_at: string
  reference_count: number
}

export interface ModelSummary {
  model_id: string
  revision: string
  name: string
  description: string
  family: string
  tags: string[]
  capabilities: string[]
  runtimes: RuntimeSummary[]
  default_runtime: string | null
  instantiated: boolean
  instance_count: number
  ready_count: number
  instances: InstanceSummary[]
}

export interface AppManifest {
  app_id: string
  name: string
  description: string
  tags: string[]
  version: string
  frontend_route: string
  api_prefix: string
  required_models: Array<{
    model_id: string
    capabilities: string[]
    preferred_runtime: string | null
    required: boolean
  }>
}

export interface AppSummary {
  manifest: AppManifest
  status: "available" | "degraded" | "unavailable"
  unavailable_reason: string | null
}

export interface SystemInfo {
  platform: string
  os_version: string
  machine: string
  processor: string
  python_version: string
  cpu_logical_count: number | null
  cpu_physical_count: number | null
  memory_total_bytes: number
  memory_available_bytes: number
  memory_percent: number
}
