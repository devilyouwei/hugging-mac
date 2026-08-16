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
  variant: string
  runtime: string
  state: string
  created_at: string
  reference_count: number
  load_metrics: LifecycleMetrics | null
}

export interface LifecycleMetrics {
  operation: "load" | "unload"
  duration_ms: number
  process_rss_before_bytes: number | null
  process_rss_after_bytes: number | null
  process_rss_change_bytes: number | null
  memory_allocated_bytes: number | null
  memory_released_bytes: number | null
  warmup_included: boolean
  measured_at: string
}

export interface UnloadResult {
  instance_id: string
  model_id: string
  revision: string
  variant: string
  runtime: string
  state: string
  metrics: LifecycleMetrics
}

export interface ModelArtifactResource {
  artifact_id: string
  format: string
  runtime: string | null
  provisioning: "download" | "convert"
  available: boolean
  size_bytes: number | null
}

export interface RuntimeResourceStatus {
  runtime: string
  available: boolean
  size_bytes: number
  artifact_ids: string[]
}

export interface ConversionTargetStatus {
  target_format: string
  runtime: string | null
  artifact_id: string
  available: boolean
  size_bytes: number | null
}

export interface ModelResourceStatus {
  model_id: string
  revision: string
  variant: string
  artifacts: ModelArtifactResource[]
  runtimes: RuntimeResourceStatus[]
  conversion_targets: ConversionTargetStatus[]
  total_size_bytes: number
}

export interface ArtifactInventoryItem {
  variant: string
  runtime: string
  artifact_id: string
  format: string
  convertible: boolean
  shared: boolean
  available: boolean
  size_bytes: number | null
}

export interface ResourceOption extends Omit<ArtifactInventoryItem, "convertible"> {
  resource_id: string
  source: { kind: string; repo_id?: string; filename?: string; url?: string }
}

export interface ModelInventory {
  model_id: string
  revision: string
  resources: ResourceOption[]
  artifacts: ArtifactInventoryItem[]
}

export interface ModelSummary {
  model_id: string
  revision: string
  variants: VariantSummary[]
  default_variant: string
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

export interface VariantSummary {
  name: string
  display_name: string
  description: string
  metadata: Record<string, string | number | boolean>
  default: boolean
  instance_count: number
  ready_count: number
}

export interface AppManifest {
  app_id: string
  name: string
  description: string
  category: "application" | "game"
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
  chip_name: string | null
  python_version: string
  cpu_logical_count: number | null
  cpu_physical_count: number | null
  cpu_performance_cores: number | null
  cpu_efficiency_cores: number | null
  gpu_cores: number | null
  neural_engine_cores: number | null
  memory_total_bytes: number
  memory_available_bytes: number
  memory_percent: number
}
