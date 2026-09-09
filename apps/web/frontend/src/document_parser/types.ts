export type JobKind = "markdown" | "structure"

export interface Job {
  id: string
  document_id: string
  kind: JobKind
  model_id: string
  model_variant: string
  model_runtime: string
  layout_model_id: string | null
  layout_model_variant: string | null
  layout_model_runtime: string | null
  state: string
  stage: string
  completed_units: number
  total_units: number
  progress: number
  cancel_requested: boolean
  error: string | null
  event_sequence: number
  created_at: string
  updated_at: string
}

export interface ParserModel {
  model_id: string
  name: string
  description: string
  variant: string
  runtime: string
  ready: boolean
  source_url: string | null
}

export interface DocumentRecord {
  id: string
  name: string
  source_kind: "pdf" | "images"
  status: string
  page_count: number
  disk_bytes: number
  current_job_id: string | null
  created_at: string
  updated_at: string
  job: Job | null
  jobs: Record<JobKind, Job | null>
}

export interface DocumentCreated {
  document: DocumentRecord
  job: Job | null
}

export interface DocumentOutput {
  document_id: string
  kind: JobKind
  content: string
  digest: string | null
  structured: StructuredDocument | null
  warnings: string[]
  updated_at: string | null
}

export interface OutputStreamEvent {
  kind: JobKind
  page: number
  attempt?: number
  delta?: string
  markdown?: string
  content?: string
  cached?: boolean
}

export interface PageLayoutRegion {
  box: { x1: number; y1: number; x2: number; y2: number }
  polygon: { x: number; y: number }[]
  confidence: number
  class_id: number
  label: string
  order: number
}

export interface PageLayout {
  page: number
  display_page: number
  page_count: number
  model_id: string | null
  runtime?: string
  device?: string
  image_size?: { width: number; height: number }
  region_count?: number
  region_counts?: Record<string, number>
  regions?: PageLayoutRegion[]
  cached?: boolean
  analyzing?: boolean
  error?: string
  current_region?: { index: number; total: number; label: string }
}

export interface AuthorInfo {
  name: string
  affiliations: string[]
  emails: string[]
  identifiers: string[]
}

export interface OutlineItem {
  id: string
  title: string
  level: number
  page: number
  children: OutlineItem[]
}

export interface StructuredDocument {
  document_id: string
  source_markdown_digest: string
  stale: boolean
  title: string | null
  authors: AuthorInfo[]
  affiliations: string[]
  abstract: string | null
  keywords: string[]
  outline: OutlineItem[]
  warnings: string[]
  updated_at: string | null
}

export interface JobSnapshot {
  job: Job
  event_sequence: number
  draft: {
    pages?: Record<string, string>
    warnings?: string[]
    structured?: StructuredDocument
    current_layout?: PageLayout
  }
}
