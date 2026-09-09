import { API_BASE, request } from "@/api/client"
import type {
  DocumentCreated,
  DocumentOutput,
  DocumentRecord,
  Job,
  JobKind,
  JobSnapshot,
  ParserModel,
} from "./types"

const PREFIX = "/api/v1/apps/document-parser"

export async function uploadDocument(files: File[]): Promise<DocumentCreated> {
  const body = new FormData()
  files.forEach((file) => body.append("files", file, file.name))
  const response = await fetch(`${API_BASE}${PREFIX}/documents`, { method: "POST", body })
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as {
      detail?: string
      error?: { message?: string }
    }
    throw new Error(payload.error?.message ?? payload.detail ?? `Upload failed (${response.status})`)
  }
  return ((await response.json()) as { data: DocumentCreated }).data
}

export async function fetchDocuments(): Promise<DocumentRecord[]> {
  return (await request<DocumentRecord[]>(`${PREFIX}/documents`)).data
}

export async function fetchDocument(id: string): Promise<DocumentRecord> {
  return (await request<DocumentRecord>(`${PREFIX}/documents/${id}`)).data
}

export async function fetchParserModels(): Promise<ParserModel[]> {
  return (await request<ParserModel[]>(`${PREFIX}/models`)).data
}

export async function fetchLayoutModels(): Promise<ParserModel[]> {
  return (await request<ParserModel[]>(`${PREFIX}/layout-models`)).data
}

export async function createJob(documentId: string, kind: JobKind, modelId?: string, layoutModelId?: string): Promise<Job> {
  return (
    await request<Job>(`${PREFIX}/documents/${documentId}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind,
        model_id: kind === "markdown" ? modelId : null,
        layout_model_id: kind === "markdown" ? layoutModelId || null : null,
      }),
    })
  ).data
}

export async function fetchJobs(activeOnly = false): Promise<Job[]> {
  return (
    await request<Job[]>(`${PREFIX}/jobs${activeOnly ? "?active_only=true" : ""}`)
  ).data
}

export async function fetchJobSnapshot(id: string): Promise<JobSnapshot> {
  return (await request<JobSnapshot>(`${PREFIX}/jobs/${id}/snapshot`)).data
}

export async function cancelJob(id: string): Promise<Job> {
  return (await request<Job>(`${PREFIX}/jobs/${id}/cancel`, { method: "POST" })).data
}

export async function fetchOutput(id: string, kind: JobKind): Promise<DocumentOutput> {
  return (await request<DocumentOutput>(`${PREFIX}/documents/${id}/outputs/${kind}`)).data
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}${PREFIX}/documents/${id}`, { method: "DELETE" })
  if (!response.ok) throw new Error(`Delete failed (${response.status})`)
}

export const markdownUrl = (id: string) => `${API_BASE}${PREFIX}/documents/${id}/result.md`
export const pageImageUrl = (id: string, page: number) =>
  `${API_BASE}${PREFIX}/documents/${id}/pages/${page}/image`
export const jobEventsUrl = (id: string, after = 0) =>
  `${API_BASE}${PREFIX}/jobs/${id}/events?after=${after}`
