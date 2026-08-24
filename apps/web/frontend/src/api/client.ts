import type { ApiResponse } from "./types"

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

interface ErrorEnvelope {
  error?: {
    code?: string
    message?: string
    trace_id?: string | null
  }
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly traceId: string | null

  constructor(
    message: string,
    {
      status,
      code = "request_failed",
      traceId = null,
    }: { status: number; code?: string; traceId?: string | null },
  ) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.code = code
    this.traceId = traceId
  }
}

export async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<ApiResponse<T>> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    let envelope: ErrorEnvelope = {}
    try {
      envelope = (await response.json()) as ErrorEnvelope
    } catch {
      // A non-JSON proxy/runtime error still receives a stable client error.
    }
    throw new ApiError(
      envelope.error?.message ?? `Request failed with status ${response.status}`,
      {
        status: response.status,
        code: envelope.error?.code,
        traceId: envelope.error?.trace_id,
      },
    )
  }
  return (await response.json()) as ApiResponse<T>
}
