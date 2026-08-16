import { ApiError, request } from "@/api/client"

import type {
  ChatStreamEvent,
  ConversationMessage,
  LoadedModels,
  Setup,
  StreamingTranscript,
  Transcript,
} from "./types"

const PREFIX = "/api/v1/games/digital-human"
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

function setupQuery(llmVariant: string | undefined, asrModelId: string | undefined): string {
  const query = new URLSearchParams()
  if (llmVariant) query.set("llm_variant", llmVariant)
  if (asrModelId) query.set("asr_model_id", asrModelId)
  const value = query.toString()
  return value ? `?${value}` : ""
}

export async function fetchSetup(llmVariant?: string, asrModelId?: string): Promise<Setup> {
  return (await request<Setup>(`${PREFIX}/setup${setupQuery(llmVariant, asrModelId)}`)).data
}

export async function loadModels(llmVariant: string, asrModelId: string): Promise<LoadedModels> {
  return (
    await request<LoadedModels>(
      `${PREFIX}/setup/load${setupQuery(llmVariant, asrModelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function loadModel(
  role: "asr" | "llm" | "tts",
  llmVariant: string,
  asrModelId: string,
): Promise<string> {
  return (
    await request<string>(
      `${PREFIX}/setup/models/${role}/load${setupQuery(llmVariant, asrModelId)}`,
      { method: "POST" },
    )
  ).data
}

export async function transcribe(
  audio: Blob,
  modelId: string,
  instanceId: string,
  vadInstanceId?: string | null,
  enhancementInstanceId?: string | null,
  signal?: AbortSignal,
): Promise<Transcript> {
  const form = new FormData()
  const extension = audio.type.includes("mp4") ? "m4a" : audio.type.includes("webm") ? "webm" : "wav"
  form.append("file", audio, `utterance.${extension}`)
  form.append("model_id", modelId)
  form.append("instance_id", instanceId)
  if (vadInstanceId) form.append("vad_instance_id", vadInstanceId)
  if (enhancementInstanceId) form.append("enhancement_instance_id", enhancementInstanceId)
  return (
    await request<Transcript>(`${PREFIX}/transcribe`, { method: "POST", body: form, signal })
  ).data
}

type AsrSocketEvent = StreamingTranscript & { type: string; message?: string }

export class DigitalHumanAsrSocket {
  private readonly socket: WebSocket
  private utteranceActive = false
  private partialHandler: ((event: StreamingTranscript) => void) | null = null
  private errorHandler: ((error: Error) => void) | null = null
  private finalResolve: ((event: StreamingTranscript) => void) | null = null
  private finalReject: ((reason?: unknown) => void) | null = null

  private constructor(socket: WebSocket) {
    this.socket = socket
    this.socket.addEventListener("message", (message) => this.handleMessage(message))
    this.socket.addEventListener("close", () => {
      const error = new Error("ASR WebSocket closed before the final transcript")
      const handler = this.errorHandler
      this.rejectFinal(error)
      handler?.(error)
    })
  }

  static connect(): Promise<DigitalHumanAsrSocket> {
    const url = new URL(`${API_BASE}${PREFIX}/asr/ws`, window.location.origin)
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
    const socket = new WebSocket(url)
    socket.binaryType = "arraybuffer"
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        cleanup()
        socket.close()
        reject(new Error("ASR WebSocket connection timed out"))
      }, 5000)
      const cleanup = () => {
        window.clearTimeout(timeout)
        socket.removeEventListener("open", handleOpen)
        socket.removeEventListener("error", handleError)
      }
      const handleOpen = () => {
        cleanup()
        resolve(new DigitalHumanAsrSocket(socket))
      }
      const handleError = () => {
        cleanup()
        reject(new Error(`Unable to open ASR WebSocket at ${url.host}`))
      }
      socket.addEventListener("open", handleOpen, { once: true })
      socket.addEventListener("error", handleError, { once: true })
    })
  }

  startUtterance(
    modelId: string,
    instanceId: string,
    sampleRate: number,
    onPartial: (event: StreamingTranscript) => void,
    onError: (error: Error) => void,
  ) {
    if (this.utteranceActive) throw new Error("An ASR utterance is already active")
    this.utteranceActive = true
    this.partialHandler = onPartial
    this.errorHandler = onError
    this.socket.send(JSON.stringify({
      type: "start_utterance",
      model_id: modelId,
      instance_id: instanceId,
      sample_rate: sampleRate,
    }))
  }

  sendAudio(pcm16: ArrayBuffer) {
    if (this.socket.readyState !== WebSocket.OPEN) throw new Error("ASR WebSocket is not open")
    this.socket.send(pcm16)
  }

  finishUtterance(): Promise<StreamingTranscript> {
    if (!this.utteranceActive || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("ASR WebSocket is not ready to finish"))
    }
    return new Promise<StreamingTranscript>((resolve, reject) => {
      this.finalResolve = resolve
      this.finalReject = reject
      try {
        this.socket.send(JSON.stringify({ type: "finish_utterance" }))
      } catch (error) {
        this.rejectFinal(error)
      }
    })
  }

  cancelUtterance() {
    if (this.utteranceActive && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "cancel_utterance" }))
    }
    this.partialHandler = null
    this.utteranceActive = false
    this.rejectFinal(new DOMException("Aborted", "AbortError"))
  }

  close() {
    this.cancelUtterance()
    this.socket.close()
  }

  private handleMessage(message: MessageEvent<string>) {
    const event = JSON.parse(message.data) as AsrSocketEvent
    if (event.type === "partial") {
      this.partialHandler?.(event)
    } else if (event.type === "final") {
      const resolve = this.finalResolve
      this.clearUtterance()
      resolve?.(event)
    } else if (event.type === "error") {
      const error = new Error(event.message || "Streaming ASR failed")
      this.errorHandler?.(error)
      this.rejectFinal(error)
    }
  }

  private rejectFinal(reason: unknown) {
    const reject = this.finalReject
    this.clearUtterance()
    reject?.(reason)
  }

  private clearUtterance() {
    this.partialHandler = null
    this.errorHandler = null
    this.utteranceActive = false
    this.finalResolve = null
    this.finalReject = null
  }
}

export async function streamReply(
  options: {
    image: Blob
    instanceId: string
    prompt: string
    history: ConversationMessage[]
  },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append("image", options.image, "webcam.jpg")
  form.append("instance_id", options.instanceId)
  form.append("prompt", options.prompt)
  form.append("history_json", JSON.stringify(options.history))
  const response = await fetch(`${API_BASE}${PREFIX}/chat/stream`, {
    method: "POST",
    body: form,
    signal,
  })
  if (!response.ok || !response.body) {
    throw new ApiError("Digital human generation failed", { status: response.status })
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
      if (data) onEvent(JSON.parse(data) as ChatStreamEvent)
    }
    if (done) break
  }
}

export async function synthesize(
  text: string,
  voice: string,
  language: string,
  instanceId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const form = new FormData()
  form.append("text", text)
  form.append("voice", voice)
  form.append("language", language)
  form.append("instance_id", instanceId)
  const response = await fetch(`${API_BASE}${PREFIX}/synthesize`, {
    method: "POST",
    body: form,
    signal,
  })
  if (!response.ok) throw new ApiError("Speech synthesis failed", { status: response.status })
  return response.blob()
}
