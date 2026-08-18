const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""

export interface StreamingTranscript {
  text: string
  delta: string
  audio_seconds: number
  generated_tokens: number
  detected_language: string | null
  inference_ms: number | null
  preprocess_ms: number | null
  enhancement_inference_ms: number | null
  is_final: boolean
}

type AsrSocketEvent = StreamingTranscript & { type: string; message?: string; utterance_id?: number }

export class ChatAsrSocket {
  private utteranceActive = false
  private partialHandler: ((event: StreamingTranscript) => void) | null = null
  private errorHandler: ((error: Error) => void) | null = null
  private finalResolve: ((event: StreamingTranscript) => void) | null = null
  private finalReject: ((reason?: unknown) => void) | null = null
  private inferenceMs = 0
  private preprocessMs = 0
  private nextUtteranceId = 0
  private utteranceId: number | null = null

  private constructor(private readonly socket: WebSocket) {
    socket.addEventListener("message", (message) => this.handleMessage(message))
    socket.addEventListener("close", () => {
      const error = new Error("ASR WebSocket closed before the final transcript")
      this.errorHandler?.(error)
      this.rejectFinal(error)
    })
  }

  static connect(): Promise<ChatAsrSocket> {
    const url = new URL(`${API_BASE}/api/v1/apps/chat/asr/ws`, window.location.origin)
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
        socket.removeEventListener("open", opened)
        socket.removeEventListener("error", failed)
      }
      const opened = () => { cleanup(); resolve(new ChatAsrSocket(socket)) }
      const failed = () => { cleanup(); reject(new Error("Unable to open the ASR WebSocket")) }
      socket.addEventListener("open", opened, { once: true })
      socket.addEventListener("error", failed, { once: true })
    })
  }

  start(instanceId: string, sampleRate: number, onPartial: (event: StreamingTranscript) => void, onError: (error: Error) => void) {
    if (this.utteranceActive) throw new Error("An ASR utterance is already active")
    this.utteranceActive = true
    this.utteranceId = ++this.nextUtteranceId
    this.inferenceMs = 0
    this.preprocessMs = 0
    this.partialHandler = onPartial
    this.errorHandler = onError
    this.socket.send(JSON.stringify({ type: "start_utterance", utterance_id: this.utteranceId, instance_id: instanceId, sample_rate: sampleRate }))
  }

  send(pcm16: ArrayBuffer) {
    if (this.socket.readyState !== WebSocket.OPEN) throw new Error("ASR WebSocket is not open")
    this.socket.send(pcm16)
  }

  finish(): Promise<StreamingTranscript> {
    if (!this.utteranceActive) return Promise.reject(new Error("No ASR utterance is active"))
    return new Promise((resolve, reject) => {
      this.finalResolve = resolve
      this.finalReject = reject
      this.socket.send(JSON.stringify({ type: "finish_utterance", utterance_id: this.utteranceId }))
    })
  }

  cancel() {
    if (this.utteranceActive && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "cancel_utterance", utterance_id: this.utteranceId }))
    }
    this.rejectFinal(new DOMException("Aborted", "AbortError"))
  }

  close() { this.cancel(); this.socket.close() }

  private handleMessage(message: MessageEvent<string>) {
    const event = JSON.parse(message.data) as AsrSocketEvent
    if (event.utterance_id !== undefined && event.utterance_id !== this.utteranceId) return
    if (event.type === "partial") {
      this.inferenceMs += event.inference_ms ?? 0
      this.preprocessMs += event.preprocess_ms ?? 0
      this.partialHandler?.(event)
    }
    else if (event.type === "final") {
      event.inference_ms = this.inferenceMs + (event.inference_ms ?? 0)
      event.preprocess_ms = this.preprocessMs + (event.preprocess_ms ?? 0)
      const resolve = this.finalResolve
      this.clear()
      resolve?.(event)
    } else if (event.type === "error") {
      const error = new Error(event.message || "Streaming ASR failed")
      this.errorHandler?.(error)
      this.rejectFinal(error)
    }
  }

  private rejectFinal(reason: unknown) {
    const reject = this.finalReject
    this.clear()
    reject?.(reason)
  }

  private clear() {
    this.utteranceActive = false
    this.utteranceId = null
    this.partialHandler = null
    this.errorHandler = null
    this.finalResolve = null
    this.finalReject = null
  }
}
