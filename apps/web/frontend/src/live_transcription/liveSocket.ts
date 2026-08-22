export interface LiveStartConfig {
  modelId: string
  instanceId: string
  sampleRate: number
  useVad: boolean
  vadInstanceId: string | null
  useEnhancement: boolean
  enhancementInstanceId: string | null
  vadThreshold: number
  sensitivity: number
  streamingChunkSeconds: number
}

export interface LiveEvent {
  type: "ready" | "speech_start" | "speech_end" | "partial" | "transcript" | "stopped" | "error"
  message?: string
  utterance_id?: number
  text?: string
  started_at_ms?: number
  ended_at_ms?: number
  duration_seconds?: number
  generated_tokens?: number
  inference_ms?: number | null
  vad_inference_ms?: number | null
  enhancement_inference_ms?: number | null
  languages?: string[]
  emotion?: string | null
  events?: string[]
  speech_duration_seconds?: number | null
}

const PREFIX = import.meta.env.VITE_API_BASE_URL ?? ""
const HARD_BACKPRESSURE_BYTES = 4 * 1024 * 1024

export class LiveTranscriptionSocket {
  private closed = false

  private constructor(
    private readonly socket: WebSocket,
    private readonly onEvent: (event: LiveEvent) => void,
    private readonly onAudio: (utteranceId: number, audio: Blob) => void,
    private readonly onError: (error: Error) => void,
  ) {
    socket.addEventListener("message", (message) => this.handleMessage(message))
    socket.addEventListener("close", () => {
      if (!this.closed) onError(new Error("Live transcription WebSocket disconnected"))
    })
  }

  static connect(config: LiveStartConfig, onEvent: (event: LiveEvent) => void,
    onAudio: (utteranceId: number, audio: Blob) => void,
    onError: (error: Error) => void): Promise<LiveTranscriptionSocket> {
    const url = new URL(`${PREFIX}/api/v1/apps/live-transcription/live/ws`, window.location.origin)
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
    const socket = new WebSocket(url)
    socket.binaryType = "arraybuffer"
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => { socket.close(); reject(new Error("Connection timed out")) }, 5000)
      const opened = () => socket.send(JSON.stringify({
        type: "start", model_id: config.modelId, instance_id: config.instanceId,
        sample_rate: config.sampleRate, use_vad: config.useVad,
        vad_instance_id: config.vadInstanceId, use_enhancement: config.useEnhancement,
        enhancement_instance_id: config.enhancementInstanceId,
        vad_threshold: config.vadThreshold, sensitivity: config.sensitivity,
        streaming_chunk_seconds: config.streamingChunkSeconds,
      }))
      const ready = (message: MessageEvent<string | ArrayBuffer>) => {
        if (typeof message.data !== "string") return
        const event = JSON.parse(message.data) as LiveEvent
        if (event.type !== "ready" && event.type !== "error") return
        window.clearTimeout(timeout)
        socket.removeEventListener("message", ready)
        if (event.type === "error") { socket.close(); reject(new Error(event.message || "Unable to start")); return }
        const live = new LiveTranscriptionSocket(socket, onEvent, onAudio, onError)
        onEvent(event)
        resolve(live)
      }
      socket.addEventListener("open", opened, { once: true })
      socket.addEventListener("message", ready)
      socket.addEventListener("error", () => reject(new Error("Unable to open WebSocket")), { once: true })
    })
  }

  send(samples: Float32Array) {
    if (this.closed || this.socket.readyState !== WebSocket.OPEN) return
    if (this.socket.bufferedAmount > HARD_BACKPRESSURE_BYTES) {
      this.onError(new Error("Audio upload cannot keep up with recording"))
      this.close()
      return
    }
    const pcm = new ArrayBuffer(samples.length * 2)
    const view = new DataView(pcm)
    for (let index = 0; index < samples.length; index += 1) {
      const value = Math.max(-1, Math.min(1, samples[index]))
      view.setInt16(index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true)
    }
    this.socket.send(pcm)
  }

  stop() {
    if (!this.closed && this.socket.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "stop" }))
  }

  close() {
    this.closed = true
    this.socket.close()
  }

  private handleMessage(message: MessageEvent<string | ArrayBuffer>) {
    if (typeof message.data === "string") {
      const event = JSON.parse(message.data) as LiveEvent
      if (event.type === "error") this.onError(new Error(event.message || "Live transcription failed"))
      else this.onEvent(event)
      return
    }
    const bytes = new Uint8Array(message.data)
    if (bytes.byteLength < 8) return
    const header = new DataView(bytes.buffer, bytes.byteOffset, 8)
    const utteranceId = Number(header.getBigUint64(0, false))
    this.onAudio(utteranceId, new Blob([bytes.slice(8)], { type: "audio/wav" }))
  }
}
