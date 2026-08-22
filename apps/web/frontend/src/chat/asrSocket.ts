const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ""
const HARD_BACKPRESSURE_BYTES = 4 * 1024 * 1024

export interface ChatAsrEvent {
  type: "ready" | "speech_start" | "speech_end" | "partial" | "transcript" | "stopped" | "error"
  message?: string
  utterance_id?: number
  text?: string
  preprocess_ms?: number | null
  inference_ms?: number | null
  vad_inference_ms?: number | null
}

interface ChatAsrConfig {
  instanceId: string
  sampleRate: number
  streamingChunkSeconds: number
}

export class ChatAsrSocket {
  private closed = false

  private constructor(
    private readonly socket: WebSocket,
    private readonly onEvent: (event: ChatAsrEvent) => void,
    private readonly onError: (error: Error) => void,
  ) {
    socket.addEventListener("message", (message) => this.handleMessage(message))
    socket.addEventListener("close", () => {
      if (!this.closed) onError(new Error("Chat ASR WebSocket disconnected"))
    })
  }

  static connect(
    config: ChatAsrConfig,
    onEvent: (event: ChatAsrEvent) => void,
    onError: (error: Error) => void,
  ): Promise<ChatAsrSocket> {
    const url = new URL(`${API_BASE}/api/v1/apps/chat/asr/ws`, window.location.origin)
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
    const socket = new WebSocket(url)
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        socket.close()
        reject(new Error("Chat ASR connection timed out"))
      }, 5000)
      const opened = () => socket.send(JSON.stringify({
        type: "start",
        instance_id: config.instanceId,
        sample_rate: config.sampleRate,
        streaming_chunk_seconds: config.streamingChunkSeconds,
      }))
      const ready = (message: MessageEvent<string | ArrayBuffer>) => {
        if (typeof message.data !== "string") return
        const event = JSON.parse(message.data) as ChatAsrEvent
        if (event.type !== "ready" && event.type !== "error") return
        window.clearTimeout(timeout)
        socket.removeEventListener("message", ready)
        if (event.type === "error") {
          socket.close()
          reject(new Error(event.message || "Unable to start Chat ASR"))
          return
        }
        const connection = new ChatAsrSocket(socket, onEvent, onError)
        onEvent(event)
        resolve(connection)
      }
      socket.addEventListener("open", opened, { once: true })
      socket.addEventListener("message", ready)
      socket.addEventListener(
        "error",
        () => reject(new Error("Unable to open the Chat ASR WebSocket")),
        { once: true },
      )
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
      const value = Math.max(-1, Math.min(1, samples[index]!))
      view.setInt16(index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true)
    }
    this.socket.send(pcm)
  }

  close() {
    this.closed = true
    this.socket.close()
  }

  private handleMessage(message: MessageEvent<string | ArrayBuffer>) {
    if (typeof message.data !== "string") return
    const event = JSON.parse(message.data) as ChatAsrEvent
    if (event.type === "error") {
      this.onError(new Error(event.message || "Chat ASR failed"))
      this.close()
    } else this.onEvent(event)
  }
}
