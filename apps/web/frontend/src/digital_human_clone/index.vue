<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue"

import {
  fetchSetup,
  loadModel as loadDigitalHumanModel,
  loadModels as loadDigitalHumanModels,
  streamReply,
  synthesize,
  transcribe,
} from "./api"
import type { ConversationMessage, LoadedModels, ModelState } from "./types"
import { encodeWave } from "./wav"
import { errorMessage, unloadModel } from "@/modelLifecycle"

const KOKORO_VOICES = [
  "af_alloy",
  "af_aoede",
  "af_heart",
  "af_bella",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  "bf_alice",
  "bf_emma",
  "bf_isabella",
  "bf_lily",
  "bm_daniel",
  "bm_fable",
  "bm_george",
  "bm_lewis",
  "ef_dora",
  "em_alex",
  "em_santa",
  "ff_siwis",
  "hf_alpha",
  "hf_beta",
  "hm_omega",
  "hm_psi",
  "if_sara",
  "im_nicola",
  "jf_alpha",
  "jf_gongitsune",
  "jf_nezumi",
  "jf_tebukuro",
  "jm_kumo",
  "pf_dora",
  "pm_alex",
  "pm_santa",
  "zf_xiaobei",
  "zf_xiaoni",
  "zf_xiaoxiao",
  "zf_xiaoyi",
  "zm_yunjian",
  "zm_yunxi",
  "zm_yunxia",
  "zm_yunyang",
]
const KOKORO_LANGUAGES = [
  { code: "a", name: "American English" },
  { code: "b", name: "British English" },
  { code: "e", name: "Spanish" },
  { code: "f", name: "French" },
  { code: "h", name: "Hindi" },
  { code: "i", name: "Italian" },
  { code: "p", name: "Portuguese" },
  { code: "j", name: "Japanese" },
  { code: "z", name: "Chinese" },
]
const SILENCE_SECONDS = 0.7
const MIN_SPEECH_SECONDS = 0.35
const MAX_SPEECH_SECONDS = 20
const MAX_HISTORY_MESSAGES = 15

interface TtsTiming {
  index: number
  text: string
  durationMs: number
}

interface TurnTiming {
  asrTotalMs: number | null
  vadMs: number | null
  enhancementMs: number | null
  asrInferenceMs: number | null
  llmMs: number | null
  ttsSegments: TtsTiming[]
  ttsWallMs: number | null
  totalStageMs: number | null
  endToEndMs: number | null
}

const models = ref<ModelState[]>([])
const llmVariants = ref<Array<{ name: string; display_name: string; description: string }>>([])
const llmVariant = ref("")
const loaded = ref<LoadedModels | null>(null)
const loadingModels = ref(false)
const loadingRole = ref<ModelState["role"] | null>(null)
const voiceId = ref("af_heart")
const language = ref("a")
const active = ref(false)
const speaking = ref(false)
const inputLevel = ref(0)
const phase = ref("Waiting for setup")
const userSubtitle = ref("")
const assistantSubtitle = ref("")
const history = ref<ConversationMessage[]>([])
const error = ref("")
const lifecycleMessages = ref<Record<string, { type: "success" | "error"; text: string }>>({})
const turnTiming = ref<TurnTiming | null>(null)
const video = ref<HTMLVideoElement | null>(null)
const subtitleEnd = ref<HTMLElement | null>(null)

let mediaStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let sourceNode: MediaStreamAudioSourceNode | null = null
let analyserNode: AnalyserNode | null = null
let silentGain: GainNode | null = null
let vadFrame = 0
let recorder: MediaRecorder | null = null
let recorderChunks: Blob[] = []
let speechStartedAt = 0
let silenceStartedAt = 0
let responseRequest: AbortController | null = null
const asrRequests = new Map<number, AbortController>()
const playbackSources = new Map<
  AudioBufferSourceNode,
  { gain: GainNode; startsAt: number; endsAt: number }
>()
let playbackScheduledUntil = 0
let turn = 0
let utteranceSequence = 0
let latestUtterance = 0

const modelsReady = computed(() => Boolean(loaded.value))
const resourcesReady = computed(() => models.value.length === 3 && models.value.every((item) => item.resources_ready))
const availableVoices = computed(() =>
  KOKORO_VOICES.filter((voice) => voice.startsWith(language.value)),
)
const canStart = computed(() => resourcesReady.value && Boolean(voiceId.value) && !active.value && !loadingModels.value)

watch(language, () => {
  if (!availableVoices.value.includes(voiceId.value)) {
    voiceId.value = availableVoices.value[0] ?? ""
  }
})

function voiceLabel(voice: string): string {
  const gender = voice[1] === "f" ? "Female" : voice[1] === "m" ? "Male" : "Voice"
  const name = voice.slice(3).replaceAll("_", " ")
  return `${name} · ${gender}`
}

function formatTime(value: number | null): string {
  if (value === null) return "—"
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

async function refreshSetup() {
  try {
    const setup = await fetchSetup(llmVariant.value || undefined)
    models.value = setup.models
    llmVariants.value = setup.llm_variants
    llmVariant.value = setup.selected_llm_variant
    const byRole = Object.fromEntries(setup.models.map((model) => [model.role, model]))
    if (byRole.asr?.ready_instance_id && byRole.llm?.ready_instance_id && byRole.tts?.ready_instance_id) {
      loaded.value = {
        asr_instance_id: byRole.asr.ready_instance_id,
        llm_instance_id: byRole.llm.ready_instance_id,
        tts_instance_id: byRole.tts.ready_instance_id,
        vad_instance_id: setup.vad_instance_id,
        enhancement_instance_id: setup.enhancement_instance_id,
      }
      phase.value = "Models ready"
    } else {
      loaded.value = null
      phase.value = "Models unavailable"
    }
  } catch (caught) {
    showError(caught, "Failed to read model status")
  }
}

async function loadAllModels(): Promise<boolean> {
  if (!llmVariant.value || loadingModels.value) return false
  loadingModels.value = true
  error.value = ""
  lifecycleMessages.value = {}
  try {
    loaded.value = await loadDigitalHumanModels(llmVariant.value)
    phase.value = "Models ready"
    await refreshSetup()
    return true
  } catch (caught) {
    lifecycleMessages.value = { all: { type: "error", text: errorMessage(caught, "模型加载失败") } }
    return false
  } finally {
    loadingModels.value = false
  }
}

async function toggleModel(role: ModelState["role"]) {
  if (!llmVariant.value || loadingRole.value) return
  const selected = models.value.find((item) => item.role === role)
  loadingRole.value = role
  error.value = ""
  delete lifecycleMessages.value[role]
  try {
    if (selected?.ready_instance_id) {
      await unloadModel(selected.ready_instance_id)
      lifecycleMessages.value[role] = { type: "success", text: "卸载成功" }
    } else {
      await loadDigitalHumanModel(role, llmVariant.value)
    }
    await refreshSetup()
  } catch (caught) {
    lifecycleMessages.value[role] = { type: "error", text: errorMessage(caught, "模型操作失败") }
  } finally {
    loadingRole.value = null
  }
}

async function startConversation() {
  if (!canStart.value) return
  error.value = ""
  try {
    if (!modelsReady.value && !await loadAllModels()) return
    stopAudioGraph()
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    })
    if (video.value) {
      video.value.srcObject = mediaStream
      await video.value.play()
    }
    active.value = true
    await createAudioCapture()
    phase.value = "Listening"
  } catch (caught) {
    showError(caught, "Camera or microphone access failed")
    stopConversation()
  }
}

function sampleVoiceActivity() {
  if (!active.value || !analyserNode) return
  const samples = new Float32Array(analyserNode.fftSize)
  analyserNode.getFloatTimeDomainData(samples)
  const rms = updateLevel(samples)
  const threshold = isPlaybackActive() ? 0.035 : 0.018
  const voiced = rms >= threshold
  if (!speaking.value) {
    if (voiced) beginUtterance()
  } else {
    if (voiced) silenceStartedAt = 0
    else if (!silenceStartedAt) silenceStartedAt = performance.now()
    const duration = (performance.now() - speechStartedAt) / 1000
    const silence = silenceStartedAt ? (performance.now() - silenceStartedAt) / 1000 : 0
    if (silence >= SILENCE_SECONDS || duration >= MAX_SPEECH_SECONDS) finishUtterance()
  }
  vadFrame = requestAnimationFrame(sampleVoiceActivity)
}

function beginUtterance() {
  if (!mediaStream || recorder) return
  const audioTracks = mediaStream.getAudioTracks()
  if (!audioTracks.length) throw new Error("The webcam session has no microphone track")
  recorderChunks = []
  try {
    recorder = new MediaRecorder(new MediaStream(audioTracks), mediaRecorderOptions())
  } catch (caught) {
    stopConversation()
    showError(caught, "This browser cannot record microphone audio")
    return
  }
  recorder.ondataavailable = (event) => {
    if (event.data.size) recorderChunks.push(event.data)
  }
  recorder.onstop = () => void submitRecordedUtterance()
  recorder.start(100)
  speechStartedAt = performance.now()
  silenceStartedAt = 0
  speaking.value = true
  if (!isPlaybackActive() && !responseRequest) phase.value = "You are speaking"
}

function finishUtterance() {
  if (!recorder || recorder.state === "inactive") return
  speaking.value = false
  recorder.stop()
}

async function submitRecordedUtterance() {
  const duration = (performance.now() - speechStartedAt) / 1000
  const type = recorder?.mimeType || recorderChunks[0]?.type || "audio/webm"
  const encodedAudio = new Blob(recorderChunks, { type })
  recorder = null
  recorderChunks = []
  speechStartedAt = 0
  silenceStartedAt = 0
  if (!active.value || duration < MIN_SPEECH_SECONDS || !encodedAudio.size) {
    if (active.value && !isPlaybackActive() && !responseRequest) phase.value = "Listening"
    return
  }
  if (!audioContext) return
  let audio: Blob
  try {
    const decoded = await audioContext.decodeAudioData(await encodedAudio.arrayBuffer())
    audio = encodeWave([new Float32Array(decoded.getChannelData(0))], decoded.sampleRate)
  } catch (caught) {
    showError(caught, "Unable to decode microphone audio")
    return
  }
  const utteranceId = ++utteranceSequence
  latestUtterance = utteranceId
  void recognizeUtterance(audio, utteranceId)
}

async function recognizeUtterance(audio: Blob, utteranceId: number) {
  if (!loaded.value || !voiceId.value) return
  const controller = new AbortController()
  let responseController: AbortController | null = null
  const turnStartedAt = performance.now()
  turnTiming.value = {
    asrTotalMs: null,
    vadMs: null,
    enhancementMs: null,
    asrInferenceMs: null,
    llmMs: null,
    ttsSegments: [],
    ttsWallMs: null,
    totalStageMs: null,
    endToEndMs: null,
  }
  asrRequests.set(utteranceId, controller)
  try {
    const asrStartedAt = performance.now()
    const transcript = await transcribe(
      audio,
      loaded.value.asr_instance_id,
      loaded.value.vad_instance_id,
      loaded.value.enhancement_instance_id,
      controller.signal,
    )
    const asrTotalMs = performance.now() - asrStartedAt
    if (utteranceId === latestUtterance && turnTiming.value) {
      turnTiming.value.asrTotalMs = asrTotalMs
      turnTiming.value.vadMs = transcript.vad_inference_ms
      turnTiming.value.enhancementMs = transcript.enhancement_inference_ms
      turnTiming.value.asrInferenceMs = transcript.inference_ms
    }
    if (!active.value || utteranceId !== latestUtterance || !isValidSpeechText(transcript.text)) {
      return
    }
    interruptResponse()
    const currentTurn = turn
    userSubtitle.value = transcript.text
    assistantSubtitle.value = ""
    phase.value = "Your digital human is observing and thinking"
    const frame = await captureFrame()
    const priorHistory = history.value.slice(-MAX_HISTORY_MESSAGES)
    responseController = new AbortController()
    responseRequest = responseController
    let answer = ""
    let speechBuffer = ""
    let speechError: unknown = null
    let scheduleQueue: Promise<void> = Promise.resolve()
    const playbackCompletions: Promise<void>[] = []
    let firstTtsStartedAt: number | null = null
    let lastTtsFinishedAt: number | null = null
    let ttsSegmentSequence = 0
    const queueSpeech = (segment: string) => {
      const text = segment.trim()
      if (!text) return
      if (speechError || currentTurn !== turn || responseController?.signal.aborted) return
      phase.value = "Generating speech"
      const segmentIndex = ++ttsSegmentSequence
      const ttsStartedAt = performance.now()
      firstTtsStartedAt ??= ttsStartedAt
      const preparedAudio = synthesize(
        text,
        voiceId.value,
        language.value,
        loaded.value!.tts_instance_id,
        responseController!.signal,
      ).then((blob) => {
        const finishedAt = performance.now()
        lastTtsFinishedAt = Math.max(lastTtsFinishedAt ?? 0, finishedAt)
        if (currentTurn === turn && turnTiming.value) {
          turnTiming.value.ttsSegments.push({
            index: segmentIndex,
            text,
            durationMs: finishedAt - ttsStartedAt,
          })
        }
        return prepareSpeech(blob)
      })
      scheduleQueue = scheduleQueue.then(async () => {
        const speech = await preparedAudio
        if (currentTurn !== turn || responseController?.signal.aborted) return
        playbackCompletions.push(scheduleSpeech(speech))
      }).catch((caught) => {
        speechError = caught
      })
    }
    const llmStartedAt = performance.now()
    await streamReply(
      {
        image: frame,
        instanceId: loaded.value.llm_instance_id,
        prompt: transcript.text,
        history: priorHistory,
      },
      (event) => {
        if (currentTurn !== turn || !event.delta) return
        answer += event.delta
        assistantSubtitle.value = answer
        speechBuffer += event.delta
        const parsed = splitSpeechBuffer(speechBuffer)
        speechBuffer = parsed.remainder
        parsed.segments.forEach(queueSpeech)
      },
      responseController.signal,
    )
    const llmMs = performance.now() - llmStartedAt
    if (currentTurn === turn && turnTiming.value) turnTiming.value.llmMs = llmMs
    if (currentTurn !== turn || !answer.trim()) return
    queueSpeech(speechBuffer)
    speechBuffer = ""
    const completedAnswer = answer.trim()
    history.value.push(
      { role: "user", content: transcript.text },
      { role: "assistant", content: completedAnswer },
    )
    history.value = history.value.slice(-MAX_HISTORY_MESSAGES)
    await scheduleQueue
    if (currentTurn === turn && turnTiming.value) {
      const ttsWallMs = firstTtsStartedAt !== null && lastTtsFinishedAt !== null
        ? lastTtsFinishedAt - firstTtsStartedAt
        : 0
      turnTiming.value.ttsWallMs = ttsWallMs
      turnTiming.value.totalStageMs =
        (turnTiming.value.asrTotalMs ?? 0) + llmMs + ttsWallMs
      turnTiming.value.endToEndMs = performance.now() - turnStartedAt
    }
    await Promise.all(playbackCompletions)
    if (speechError) throw speechError
  } catch (caught) {
    if (!isAbort(caught) && active.value && utteranceId === latestUtterance) {
      showError(caught, "This conversation turn failed")
    }
  } finally {
    asrRequests.delete(utteranceId)
    if (responseRequest === responseController) responseRequest = null
    if (
      active.value
      && utteranceId === latestUtterance
      && !isPlaybackActive()
      && !responseRequest
    ) {
      phase.value = "Listening"
    }
    void nextTick(() => subtitleEnd.value?.scrollIntoView({ behavior: "smooth" }))
  }
}

async function captureFrame(): Promise<Blob> {
  const source = video.value
  if (!source || !source.videoWidth) throw new Error("The camera image is not ready")
  const scale = Math.min(1, 768 / source.videoWidth)
  const canvas = document.createElement("canvas")
  canvas.width = Math.round(source.videoWidth * scale)
  canvas.height = Math.round(source.videoHeight * scale)
  const context = canvas.getContext("2d")
  if (!context) throw new Error("Unable to read the camera image")
  context.translate(canvas.width, 0)
  context.scale(-1, 1)
  context.drawImage(source, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Camera capture failed")), "image/jpeg", 0.72),
  )
}

interface PreparedSpeech {
  buffer: AudioBuffer
  duration: number
  offset: number
}

async function prepareSpeech(blob: Blob): Promise<PreparedSpeech> {
  if (!audioContext) throw new Error("The audio context is unavailable")
  const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer())
  const { offset, duration } = audibleRange(buffer)
  return { buffer, offset, duration }
}

function scheduleSpeech(speech: PreparedSpeech): Promise<void> {
  if (!audioContext) return Promise.reject(new Error("The audio context is unavailable"))
  const context = audioContext
  const crossfade = 0.008
  const now = context.currentTime
  const continuous = playbackScheduledUntil > now + 0.01
  const startsAt = continuous
    ? Math.max(now + 0.005, playbackScheduledUntil - crossfade)
    : now + 0.015
  const endsAt = startsAt + speech.duration
  const source = context.createBufferSource()
  const gain = context.createGain()
  source.buffer = speech.buffer
  source.connect(gain)
  gain.connect(context.destination)
  gain.gain.setValueAtTime(0, startsAt)
  gain.gain.linearRampToValueAtTime(1, startsAt + Math.min(crossfade, speech.duration / 3))
  gain.gain.setValueAtTime(1, Math.max(startsAt, endsAt - crossfade))
  gain.gain.linearRampToValueAtTime(0, endsAt)
  playbackScheduledUntil = endsAt
  playbackSources.set(source, { gain, startsAt, endsAt })
  phase.value = "Your digital human is speaking. You can interrupt at any time"
  return new Promise<void>((resolve) => {
    source.onended = () => {
      source.disconnect()
      gain.disconnect()
      playbackSources.delete(source)
      resolve()
    }
    source.start(startsAt, speech.offset, speech.duration)
  })
}

function audibleRange(buffer: AudioBuffer): { offset: number; duration: number } {
  const threshold = 0.0015
  let first = buffer.length
  let last = 0
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const samples = buffer.getChannelData(channel)
    let start = 0
    while (start < samples.length && Math.abs(samples[start] ?? 0) < threshold) start += 1
    let end = samples.length - 1
    while (end > start && Math.abs(samples[end] ?? 0) < threshold) end -= 1
    first = Math.min(first, start)
    last = Math.max(last, end)
  }
  if (first >= buffer.length) return { offset: 0, duration: buffer.duration }
  const paddingBefore = Math.round(buffer.sampleRate * 0.01)
  const paddingAfter = Math.round(buffer.sampleRate * 0.035)
  const startFrame = Math.max(0, first - paddingBefore)
  const endFrame = Math.min(buffer.length, last + paddingAfter + 1)
  return {
    offset: startFrame / buffer.sampleRate,
    duration: Math.max(1, endFrame - startFrame) / buffer.sampleRate,
  }
}

function splitSpeechBuffer(value: string): { segments: string[]; remainder: string } {
  const segments: string[] = []
  const boundary = /(?:[.!?;。！？；…]+["'”’）)\]}》】]*|\n+)/gu
  let start = 0
  for (const match of value.matchAll(boundary)) {
    const end = (match.index ?? 0) + match[0].length
    const segment = value.slice(start, end).trim()
    if (segment) segments.push(segment)
    start = end
  }
  return { segments, remainder: value.slice(start) }
}

function interruptResponse() {
  turn += 1
  responseRequest?.abort()
  responseRequest = null
  stopPlayback()
  assistantSubtitle.value = ""
}

function stopPlayback() {
  for (const [source, item] of playbackSources) {
    try {
      source.stop()
    } catch {
      source.disconnect()
      item.gain.disconnect()
      playbackSources.delete(source)
    }
  }
  playbackScheduledUntil = 0
}

function isPlaybackActive(): boolean {
  if (!audioContext) return false
  const now = audioContext.currentTime
  return Array.from(playbackSources.values()).some(
    (item) => item.startsAt <= now + 0.02 && item.endsAt > now,
  )
}

function stopConversation() {
  interruptResponse()
  for (const request of asrRequests.values()) request.abort()
  asrRequests.clear()
  latestUtterance = ++utteranceSequence
  active.value = false
  speaking.value = false
  if (video.value) video.value.srcObject = null
  stopAudioGraph()
  phase.value = modelsReady.value ? "Conversation paused" : "Waiting for setup"
}

async function createAudioCapture() {
  if (!mediaStream) throw new Error("Media stream is unavailable")
  if (!mediaStream.getAudioTracks().length) throw new Error("Microphone access was not granted")
  audioContext = new AudioContext({ latencyHint: "interactive" })
  await audioContext.resume()
  sourceNode = audioContext.createMediaStreamSource(new MediaStream(mediaStream.getAudioTracks()))
  analyserNode = audioContext.createAnalyser()
  analyserNode.fftSize = 2048
  analyserNode.smoothingTimeConstant = 0.2
  silentGain = audioContext.createGain()
  silentGain.gain.value = 0.00001
  sourceNode.connect(analyserNode)
  analyserNode.connect(silentGain)
  silentGain.connect(audioContext.destination)
  vadFrame = requestAnimationFrame(sampleVoiceActivity)
}

function stopAudioGraph() {
  if (vadFrame) cancelAnimationFrame(vadFrame)
  vadFrame = 0
  if (recorder && recorder.state !== "inactive") {
    recorder.ondataavailable = null
    recorder.onstop = null
    recorder.stop()
  }
  sourceNode?.disconnect()
  analyserNode?.disconnect()
  silentGain?.disconnect()
  mediaStream?.getTracks().forEach((track) => track.stop())
  void audioContext?.close()
  mediaStream = null
  audioContext = null
  sourceNode = null
  analyserNode = null
  silentGain = null
  recorder = null
  recorderChunks = []
  speechStartedAt = 0
  silenceStartedAt = 0
  inputLevel.value = 0
}

function mediaRecorderOptions(): MediaRecorderOptions | undefined {
  for (const mimeType of ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"]) {
    if (MediaRecorder.isTypeSupported(mimeType)) return { mimeType }
  }
  return undefined
}

function updateLevel(chunk: Float32Array): number {
  let energy = 0
  for (const sample of chunk) energy += sample * sample
  const rms = Math.sqrt(energy / chunk.length)
  inputLevel.value = Math.min(1, rms / 0.1)
  return rms
}

function showError(caught: unknown, fallback: string) {
  error.value = caught instanceof Error ? caught.message : fallback
  phase.value = fallback
}

function isAbort(caught: unknown): boolean {
  return caught instanceof DOMException && caught.name === "AbortError"
}

function isValidSpeechText(text: string): boolean {
  return /[\p{L}\p{N}]/u.test(text.trim())
}

onMounted(refreshSetup)
onBeforeUnmount(() => {
  stopConversation()
})
</script>

<template>
  <div class="page inner-page digital-human-page">
    <header class="digital-human-header">
      <div class="hero-title">
        <RouterLink class="back-link" to="/games">← Neural Games</RouterLink>
        <p class="kicker">LOCAL MULTIMODAL DIGITAL HUMAN</p>
        <h1>Digital <em>Human</em></h1>
        <div class="phase-pill"><i :class="{ live: active }"></i>{{ phase }}</div>
      </div>
      <section class="setup-panel">
        <div class="setup-block model-block">
          <div class="step-heading"><span>01</span><h2>Local models</h2></div>
          <div class="model-stack">
            <div v-for="model in models" :key="model.role" class="mini-model">
              <b>{{ model.role.toUpperCase() }}</b>
              <select
                v-if="model.role === 'llm'"
                v-model="llmVariant"
                class="mini-model-select"
                aria-label="Language model variant"
                :disabled="active"
                @change="refreshSetup"
              >
                <option v-for="item in llmVariants" :key="item.name" :value="item.name">
                  {{ item.display_name }}
                </option>
              </select>
              <span v-else>{{ model.model_id.split('/').at(-1) }}</span>
              <span class="runtime-badge">{{ model.runtime }}</span>
              <i :class="{ ready: model.ready_instance_id }">{{ model.ready_instance_id ? "LOADED" : model.resources_ready ? "LOCAL" : "NEEDED" }}</i>
              <button
                type="button"
                :disabled="active || (!model.ready_instance_id && !model.resources_ready) || Boolean(loadingRole) || loadingModels"
                @click="toggleModel(model.role)"
              >
                {{ loadingRole === model.role ? "Wait…" : model.ready_instance_id ? "Unload" : "Load" }}
              </button>
              <small v-if="lifecycleMessages[model.role]" :class="`lifecycle-${lifecycleMessages[model.role]?.type}`">{{ lifecycleMessages[model.role]?.text }}</small>
            </div>
          </div>
          <button class="action-button" type="button" :disabled="active || loadingModels || modelsReady" @click="loadAllModels">
            {{ loadingModels ? "Loading all models…" : modelsReady ? "All models loaded" : "Load all models" }}
          </button>
          <small v-if="lifecycleMessages.all" :class="`lifecycle-${lifecycleMessages.all?.type}`">{{ lifecycleMessages.all?.text }}</small>
          <p v-if="!modelsReady" class="error-copy">Models must be downloaded in Models before they can be loaded here.</p>
        </div>
        <div class="setup-block voice-block">
          <div class="step-heading"><span>02</span><h2>Kokoro speech</h2></div>
          <div class="voice-options">
            <label class="reference-copy">
              <span>LANGUAGE</span>
              <select v-model="language" :disabled="active">
                <option v-for="item in KOKORO_LANGUAGES" :key="item.code" :value="item.code">
                  {{ item.name }}
                </option>
              </select>
            </label>
            <label class="reference-copy">
              <span>ROLE / VOICE</span>
              <select v-model="voiceId" :disabled="active">
                <option v-for="item in availableVoices" :key="item" :value="item">
                  {{ voiceLabel(item) }}
                </option>
              </select>
            </label>
          </div>
          <small class="kokoro-note">82M · {{ models.find((item) => item.role === 'tts')?.runtime ?? 'runtime pending' }} · 24 kHz</small>
        </div>
      </section>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>

    <main class="digital-human-grid">
      <section class="stage-panel">
        <div class="video-stage" :class="{ active }">
          <video ref="video" muted playsinline></video>
          <div v-if="!active" class="camera-placeholder">
            <span>◉</span><p>点击 Start conversation；<br />未加载的模型会自动加载</p>
          </div>
          <div class="scanline"></div>
          <div class="live-label"><i></i>{{ active ? "LIVE" : "STANDBY" }}</div>
          <div class="subtitles">
            <p v-if="userSubtitle"><span>YOU</span>{{ userSubtitle }}</p>
            <p v-if="assistantSubtitle" class="digital-human-line"><span>DIGITAL HUMAN</span>{{ assistantSubtitle }}</p>
            <i ref="subtitleEnd"></i>
          </div>
        </div>
        <section v-if="turnTiming" class="result-panel" aria-label="Turn timing results">
          <div class="result-heading">
            <span>RESULT</span>
            <b>Turn generation timing</b>
          </div>
          <div class="timing-stage timing-asr">
            <strong>YOU · ASR</strong>
            <b>{{ formatTime(turnTiming.asrTotalMs) }}</b>
            <small>VAD {{ formatTime(turnTiming.vadMs) }}</small>
            <small>Enhance {{ formatTime(turnTiming.enhancementMs) }}</small>
            <small>ASR {{ formatTime(turnTiming.asrInferenceMs) }}</small>
          </div>
          <div class="timing-stage">
            <strong>LLM</strong>
            <b>{{ formatTime(turnTiming.llmMs) }}</b>
            <small>stream generation</small>
          </div>
          <div class="timing-stage timing-tts">
            <strong>TTS · PARALLEL</strong>
            <b>{{ formatTime(turnTiming.ttsWallMs) }}</b>
            <div class="tts-timings">
              <small v-if="!turnTiming.ttsSegments.length">waiting for segments</small>
              <small v-for="item in [...turnTiming.ttsSegments].sort((a, b) => a.index - b.index)" :key="`${item.index}-${item.text}`" :title="item.text">
                #{{ item.index }} {{ formatTime(item.durationMs) }}
              </small>
            </div>
          </div>
          <div class="timing-total">
            <small>STAGE TOTAL</small>
            <b>{{ formatTime(turnTiming.totalStageMs) }}</b>
            <span>E2E {{ formatTime(turnTiming.endToEndMs) }}</span>
          </div>
        </section>
        <div class="stage-controls">
          <button v-if="!active" class="start-button" :disabled="!canStart" @click="startConversation">Start conversation</button>
          <button v-else class="stop-button" @click="stopConversation">End conversation</button>
          <p>{{ active ? "Just start speaking. Confirmed speech interrupts the digital human; noise is ignored." : "One camera frame is captured per turn and processed entirely by the local model." }}</p>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.digital-human-page { padding-bottom: 4rem; }
.digital-human-header { align-items: stretch; border-bottom: 1px solid var(--line); display: grid; gap: clamp(1rem, 3vw, 3rem); grid-template-columns: minmax(17rem, .7fr) minmax(32rem, 1.3fr); padding: .65rem 0 1.25rem; }
.hero-title { display: flex; flex-direction: column; }
.digital-human-header h1 { font-size: clamp(2.4rem, 4.5vw, 4.4rem); letter-spacing: -.065em; line-height: .95; margin: .55rem 0 .8rem; white-space: nowrap; }
.digital-human-header h1 em { color: var(--orange); font-style: normal; }
.phase-pill { align-items: center; border: 1px solid var(--line); display: flex; font-family: var(--font-mono); font-size: .68rem; gap: .65rem; margin-top: auto; max-width: 24rem; padding: .75rem 1rem; }
.phase-pill i { background: #999; border-radius: 50%; height: .5rem; width: .5rem; }
.phase-pill i.live { animation: pulse 1.2s infinite; background: var(--signal); }
.digital-human-grid { margin: 0 auto; max-width: 108rem; padding-top: 2rem; width: 100%; }
.setup-panel { align-self: end; border-left: 1px solid var(--line); display: grid; gap: 1.5rem; grid-template-columns: minmax(20rem, 1.25fr) minmax(17rem, .75fr); padding-left: clamp(1.5rem, 3vw, 3rem); }
.setup-block { min-width: 0; }
.voice-block { border-left: 1px solid var(--line); padding-left: 1.5rem; }
.step-heading { align-items: center; display: flex; gap: .8rem; margin-bottom: 1rem; }
.step-heading span { background: var(--ink); color: var(--paper); font: .68rem var(--font-mono); padding: .3rem .45rem; }
.step-heading h2 { font-size: 1.15rem; margin: 0; }
.model-stack { display: grid; gap: .45rem; margin-bottom: 1rem; }
.mini-model { align-items: center; background: var(--paper-deep); display: grid; font: .65rem var(--font-mono); gap: .6rem; grid-template-columns: 2.8rem minmax(8rem, 1fr) minmax(5.5rem, auto) 4.2rem 4.5rem; height: 2.7rem; overflow: hidden; padding: 0 .65rem; position: relative; }
.mini-model span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mini-model .runtime-badge { border: 1px solid var(--line); color: var(--muted); font-size: .57rem; padding: .25rem .4rem; text-align: center; text-transform: uppercase; }
.mini-model-select { background: transparent; border: 0; color: inherit; font: inherit; min-width: 0; padding: 0; width: 100%; }
.mini-model i { color: var(--orange); font-size: .57rem; font-style: normal; }
.mini-model i.ready { color: #548500; }
.mini-model button { background: transparent; border: 1px solid var(--line); cursor: pointer; font: inherit; padding: .35rem .5rem; }
.mini-model button:disabled { cursor: not-allowed; }
.action-button, .start-button, .stop-button { border: 0; cursor: pointer; font-weight: 800; padding: .9rem 1rem; width: 100%; }
.action-button { background: var(--ink); color: var(--paper); margin-top: .8rem; }
button:disabled { cursor: not-allowed; opacity: .38; }
.reference-copy { display: grid; gap: .45rem; }
.reference-copy > span { font: .65rem var(--font-mono); }
.reference-copy select { background: transparent; border: 1px solid var(--line); font-size: 1rem; padding: .8rem; width: 100%; }
.reference-copy small { color: var(--muted); justify-self: end; }
.voice-options { display: grid; gap: .8rem; }
.kokoro-note { color: var(--muted); display: block; font: .6rem var(--font-mono); margin-top: .7rem; text-align: right; }
.video-stage { aspect-ratio: 16/9; background: #181916; margin: 0 auto; max-height: calc(100vh - 12rem); min-height: 32rem; overflow: hidden; position: relative; width: min(100%, 100rem); }
.video-stage video { height: 100%; object-fit: cover; transform: scaleX(-1); width: 100%; }
.camera-placeholder { color: #aaa; left: 50%; position: absolute; text-align: center; top: 50%; transform: translate(-50%, -50%); }
.camera-placeholder span { color: var(--signal); font-size: 3rem; }
.camera-placeholder p { font: .72rem/1.6 var(--font-mono); }
.scanline { background: linear-gradient(transparent 50%, rgba(184,242,60,.05) 50%); background-size: 100% 4px; inset: 0; pointer-events: none; position: absolute; }
.live-label { align-items: center; background: rgba(0,0,0,.72); color: white; display: flex; font: .6rem var(--font-mono); gap: .4rem; left: 1rem; padding: .4rem .55rem; position: absolute; top: 1rem; }
.live-label i { background: var(--orange); border-radius: 50%; height: .45rem; width: .45rem; }
.subtitles { bottom: 1rem; display: grid; gap: .5rem; left: 1rem; max-height: 40%; overflow: auto; position: absolute; right: 1rem; }
.subtitles p { background: rgba(0,0,0,.78); color: white; font-size: clamp(.82rem, 1.4vw, 1.1rem); line-height: 1.45; margin: 0; padding: .65rem .8rem; }
.subtitles p span { color: var(--blue); font: .55rem var(--font-mono); margin-right: .7rem; }
.subtitles .digital-human-line span { color: var(--signal); }
.result-panel { border: 1px solid var(--line); display: grid; grid-template-columns: minmax(10rem, .8fr) minmax(15rem, 1.2fr) minmax(9rem, .75fr) minmax(15rem, 1.35fr) minmax(10rem, .8fr); margin: 1rem auto 0; max-width: 100rem; min-height: 5.25rem; }
.result-heading, .timing-stage, .timing-total { display: flex; flex-direction: column; justify-content: center; min-width: 0; padding: .7rem 1rem; }
.result-heading, .timing-stage { border-right: 1px solid var(--line); }
.result-heading span, .timing-stage strong, .timing-total small { color: var(--muted); font: .56rem var(--font-mono); letter-spacing: .08em; }
.result-heading b { font-size: .78rem; margin-top: .35rem; }
.timing-stage b, .timing-total b { font: 1.05rem var(--font-mono); margin: .25rem 0; }
.timing-stage > small, .timing-total span, .tts-timings small { color: var(--muted); font: .55rem var(--font-mono); }
.timing-asr { display: grid; gap: .18rem .7rem; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.timing-asr strong, .timing-asr b { grid-column: 1 / -1; }
.tts-timings { display: flex; gap: .5rem; max-width: 100%; overflow-x: auto; white-space: nowrap; }
.timing-total { background: var(--ink); color: var(--paper); }
.timing-total span { color: color-mix(in srgb, var(--paper) 65%, transparent); }
.stage-controls { align-items: center; display: grid; gap: 1rem; grid-template-columns: 14rem 1fr; margin: 1rem auto 0; max-width: 100rem; }
.stage-controls p { color: var(--muted); font-size: .75rem; line-height: 1.5; margin: 0; }
.start-button { background: var(--signal); }
.mini-model > small { background: var(--paper-deep); grid-column: 2 / -1; overflow: hidden; position: absolute; right: 5.8rem; text-overflow: ellipsis; white-space: nowrap; }
.lifecycle-success { color:#3c8b2f; font:.5rem var(--font-mono); }.lifecycle-error { color:#cf3f27; font:.5rem var(--font-mono); }
.stop-button { background: var(--orange); }
@media (max-width: 1050px) { .digital-human-header { grid-template-columns: 1fr; } .digital-human-header h1 { white-space:normal; } .setup-panel { border-left: 0; padding-left: 0; } .phase-pill { margin-top: 0; } }
@media (max-width: 900px) { .result-panel { grid-template-columns: 1fr 1fr; } .result-heading, .timing-stage { border-bottom: 1px solid var(--line); } .timing-total { grid-column: 1 / -1; } }
@media (max-width: 700px) { .setup-panel { grid-template-columns: 1fr; } .voice-block { border-left: 0; border-top: 1px solid var(--line); padding: 1.5rem 0 0; } .video-stage { min-height: 20rem; } .stage-controls { grid-template-columns: 1fr; } }
</style>
