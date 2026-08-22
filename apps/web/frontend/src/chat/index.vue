<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue"

import { fetchChatModels, fetchLoadedChatModel, loadChatModel, streamChatMessage } from "./api"
import { ChatAsrSocket } from "./asrSocket"
import ChatAudioPlayer from "./ChatAudioPlayer.vue"
import { encodePcm16, mergeWavBlobs } from "./wav"
import { errorMessage, unloadModel } from "@/modelLifecycle"
import { fetchAsrModels, loadAsrModel } from "@/live_transcription/api"
import { fetchTtsModels, loadTtsModel, synthesizeSpeech } from "@/text_to_speech/api"
import type { AsrModel, LoadedAsrModel } from "@/live_transcription/types"
import type { LoadedTtsModel, TtsModel } from "@/text_to_speech/types"
import type { ChatModel, ConversationMessage, LoadedChatModel } from "./types"

const model = ref<LoadedChatModel | null>(null)
const models = ref<ChatModel[]>([])
const selectedModelId = ref("")
const messages = ref<ConversationMessage[]>([])
const prompt = ref("")
const images = ref<Array<{ file: File; url: string }>>([])
const loadingModel = ref(false)
const sending = ref(false)
const error = ref("")
const lifecycleMessage = ref<{ type: "error"; text: string } | null>(null)
const maxTokens = ref(512)
const temperature = ref(0)
const enableThinking = ref(false)
const asrEnabled = ref(true)
const ttsEnabled = ref(true)
const cameraEnabled = ref(false)
const asrModel = ref<AsrModel | null>(null)
const loadedAsr = ref<LoadedAsrModel | null>(null)
const ttsModel = ref<TtsModel | null>(null)
const loadedTts = ref<LoadedTtsModel | null>(null)
const voice = ref("af_heart")
const language = ref("en-us")
const asrDraft = ref("")
const video = ref<HTMLVideoElement | null>(null)
const cameraPosition = reactive({ x: 0, y: 72 })
const cameraSize = ref(300)
const conversation = ref<HTMLElement | null>(null)
const autoFollow = ref(true)
let activeRequest: AbortController | null = null
let nextId = 0
let activeTurn = 0
let activeAssistant: ConversationMessage | null = null
let mediaStream: MediaStream | null = null
let cameraStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let audioSource: MediaStreamAudioSourceNode | null = null
let audioProcessor: ScriptProcessorNode | null = null
let audioGain: GainNode | null = null
let asrSocket: ChatAsrSocket | null = null
let speaking = false
let asrFinalizing = false
let silenceStartedAt = 0
let speechStartedAt = 0
let asrChunks: Float32Array[] = []
let asrSamples = 0
let preRoll: Float32Array[] = []
let preRollSamples = 0
let currentAudio: HTMLAudioElement | null = null
let currentAudioResolve: (() => void) | null = null
let playbackGeneration = 0
const ttsRequests = new Set<AbortController>()
let vadInferenceMs = 0

interface TtsSegmentResult {
  index: number
  text: string
  blob: Blob
  durationSeconds: number
  generationMs: number
}

interface TtsPipeline {
  push: (delta: string) => void
  finish: () => Promise<void>
  cancel: () => void
}

interface AsrTiming {
  totalMs: number
  vadMs: number
  enhancementMs: number | null
  inferenceMs: number | null
}

const ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
const TTS_MODEL_ID = "hexgrad/kokoro"
const ASR_CHUNK_SECONDS = 2.24
const PRE_ROLL_SECONDS = 0.5
const SILENCE_SECONDS = 0.7
const MIN_SPEECH_SECONDS = 0.15
const VISION_FRAME_SIZE = 448
const KOKORO_LANGUAGE_PREFIX: Record<string, string> = {
  "en-us": "a",
  "en-gb": "b",
  es: "e",
  fr: "f",
  hi: "h",
  it: "i",
  pt: "p",
  ja: "j",
  zh: "z",
}
const LANGUAGE_LABELS: Record<string, string> = {
  "en-us": "English · US",
  "en-gb": "English · UK",
  es: "Español",
  fr: "Français",
  hi: "हिन्दी",
  it: "Italiano",
  pt: "Português",
  ja: "日本語",
  zh: "中文",
}

const ready = computed(() => model.value?.state === "ready")
const selectedProfile = computed(() =>
  models.value.find((item) => item.profile_id === selectedModelId.value),
)
const selectedResourceAvailable = computed(() =>
  Boolean(
    selectedProfile.value?.resource.artifacts.find(
      (item) => item.artifact_id === selectedProfile.value?.required_artifact_id,
    )?.available,
  ),
)
const supportsImages = computed(() => selectedProfile.value?.supports_images ?? false)
const canSend = computed(
  () => Boolean(prompt.value.trim()) && !loadingModel.value && Boolean(selectedResourceAvailable.value),
)
const asrAvailable = computed(() => Boolean(asrModel.value?.resource.runtimes.some((item) => item.runtime === "coreml" && item.available)))
const ttsAvailable = computed(() => Boolean(ttsModel.value?.resource.runtimes?.some((item) => item.runtime === "coreml" && item.available)))
const ttsLanguages = computed(() =>
  (ttsModel.value?.languages ?? []).filter((item) => KOKORO_LANGUAGE_PREFIX[item]),
)
const ttsVoices = computed(() => {
  const prefix = KOKORO_LANGUAGE_PREFIX[language.value]
  return (ttsModel.value?.voices ?? []).filter((item) => item.startsWith(prefix ?? ""))
})
const loadedProfile = computed(() => {
  if (model.value?.state === "ready") {
    return models.value.find((item) =>
      item.model_id === model.value?.model_id && item.variant === model.value?.variant,
    ) ?? null
  }
  return models.value.find((item) => item.ready_instance_id) ?? null
})
const loadedRuntime = computed(() =>
  model.value?.state === "ready" ? model.value.runtime : loadedProfile.value?.runtime,
)

async function refreshModel() {
  try {
    const [chatModels, asrModels, ttsModels] = await Promise.all([
      fetchChatModels(),
      fetchAsrModels(),
      fetchTtsModels(),
    ])
    models.value = chatModels
    asrModel.value = asrModels.find((item) => item.model_id === ASR_MODEL_ID) ?? null
    ttsModel.value = ttsModels.find((item) => item.model_id === TTS_MODEL_ID) ?? null
    syncTtsSelection()
    if (!asrModel.value?.resource.runtimes.some((item) => item.runtime === "coreml" && item.available)) asrEnabled.value = false
    if (!ttsModel.value?.resource.runtimes?.some((item) => item.runtime === "coreml" && item.available)) ttsEnabled.value = false
    loadedAsr.value = asrModel.value?.ready_instance_id
      ? { instance_id: asrModel.value.ready_instance_id, model_id: ASR_MODEL_ID, variant: asrModel.value.variant, runtime: asrModel.value.runtime, device: "", state: "ready" }
      : null
    loadedTts.value = ttsModel.value?.ready_instance_id
      ? { instance_id: ttsModel.value.ready_instance_id, model_id: TTS_MODEL_ID, variant: ttsModel.value.variant, runtime: ttsModel.value.runtime, device: "", state: "ready" }
      : null
    if (!models.value.some((item) => item.profile_id === selectedModelId.value)) {
      selectedModelId.value = models.value.find(modelResourceAvailable)?.profile_id ?? ""
    }
    model.value = selectedModelId.value
      ? await fetchLoadedChatModel(selectedModelId.value)
      : null
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

function syncTtsSelection() {
  const languages = ttsModel.value?.languages ?? []
  if (!languages.includes(language.value)) language.value = languages[0] ?? "en-us"
  const voices = (ttsModel.value?.voices ?? []).filter((item) =>
    item.startsWith(KOKORO_LANGUAGE_PREFIX[language.value] ?? ""),
  )
  if (!voices.includes(voice.value)) voice.value = voices[0] ?? ttsModel.value?.voices[0] ?? "af_heart"
}

function changeTtsLanguage() {
  const firstVoice = ttsVoices.value[0]
  if (firstVoice) voice.value = firstVoice
}

function voiceLabel(value: string): string {
  const name = value.split("_").slice(1).join(" ")
  const displayName = name.replace(/\b\w/g, (letter) => letter.toUpperCase())
  const gender = value[1] === "f" ? "Female" : value[1] === "m" ? "Male" : "Voice"
  return `${displayName} · ${gender}`
}

async function ensureSelectedModelLoaded(): Promise<LoadedChatModel> {
  if (ready.value && model.value) return model.value
  if (!selectedModelId.value || !selectedResourceAvailable.value) {
    throw new Error("模型资产不可用，请先在 Models 页面下载模型。")
  }
  loadingModel.value = true
  error.value = ""
  lifecycleMessage.value = null
  try {
    model.value = await loadChatModel(selectedModelId.value)
    const profile = models.value.find((item) => item.profile_id === selectedModelId.value)
    if (profile) profile.ready_instance_id = model.value.instance_id
    return model.value
  } catch (caught) {
    const message = errorMessage(caught, "模型加载失败")
    lifecycleMessage.value = { type: "error", text: message }
    throw caught
  } finally {
    loadingModel.value = false
  }
}

async function selectModel() {
  lifecycleMessage.value = null
  interruptTurn()
  images.value.forEach((item) => URL.revokeObjectURL(item.url))
  images.value = []
  error.value = ""
  try {
    model.value = await fetchLoadedChatModel(selectedModelId.value)
  } catch (caught) {
    model.value = null
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

async function toggleSelectedModel() {
  if (loadingModel.value) return
  if (!ready.value || !model.value) {
    try { await ensureSelectedModelLoaded() } catch { /* inline feedback */ }
    return
  }
  loadingModel.value = true
  lifecycleMessage.value = null
  try {
    const instanceId = model.value.instance_id
    await unloadModel(instanceId)
    for (const profile of models.value) {
      if (profile.ready_instance_id === instanceId) profile.ready_instance_id = null
    }
    model.value = null
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "模型卸载失败") }
  } finally {
    loadingModel.value = false
  }
}

function selectImages(event: Event) {
  if (!supportsImages.value) return
  const input = event.target as HTMLInputElement
  for (const file of Array.from(input.files ?? [])) {
    if (images.value.length >= 4) break
    images.value.push({ file, url: URL.createObjectURL(file) })
  }
  input.value = ""
}

function removeImage(index: number) {
  const item = images.value[index]
  if (item) URL.revokeObjectURL(item.url)
  images.value.splice(index, 1)
}

function resetConversation() {
  interruptTurn()
  messages.value.forEach((message) =>
    message.images.forEach((image) => URL.revokeObjectURL(image.url)),
  )
  messages.value.forEach((message) => {
    if (message.audio?.url) URL.revokeObjectURL(message.audio.url)
  })
  messages.value = []
  error.value = ""
  autoFollow.value = true
}

function modelResourceAvailable(item: ChatModel): boolean {
  return Boolean(item.resource.artifacts.find((artifact) => artifact.artifact_id === item.required_artifact_id)?.available)
}

function interruptTurn() {
  activeTurn += 1
  activeRequest?.abort()
  activeRequest = null
  for (const controller of ttsRequests) controller.abort()
  ttsRequests.clear()
  stopAutoPlayback()
  document.querySelectorAll<HTMLAudioElement>(".voice-player-audio").forEach((audio) => audio.pause())
  if (activeAssistant?.status === "streaming") {
    activeAssistant.status = "interrupted"
  }
  activeAssistant = null
  sending.value = false
}

function stopAutoPlayback() {
  playbackGeneration += 1
  currentAudio?.pause()
  currentAudioResolve?.()
  currentAudioResolve = null
  currentAudio = null
}

async function scrollToBottom(behavior: ScrollBehavior = "smooth") {
  await nextTick()
  const target = conversation.value
  if (!target) return
  target.scrollTo({ top: target.scrollHeight, behavior })
}

function handleConversationScroll() {
  const target = conversation.value
  if (!target) return
  autoFollow.value = target.scrollHeight - target.scrollTop - target.clientHeight < 96
}

function resumeAutoFollow() {
  autoFollow.value = true
  void scrollToBottom()
}

async function followLatestContent() {
  if (autoFollow.value) await scrollToBottom("auto")
}

async function submit() {
  if (!canSend.value) return
  const text = prompt.value.trim()
  const attachments = images.value.splice(0)
  prompt.value = ""
  cancelAsrUtterance()
  await startTurn(text, attachments, "text")
}

async function startTurn(
  text: string,
  attachments: Array<{ file: File; url: string }> = [],
  source: "text" | "asr" = "text",
  asrTiming?: AsrTiming,
) {
  if (!text.trim()) return
  interruptTurn()
  const turnId = activeTurn
  try {
    await ensureSelectedModelLoaded()
  } catch {
    return
  }
  if (turnId !== activeTurn) return
  if (!model.value) return
  const history = messages.value
    .filter((message) => message.status !== "interrupted")
    .map(({ role, content }) => ({ role, content }))
  const cameraFrame = cameraEnabled.value ? await captureCameraFrame() : null
  if (cameraFrame) attachments.push(cameraFrame)
  const userMessage = reactive<ConversationMessage>({
    id: ++nextId,
    role: "user",
    content: text,
    images: attachments.map((item) => ({ name: item.file.name, url: item.url })),
    status: "complete",
    source,
    asrTiming,
  })
  messages.value.push(userMessage)
  const assistantMessage = reactive<ConversationMessage>({
    id: ++nextId,
    role: "assistant",
    content: "",
    images: [],
    status: "streaming",
  })
  messages.value.push(assistantMessage)
  activeAssistant = assistantMessage
  sending.value = true
  autoFollow.value = true
  error.value = ""
  await scrollToBottom()
  const request = new AbortController()
  activeRequest = request
  const ttsInstance = ttsEnabled.value
    ? ensureTtsLoaded()
    : Promise.resolve<LoadedTtsModel | null>(null)
  const ttsPipeline = ttsEnabled.value
    ? createTtsPipeline(assistantMessage, turnId, ttsInstance)
    : null
  try {
    const finalEvent = await streamChatMessage(
      {
        instanceId: model.value.instance_id,
        prompt: text,
        history,
        images: attachments.map((item) => item.file),
        maxTokens: maxTokens.value,
        temperature: temperature.value,
        enableThinking: enableThinking.value,
      },
      (event) => {
        if (turnId !== activeTurn) return
        if (event.delta) {
          assistantMessage.content += event.delta
          ttsPipeline?.push(event.delta)
        }
        void followLatestContent()
      },
      request.signal,
    )
    if (turnId !== activeTurn) return
    const inferenceMs = finalEvent.inference_ms
    const generatedTokens = finalEvent.generated_tokens
    assistantMessage.llmTiming = {
      runtime: finalEvent.runtime,
      generatedTokens,
      inferenceMs,
      tokensPerSecond: generatedTokens !== null && inferenceMs !== null && inferenceMs > 0
        ? generatedTokens / (inferenceMs / 1000)
        : null,
    }
    assistantMessage.status = "complete"
    await ttsPipeline?.finish()
  } catch (caught) {
    ttsPipeline?.cancel()
    if (!(caught instanceof DOMException && caught.name === "AbortError")) {
      error.value = caught instanceof Error ? caught.message : "消息发送失败"
      if (!assistantMessage.content) {
        messages.value = messages.value.filter((item) => item.id !== assistantMessage.id)
      }
    }
  } finally {
    if (activeRequest === request) activeRequest = null
    if (turnId === activeTurn) sending.value = false
    if (activeAssistant === assistantMessage) activeAssistant = null
    await followLatestContent()
  }
}

function createTtsPipeline(
  message: ConversationMessage,
  turnId: number,
  instancePromise: Promise<LoadedTtsModel | null>,
): TtsPipeline {
  let speechBuffer = ""
  let segmentIndex = 0
  let cancelled = false
  const controllers = new Set<AbortController>()
  const results: Array<Promise<TtsSegmentResult>> = []
  const playbackId = ++playbackGeneration
  let playbackChain = Promise.resolve()

  const queue = (text: string) => {
    const segment = text.trim()
    if (!segment || cancelled || turnId !== activeTurn) return
    const index = ++segmentIndex
    message.audio ??= { status: "generating", url: null, durationSeconds: 0 }
    const generated = (async (): Promise<TtsSegmentResult> => {
      const instance = await instancePromise
      if (!instance || cancelled || turnId !== activeTurn) throw new DOMException("Aborted", "AbortError")
      const controller = new AbortController()
      controllers.add(controller)
      ttsRequests.add(controller)
      const startedAt = performance.now()
      try {
        const response = await synthesizeSpeech({
          model_id: TTS_MODEL_ID,
          instance_id: instance.instance_id,
          text: segment,
          voice: voice.value,
          language: language.value,
          speed: 1,
        }, controller.signal)
        return {
          index,
          text: segment,
          blob: response.blob,
          durationSeconds: Number(response.headers.get("x-duration-seconds") ?? 0),
          generationMs: performance.now() - startedAt,
        }
      } finally {
        controllers.delete(controller)
        ttsRequests.delete(controller)
      }
    })()
    results.push(generated)
    playbackChain = playbackChain.then(async () => {
      const result = await generated
      if (cancelled || turnId !== activeTurn || playbackId !== playbackGeneration) return
      await playAudioBlob(result.blob, turnId, playbackId)
    }).catch((caught) => {
      if (!(caught instanceof DOMException && caught.name === "AbortError")) {
        error.value = errorMessage(caught, "TTS 生成失败")
      }
    })
  }

  return {
    push(delta: string) {
      speechBuffer += delta
      const parsed = extractSpeechSegments(speechBuffer)
      speechBuffer = parsed.remainder
      parsed.segments.forEach(queue)
    },
    async finish() {
      queue(speechBuffer)
      speechBuffer = ""
      if (!results.length) return
      try {
        const generated = await Promise.all(results)
        if (cancelled || turnId !== activeTurn) return
        const merged = await mergeWavBlobs(generated.map((item) => item.blob))
        if (cancelled || turnId !== activeTurn) return
        const url = URL.createObjectURL(merged)
        if (message.audio?.url) URL.revokeObjectURL(message.audio.url)
        message.audio = {
          status: "ready",
          url,
          durationSeconds: generated.reduce((total, item) => total + item.durationSeconds, 0),
        }
        message.ttsTiming = {
          runtime: (await instancePromise)?.runtime ?? "coreml",
          segmentCount: generated.length,
          firstAudioMs: generated[0]!.generationMs,
          slowestSegmentMs: Math.max(...generated.map((item) => item.generationMs)),
        }
      } catch (caught) {
        if (!(caught instanceof DOMException && caught.name === "AbortError")) {
          message.audio = undefined
          error.value = errorMessage(caught, "TTS 生成失败")
        }
      }
    },
    cancel() {
      cancelled = true
      controllers.forEach((controller) => controller.abort())
      controllers.clear()
    },
  }
}

function extractSpeechSegments(value: string): { segments: string[]; remainder: string } {
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

async function playAudioBlob(blob: Blob, turnId: number, playbackId: number) {
  if (turnId !== activeTurn || playbackId !== playbackGeneration || !ttsEnabled.value) return
  const url = URL.createObjectURL(blob)
  const audio = new Audio(url)
  currentAudio = audio
  try {
    await audio.play()
    await new Promise<void>((resolve) => {
      currentAudioResolve = resolve
      audio.onended = () => resolve()
      audio.onerror = () => resolve()
    })
  } catch {
    // Browser autoplay policies may require the user to use the visible player.
  } finally {
    if (currentAudio === audio) currentAudio = null
    currentAudioResolve = null
    URL.revokeObjectURL(url)
  }
}

async function ensureTtsLoaded(): Promise<LoadedTtsModel | null> {
  if (!ttsEnabled.value || !ttsAvailable.value || !ttsModel.value) return null
  if (loadedTts.value?.state === "ready") return loadedTts.value
  loadedTts.value = await loadTtsModel(TTS_MODEL_ID, "coreml")
  return loadedTts.value
}

async function toggleTts() {
  if (ttsEnabled.value) {
    if (!ttsAvailable.value) {
      ttsEnabled.value = false
      error.value = "Kokoro Core ML 资源未下载。"
      return
    }
    try { await ensureTtsLoaded() } catch (caught) {
      ttsEnabled.value = false
      error.value = errorMessage(caught, "TTS 模型加载失败")
    }
    return
  }
  for (const controller of ttsRequests) controller.abort()
  ttsRequests.clear()
  stopAutoPlayback()
  if (loadedTts.value) {
    await unloadModel(loadedTts.value.instance_id).catch(() => undefined)
    loadedTts.value = null
  }
}

async function toggleAsr() {
  if (!asrEnabled.value) {
    stopAsr()
    if (loadedAsr.value) {
      await unloadModel(loadedAsr.value.instance_id).catch(() => undefined)
      loadedAsr.value = null
    }
    return
  }
  if (!asrAvailable.value || !asrModel.value) {
    asrEnabled.value = false
    error.value = "Nemotron Core ML 资源未下载。"
    return
  }
  try {
    loadedAsr.value ??= await loadAsrModel(ASR_MODEL_ID, asrModel.value.variant, "coreml")
    asrSocket = await ChatAsrSocket.connect()
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: false, channelCount: 1 },
      video: false,
    })
    audioContext = new AudioContext({ latencyHint: "interactive" })
    await audioContext.resume()
    audioSource = audioContext.createMediaStreamSource(mediaStream)
    audioProcessor = audioContext.createScriptProcessor(4096, 1, 1)
    audioGain = audioContext.createGain()
    audioGain.gain.value = 0.00001
    audioProcessor.onaudioprocess = processAsrAudio
    audioSource.connect(audioProcessor)
    audioProcessor.connect(audioGain)
    audioGain.connect(audioContext.destination)
  } catch (caught) {
    stopAsr()
    asrEnabled.value = false
    error.value = errorMessage(caught, "语音输入启动失败")
  }
}

function processAsrAudio(event: AudioProcessingEvent) {
  if (!asrEnabled.value || !audioContext) return
  const vadStartedAt = performance.now()
  const chunk = new Float32Array(event.inputBuffer.getChannelData(0))
  let energy = 0
  for (const sample of chunk) energy += sample * sample
  const voiced = Math.sqrt(energy / chunk.length) >= 0.018
  vadInferenceMs += performance.now() - vadStartedAt
  if (!speaking) {
    preRoll.push(chunk)
    preRollSamples += chunk.length
    const limit = audioContext.sampleRate * PRE_ROLL_SECONDS
    while (preRollSamples > limit && preRoll.length > 1) preRollSamples -= preRoll.shift()!.length
    if (voiced) beginAsrUtterance()
    return
  }
  asrChunks.push(chunk)
  asrSamples += chunk.length
  if (asrSamples / audioContext.sampleRate >= ASR_CHUNK_SECONDS) flushAsrAudio()
  if (voiced) silenceStartedAt = 0
  else if (!silenceStartedAt) silenceStartedAt = performance.now()
  const silence = silenceStartedAt ? (performance.now() - silenceStartedAt) / 1000 : 0
  if (silence >= SILENCE_SECONDS) finishAsrUtterance()
}

function beginAsrUtterance() {
  if (!asrSocket || !loadedAsr.value || !audioContext || speaking) return
  if (asrFinalizing) {
    asrSocket.cancel()
    asrFinalizing = false
  }
  speaking = true
  speechStartedAt = performance.now()
  silenceStartedAt = 0
  asrChunks = preRoll
  asrSamples = preRollSamples
  preRoll = []
  preRollSamples = 0
  asrDraft.value = "Listening…"
  asrSocket.start(
    loadedAsr.value.instance_id,
    audioContext.sampleRate,
    (event) => { if (event.text.trim()) asrDraft.value = event.text },
    (caught) => { error.value = caught.message },
  )
}

function flushAsrAudio() {
  if (!asrSocket || !asrChunks.length) return
  asrSocket.send(encodePcm16(asrChunks))
  asrChunks = []
  asrSamples = 0
}

function finishAsrUtterance() {
  if (!speaking || !asrSocket) return
  const duration = (performance.now() - speechStartedAt) / 1000
  speaking = false
  silenceStartedAt = 0
  if (duration < MIN_SPEECH_SECONDS) { cancelAsrUtterance(); return }
  flushAsrAudio()
  const utteranceVadMs = vadInferenceMs
  vadInferenceMs = 0
  asrFinalizing = true
  void asrSocket.finish().then((result) => {
    asrDraft.value = ""
    if (/[\p{L}\p{N}]/u.test(result.text.trim())) {
      void startTurn(result.text.trim(), [], "asr", {
        totalMs: utteranceVadMs + (result.preprocess_ms ?? 0) + (result.inference_ms ?? 0),
        vadMs: utteranceVadMs,
        enhancementMs: result.enhancement_inference_ms,
        inferenceMs: (result.preprocess_ms ?? 0) + (result.inference_ms ?? 0),
      })
    }
  }).catch((caught) => {
    if (!(caught instanceof DOMException && caught.name === "AbortError")) error.value = errorMessage(caught, "ASR 失败")
  }).finally(() => {
    asrFinalizing = false
  })
}

function cancelAsrUtterance() {
  asrSocket?.cancel()
  speaking = false
  asrFinalizing = false
  asrChunks = []
  asrSamples = 0
  silenceStartedAt = 0
  speechStartedAt = 0
  asrDraft.value = ""
  vadInferenceMs = 0
}

function formatTime(value: number | null): string {
  if (value === null) return "—"
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

function stopAsr() {
  cancelAsrUtterance()
  asrSocket?.close()
  asrSocket = null
  if (audioProcessor) audioProcessor.onaudioprocess = null
  audioProcessor?.disconnect()
  audioSource?.disconnect()
  audioGain?.disconnect()
  mediaStream?.getTracks().forEach((track) => track.stop())
  void audioContext?.close()
  mediaStream = null
  audioContext = null
  audioSource = null
  audioProcessor = null
  audioGain = null
  preRoll = []
  preRollSamples = 0
}

async function toggleCamera() {
  if (!cameraEnabled.value) { stopCamera(); return }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    })
    await nextTick()
    if (video.value) { video.value.srcObject = cameraStream; await video.value.play() }
    cameraPosition.x = Math.max(12, window.innerWidth - cameraSize.value - 24)
  } catch (caught) {
    cameraEnabled.value = false
    stopCamera()
    error.value = errorMessage(caught, "摄像头启动失败")
  }
}

function stopCamera() {
  if (video.value) video.value.srcObject = null
  cameraStream?.getTracks().forEach((track) => track.stop())
  cameraStream = null
}

function closeCamera() {
  cameraEnabled.value = false
  stopCamera()
}

async function captureCameraFrame(): Promise<{ file: File; url: string } | null> {
  const source = video.value
  if (!cameraEnabled.value || !source?.videoWidth || !source.videoHeight) return null
  const scale = Math.min(VISION_FRAME_SIZE / source.videoWidth, VISION_FRAME_SIZE / source.videoHeight)
  const width = Math.round(source.videoWidth * scale)
  const height = Math.round(source.videoHeight * scale)
  const canvas = document.createElement("canvas")
  canvas.width = VISION_FRAME_SIZE
  canvas.height = VISION_FRAME_SIZE
  const context = canvas.getContext("2d")
  if (!context) return null
  context.fillStyle = "#000"
  context.fillRect(0, 0, canvas.width, canvas.height)
  context.translate(canvas.width, 0)
  context.scale(-1, 1)
  context.drawImage(source, Math.floor((VISION_FRAME_SIZE - width) / 2), Math.floor((VISION_FRAME_SIZE - height) / 2), width, height)
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Camera capture failed")), "image/jpeg", 0.72))
  const file = new File([blob], "webcam.jpg", { type: "image/jpeg" })
  return { file, url: URL.createObjectURL(blob) }
}

function startCameraDrag(event: PointerEvent) {
  const target = event.currentTarget as HTMLElement
  const panel = target.closest(".chat-camera") as HTMLElement | null
  if (!panel) return
  const rect = panel.getBoundingClientRect()
  const offsetX = event.clientX - rect.left
  const offsetY = event.clientY - rect.top
  const move = (moveEvent: PointerEvent) => {
    cameraPosition.x = Math.max(0, Math.min(window.innerWidth - panel.offsetWidth, moveEvent.clientX - offsetX))
    cameraPosition.y = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, moveEvent.clientY - offsetY))
  }
  const end = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", end) }
  window.addEventListener("pointermove", move)
  window.addEventListener("pointerup", end, { once: true })
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    void submit()
  }
}

onMounted(async () => {
  await refreshModel()
  if (asrEnabled.value && asrAvailable.value) await toggleAsr()
  if (ttsEnabled.value && ttsAvailable.value) await toggleTts()
})
onBeforeUnmount(() => {
  interruptTurn()
  stopAsr()
  stopCamera()
  images.value.forEach((item) => URL.revokeObjectURL(item.url))
  messages.value.forEach((message) =>
    message.images.forEach((image) => URL.revokeObjectURL(image.url)),
  )
  messages.value.forEach((message) => {
    if (message.audio?.url) URL.revokeObjectURL(message.audio.url)
  })
})
</script>

<template>
  <div class="chat-page">
    <aside class="chat-sidebar">
      <RouterLink class="chat-brand" to="/apps">← Neural Apps</RouterLink>

      <label class="chat-model-select">
        MODEL
        <span class="chat-model-control">
          <select v-model="selectedModelId" :disabled="loadingModel" @change="selectModel">
            <option v-for="item in models" :key="item.profile_id" :value="item.profile_id" :disabled="!modelResourceAvailable(item)">
              {{ item.display_name }}
            </option>
          </select>
          <button type="button" :disabled="loadingModel || (!ready && !selectedResourceAvailable)" @click="toggleSelectedModel">
            {{ loadingModel ? "WAIT…" : ready ? "UNLOAD" : "LOAD" }}
          </button>
        </span>
      </label>
      <small v-if="lifecycleMessage" :class="`lifecycle-${lifecycleMessage.type}`">{{ lifecycleMessage.text }}</small>

      <div class="chat-loaded-model" :class="{ empty: !loadedProfile }">
        <span>CURRENTLY LOADED</span>
        <strong>{{ loadedProfile?.display_name ?? "None" }}</strong>
        <small>{{ loadedRuntime?.toUpperCase() ?? "—" }}</small>
      </div>
      <p v-if="!ready && !selectedResourceAvailable" class="chat-error">模型不可用。请前往 Models 页面下载模型后重试。</p>

      <details class="chat-settings">
        <summary>Generation settings</summary>
        <label>Max tokens <output>{{ maxTokens }}</output></label>
        <input v-model.number="maxTokens" type="range" min="64" max="2048" step="64" />
        <label>Temperature <output>{{ temperature.toFixed(1) }}</output></label>
        <input v-model.number="temperature" type="range" min="0" max="1.5" step="0.1" />
        <label class="chat-toggle"><input v-model="enableThinking" type="checkbox" /> Thinking mode</label>
      </details>
      <section class="chat-capabilities" aria-label="Conversation capabilities">
        <label class="chat-capability" :class="{ unavailable: !asrAvailable }">
          <input v-model="asrEnabled" type="checkbox" :disabled="!asrAvailable" @change="toggleAsr" />
          <span><b>ASR</b><small>{{ loadedAsr ? loadedAsr.runtime.toUpperCase() : "VOICE INPUT" }}</small></span>
        </label>
        <div class="chat-tts-capability">
          <label class="chat-capability" :class="{ unavailable: !ttsAvailable }">
            <input v-model="ttsEnabled" type="checkbox" :disabled="!ttsAvailable" @change="toggleTts" />
            <span><b>TTS</b><small>{{ loadedTts ? loadedTts.runtime.toUpperCase() : "VOICE OUTPUT" }}</small></span>
          </label>
          <div v-if="ttsEnabled && ttsAvailable" class="chat-tts-options">
            <label>
              <span>LANGUAGE</span>
              <select v-model="language" aria-label="TTS language" @change="changeTtsLanguage">
                <option v-for="item in ttsLanguages" :key="item" :value="item">
                  {{ LANGUAGE_LABELS[item] ?? item }}
                </option>
              </select>
            </label>
            <label>
              <span>VOICE</span>
              <select v-model="voice" aria-label="TTS voice">
                <option v-for="item in ttsVoices" :key="item" :value="item">
                  {{ voiceLabel(item) }}
                </option>
              </select>
            </label>
          </div>
        </div>
        <label class="chat-capability" :class="{ unavailable: !supportsImages }">
          <input v-model="cameraEnabled" type="checkbox" :disabled="!supportsImages" @change="toggleCamera" />
          <span><b>CAMERA</b><small>448 × 448 FRAME</small></span>
        </label>
      </section>
      <button class="chat-clear" :disabled="!messages.length" @click="resetConversation">
        ＋ New conversation
      </button>
    </aside>

    <main class="chat-workspace">
      <header class="chat-topbar">
        <div><span class="status-dot"></span> ON-DEVICE SESSION</div>
        <span>{{ messages.length }} MESSAGES</span>
      </header>

      <aside
        v-if="cameraEnabled"
        class="chat-camera"
        :style="{ left: `${cameraPosition.x}px`, top: `${cameraPosition.y}px`, width: `${cameraSize}px` }"
      >
        <header @pointerdown="startCameraDrag">
          <span>LIVE CAMERA</span>
          <button type="button" aria-label="关闭摄像头" @pointerdown.stop @click="closeCamera">×</button>
        </header>
        <video ref="video" autoplay muted playsinline></video>
        <small>每次发送仅附加当前画面</small>
      </aside>

      <section ref="conversation" class="chat-conversation" aria-live="polite" @scroll.passive="handleConversationScroll">
        <div v-if="!messages.length" class="chat-empty">
          <span>Q / 35</span>
          <h2>What can we<br /><i>explore?</i></h2>
          <p>Ask a question, continue a thought<span v-if="supportsImages">, or attach an image for visual understanding</span>.</p>
          <div class="chat-suggestions">
            <button @click="prompt = '解释一下 Transformer 的注意力机制。'">解释一个复杂概念</button>
            <button v-if="supportsImages" @click="prompt = '请分析我上传的图片，并指出关键细节。'">分析一张图片</button>
            <button @click="prompt = '帮我设计一个清晰的项目实施计划。'">制定项目计划</button>
          </div>
        </div>

        <article v-for="message in messages" :key="message.id" :class="['chat-message', message.role]">
          <div class="chat-avatar">{{ message.role === "user" ? "YOU" : "Q" }}</div>
          <div class="chat-bubble">
            <div v-if="message.images.length" class="chat-message-images">
              <img v-for="image in message.images" :key="image.url" :src="image.url" :alt="image.name" />
            </div>
            <p v-if="message.content">{{ message.content }}</p>
            <div v-else-if="message.role === 'assistant'" class="chat-thinking"><i></i><i></i><i></i></div>
            <ChatAudioPlayer
              v-if="message.audio"
              :src="message.audio.url"
              :status="message.audio.status"
              :duration-seconds="message.audio.durationSeconds"
              @play="stopAutoPlayback"
            />
            <span v-if="message.status === 'interrupted'" class="chat-interrupted">INTERRUPTED</span>
            <div v-if="message.source === 'asr' && message.asrTiming" class="chat-message-meta">
              <span>ASR {{ formatTime(message.asrTiming.totalMs) }}</span>
              <span>VAD {{ formatTime(message.asrTiming.vadMs) }}</span>
              <span v-if="message.asrTiming.enhancementMs !== null">ENHANCE {{ formatTime(message.asrTiming.enhancementMs) }}</span>
              <span>MODEL {{ formatTime(message.asrTiming.inferenceMs) }}</span>
            </div>
            <div v-if="message.llmTiming" class="chat-message-meta">
              <span>{{ message.llmTiming.runtime.toUpperCase() }}</span>
              <span>LLM {{ formatTime(message.llmTiming.inferenceMs) }}</span>
              <span>{{ message.llmTiming.generatedTokens ?? "—" }} TOKENS</span>
              <span>{{ message.llmTiming.tokensPerSecond?.toFixed(1) ?? "—" }} TOK/S</span>
            </div>
            <div v-if="message.audio?.status === 'generating'" class="chat-message-meta">
              <span>TTS STREAMING</span><span>COREML</span>
            </div>
            <div v-else-if="message.ttsTiming" class="chat-message-meta">
              <span>{{ message.ttsTiming.runtime.toUpperCase() }}</span>
              <span>TTS FIRST {{ formatTime(message.ttsTiming.firstAudioMs) }}</span>
              <span>SLOWEST {{ formatTime(message.ttsTiming.slowestSegmentMs) }}</span>
              <span>{{ message.ttsTiming.segmentCount }} SEGMENTS → 1 TRACK</span>
            </div>
          </div>
        </article>
      </section>

      <button v-if="!autoFollow" class="chat-scroll-latest" type="button" @click="resumeAutoFollow">
        ↓ Latest
      </button>

      <div v-if="error" class="chat-error" role="alert">{{ error }}</div>
      <div v-if="asrDraft" class="chat-asr-draft"><span></span>{{ asrDraft }}</div>
      <footer class="chat-composer">
        <div v-if="images.length" class="chat-previews">
          <figure v-for="(image, index) in images" :key="image.url">
            <img :src="image.url" :alt="image.file.name" />
            <button aria-label="移除图片" @click="removeImage(index)">×</button>
          </figure>
        </div>
        <div class="chat-input-row">
          <label v-if="supportsImages" class="chat-attach" :class="{ disabled: loadingModel || images.length >= 4 }">
            <input type="file" accept="image/png,image/jpeg,image/webp" multiple :disabled="loadingModel || images.length >= 4" @change="selectImages" />
            <span>＋</span><small>IMAGE</small>
          </label>
          <textarea v-model="prompt" rows="1" maxlength="32000" :disabled="loadingModel" placeholder="输入消息…（新消息会打断当前回复）" @keydown="handleKeydown"></textarea>
          <button class="chat-send" :disabled="!canSend" @click="submit">
            <span>↑</span>
          </button>
        </div>
        <small class="chat-hint">ENTER TO SEND · SHIFT + ENTER FOR NEW LINE<span v-if="supportsImages"> · UP TO 4 IMAGES</span></small>
      </footer>
    </main>
  </div>
</template>

<style scoped>
.chat-page { background:#0d0e0d; color:#f4f2e9; display:grid; grid-template-columns:19rem minmax(0,1fr); height:100%; min-height:0; overflow:hidden; }
.chat-sidebar { border-right:1px solid #343630; display:flex; flex-direction:column; gap:1.15rem; min-height:0; overflow:auto; padding:2rem; }
.chat-brand,.chat-loaded-model,.chat-model-link,.chat-settings,.chat-clear,.chat-topbar,.chat-hint,.chat-bubble small { font-family:var(--font-mono); }
.chat-brand { color:#aaa99f; font-size:.68rem; letter-spacing:.06em; }
.chat-loaded-model { align-items:center; display:grid; gap:.45rem; grid-template-columns:auto minmax(0,1fr) auto; }
.chat-loaded-model>span { color:#77796f; font-size:.48rem; letter-spacing:.08em; }
.chat-loaded-model strong { color:#f4f2e9; font-size:.58rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.chat-loaded-model small { background:#b8f23c18; border:1px solid #718c37; color:var(--signal); font-size:.48rem; padding:.24rem .38rem; }
.chat-loaded-model.empty strong,.chat-loaded-model.empty small { color:#77796f; }
.chat-loaded-model.empty small { background:transparent; border-color:#343630; }
.chat-primary { background:var(--signal); border:0; color:#0d0e0d; cursor:pointer; font-weight:800; padding:.9rem; }.chat-primary:disabled { opacity:.5; }
.chat-model-link { color:var(--signal); font-size:.6rem; text-align:center; }
.chat-model-select { color:#85877f; display:flex; flex-direction:column; font:.56rem var(--font-mono); gap:.45rem; }.chat-model-select select { background:#191b18; border:1px solid #343630; color:#f4f2e9; font:inherit; padding:.7rem; width:100%; }
.chat-model-control { display:flex; gap:.4rem; }.chat-model-control select { min-width:0; }.chat-model-control button { background:var(--signal); border:0; cursor:pointer; font:700 .52rem var(--font-mono); padding:0 .6rem; }.chat-model-control button:disabled { cursor:not-allowed; opacity:.4; }
.lifecycle-error { color:#ff8066; font: .55rem var(--font-mono); }
.chat-delete { background:none; border:1px solid #5d3934; color:#e7a59d; cursor:pointer; font:.58rem var(--font-mono); padding:.7rem; }.chat-delete:disabled { opacity:.5; }
.chat-settings { border-top:1px solid #343630; color:#aaa99f; font-size:.62rem; padding-top:1rem; }.chat-settings summary { cursor:pointer; margin-bottom:1rem; }.chat-settings label { display:flex; justify-content:space-between; margin-top:.7rem; }.chat-settings input[type=range] { accent-color:var(--signal); width:100%; }.chat-toggle { justify-content:flex-start!important; gap:.5rem; }
.chat-capabilities { border-top:1px solid #343630; display:grid; gap:.45rem; padding-top:1rem; }
.chat-capability { align-items:center; background:#151714; border:1px solid #343630; cursor:pointer; display:flex; gap:.65rem; padding:.65rem .75rem; }
.chat-capability:has(input:checked) { border-color:#718c37; }
.chat-capability.unavailable { cursor:not-allowed; opacity:.35; }
.chat-capability input { accent-color:var(--signal); }
.chat-capability>span { display:flex; flex:1; justify-content:space-between; }
.chat-capability b,.chat-capability small { font:600 .55rem var(--font-mono); }
.chat-capability small { color:#77796f; }
.chat-tts-capability { display:grid; gap:0; }
.chat-tts-options { background:#11130f; border:1px solid #343630; border-top:0; display:grid; gap:.5rem; padding:.6rem; }
.chat-tts-options label { display:grid; gap:.3rem; min-width:0; }
.chat-tts-options label>span { color:#77796f; font:600 .47rem var(--font-mono); letter-spacing:.08em; }
.chat-tts-options select { appearance:none; background:#191b18 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='7' viewBox='0 0 12 7'%3E%3Cpath d='m1 1 5 5 5-5' fill='none' stroke='%23898d80' stroke-width='1.5'/%3E%3C/svg%3E") no-repeat right .6rem center; border:1px solid #343630; border-radius:.25rem; color:#f4f2e9; font:600 .55rem var(--font-mono); min-width:0; padding:.52rem 1.8rem .52rem .6rem; width:100%; }
.chat-tts-options select:focus { border-color:var(--signal); outline:0; }
.chat-clear { background:none; border:1px solid #343630; color:#f4f2e9; cursor:pointer; font-size:.62rem; margin-top:auto; padding:.8rem; }
.chat-workspace { display:flex; flex-direction:column; height:100%; min-height:0; min-width:0; overflow:hidden; position:relative; }
.chat-topbar { border-bottom:1px solid #343630; color:#85877f; display:flex; font-size:.58rem; justify-content:space-between; letter-spacing:.08em; padding:1rem 2rem; }.chat-topbar div { align-items:center; display:flex; gap:.6rem; }
.chat-conversation { flex:1; min-height:0; overscroll-behavior:contain; overflow-x:hidden; overflow-y:auto; padding:3rem max(4vw,2rem); scrollbar-gutter:stable; }
.chat-empty { margin:7vh auto 0; max-width:43rem; }.chat-empty>span { color:var(--signal); font:700 .7rem var(--font-mono); }.chat-empty h2 { font-size:clamp(3rem,6vw,6rem); letter-spacing:-.07em; line-height:.83; margin:1rem 0 1.5rem; }.chat-empty h2 i { color:#a6a79e; font-weight:400; }.chat-empty>p { color:#999a92; line-height:1.6; max-width:34rem; }
.chat-suggestions { display:grid; gap:.6rem; grid-template-columns:repeat(3,1fr); margin-top:2rem; }.chat-suggestions button { background:#151714; border:1px solid #343630; color:#d7d6cf; cursor:pointer; font-size:.7rem; padding:1rem; text-align:left; }.chat-suggestions button:hover { border-color:var(--signal); }
.chat-message { align-items:flex-end; animation:bubble-in-left .32s cubic-bezier(.2,.8,.25,1) both; display:flex; gap:.75rem; margin:0 auto 1.35rem; max-width:58rem; width:100%; }.chat-message.user { animation-name:bubble-in-right; flex-direction:row-reverse; }.chat-avatar { align-items:center; background:#292d27; border:1px solid #3c4138; border-radius:50%; box-shadow:0 .35rem 1rem #0004; display:flex; flex:0 0 auto; font:700 .5rem var(--font-mono); height:2.35rem; justify-content:center; width:2.35rem; }.chat-message.assistant .chat-avatar { background:var(--signal); border-color:var(--signal); color:#111; font-size:.95rem; }.chat-message.user .chat-avatar { background:#30352e; color:#e5e8df; }.chat-bubble { background:linear-gradient(145deg,#181b17,#141612); border:1px solid #2f332c; border-radius:1.15rem 1.15rem 1.15rem .3rem; box-shadow:0 .55rem 1.6rem #0002; max-width:min(78%,46rem); min-width:7rem; padding:.85rem 1rem .7rem; position:relative; }.chat-message.user .chat-bubble { background:linear-gradient(145deg,#28341e,#20291a); border-color:#3f512b; border-radius:1.15rem 1.15rem .3rem 1.15rem; }.chat-bubble p { font-size:.95rem; line-height:1.65; margin:0; white-space:pre-wrap; }.chat-bubble::after { border-bottom:.45rem solid transparent; border-right:.55rem solid #171a15; bottom:.1rem; content:""; left:-.45rem; position:absolute; }.chat-message.user .chat-bubble::after { border-left:.55rem solid #222c1b; border-right:0; left:auto; right:-.45rem; }
.chat-message-images { display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:.75rem; }.chat-message-images img { background:#090a09; border:1px solid #3d4436; border-radius:.8rem; height:10rem; object-fit:contain; width:12rem; }
.chat-message-meta { border-top:1px solid #34392f; color:#858b7d; display:flex; flex-wrap:wrap; font:600 .48rem var(--font-mono); gap:.35rem .75rem; letter-spacing:.035em; margin-top:.7rem; padding-top:.55rem; }.chat-message.user .chat-message-meta { border-color:#465a30; color:#a6b694; }.chat-interrupted { color:#d49a60; display:block; font:.52rem var(--font-mono); margin-top:.55rem; }
.chat-thinking { display:flex; gap:.3rem; padding-top:.8rem; }.chat-thinking i { animation:pulse 1s infinite alternate; background:var(--signal); border-radius:50%; height:.4rem; width:.4rem; }.chat-thinking i:nth-child(2){animation-delay:.2s}.chat-thinking i:nth-child(3){animation-delay:.4s}@keyframes pulse{to{opacity:.2;transform:translateY(-.2rem)}}
.chat-error { background:#5d211b; color:#ffd9d3; font-size:.72rem; margin:0 max(4vw,2rem) 1rem; padding:.75rem 1rem; }
.chat-asr-draft { align-items:center; color:#c9cbc1; display:flex; font:.62rem var(--font-mono); gap:.55rem; margin:0 max(4vw,2rem) .7rem; }.chat-asr-draft span { animation:pulse 1s infinite alternate; background:var(--signal); border-radius:50%; height:.5rem; width:.5rem; }
.chat-camera { background:#11130f; border:1px solid #595c52; box-shadow:0 1rem 3rem #000b; min-height:12rem; min-width:13rem; overflow:hidden; position:fixed; resize:both; z-index:20; }.chat-camera header { align-items:center; background:#1c1f19; cursor:move; display:flex; font:600 .55rem var(--font-mono); justify-content:space-between; padding:.55rem .7rem; touch-action:none; user-select:none; }.chat-camera header button { background:none; border:0; color:#ddd; cursor:pointer; font-size:1rem; }.chat-camera video { aspect-ratio:16/9; display:block; height:calc(100% - 3.8rem); min-height:8rem; object-fit:cover; transform:scaleX(-1); width:100%; }.chat-camera>small { color:#77796f; display:block; font:.48rem var(--font-mono); padding:.4rem .7rem; }
.chat-scroll-latest { background:#262923; border:1px solid #52554c; border-radius:2rem; bottom:7.4rem; color:#f4f2e9; cursor:pointer; font:600 .62rem var(--font-mono); left:50%; padding:.65rem 1rem; position:absolute; transform:translateX(-50%); z-index:3; }
.chat-composer { background:linear-gradient(180deg,#0d0e0d00,#0d0e0d 25%); border-top:1px solid #292c27; padding:1rem max(4vw,2rem) 1.4rem; }.chat-input-row { align-items:end; background:#191c17; border:1px solid #3b4037; border-radius:1.35rem; box-shadow:0 .7rem 2rem #0003; display:grid; grid-template-columns:auto minmax(0,1fr) auto; padding:.5rem; transition:border-color .2s,box-shadow .2s; }.chat-input-row:focus-within { border-color:#75894e; box-shadow:0 0 0 3px #b8f23c0c,0 .7rem 2rem #0003; }.chat-attach { align-items:center; cursor:pointer; display:flex; gap:.35rem; padding:.6rem; }.chat-attach input { display:none; }.chat-attach span { font-size:1.4rem; }.chat-attach small { color:#8c8d85; font:.52rem var(--font-mono); }.chat-attach.disabled { opacity:.35; }.chat-input-row textarea { background:none; border:0; color:#f4f2e9; font:1rem var(--font-display); max-height:10rem; min-height:2.6rem; outline:0; padding:.7rem; resize:vertical; }.chat-send { align-items:center; background:var(--signal); border:0; border-radius:50%; color:#111; cursor:pointer; display:flex; font-size:1.4rem; height:2.7rem; justify-content:center; transition:transform .18s,box-shadow .18s; width:2.7rem; }.chat-send:not(:disabled):hover { box-shadow:0 0 1rem #b8f23c55; transform:translateY(-2px); }.chat-send:disabled { background:#42443e; color:#85877f; cursor:default; }.chat-hint { color:#62645d; display:block; font-size:.5rem; letter-spacing:.08em; margin-top:.55rem; text-align:center; }
.chat-previews { display:flex; gap:.6rem; margin-bottom:.7rem; }.chat-previews figure { height:4.5rem; margin:0; position:relative; width:4.5rem; }.chat-previews img { border-radius:.4rem; height:100%; object-fit:cover; width:100%; }.chat-previews button { background:#111; border:1px solid #555; border-radius:50%; color:white; cursor:pointer; height:1.3rem; position:absolute; right:-.3rem; top:-.3rem; width:1.3rem; }
@keyframes bubble-in-left { from { opacity:0; transform:translate(-.7rem,.4rem) scale(.98); } to { opacity:1; transform:none; } }
@keyframes bubble-in-right { from { opacity:0; transform:translate(.7rem,.4rem) scale(.98); } to { opacity:1; transform:none; } }
@media(max-width:800px){.chat-page{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr)}.chat-sidebar{border-bottom:1px solid #343630;border-right:0;display:grid;grid-template-columns:1fr auto;height:auto;overflow:auto;padding:1rem 1.2rem}.chat-settings,.chat-clear{display:none}.chat-brand{align-self:center}.chat-loaded-model{grid-column:2;grid-row:1}.chat-model-select{grid-column:1/-1}.chat-capabilities{grid-column:1/-1;grid-template-columns:repeat(3,minmax(0,1fr))}.chat-capability>span{align-items:flex-start;flex-direction:column;gap:.2rem}.chat-workspace{height:100%;min-height:0}.chat-suggestions{grid-template-columns:1fr}.chat-conversation{padding:2rem 1rem}.chat-message{gap:.5rem}.chat-bubble{max-width:85%}.chat-composer{padding:1rem}}

/* Aurora Glass chat workspace */
.chat-page {
  background:
    radial-gradient(circle at 72% 10%, rgb(34 101 190 / 18%), transparent 34rem),
    linear-gradient(145deg, #090b10, #0d1017 48%, #090b10);
  color:#f5f5f7;
  grid-template-columns:20rem minmax(0,1fr);
  padding:10px 12px 12px;
}
.chat-sidebar {
  background:linear-gradient(145deg,rgb(44 48 59 / 68%),rgb(24 27 34 / 58%));
  border:1px solid rgb(255 255 255 / 11%);
  border-radius:22px;
  box-shadow:0 1px 1px rgb(255 255 255 / 8%) inset,0 22px 60px rgb(0 0 0 / 28%);
  gap:1rem;
  padding:1.3rem;
  -webkit-backdrop-filter:blur(30px) saturate(155%);
  backdrop-filter:blur(30px) saturate(155%);
}
.chat-brand { color:#aeb5c2; font-family:var(--font-display); font-weight:650; letter-spacing:0; }
.chat-model-select { color:#8f96a3; font-family:var(--font-display); font-size:.63rem; font-weight:650; letter-spacing:.08em; }
.chat-model-select select,.chat-tts-options select {
  background:rgb(255 255 255 / 7%);
  border:1px solid rgb(255 255 255 / 10%);
  border-radius:11px;
  color:#f5f5f7;
  min-height:38px;
}
.chat-model-control button {
  background:linear-gradient(180deg,#248fff,#0879e9);
  border-radius:11px;
  color:white;
  font-family:var(--font-display);
  padding:0 .75rem;
}
.chat-loaded-model {
  background:rgb(255 255 255 / 5%);
  border:1px solid rgb(255 255 255 / 8%);
  border-radius:13px;
  padding:.75rem;
}
.chat-loaded-model strong { color:#f5f5f7; font-family:var(--font-display); font-size:.68rem; }
.chat-loaded-model small { background:rgb(48 209 88 / 11%); border-color:rgb(48 209 88 / 22%); border-radius:999px; color:#57da71; }
.chat-settings,.chat-capabilities { border-color:rgb(255 255 255 / 9%); }
.chat-settings summary { color:#c5c9d1; font-family:var(--font-display); font-weight:600; }
.chat-capability {
  background:rgb(255 255 255 / 4%);
  border-color:rgb(255 255 255 / 8%);
  border-radius:12px;
  transition:background .18s ease,border-color .18s ease,transform .18s ease;
}
.chat-capability:hover { background:rgb(255 255 255 / 7%); transform:translateY(-1px); }
.chat-capability:has(input:checked) { background:rgb(10 132 255 / 9%); border-color:rgb(10 132 255 / 28%); }
.chat-capability input,.chat-settings input { accent-color:#0a84ff; }
.chat-tts-options { background:rgb(0 0 0 / 13%); border-color:rgb(255 255 255 / 8%); border-radius:0 0 12px 12px; }
.chat-clear {
  background:rgb(255 255 255 / 5%);
  border-color:rgb(255 255 255 / 9%);
  border-radius:12px;
  color:#f5f5f7;
  font-family:var(--font-display);
  font-weight:600;
}
.chat-workspace { margin-left:10px; }
.chat-topbar {
  background:linear-gradient(180deg,rgb(10 12 18 / 92%),rgb(10 12 18 / 60%));
  border:0;
  color:#9299a6;
  font-family:var(--font-display);
  padding:.8rem 1.4rem;
  -webkit-backdrop-filter:blur(18px);
  backdrop-filter:blur(18px);
}
.chat-conversation { padding:2rem max(4vw,2rem) 1rem; }
.chat-empty { margin-top:6vh; }
.chat-empty>span { color:#4da3ff; }
.chat-empty h2 { background:linear-gradient(150deg,#fff 25%,#9ca9bc); -webkit-background-clip:text; background-clip:text; color:transparent; }
.chat-empty h2 i { color:#8793a6; }
.chat-suggestions button {
  background:linear-gradient(145deg,rgb(255 255 255 / 8%),rgb(255 255 255 / 4%));
  border:1px solid rgb(255 255 255 / 10%);
  border-radius:15px;
  color:#e3e6eb;
  min-height:74px;
  transition:transform .2s var(--ease-spring),background .2s ease,border-color .2s ease;
}
.chat-suggestions button:hover { background:rgb(10 132 255 / 12%); border-color:rgb(10 132 255 / 32%); transform:translateY(-3px); }
.chat-avatar { background:rgb(255 255 255 / 9%); border-color:rgb(255 255 255 / 12%); box-shadow:0 8px 20px rgb(0 0 0 / 22%); }
.chat-message.assistant .chat-avatar { background:linear-gradient(145deg,#2f9aff,#6d65f4); border-color:rgb(255 255 255 / 20%); color:white; }
.chat-message.user .chat-avatar { background:linear-gradient(145deg,#454b58,#272b33); }
.chat-bubble {
  background:linear-gradient(145deg,rgb(48 53 65 / 78%),rgb(28 31 39 / 72%));
  border-color:rgb(255 255 255 / 10%);
  border-radius:19px 19px 19px 6px;
  box-shadow:0 14px 35px rgb(0 0 0 / 18%);
  -webkit-backdrop-filter:blur(18px);
  backdrop-filter:blur(18px);
}
.chat-message.user .chat-bubble { background:linear-gradient(145deg,#167fe9,#0869ce); border-color:rgb(102 181 255 / 34%); border-radius:19px 19px 6px 19px; }
.chat-bubble::after,.chat-message.user .chat-bubble::after { display:none; }
.chat-message-meta { border-color:rgb(255 255 255 / 10%); color:#929ba9; }
.chat-message.user .chat-message-meta { border-color:rgb(255 255 255 / 18%); color:#c9e2ff; }
.chat-thinking i { background:#4da3ff; }
.chat-error { background:rgb(255 69 58 / 15%); border:1px solid rgb(255 69 58 / 24%); border-radius:12px; color:#ffb4ae; }
.chat-camera { background:rgb(24 27 34 / 82%); border-color:rgb(255 255 255 / 14%); border-radius:18px; box-shadow:0 24px 70px rgb(0 0 0 / 48%); }
.chat-camera header { background:rgb(255 255 255 / 7%); }
.chat-scroll-latest { background:rgb(42 46 56 / 82%); border-color:rgb(255 255 255 / 14%); box-shadow:0 10px 30px rgb(0 0 0 / 28%); font-family:var(--font-display); -webkit-backdrop-filter:blur(18px); backdrop-filter:blur(18px); }
.chat-composer { background:linear-gradient(180deg,transparent,#090b10 32%); border:0; padding:1rem max(4vw,2rem) 1.15rem; }
.chat-input-row {
  background:rgb(39 43 52 / 76%);
  border-color:rgb(255 255 255 / 13%);
  border-radius:20px;
  box-shadow:0 1px 1px rgb(255 255 255 / 7%) inset,0 18px 48px rgb(0 0 0 / 32%);
  -webkit-backdrop-filter:blur(28px) saturate(160%);
  backdrop-filter:blur(28px) saturate(160%);
}
.chat-input-row:focus-within { border-color:rgb(41 151 255 / 58%); box-shadow:0 0 0 3px rgb(10 132 255 / 13%),0 20px 55px rgb(0 0 0 / 35%); }
.chat-send { background:linear-gradient(180deg,#2b98ff,#0877e7); color:white; box-shadow:0 7px 18px rgb(0 106 230 / 35%); }
.chat-send:not(:disabled):hover { box-shadow:0 10px 28px rgb(0 126 255 / 46%); transform:translateY(-2px) scale(1.04); }
.chat-send:disabled { background:#3a3e47; color:#777e8a; box-shadow:none; }
.chat-hint { color:#606775; font-family:var(--font-display); }
@media(max-width:800px){
  .chat-page{padding:7px;grid-template-columns:1fr;}
  .chat-sidebar{border:1px solid rgb(255 255 255 / 10%);border-radius:18px;padding:.85rem;}
  .chat-workspace{margin-left:0;}
}

/* Follow the global system appearance instead of forcing a dark workspace. */
.chat-page {
  background:
    radial-gradient(circle at 72% 8%, rgb(10 132 255 / 11%), transparent 34rem),
    radial-gradient(circle at 18% 88%, rgb(94 92 230 / 7%), transparent 30rem),
    var(--paper);
  color:var(--ink);
}
.chat-sidebar {
  background:color-mix(in srgb,var(--glass-strong) 88%,transparent);
  border-color:var(--glass-border);
  box-shadow:var(--shadow-glass);
}
.chat-brand,.chat-model-select,.chat-topbar,.chat-hint { color:var(--muted); }
.chat-loaded-model {
  background:color-mix(in srgb,var(--surface-solid) 62%,transparent);
  border-color:var(--line);
}
.chat-loaded-model>span,.chat-loaded-model.empty strong,.chat-loaded-model.empty small { color:var(--muted); }
.chat-loaded-model strong { color:var(--ink); }
.chat-loaded-model small {
  background:var(--signal-soft);
  border-color:color-mix(in srgb,var(--signal) 32%,var(--line));
  color:#248a3d;
}
.chat-loaded-model.empty small { background:transparent; border-color:var(--line); }
.chat-primary,.chat-model-control button {
  background:linear-gradient(180deg,#2997ff,#0878e8);
  color:white;
}
.chat-model-link { color:var(--accent); }
.chat-model-select select,.chat-tts-options select {
  background:color-mix(in srgb,var(--surface-solid) 78%,transparent);
  border-color:var(--line);
  color:var(--ink);
}
.chat-delete {
  border-color:color-mix(in srgb,var(--danger) 30%,var(--line));
  color:color-mix(in srgb,var(--danger) 78%,var(--ink));
}
.chat-settings,.chat-capabilities { border-color:var(--line); color:var(--muted); }
.chat-settings summary { color:var(--ink); }
.chat-capability {
  background:color-mix(in srgb,var(--surface-solid) 58%,transparent);
  border-color:var(--line);
  color:var(--ink);
}
.chat-capability:hover { background:color-mix(in srgb,var(--accent) 6%,var(--surface-solid)); }
.chat-capability:has(input:checked) {
  background:color-mix(in srgb,var(--accent) 9%,var(--surface-solid));
  border-color:rgb(10 132 255 / 30%);
}
.chat-capability small,.chat-tts-options label>span { color:var(--muted); }
.chat-tts-options {
  background:color-mix(in srgb,var(--surface-solid) 44%,transparent);
  border-color:var(--line);
}
.chat-clear {
  background:color-mix(in srgb,var(--surface-solid) 52%,transparent);
  border-color:var(--line);
  color:var(--ink);
}
.chat-topbar {
  background:color-mix(in srgb,var(--glass-strong) 82%,transparent);
  border-bottom:1px solid var(--line);
  color:var(--muted);
}
.chat-empty>span { color:var(--accent); }
.chat-empty h2 {
  background:linear-gradient(150deg,var(--ink) 20%,color-mix(in srgb,var(--ink) 34%,var(--muted)));
  -webkit-background-clip:text;
  background-clip:text;
  color:transparent;
}
.chat-empty h2 i { color:var(--muted); }
.chat-empty>p { color:color-mix(in srgb,var(--ink) 62%,var(--muted)); }
.chat-suggestions button {
  background:linear-gradient(145deg,color-mix(in srgb,var(--surface-solid) 88%,transparent),color-mix(in srgb,var(--surface-solid) 58%,transparent));
  border-color:var(--line);
  box-shadow:0 1px 0 rgb(255 255 255 / 44%) inset,0 10px 28px rgb(26 39 66 / 7%);
  color:var(--ink);
}
.chat-suggestions button:hover {
  background:color-mix(in srgb,var(--accent) 8%,var(--surface-solid));
  border-color:rgb(10 132 255 / 28%);
}
.chat-avatar {
  background:color-mix(in srgb,var(--surface-solid) 86%,transparent);
  border-color:var(--line);
  box-shadow:0 8px 20px rgb(28 39 60 / 10%);
  color:var(--ink);
}
.chat-message.user .chat-avatar { background:linear-gradient(145deg,#7f8796,#4f5663); color:white; }
.chat-bubble {
  background:color-mix(in srgb,var(--surface-solid) 86%,transparent);
  border-color:var(--line);
  box-shadow:0 14px 35px rgb(26 39 66 / 9%);
  color:var(--ink);
}
.chat-message.user .chat-bubble {
  background:linear-gradient(145deg,#218ef9,#0875df);
  border-color:rgb(0 105 220 / 24%);
  color:white;
}
.chat-message-meta { border-color:var(--line); color:var(--muted); }
.chat-message.user .chat-message-meta { border-color:rgb(255 255 255 / 24%); color:rgb(255 255 255 / 76%); }
.chat-message-images img { background:color-mix(in srgb,var(--ink) 4%,transparent); border-color:var(--line); }
.chat-error { background:rgb(255 69 58 / 10%); color:color-mix(in srgb,var(--danger) 72%,var(--ink)); }
.chat-asr-draft { color:var(--muted); }
.chat-camera {
  background:color-mix(in srgb,var(--surface-solid) 88%,transparent);
  border-color:var(--line);
  box-shadow:0 24px 70px rgb(24 34 52 / 22%);
  color:var(--ink);
}
.chat-camera header { background:color-mix(in srgb,var(--ink) 5%,transparent); }
.chat-camera header button { color:var(--ink); }
.chat-camera>small { color:var(--muted); }
.chat-scroll-latest {
  background:color-mix(in srgb,var(--glass-strong) 90%,transparent);
  border-color:var(--glass-border);
  box-shadow:0 10px 30px rgb(24 34 52 / 14%);
  color:var(--ink);
}
.chat-composer {
  background:linear-gradient(180deg,transparent,color-mix(in srgb,var(--paper) 96%,transparent) 32%);
}
.chat-input-row {
  background:color-mix(in srgb,var(--surface-solid) 84%,transparent);
  border-color:var(--line);
  box-shadow:0 1px 0 rgb(255 255 255 / 55%) inset,0 18px 48px rgb(25 39 67 / 11%);
}
.chat-input-row:focus-within { border-color:rgb(10 132 255 / 48%); box-shadow:0 0 0 3px rgb(10 132 255 / 12%),0 20px 48px rgb(25 39 67 / 12%); }
.chat-input-row textarea { color:var(--ink); }
.chat-input-row textarea::placeholder { color:color-mix(in srgb,var(--muted) 72%,transparent); }
.chat-attach small,.chat-hint { color:var(--muted); }
.chat-send:disabled { background:color-mix(in srgb,var(--muted) 28%,var(--surface-solid)); color:var(--muted); }
.chat-previews button { background:var(--surface-solid); border-color:var(--line); color:var(--ink); }

@media(max-width:800px){
  .chat-sidebar{border-color:var(--glass-border);}
}
</style>
