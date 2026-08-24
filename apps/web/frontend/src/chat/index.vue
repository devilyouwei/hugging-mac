<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue"

import { fetchChatModels, fetchLoadedChatModel, loadChatModel, streamChatMessage } from "./api"
import { ChatAsrSocket } from "./asrSocket"
import type { ChatAsrEvent } from "./asrSocket"
import ChatAudioPlayer from "./ChatAudioPlayer.vue"
import { mergeWavBlobs } from "./wav"
import { errorMessage, unloadModel } from "@/modelLifecycle"
import { fetchAsrModels, loadAsrModel } from "@/live_transcription/api"
import { fetchTtsModels, loadTtsModel, synthesizeSpeech } from "@/text_to_speech/api"
import type { AsrModel, LoadedAsrModel } from "@/live_transcription/types"
import type { LoadedTtsModel, TtsModel } from "@/text_to_speech/types"
import type { ChatModel, ConversationMessage, LoadedChatModel } from "./types"

const model = ref<LoadedChatModel | null>(null)
const models = ref<ChatModel[]>([])
const selectedModelId = ref("")
const selectedLlmModelId = ref("")
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
const asrEnabled = ref(false)
const ttsEnabled = ref(false)
const cameraEnabled = ref(false)
const asrModels = ref<AsrModel[]>([])
const selectedAsrModelId = ref("")
const selectedAsrVariant = ref("")
const loadedAsr = ref<LoadedAsrModel | null>(null)
const ttsModels = ref<TtsModel[]>([])
const selectedTtsModelId = ref("")
const loadedTts = ref<LoadedTtsModel | null>(null)
const voice = ref("af_heart")
const language = ref("en-us")
const asrDraft = ref("")
const video = ref<HTMLVideoElement | null>(null)
const cameraPosition = reactive({ x: 0, y: 72 })
const cameraSize = ref(300)
const conversation = ref<HTMLElement | null>(null)
const autoFollow = ref(true)
const isDraggingImages = ref(false)
const audioLevel = ref(0)
const wavePhase = ref(0)
let activeRequest: AbortController | null = null
let imageDragDepth = 0
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
let currentAudio: HTMLAudioElement | null = null
let currentAudioResolve: (() => void) | null = null
let playbackGeneration = 0
const ttsRequests = new Set<AbortController>()

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
  inferenceMs: number | null
}

const DEFAULT_ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
const DEFAULT_TTS_MODEL_ID = "hexgrad/kokoro"
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
const llmModels = computed(() =>
  Array.from(new Map(models.value.map((item) => [item.model_id, item])).values()),
)
const llmVariants = computed(() =>
  models.value.filter((item) => item.model_id === selectedLlmModelId.value),
)
const asrModel = computed(() =>
  asrModels.value.find((item) => item.model_id === selectedAsrModelId.value) ?? null,
)
const asrVariants = computed(() => asrModel.value?.variants ?? [])
const selectedAsrVariantInfo = computed(() =>
  asrVariants.value.find((item) => item.name === selectedAsrVariant.value) ?? null,
)
const ttsModel = computed(() =>
  ttsModels.value.find((item) => item.model_id === selectedTtsModelId.value) ?? null,
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
const voiceWaveBars = computed(() =>
  Array.from({ length: 68 }, (_, index) => {
    const center = 1 - Math.abs(index - 33.5) / 34
    const texture = .38 + Math.abs(Math.sin(index * 1.73 + wavePhase.value)) * .62
    return Math.max(4, 7 + audioLevel.value * 72 * center * texture)
  }),
)
const asrAvailable = computed(() => Boolean(selectedAsrVariantInfo.value?.available))
const ttsAvailable = computed(() => Boolean(ttsModel.value?.resource.runtimes?.some((item) => item.available)))
const ttsLanguages = computed(() =>
  selectedTtsModelId.value === DEFAULT_TTS_MODEL_ID
    ? (ttsModel.value?.languages ?? []).filter((item) => KOKORO_LANGUAGE_PREFIX[item])
    : (ttsModel.value?.languages ?? []),
)
const ttsVoices = computed(() => {
  if (selectedTtsModelId.value !== DEFAULT_TTS_MODEL_ID) return ttsModel.value?.voices ?? []
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
    const [chatModels, asrCatalog, ttsCatalog] = await Promise.all([
      fetchChatModels(),
      fetchAsrModels(),
      fetchTtsModels(),
    ])
    models.value = chatModels
    asrModels.value = asrCatalog
    ttsModels.value = ttsCatalog
    if (!asrCatalog.some((item) => item.model_id === selectedAsrModelId.value)) {
      selectedAsrModelId.value = asrCatalog.find((item) => item.model_id === DEFAULT_ASR_MODEL_ID)?.model_id
        ?? asrCatalog.find((item) => item.variants.some((variant) => variant.available))?.model_id
        ?? asrCatalog[0]?.model_id
        ?? ""
    }
    syncAsrVariant()
    if (!ttsCatalog.some((item) => item.model_id === selectedTtsModelId.value)) {
      selectedTtsModelId.value = ttsCatalog.find((item) => item.model_id === DEFAULT_TTS_MODEL_ID)?.model_id
        ?? ttsCatalog.find((item) => item.resource.runtimes?.some((runtime) => runtime.available))?.model_id
        ?? ttsCatalog[0]?.model_id
        ?? ""
    }
    syncTtsSelection()
    if (!asrAvailable.value) asrEnabled.value = false
    if (!ttsAvailable.value) ttsEnabled.value = false
    const readyAsr = asrModel.value?.ready_instances.find((item) =>
      item.variant === selectedAsrVariant.value && item.runtime === preferredRuntime(selectedAsrVariantInfo.value?.available_runtimes ?? []),
    )
    loadedAsr.value = readyAsr
      ? { instance_id: readyAsr.instance_id, model_id: selectedAsrModelId.value, variant: readyAsr.variant, runtime: readyAsr.runtime, device: "", state: "ready" }
      : null
    const preferredTtsRuntime = preferredRuntime(ttsModel.value?.resource.runtimes?.filter((item) => item.available).map((item) => item.runtime) ?? [])
    const readyTts = ttsModel.value?.ready_instances?.find((item) => item.runtime === preferredTtsRuntime)
    loadedTts.value = readyTts
      ? { instance_id: readyTts.instance_id, model_id: selectedTtsModelId.value, variant: ttsModel.value?.variant ?? "", runtime: readyTts.runtime, device: "", state: "ready" }
      : null
    if (!models.value.some((item) => item.profile_id === selectedModelId.value)) {
      selectedModelId.value = models.value.find(modelResourceAvailable)?.profile_id ?? ""
    }
    selectedLlmModelId.value = selectedProfile.value?.model_id ?? models.value[0]?.model_id ?? ""
    model.value = selectedModelId.value
      ? await fetchLoadedChatModel(selectedModelId.value)
      : null
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

function preferredRuntime(runtimes: string[]): string {
  const priority = ["coreml", "pytorch", "mlx"]
  return priority.find((runtime) => runtimes.includes(runtime)) ?? runtimes[0] ?? ""
}

function syncAsrVariant() {
  const variants = asrModel.value?.variants ?? []
  if (!variants.some((item) => item.name === selectedAsrVariant.value)) {
    selectedAsrVariant.value = variants.find((item) => item.available)?.name ?? variants[0]?.name ?? ""
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

async function changeLlmModel() {
  selectedModelId.value = llmVariants.value.find(modelResourceAvailable)?.profile_id
    ?? llmVariants.value[0]?.profile_id
    ?? ""
  await selectModel()
}

async function changeAsrModel() {
  if (asrEnabled.value) {
    asrEnabled.value = false
    await toggleAsr()
  }
  loadedAsr.value = null
  selectedAsrVariant.value = ""
  syncAsrVariant()
}

async function changeAsrVariant() {
  if (asrEnabled.value) {
    asrEnabled.value = false
    await toggleAsr()
  }
  loadedAsr.value = null
}

async function changeTtsModel() {
  if (ttsEnabled.value) {
    ttsEnabled.value = false
    await toggleTts()
  }
  loadedTts.value = null
  syncTtsSelection()
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

function addImages(files: Iterable<File>) {
  if (!supportsImages.value || loadingModel.value) return
  const acceptedTypes = new Set(["image/png", "image/jpeg", "image/webp"])
  for (const file of files) {
    if (images.value.length >= 4) break
    if (!acceptedTypes.has(file.type)) continue
    images.value.push({ file, url: URL.createObjectURL(file) })
  }
}

function selectImages(event: Event) {
  const input = event.target as HTMLInputElement
  addImages(Array.from(input.files ?? []))
  input.value = ""
}

function containsDraggedFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files")
}

function handleImageDragEnter(event: DragEvent) {
  if (!containsDraggedFiles(event) || !supportsImages.value || loadingModel.value) return
  event.preventDefault()
  imageDragDepth += 1
  isDraggingImages.value = true
}

function handleImageDragOver(event: DragEvent) {
  if (!containsDraggedFiles(event) || !supportsImages.value || loadingModel.value) return
  event.preventDefault()
  if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"
}

function handleImageDragLeave(event: DragEvent) {
  if (!containsDraggedFiles(event) || !isDraggingImages.value) return
  imageDragDepth = Math.max(0, imageDragDepth - 1)
  if (imageDragDepth === 0) isDraggingImages.value = false
}

function handleImageDrop(event: DragEvent) {
  if (!containsDraggedFiles(event)) return
  event.preventDefault()
  imageDragDepth = 0
  isDraggingImages.value = false
  addImages(Array.from(event.dataTransfer?.files ?? []))
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
          model_id: selectedTtsModelId.value,
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
  if (loadedTts.value?.state === "ready" && loadedTts.value.model_id === selectedTtsModelId.value) return loadedTts.value
  const runtime = preferredRuntime(ttsModel.value.resource.runtimes?.filter((item) => item.available).map((item) => item.runtime) ?? [])
  loadedTts.value = await loadTtsModel(
    selectedTtsModelId.value,
    ttsModel.value.variant,
    runtime,
  )
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
    const socket = asrSocket
    stopAsrCapture()
    await socket?.stop()
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
    const runtime = preferredRuntime(selectedAsrVariantInfo.value?.available_runtimes ?? [])
    if (
      loadedAsr.value?.model_id !== selectedAsrModelId.value
      || loadedAsr.value.variant !== selectedAsrVariant.value
      || loadedAsr.value.runtime !== runtime
    ) {
      loadedAsr.value = await loadAsrModel(selectedAsrModelId.value, selectedAsrVariant.value, runtime)
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: false, channelCount: 1 },
      video: false,
    })
    audioContext = new AudioContext({ latencyHint: "interactive" })
    await audioContext.resume()
    asrSocket = await ChatAsrSocket.connect(
      {
        instanceId: loadedAsr.value.instance_id,
        modelId: selectedAsrModelId.value,
        sampleRate: audioContext.sampleRate,
        streamingChunkSeconds: selectedAsrVariantInfo.value?.streaming_chunk_seconds ?? asrModel.value.streaming_chunk_seconds ?? 2.24,
      },
      handleAsrEvent,
      (caught) => { error.value = caught.message },
    )
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

async function toggleAsrFromComposer() {
  asrEnabled.value = !asrEnabled.value
  await toggleAsr()
}

function processAsrAudio(event: AudioProcessingEvent) {
  if (!asrEnabled.value || !asrSocket) return
  const samples = event.inputBuffer.getChannelData(0)
  let energy = 0
  for (let index = 0; index < samples.length; index += 1) energy += samples[index]! * samples[index]!
  const rms = Math.sqrt(energy / samples.length)
  const normalized = Math.min(1, Math.max(0, (rms - .008) * 9))
  audioLevel.value = audioLevel.value * .56 + normalized * .44
  wavePhase.value += .32 + normalized * .48
  asrSocket.send(samples)
}

function handleAsrEvent(event: ChatAsrEvent) {
  if (event.type === "speech_start") {
    asrDraft.value = "Listening…"
  } else if (event.type === "partial" && event.text?.trim()) {
    asrDraft.value = event.text
  } else if (event.type === "transcript") {
    asrDraft.value = ""
    const text = event.text?.trim() ?? ""
    if (/[\p{L}\p{N}]/u.test(text)) {
      const vadMs = event.vad_inference_ms ?? 0
      const inferenceMs = (event.preprocess_ms ?? 0) + (event.inference_ms ?? 0)
      void startTurn(text, [], "asr", {
        totalMs: vadMs + inferenceMs,
        vadMs,
        inferenceMs,
      })
    }
  } else if (event.type === "stopped") {
    asrDraft.value = ""
  }
}

function formatTime(value: number | null): string {
  if (value === null) return "—"
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

function stopAsr() {
  asrDraft.value = ""
  audioLevel.value = 0
  asrSocket?.close()
  asrSocket = null
  stopAsrCapture()
}

function stopAsrCapture() {
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
})
onBeforeUnmount(() => {
  imageDragDepth = 0
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

      <section class="chat-model-stack" aria-label="Conversation models">
        <article class="chat-model-card llm">
          <header><span class="model-glyph">✦</span><div><b>Language model</b><small>REASON & RESPOND</small></div><i :class="{ ready }">{{ ready ? "READY" : "OFF" }}</i></header>
          <label>MODEL
            <select v-model="selectedLlmModelId" :disabled="loadingModel" @change="changeLlmModel">
              <option v-for="item in llmModels" :key="item.model_id" :value="item.model_id">{{ item.model_id.split('/').pop() }}</option>
            </select>
          </label>
          <div class="model-variant-row">
            <label>VARIANT
              <select v-model="selectedModelId" :disabled="loadingModel" @change="selectModel">
                <option v-for="item in llmVariants" :key="item.profile_id" :value="item.profile_id" :disabled="!modelResourceAvailable(item)">{{ item.display_name }}</option>
              </select>
            </label>
            <button type="button" :disabled="loadingModel || (!ready && !selectedResourceAvailable)" @click="toggleSelectedModel">{{ loadingModel ? "…" : ready ? "UNLOAD" : "LOAD" }}</button>
          </div>
          <p v-if="loadedProfile"><span></span>{{ loadedProfile.display_name }}</p>
        </article>

        <article class="chat-model-card asr">
          <header><span class="model-glyph">◉</span><div><b>Speech to text</b><small>VOICE INPUT</small></div><i :class="{ ready: asrAvailable }">{{ asrAvailable ? "AVAILABLE" : "MISSING" }}</i></header>
          <label>MODEL
            <select v-model="selectedAsrModelId" @change="changeAsrModel">
              <option v-for="item in asrModels" :key="item.model_id" :value="item.model_id">{{ item.short_name }}</option>
            </select>
          </label>
          <label>VARIANT
            <select v-model="selectedAsrVariant" @change="changeAsrVariant">
              <option v-for="item in asrVariants" :key="item.name" :value="item.name" :disabled="!item.available">{{ item.display_name }}</option>
            </select>
          </label>
        </article>

        <article class="chat-model-card tts">
          <header><span class="model-glyph">♪</span><div><b>Text to speech</b><small>VOICE OUTPUT</small></div><label class="model-switch"><input v-model="ttsEnabled" type="checkbox" :disabled="!ttsAvailable" @change="toggleTts" /><span></span></label></header>
          <label>MODEL
            <select v-model="selectedTtsModelId" @change="changeTtsModel">
              <option v-for="item in ttsModels" :key="item.model_id" :value="item.model_id" :disabled="!item.resource.runtimes?.some((runtime) => runtime.available)">{{ item.short_name }}</option>
            </select>
          </label>
          <label>VARIANT
            <select :value="ttsModel?.variant" disabled><option>{{ ttsModel?.variant ?? "—" }}</option></select>
          </label>
          <div v-if="ttsEnabled && ttsAvailable && (ttsLanguages.length || ttsVoices.length)" class="chat-tts-options">
            <label v-if="ttsLanguages.length"><span>LANGUAGE</span><select v-model="language" aria-label="TTS language" @change="changeTtsLanguage"><option v-for="item in ttsLanguages" :key="item" :value="item">{{ LANGUAGE_LABELS[item] ?? item }}</option></select></label>
            <label v-if="ttsVoices.length"><span>VOICE</span><select v-model="voice" aria-label="TTS voice"><option v-for="item in ttsVoices" :key="item" :value="item">{{ voiceLabel(item) }}</option></select></label>
          </div>
        </article>
      </section>
      <small v-if="lifecycleMessage" :class="`lifecycle-${lifecycleMessage.type}`">{{ lifecycleMessage.text }}</small>
      <p v-if="!ready && !selectedResourceAvailable" class="chat-error">模型资源尚未下载，请先前往 Models 页面。</p>

      <details class="chat-settings">
        <summary>Generation settings</summary>
        <label>Max tokens <output>{{ maxTokens }}</output></label>
        <input v-model.number="maxTokens" type="range" min="64" max="2048" step="64" />
        <label>Temperature <output>{{ temperature.toFixed(1) }}</output></label>
        <input v-model.number="temperature" type="range" min="0" max="1.5" step="0.1" />
        <label class="chat-toggle"><input v-model="enableThinking" type="checkbox" /> Thinking mode</label>
      </details>
      <section class="chat-capabilities" aria-label="Conversation capabilities">
        <label class="chat-capability" :class="{ unavailable: !supportsImages }">
          <input v-model="cameraEnabled" type="checkbox" :disabled="!supportsImages" @change="toggleCamera" />
          <span><b>CAMERA</b><small>448 × 448 FRAME</small></span>
        </label>
      </section>
      <button class="chat-clear" :disabled="!messages.length" @click="resetConversation">
        ＋ New conversation
      </button>
    </aside>

    <main
      class="chat-workspace"
      :class="{ 'is-dragging-images': isDraggingImages }"
      @dragenter="handleImageDragEnter"
      @dragover="handleImageDragOver"
      @dragleave="handleImageDragLeave"
      @drop="handleImageDrop"
    >
      <header class="chat-topbar">
        <div><span class="status-dot"></span> ON-DEVICE SESSION</div>
        <span>{{ messages.length }} MESSAGES</span>
      </header>

      <div v-if="isDraggingImages" class="chat-drop-overlay" aria-hidden="true">
        <span>＋</span>
        <strong>拖放图片到这里</strong>
        <small>PNG、JPEG 或 WEBP · 最多 4 张</small>
      </div>

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
      <footer class="chat-composer" :class="{ 'is-listening': asrEnabled }">
        <div v-if="images.length" class="chat-previews">
          <figure v-for="(image, index) in images" :key="image.url">
            <img :src="image.url" :alt="image.file.name" />
            <button aria-label="移除图片" @click="removeImage(index)">×</button>
          </figure>
        </div>
        <div class="chat-input-row">
          <div
            v-if="asrEnabled"
            class="chat-voice-wave"
            :style="{ '--voice-energy': audioLevel.toFixed(3) }"
            aria-hidden="true"
          >
            <div class="chat-voice-glow"></div>
            <span
              v-for="(height, index) in voiceWaveBars"
              :key="index"
              :style="{ '--wave-height': `${height}px`, '--wave-delay': `${index * -18}ms` }"
            ></span>
          </div>
          <label v-if="supportsImages" class="chat-attach" :class="{ disabled: loadingModel || images.length >= 4 }">
            <input type="file" accept="image/png,image/jpeg,image/webp" multiple :disabled="loadingModel || images.length >= 4" @change="selectImages" />
            <span>＋</span><small>IMAGE</small>
          </label>
          <textarea v-model="prompt" rows="1" maxlength="32000" :disabled="loadingModel" placeholder="输入消息…（新消息会打断当前回复）" @keydown="handleKeydown"></textarea>
          <button
            class="chat-mic"
            :class="{ active: asrEnabled }"
            type="button"
            :disabled="!asrAvailable"
            :aria-label="asrEnabled ? '关闭语音输入' : '开启语音输入'"
            :aria-pressed="asrEnabled"
            @click="toggleAsrFromComposer"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 15.25a3.75 3.75 0 0 0 3.75-3.75v-5a3.75 3.75 0 0 0-7.5 0v5A3.75 3.75 0 0 0 12 15.25Z" />
              <path d="M5.75 11.25v.25a6.25 6.25 0 0 0 12.5 0v-.25M12 17.75v3M9.25 20.75h5.5" />
            </svg>
          </button>
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
.chat-composer { background:linear-gradient(180deg,#0d0e0d00,#0d0e0d 25%); border-top:1px solid #292c27; isolation:isolate; padding:1rem max(4vw,2rem) 1.4rem; position:relative; }.chat-input-row { align-items:center; background:#191c17; border:1px solid #3b4037; border-radius:1.35rem; box-shadow:0 .7rem 2rem #0003; display:grid; grid-template-columns:auto minmax(0,1fr) auto auto; padding:.5rem; position:relative; transition:border-color .2s,box-shadow .2s; z-index:2; }.chat-input-row:focus-within { border-color:#75894e; box-shadow:0 0 0 3px #b8f23c0c,0 .7rem 2rem #0003; }.chat-attach { align-items:center; cursor:pointer; display:flex; gap:.35rem; padding:.6rem; }.chat-attach input { display:none; }.chat-attach span { font-size:1.4rem; }.chat-attach small { color:#8c8d85; font:.52rem var(--font-mono); }.chat-attach.disabled { opacity:.35; }.chat-input-row textarea { align-self:center; background:none; border:0; box-sizing:border-box; color:#f4f2e9; field-sizing:content; font:1rem/1.35 var(--font-display); max-height:10rem; min-height:2.7rem; outline:0; overflow-y:auto; padding:.675rem .7rem; resize:none; }.chat-send,.chat-mic { align-items:center; border:0; border-radius:50%; cursor:pointer; display:flex; height:2.7rem; justify-content:center; transition:transform .18s,box-shadow .22s,background .22s,color .22s; width:2.7rem; }.chat-send { background:var(--signal); color:#111; font-size:1.4rem; }.chat-send:not(:disabled):hover { box-shadow:0 0 1rem #b8f23c55; transform:translateY(-2px); }.chat-send:disabled { background:#42443e; color:#85877f; cursor:default; }.chat-mic { background:transparent; color:#8d94a1; margin-right:.28rem; }.chat-mic svg { fill:currentColor; height:1.25rem; overflow:visible; stroke:currentColor; stroke-linecap:round; stroke-linejoin:round; stroke-width:1.55; width:1.25rem; }.chat-mic svg path:first-child { stroke:none; }.chat-mic:hover:not(:disabled) { background:rgb(98 130 255 / 12%); color:#8eb9ff; }.chat-mic.active { animation:mic-breathe 1.8s ease-in-out infinite; background:linear-gradient(145deg,#7b61ff,#0aa8ff); box-shadow:0 0 0 1px rgb(161 206 255 / 32%) inset,0 0 18px rgb(53 133 255 / 62%),0 0 38px rgb(136 72 255 / 30%); color:white; }.chat-mic:disabled { cursor:not-allowed; opacity:.3; }.chat-hint { color:#62645d; display:block; font-size:.5rem; letter-spacing:.08em; margin-top:.55rem; position:relative; text-align:center; z-index:2; }
.chat-voice-wave { align-items:flex-end; display:flex; gap:clamp(2px,.3vw,6px); height:8.5rem; justify-content:center; left:7%; overflow:visible; padding:0; pointer-events:none; position:absolute; right:7%; top:1px; transform:translateY(-100%); z-index:3; }.chat-voice-wave::after { background:linear-gradient(90deg,transparent,rgb(99 187 255 / 55%),rgb(153 93 255 / 50%),transparent); border-radius:50%; bottom:-.18rem; box-shadow:0 0 12px rgb(86 174 255 / 72%),0 0 28px rgb(131 79 255 / 45%); content:""; height:2px; left:5%; position:absolute; right:5%; }.chat-voice-wave>span { animation:wave-shimmer 1.25s ease-in-out infinite alternate; animation-delay:var(--wave-delay); background:linear-gradient(180deg,#d9f6ff 0%,#69c8ff 30%,#8b6cff 68%,#ff5ec9 100%); border-radius:999px; box-shadow:0 0 8px rgb(105 202 255 / 82%),0 0 18px rgb(126 89 255 / 44%); height:var(--wave-height); max-height:7.5rem; min-height:5px; opacity:.9; position:relative; transform-origin:bottom; transition:height 90ms linear; width:clamp(2px,.22vw,4px); z-index:2; }.chat-voice-glow { background:radial-gradient(ellipse at center,rgb(65 174 255 / 38%),rgb(132 70 255 / 23%) 35%,rgb(255 72 194 / 12%) 55%,transparent 76%); bottom:-2.15rem; filter:blur(16px); height:9.5rem; left:2%; position:absolute; right:2%; transform:scaleY(calc(.72 + var(--voice-energy, .2))); z-index:0; }.chat-composer.is-listening .chat-input-row { border-color:rgb(91 155 255 / 46%); box-shadow:0 1px 0 rgb(255 255 255 / 10%) inset,0 -8px 28px rgb(91 84 255 / 12%),0 0 34px rgb(64 145 255 / 22%),0 18px 48px rgb(0 0 0 / 32%); }
.chat-drop-overlay { align-items:center; background:rgb(8 14 24 / 78%); border:2px dashed rgb(77 163 255 / 72%); border-radius:20px; display:flex; flex-direction:column; inset:4.2rem 1.25rem 1.25rem; justify-content:center; pointer-events:none; position:absolute; z-index:30; -webkit-backdrop-filter:blur(12px); backdrop-filter:blur(12px); }.chat-drop-overlay span { color:#4da3ff; font-size:3rem; line-height:1; }.chat-drop-overlay strong { font-size:1.05rem; margin-top:.65rem; }.chat-drop-overlay small { color:#aab2c0; font:.58rem var(--font-mono); margin-top:.45rem; }
.chat-previews { display:flex; gap:.6rem; margin-bottom:.7rem; }.chat-previews figure { height:4.5rem; margin:0; position:relative; width:4.5rem; }.chat-previews img { border-radius:.4rem; height:100%; object-fit:cover; width:100%; }.chat-previews button { background:#111; border:1px solid #555; border-radius:50%; color:white; cursor:pointer; height:1.3rem; position:absolute; right:-.3rem; top:-.3rem; width:1.3rem; }
@keyframes bubble-in-left { from { opacity:0; transform:translate(-.7rem,.4rem) scale(.98); } to { opacity:1; transform:none; } }
@keyframes bubble-in-right { from { opacity:0; transform:translate(.7rem,.4rem) scale(.98); } to { opacity:1; transform:none; } }
@keyframes wave-shimmer { from { filter:saturate(.9) brightness(.88); opacity:.68; } to { filter:saturate(1.18) brightness(1.18); opacity:1; } }
@keyframes mic-breathe { 50% { box-shadow:0 0 0 1px rgb(200 230 255 / 48%) inset,0 0 25px rgb(53 157 255 / 72%),0 0 48px rgb(151 70 255 / 38%); } }
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
  background:transparent;
  border-bottom:0;
  color:var(--muted);
  -webkit-backdrop-filter:none;
  backdrop-filter:none;
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

/* Compact model studio sidebar */
.chat-page { grid-template-columns:23rem minmax(0,1fr); }
.chat-sidebar { gap:.85rem; padding:1.15rem; scrollbar-width:thin; }
.chat-model-stack { display:grid; gap:.62rem; }
.chat-model-card { background:color-mix(in srgb,var(--surface-solid) 64%,transparent); border:1px solid var(--line); border-radius:15px; box-shadow:0 1px 0 rgb(255 255 255 / 44%) inset,0 8px 24px rgb(25 39 67 / 5%); display:grid; gap:.58rem; padding:.72rem; transition:border-color .2s,box-shadow .2s,transform .2s; }
.chat-model-card:hover { border-color:color-mix(in srgb,var(--accent) 24%,var(--line)); box-shadow:0 1px 0 rgb(255 255 255 / 48%) inset,0 12px 30px rgb(25 39 67 / 8%); transform:translateY(-1px); }
.chat-model-card>header { align-items:center; display:grid; gap:.55rem; grid-template-columns:auto minmax(0,1fr) auto; }
.model-glyph { align-items:center; background:color-mix(in srgb,var(--accent) 10%,var(--surface-solid)); border:1px solid color-mix(in srgb,var(--accent) 18%,var(--line)); border-radius:9px; color:var(--accent); display:flex; font-size:.82rem; height:1.85rem; justify-content:center; width:1.85rem; }
.chat-model-card.asr .model-glyph { background:rgb(94 92 230 / 10%); border-color:rgb(94 92 230 / 20%); color:#7775f5; }
.chat-model-card.tts .model-glyph { background:rgb(175 82 222 / 10%); border-color:rgb(175 82 222 / 20%); color:#b65ce3; }
.chat-model-card header div { display:grid; gap:.08rem; min-width:0; }
.chat-model-card header b { color:var(--ink); font-size:.68rem; font-weight:700; }
.chat-model-card header small { color:var(--muted); font:.43rem var(--font-mono); letter-spacing:.09em; }
.chat-model-card header i { background:color-mix(in srgb,var(--muted) 8%,transparent); border-radius:999px; color:var(--muted); font:700 .4rem var(--font-mono); font-style:normal; letter-spacing:.05em; padding:.27rem .4rem; }
.chat-model-card header i.ready { background:rgb(48 209 88 / 10%); color:#2aa647; }
.chat-model-card>label,.model-variant-row label { color:var(--muted); display:grid; font:650 .43rem var(--font-mono); gap:.27rem; letter-spacing:.08em; min-width:0; }
.chat-model-card select { appearance:none; background:color-mix(in srgb,var(--surface-solid) 86%,transparent) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='m1 1 4 4 4-4' fill='none' stroke='%238b93a1' stroke-width='1.4'/%3E%3C/svg%3E") no-repeat right .55rem center; border:1px solid var(--line); border-radius:9px; color:var(--ink); font:600 .58rem var(--font-display); height:2rem; min-width:0; padding:0 1.55rem 0 .58rem; text-overflow:ellipsis; width:100%; }
.chat-model-card select:focus { border-color:rgb(10 132 255 / 48%); box-shadow:0 0 0 3px rgb(10 132 255 / 9%); outline:0; }
.chat-model-card select:disabled { cursor:not-allowed; opacity:.58; }
.model-variant-row { display:grid; gap:.45rem; grid-template-columns:minmax(0,1fr) auto; }
.model-variant-row button { align-self:end; background:linear-gradient(180deg,#2997ff,#0878e8); border:0; border-radius:9px; color:white; cursor:pointer; font:700 .46rem var(--font-mono); height:2rem; padding:0 .65rem; }
.model-variant-row button:disabled { cursor:not-allowed; filter:saturate(.2); opacity:.5; }
.chat-model-card>p { align-items:center; color:var(--muted); display:flex; font:.48rem var(--font-mono); gap:.35rem; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.chat-model-card>p span { background:#30d158; border-radius:50%; box-shadow:0 0 8px rgb(48 209 88 / 55%); flex:0 0 auto; height:.38rem; width:.38rem; }
.model-switch { cursor:pointer; display:block!important; }
.model-switch input { clip:rect(0 0 0 0); position:absolute; }
.model-switch span { background:color-mix(in srgb,var(--muted) 25%,var(--surface-solid)); border-radius:999px; display:block; height:1.15rem; position:relative; transition:background .2s; width:2rem; }
.model-switch span::after { background:white; border-radius:50%; box-shadow:0 1px 4px rgb(0 0 0 / 22%); content:""; height:.85rem; left:.15rem; position:absolute; top:.15rem; transition:transform .2s var(--ease-spring); width:.85rem; }
.model-switch input:checked+span { background:linear-gradient(90deg,#5e5ce6,#af52de); }
.model-switch input:checked+span::after { transform:translateX(.85rem); }
.chat-model-card .chat-tts-options { border-radius:10px; grid-template-columns:repeat(2,minmax(0,1fr)); padding:.5rem; }
.chat-model-card .chat-tts-options select { font-size:.52rem; height:1.85rem; }
.chat-settings { margin-top:.1rem; }
.chat-capabilities { padding-top:.75rem; }

@media(max-width:800px){
  .chat-page{grid-template-columns:1fr;}
  .chat-model-stack{grid-column:1/-1;grid-template-columns:repeat(3,minmax(0,1fr));}
  .chat-model-card .chat-tts-options{grid-template-columns:1fr;}
}
@media(max-width:620px){.chat-model-stack{grid-template-columns:1fr;}}
</style>
