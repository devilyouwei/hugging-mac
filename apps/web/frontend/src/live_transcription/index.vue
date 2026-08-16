<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue"

import {
  cancelTranscriptionStream,
  detectVoiceActivity,
  fetchAsrModels,
  fetchPipelineComponents,
  loadAsrModel,
  loadEnhancementModel,
  loadVadModel,
  finishTranscriptionStream,
  startTranscriptionStream,
  transcribeStreamChunk,
  transcribeUtterance,
} from "./api"
import { errorMessage, unloadModel } from "@/modelLifecycle"
import type {
  AsrModel,
  LoadedAsrModel,
  PipelineComponent,
  StreamingSession,
  TranscriptSegment,
} from "./types"
import { encodeWave } from "./wav"

const SILENCE_SECONDS = 0.65
const MIN_UTTERANCE_SECONDS = 0.35
const MAX_UTTERANCE_SECONDS = 25
const PRE_ROLL_SECONDS = 0.18
const VAD_WINDOW_SECONDS = 0.18
const ENERGY_THRESHOLD_MIN = 0.006
const ENERGY_THRESHOLD_MAX = 0.035

const models = ref<AsrModel[]>([])
const selectedModelId = ref("")
const selectedRuntimeByModel = ref<Record<string, string>>({})
const loadedModels = ref<Record<string, LoadedAsrModel>>({})
const loadedVadModel = ref<LoadedAsrModel | null>(null)
const loadedEnhancementModel = ref<LoadedAsrModel | null>(null)
const pipelineComponents = ref<PipelineComponent[]>([])
const componentIssues = ref<Record<string, string>>({})
const loadingModel = ref(false)
const listening = ref(false)
const speaking = ref(false)
const inputLevel = ref(0)
const sensitivity = ref(0.65)
const elapsedSeconds = ref(0)
const segments = ref<TranscriptSegment[]>([])
const pendingCount = ref(0)
const error = ref("")
const lifecycleMessage = ref<{ type: "success" | "error"; text: string } | null>(null)
const transcriptEnd = ref<HTMLElement | null>(null)
const streamSession = ref<StreamingSession | null>(null)
const streamingText = ref("")
const streamingCommittedText = ref("")
const streamingStatus = ref<"idle" | "streaming" | "processing" | "complete" | "error">("idle")
const streamingInferenceMs = ref<number | null>(null)
const streamingAudioSeconds = ref(0)
const streamingCommittedAudioSeconds = ref(0)
const streamingLanguage = ref<string | null>(null)
const streamingTokens = ref(0)
const streamingCommittedTokens = ref(0)

let mediaStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let sourceNode: MediaStreamAudioSourceNode | null = null
let processorNode: ScriptProcessorNode | null = null
let silentGain: GainNode | null = null
let currentChunks: Float32Array[] = []
let currentSamples = 0
let silenceSamples = 0
let preRoll: Float32Array[] = []
let preRollSamples = 0
let vadChunks: Float32Array[] = []
let vadSamples = 0
let sessionStartedAt = 0
let elapsedTimer: number | null = null
let segmentSequence = 0
let recognitionQueue = Promise.resolve()
let vadQueue = Promise.resolve()
let vadGeneration = 0
let streamChunks: Float32Array[] = []
let streamSamples = 0
let streamQueue = Promise.resolve()
let streamGeneration = 0
const requests = new Set<AbortController>()

const selectedModel = computed(() =>
  models.value.find((model) => model.model_id === selectedModelId.value) ?? null,
)
const runtimeKey = (modelId: string, runtime: string) => `${modelId}::${runtime}`
const runtimeForModel = (model: AsrModel) =>
  selectedRuntimeByModel.value[model.model_id]
  ?? (model.resource.runtimes.some((item) => item.runtime === "coreml") ? "coreml" : model.runtime)
const runtimeAvailable = (model: AsrModel, runtime = runtimeForModel(model)) =>
  Boolean(model.resource.runtimes.find((item) => item.runtime === runtime)?.available)
const loadedInstanceFor = (model: AsrModel) =>
  loadedModels.value[runtimeKey(model.model_id, runtimeForModel(model))] ?? null
const modelIsReady = (model: AsrModel) => loadedInstanceFor(model)?.state === "ready"
const runtimeOptions = (model: AsrModel) => [...model.resource.runtimes].sort((left, right) =>
  left.runtime === "coreml" ? -1 : right.runtime === "coreml" ? 1 : 0,
)
const runtimeLabel = (runtime: string) => runtime === "pytorch-mps"
  ? "Torch MPS"
  : runtime === "coreml" ? "Core ML" : runtime
const selectedLoadedModel = computed(() =>
  selectedModel.value ? loadedInstanceFor(selectedModel.value) : null,
)
const modelReady = computed(() =>
  Boolean(
    selectedLoadedModel.value
      && selectedLoadedModel.value.model_id === selectedModelId.value
      && selectedLoadedModel.value.state === "ready",
  ),
)
const streamingMode = computed(() => Boolean(selectedModel.value?.streaming))
const selectedArtifactAvailable = computed(() => {
  const model = selectedModel.value
  if (!model) return false
  return runtimeAvailable(model)
})
const threshold = computed(() => 0.75 - sensitivity.value * 0.5)
const energyThreshold = computed(() =>
  ENERGY_THRESHOLD_MAX
  - sensitivity.value * (ENERGY_THRESHOLD_MAX - ENERGY_THRESHOLD_MIN),
)
const vadComponent = computed(() =>
  pipelineComponents.value.find((component) => component.component_id === "vad") ?? null,
)
const enhancementComponent = computed(() =>
  pipelineComponents.value.find((component) => component.component_id === "enhancement") ?? null,
)
const componentState = (component: PipelineComponent) => {
  if (component.component_id === "vad" && loadedVadModel.value?.state === "ready") return "loaded"
  if (component.component_id === "enhancement" && loadedEnhancementModel.value?.state === "ready") return "loaded"
  return component.downloaded ? "not-loaded" : "not-downloaded"
}
const componentStateLabel = (component: PipelineComponent) =>
  componentState(component).replace("-", " ").toUpperCase()
const transcriptText = computed(() =>
  streamingMode.value
    ? streamingText.value
    : segments.value
    .filter((segment) => segment.status === "complete" && segment.text)
    .map((segment) => segment.text)
    .join(" "),
)

async function loadModels() {
  try {
    models.value = await fetchAsrModels()
    for (const model of models.value) {
      selectedRuntimeByModel.value[model.model_id] = runtimeForModel(model)
      for (const ready of model.ready_instances ?? []) {
        loadedModels.value[runtimeKey(model.model_id, ready.runtime)] = {
          instance_id: ready.instance_id,
          model_id: model.model_id,
          variant: model.variant,
          runtime: ready.runtime,
          device: ready.runtime,
          state: "ready",
        }
      }
    }
    selectedModelId.value ||= models.value[0]?.model_id ?? ""
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Failed to load the ASR model list"
  }
  await refreshPipelineComponents()
}

async function refreshPipelineComponents() {
  try {
    pipelineComponents.value = await fetchPipelineComponents()
    const vad = pipelineComponents.value.find((component) => component.component_id === "vad")
    const enhancement = pipelineComponents.value.find(
      (component) => component.component_id === "enhancement",
    )
    loadedVadModel.value = vad?.loaded_model ?? null
    loadedEnhancementModel.value = enhancement?.loaded_model ?? null
    delete componentIssues.value.status
  } catch (caught) {
    componentIssues.value.status = errorMessage(caught, "Component status is unavailable")
  }
}

async function ensureSelectedModelLoaded(): Promise<LoadedAsrModel> {
  if (modelReady.value && selectedLoadedModel.value) return selectedLoadedModel.value
  if (!selectedModel.value) throw new Error("Select an ASR model first")
  if (!selectedArtifactAvailable.value) {
    throw new Error("Model assets are unavailable. Download or convert them on the Models page.")
  }
  error.value = ""
  loadingModel.value = true
  lifecycleMessage.value = null
  try {
    const runtime = runtimeForModel(selectedModel.value)
    const loaded = await loadAsrModel(selectedModel.value.model_id, runtime)
    loadedModels.value[runtimeKey(selectedModel.value.model_id, runtime)] = loaded
    return loaded
  } catch (caught) {
    const message = errorMessage(caught, "Failed to load the ASR model")
    lifecycleMessage.value = { type: "error", text: message }
    throw caught
  } finally {
    loadingModel.value = false
  }
}

async function ensureVadModelLoaded(): Promise<LoadedAsrModel | null> {
  if (loadedVadModel.value?.state === "ready") return loadedVadModel.value
  if (!vadComponent.value?.downloaded) return null
  try {
    loadedVadModel.value = await loadVadModel()
    vadComponent.value.loaded_model = loadedVadModel.value
    delete componentIssues.value.vad
    return loadedVadModel.value
  } catch (caught) {
    componentIssues.value.vad = errorMessage(
      caught,
      "Silero VAD could not be loaded; using the browser energy gate",
    )
    return null
  }
}

async function ensureEnhancementModelLoaded(): Promise<LoadedAsrModel | null> {
  if (loadedEnhancementModel.value?.state === "ready") return loadedEnhancementModel.value
  if (!enhancementComponent.value?.downloaded) return null
  try {
    loadedEnhancementModel.value = await loadEnhancementModel()
    enhancementComponent.value.loaded_model = loadedEnhancementModel.value
    delete componentIssues.value.enhancement
    return loadedEnhancementModel.value
  } catch (caught) {
    componentIssues.value.enhancement = errorMessage(
      caught,
      "DeepFilterNet3 could not be loaded; sending raw audio to ASR",
    )
    return null
  }
}

async function toggleSelectedModel() {
  if (loadingModel.value) return
  if (!modelReady.value || !selectedLoadedModel.value) {
    try { await ensureSelectedModelLoaded() } catch { /* inline feedback */ }
    return
  }
  loadingModel.value = true
  lifecycleMessage.value = null
  try {
    const instance = selectedLoadedModel.value
    await unloadModel(instance.instance_id)
    delete loadedModels.value[runtimeKey(instance.model_id, instance.runtime)]
    lifecycleMessage.value = { type: "success", text: "Model unloaded" }
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "Failed to unload model") }
  } finally {
    loadingModel.value = false
  }
}

async function toggleModel(model: AsrModel) {
  selectModel(model.model_id)
  await toggleSelectedModel()
}

function selectRuntime(model: AsrModel, runtime: string) {
  if (listening.value || pendingCount.value || loadingModel.value) return
  selectedRuntimeByModel.value[model.model_id] = runtime
  if (selectedModelId.value === model.model_id) lifecycleMessage.value = null
}

function onRuntimeChange(model: AsrModel, event: Event) {
  selectRuntime(model, (event.target as HTMLSelectElement).value)
}

function selectModel(modelId: string) {
  if (listening.value || pendingCount.value) return
  stopAudioGraph()
  selectedModelId.value = modelId
  lifecycleMessage.value = null
  error.value = ""
}

async function startListening() {
  if (listening.value) return
  try {
    await ensureSelectedModelLoaded()
  } catch (caught) {
    error.value = errorMessage(caught, "Failed to load the ASR model")
    return
  }
  if (streamingMode.value) {
    await ensureVadModelLoaded()
  } else {
    await ensureVadModelLoaded()
    await ensureEnhancementModelLoaded()
  }
  error.value = ""
  try {
    if (streamingMode.value) {
      streamGeneration += 1
      streamSession.value = null
      streamingText.value = ""
      streamingCommittedText.value = ""
      streamingInferenceMs.value = null
      streamingAudioSeconds.value = 0
      streamingCommittedAudioSeconds.value = 0
      streamingLanguage.value = null
      streamingTokens.value = 0
      streamingCommittedTokens.value = 0
      streamingStatus.value = "streaming"
      streamChunks = []
      streamSamples = 0
      streamQueue = Promise.resolve()
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: true,
        echoCancellation: true,
        // Audio preprocessing is explicit in the app pipeline. Nemotron receives
        // raw audio, while utterance-based ASR may use DeepFilterNet3.
        noiseSuppression: false,
        channelCount: 1,
      },
      video: false,
    })
    audioContext = new AudioContext()
    sourceNode = audioContext.createMediaStreamSource(mediaStream)
    processorNode = audioContext.createScriptProcessor(4096, 1, 1)
    silentGain = audioContext.createGain()
    silentGain.gain.value = 0
    processorNode.onaudioprocess = handleAudio
    sourceNode.connect(processorNode)
    processorNode.connect(silentGain)
    silentGain.connect(audioContext.destination)
    vadGeneration += 1
    listening.value = true
    sessionStartedAt = performance.now()
    elapsedTimer = window.setInterval(() => {
      elapsedSeconds.value = (performance.now() - sessionStartedAt) / 1000
    }, 250)
  } catch (caught) {
    stopAudioGraph()
    await cancelActiveStream()
    error.value = caught instanceof DOMException && caught.name === "NotAllowedError"
      ? "Microphone permission was denied. Allow access in the browser site settings."
      : caught instanceof Error ? caught.message : "Failed to start the microphone"
  }
}

function handleAudio(event: AudioProcessingEvent) {
  if (!listening.value || !audioContext) return
  const input = event.inputBuffer.getChannelData(0)
  const chunk = new Float32Array(input)
  let energy = 0
  for (const sample of chunk) energy += sample * sample
  const rms = Math.sqrt(energy / chunk.length)
  inputLevel.value = Math.min(1, rms / 0.12)
  if (streamingMode.value) {
    streamChunks.push(chunk)
    streamSamples += chunk.length
    if (!speaking.value) {
      const preRollLimit = audioContext.sampleRate * PRE_ROLL_SECONDS
      while (streamSamples > preRollLimit && streamChunks.length > 1) {
        streamSamples -= streamChunks.shift()!.length
      }
    } else if (
      streamSamples / audioContext.sampleRate
      >= (selectedModel.value?.streaming_chunk_seconds ?? 2.24)
    ) {
      flushStreamingAudio(audioContext.sampleRate)
    }

    if (loadedVadModel.value) {
      vadChunks.push(chunk)
      vadSamples += chunk.length
      if (vadSamples / audioContext.sampleRate >= VAD_WINDOW_SECONDS) {
        const analysisChunks = vadChunks
        const analysisSamples = vadSamples
        const sampleRate = audioContext.sampleRate
        vadChunks = []
        vadSamples = 0
        enqueueVadAnalysis(analysisChunks, analysisSamples, sampleRate)
      }
    } else {
      applyVadResult(rms >= energyThreshold.value, chunk.length, audioContext.sampleRate)
    }
    return
  }
  if (speaking.value) {
    currentChunks.push(chunk)
    currentSamples += chunk.length
  } else {
    preRoll.push(chunk)
    preRollSamples += chunk.length
    const preRollLimit = audioContext.sampleRate * PRE_ROLL_SECONDS
    while (preRollSamples > preRollLimit && preRoll.length > 1) {
      preRollSamples -= preRoll.shift()!.length
    }
  }

  if (loadedVadModel.value) {
    vadChunks.push(chunk)
    vadSamples += chunk.length
    if (vadSamples / audioContext.sampleRate >= VAD_WINDOW_SECONDS) {
      const analysisChunks = vadChunks
      const analysisSamples = vadSamples
      const sampleRate = audioContext.sampleRate
      vadChunks = []
      vadSamples = 0
      enqueueVadAnalysis(analysisChunks, analysisSamples, sampleRate)
    }
  } else {
    applyVadResult(rms >= energyThreshold.value, chunk.length, audioContext.sampleRate)
  }

  const duration = currentSamples / audioContext.sampleRate
  if (speaking.value && duration >= MAX_UTTERANCE_SECONDS) {
    finishUtterance(audioContext.sampleRate)
  }
}

function enqueueVadAnalysis(
  chunks: Float32Array[],
  analyzedSamples: number,
  sampleRate: number,
) {
  const instance = loadedVadModel.value
  if (!instance) return
  const generation = vadGeneration
  const audio = encodeWave(chunks, sampleRate)
  vadQueue = vadQueue.then(async () => {
    if (!listening.value || generation !== vadGeneration) return
    const controller = new AbortController()
    requests.add(controller)
    try {
      const result = await detectVoiceActivity(
        audio,
        instance.instance_id,
        threshold.value,
        controller.signal,
      )
      if (generation === vadGeneration) {
        applyVadResult(result.voiced, analyzedSamples, sampleRate)
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      componentIssues.value.vad = errorMessage(
        caught,
        "Silero VAD inference failed; switched to the browser energy gate",
      )
      loadedVadModel.value = null
    } finally {
      requests.delete(controller)
    }
  })
}

function applyVadResult(voiced: boolean, analyzedSamples: number, sampleRate: number) {
  if (!listening.value) return
  if (streamingMode.value) {
    applyStreamingVadResult(voiced, analyzedSamples, sampleRate)
    return
  }
  if (voiced) {
    silenceSamples = 0
    if (!speaking.value) {
      speaking.value = true
      currentChunks = preRoll
      currentSamples = preRollSamples
      preRoll = []
      preRollSamples = 0
    }
    return
  }
  if (!speaking.value) return
  silenceSamples += analyzedSamples
  if (silenceSamples / sampleRate >= SILENCE_SECONDS) finishUtterance(sampleRate)
}

function applyStreamingVadResult(voiced: boolean, analyzedSamples: number, sampleRate: number) {
  if (voiced) {
    silenceSamples = 0
    if (!speaking.value) {
      speaking.value = true
      beginStreamingUtterance()
    }
    return
  }
  if (!speaking.value) return
  silenceSamples += analyzedSamples
  if (silenceSamples / sampleRate < SILENCE_SECONDS) return
  silenceSamples = 0
  speaking.value = false
  finishStreamingUtterance(sampleRate)
}

function beginStreamingUtterance() {
  const model = selectedModel.value
  const instance = selectedLoadedModel.value
  if (!model || !instance) return
  const generation = streamGeneration
  streamingStatus.value = "processing"
  streamQueue = streamQueue.then(async () => {
    if (!listening.value || generation !== streamGeneration || streamSession.value) return
    streamSession.value = await startTranscriptionStream(model.model_id, instance.instance_id)
    streamingStatus.value = "streaming"
  }).catch(handleStreamingFailure)
}

function finishUtterance(sampleRate: number) {
  const chunks = currentChunks
  const duration = currentSamples / sampleRate
  currentChunks = []
  currentSamples = 0
  silenceSamples = 0
  speaking.value = false
  if (duration < MIN_UTTERANCE_SECONDS) return
  const audio = encodeWave(chunks, sampleRate)
  enqueueRecognition(audio, duration)
}

function flushStreamingAudio(sampleRate: number) {
  if (!streamChunks.length) return
  const chunks = streamChunks
  streamChunks = []
  streamSamples = 0
  enqueueStreamingAudio(encodeWave(chunks, sampleRate))
}

function enqueueStreamingAudio(audio: Blob) {
  const generation = streamGeneration
  streamingStatus.value = "processing"
  streamQueue = streamQueue.then(async () => {
    const session = streamSession.value
    if (generation !== streamGeneration || !session) return
    const controller = new AbortController()
    requests.add(controller)
    try {
      const response = await transcribeStreamChunk(
        audio,
        session,
        controller.signal,
      )
      streamingText.value = joinStreamingText(streamingCommittedText.value, response.text)
      streamingInferenceMs.value = response.inference_ms
      streamingAudioSeconds.value = streamingCommittedAudioSeconds.value + response.audio_seconds
      streamingLanguage.value = response.detected_language
      streamingTokens.value = streamingCommittedTokens.value + response.generated_tokens
      streamingStatus.value = listening.value ? "streaming" : "processing"
      void nextTick(() => transcriptEnd.value?.scrollIntoView({ behavior: "smooth" }))
    } finally {
      requests.delete(controller)
    }
  }).catch(handleStreamingFailure)
}

function finishStreamingUtterance(sampleRate: number) {
  if (streamChunks.length) flushStreamingAudio(sampleRate)
  const generation = streamGeneration
  streamQueue = streamQueue.then(async () => {
    const session = streamSession.value
    if (generation !== streamGeneration || !session) return
    streamingStatus.value = "processing"
    const response = await finishTranscriptionStream(session)
    commitStreamingResponse(response)
    streamSession.value = null
    streamingStatus.value = listening.value ? "streaming" : "complete"
  }).catch(handleStreamingFailure)
}

function joinStreamingText(committed: string, current: string) {
  return [committed.trim(), current.trim()].filter(Boolean).join("\n")
}

function commitStreamingResponse(response: Awaited<ReturnType<typeof finishTranscriptionStream>>) {
  streamingCommittedText.value = joinStreamingText(streamingCommittedText.value, response.text)
  streamingText.value = streamingCommittedText.value
  streamingInferenceMs.value = response.inference_ms
  streamingCommittedAudioSeconds.value += response.audio_seconds
  streamingAudioSeconds.value = streamingCommittedAudioSeconds.value
  streamingLanguage.value = response.detected_language ?? streamingLanguage.value
  streamingCommittedTokens.value += response.generated_tokens
  streamingTokens.value = streamingCommittedTokens.value
}

async function handleStreamingFailure(caught: unknown) {
  if (caught instanceof DOMException && caught.name === "AbortError") return
  streamingStatus.value = "error"
  error.value = errorMessage(caught, "Streaming transcription failed")
  stopAudioGraph()
  await cancelActiveStream()
}

function enqueueRecognition(audio: Blob, durationSeconds: number) {
  const model = selectedModel.value
  const instance = selectedLoadedModel.value
  if (!model || !instance) return
  const segment: TranscriptSegment = {
    id: ++segmentSequence,
    createdAt: new Date(),
    durationSeconds,
    status: "recognizing",
    text: "",
    inferenceMs: null,
    modelName: model.short_name,
    languages: [],
    emotion: null,
    events: [],
    speechDurationSeconds: null,
    vadInferenceMs: null,
    enhancementInferenceMs: null,
  }
  segments.value.push(segment)
  pendingCount.value += 1
  void nextTick(() => transcriptEnd.value?.scrollIntoView({ behavior: "smooth" }))
  recognitionQueue = recognitionQueue.then(async () => {
    const controller = new AbortController()
    requests.add(controller)
    try {
      const response = await transcribeUtterance(
        audio,
        model.model_id,
        instance.instance_id,
        loadedVadModel.value?.instance_id ?? "",
        loadedEnhancementModel.value?.instance_id ?? "",
        threshold.value,
        controller.signal,
      )
      const liveSegment = segments.value.find((item) => item.id === segment.id)
      if (!liveSegment) return
      liveSegment.text = response.text || "No clear speech recognized"
      liveSegment.inferenceMs = response.inference_ms
      liveSegment.languages = response.languages
      liveSegment.emotion = response.emotion
      liveSegment.events = response.events
      liveSegment.speechDurationSeconds = response.speech_duration_seconds
      liveSegment.vadInferenceMs = response.vad_inference_ms
      liveSegment.enhancementInferenceMs = response.enhancement_inference_ms
      liveSegment.status = "complete"
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      const liveSegment = segments.value.find((item) => item.id === segment.id)
      if (liveSegment) {
        liveSegment.text = caught instanceof Error ? caught.message : "Recognition failed"
        liveSegment.status = "error"
      }
    } finally {
      requests.delete(controller)
      pendingCount.value = Math.max(0, pendingCount.value - 1)
      void nextTick(() => transcriptEnd.value?.scrollIntoView({ behavior: "smooth" }))
    }
  })
}

async function stopListening() {
  if (streamingMode.value) {
    const sampleRate = audioContext?.sampleRate
    if (sampleRate && speaking.value && streamChunks.length) flushStreamingAudio(sampleRate)
    stopAudioGraph()
    await streamQueue
    const session = streamSession.value
    if (streamingStatus.value === "error") return
    if (!session) {
      streamingStatus.value = streamingText.value ? "complete" : "idle"
      return
    }
    try {
      const response = await finishTranscriptionStream(session)
      commitStreamingResponse(response)
      streamingStatus.value = "complete"
    } catch (caught) {
      streamingStatus.value = "error"
      error.value = errorMessage(caught, "Failed to finish the transcription stream")
      try { await cancelTranscriptionStream(session) } catch { /* best-effort cleanup */ }
    } finally {
      streamSession.value = null
    }
    return
  }
  if (audioContext && currentSamples / audioContext.sampleRate >= MIN_UTTERANCE_SECONDS) {
    finishUtterance(audioContext.sampleRate)
  }
  stopAudioGraph()
}

async function cancelActiveStream() {
  const session = streamSession.value
  if (!session) return
  streamGeneration += 1
  streamSession.value = null
  try {
    await cancelTranscriptionStream(session)
  } catch {
    // The backend also clears all sessions when the model instance unloads.
  }
}

function stopAudioGraph() {
  vadGeneration += 1
  listening.value = false
  speaking.value = false
  inputLevel.value = 0
  if (elapsedTimer != null) window.clearInterval(elapsedTimer)
  elapsedTimer = null
  processorNode?.disconnect()
  sourceNode?.disconnect()
  silentGain?.disconnect()
  if (processorNode) processorNode.onaudioprocess = null
  mediaStream?.getTracks().forEach((track) => track.stop())
  void audioContext?.close()
  mediaStream = null
  audioContext = null
  sourceNode = null
  processorNode = null
  silentGain = null
  currentChunks = []
  currentSamples = 0
  silenceSamples = 0
  preRoll = []
  preRollSamples = 0
  vadChunks = []
  vadSamples = 0
}

async function copyTranscript() {
  if (transcriptText.value) await navigator.clipboard.writeText(transcriptText.value)
}

function clearTranscript() {
  if (streamingMode.value) {
    streamingText.value = ""
    streamingStatus.value = "idle"
    streamingInferenceMs.value = null
    streamingAudioSeconds.value = 0
    streamingCommittedAudioSeconds.value = 0
    streamingLanguage.value = null
    streamingTokens.value = 0
    streamingCommittedTokens.value = 0
    streamingCommittedText.value = ""
  } else {
    segments.value = []
  }
}

onMounted(loadModels)
onBeforeUnmount(() => {
  stopAudioGraph()
  void cancelActiveStream()
  requests.forEach((request) => request.abort())
})
</script>

<template>
  <div class="page inner-page transcription-page">
    <header class="transcription-hero">
      <div class="hero-copy">
        <RouterLink class="back-link" to="/apps">← Neural Apps</RouterLink>
        <p class="kicker">MULTI-MODEL ASR · OPTIONAL VAD AND SPEECH ENHANCEMENT</p>
        <h1><span aria-hidden="true">🎙️</span> Voice, <em>made visible.</em></h1>
        <p>Select and load an ASR model, then start listening. Utterance-based ASR uses optional Silero VAD and DeepFilterNet3; Nemotron uses VAD boundaries with raw audio.</p>
      </div>
      <section class="asr-model-picker" aria-label="Select a speech recognition model">
        <article
        v-for="model in models"
        :key="model.model_id"
          :class="{
            selected: selectedModelId === model.model_id,
            loaded: modelIsReady(model),
            unavailable: !runtimeAvailable(model),
          }"
        >
          <button
            class="model-identity"
            type="button"
            :disabled="listening || pendingCount > 0"
            @click="selectModel(model.model_id)"
          >
            <span>{{ model.rich_understanding ? "🧠" : "⚡" }}</span>
            <span>
              <small>{{ model.variant }} · {{ modelIsReady(model) ? "LOADED" : model.streaming ? "STREAMING ASR" : "ASR MODEL" }}</small>
              <strong>{{ model.display_name }}</strong>
              <em>{{ model.description }}</em>
            </span>
            <i aria-hidden="true"></i>
          </button>
          <div class="model-controls">
            <label>
              <span>RUNTIME</span>
              <select
                :value="runtimeForModel(model)"
                :disabled="listening || pendingCount > 0 || loadingModel"
                @change="onRuntimeChange(model, $event)"
              >
                <option
                  v-for="runtime in runtimeOptions(model)"
                  :key="runtime.runtime"
                  :value="runtime.runtime"
                  :disabled="!runtime.available"
                >
                  {{ runtimeLabel(runtime.runtime) }}{{ runtime.available ? "" : " · unavailable" }}
                </option>
              </select>
            </label>
            <button
              type="button"
              :disabled="(!modelIsReady(model) && !runtimeAvailable(model)) || loadingModel || listening || pendingCount > 0"
              @click="toggleModel(model)"
            >
              {{ loadingModel && selectedModelId === model.model_id ? "WAIT…" : modelIsReady(model) ? "UNLOAD" : "LOAD" }}
            </button>
          </div>
          <small
            v-if="selectedModelId === model.model_id && lifecycleMessage"
            :class="`lifecycle-${lifecycleMessage.type}`"
          >{{ lifecycleMessage.text }}</small>
        </article>
      </section>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>

    <section class="transcription-studio">
      <aside class="recorder-panel">
        <div class="recorder-state">
          <div class="mic-orbit" :class="{ listening, speaking }">
            <span>🎙️</span><i></i><i></i><i></i>
          </div>
          <p class="kicker">{{ listening ? (streamingMode ? "STREAMING AUDIO" : speaking ? "VOICE DETECTED" : "LISTENING FOR SPEECH") : "MICROPHONE IDLE" }}</p>
          <strong>{{ listening ? elapsedSeconds.toFixed(1) : "0.0" }}<small> SEC</small></strong>
        </div>

        <div class="level-meter" aria-label="Microphone input level">
          <span
            v-for="bar in 24"
            :key="bar"
            :class="{ active: inputLevel * 24 >= bar, voice: speaking }"
          ></span>
        </div>

        <label class="vad-sensitivity">
          <span>
            <b>SPEECH SENSITIVITY</b>
            <small>{{ loadedVadModel ? "Silero neural VAD" : "Browser energy gate" }}</small>
          </span>
          <input v-model.number="sensitivity" type="range" min="0" max="1" step="0.05" />
          <output>{{ Math.round(sensitivity * 100) }}%</output>
        </label>

        <div class="pipeline-components" aria-label="Optional audio components">
          <article v-for="component in pipelineComponents" :key="component.component_id">
            <div>
              <strong>{{ component.display_name }}</strong>
              <small>{{ component.description }}</small>
            </div>
            <span :class="`component-${componentState(component)}`">
              {{ componentStateLabel(component) }}
            </span>
            <em v-if="componentIssues[component.component_id]">
              {{ componentIssues[component.component_id] }}
            </em>
            <em v-else-if="!component.downloaded">
              Optional · {{ component.component_id === "vad" ? "browser energy gate fallback" : "direct ASR fallback" }}
            </em>
            <em v-else-if="streamingMode && component.component_id === 'vad'">
              Speech boundary detection only · audio remains continuous within each utterance
            </em>
            <em v-else-if="streamingMode && component.component_id === 'enhancement'">
              Disabled for Nemotron streaming
            </em>
            <em v-else-if="componentState(component) === 'not-loaded'">
              Loads automatically when listening starts
            </em>
          </article>
          <p v-if="componentIssues.status">{{ componentIssues.status }}</p>
        </div>

        <button
          class="record-button"
          :class="{ recording: listening }"
          type="button"
          :disabled="loadingModel || !selectedArtifactAvailable"
          @click="listening ? stopListening() : startListening()"
        >
          <i></i>{{ listening ? "STOP LISTENING" : "START LISTENING" }}
        </button>
        <p class="recorder-note">
          <template v-if="streamingMode">
            {{ loadedVadModel ? "Silero VAD boundaries" : "Browser energy boundaries" }} → Raw audio → Nemotron ·
            {{ Math.round(SILENCE_SECONDS * 1000) }}ms pause finalizes each stream
          </template>
          <template v-else>
            {{ loadedVadModel ? "Silero VAD" : "Browser energy gate" }} →
            {{ loadedEnhancementModel ? "DeepFilterNet3 Core ML" : "Direct audio" }} → ASR ·
            {{ Math.round(SILENCE_SECONDS * 1000) }}ms pause splitting · {{ MAX_UTTERANCE_SECONDS }}s maximum utterance
          </template>
        </p>
      </aside>

      <article class="transcript-panel">
        <header>
          <div>
            <p class="kicker">{{ streamingMode ? "LIVE STREAM" : "LIVE TRANSCRIPT" }}</p>
            <span v-if="streamingMode">
              {{ streamingStatus }} · {{ streamingAudioSeconds.toFixed(1) }}s audio · {{ streamingTokens }} tokens
            </span>
            <span v-else>{{ segments.length }} utterances · {{ pendingCount }} processing</span>
          </div>
          <div>
            <button type="button" :disabled="!transcriptText" @click="copyTranscript">COPY</button>
            <button type="button" :disabled="!transcriptText || listening" @click="clearTranscript">CLEAR</button>
          </div>
        </header>

        <div class="transcript-scroll">
          <div v-if="streamingMode" class="streaming-transcript" :class="`streaming-${streamingStatus}`">
            <header>
              <span>STATEFUL RNN-T</span>
              <span v-if="streamingLanguage">LANGUAGE · {{ streamingLanguage }}</span>
              <span>CHUNK · {{ selectedModel?.streaming_chunk_seconds?.toFixed(2) ?? "2.24" }}s</span>
            </header>
            <p v-if="streamingText">
              {{ streamingText }}<i v-if="listening || streamingStatus === 'processing'" aria-hidden="true"></i>
            </p>
            <div v-else class="streaming-placeholder">
              <span aria-hidden="true">⌁</span>
              <strong>{{ listening ? "Listening for the first streaming tokens…" : "Continuous text will appear here." }}</strong>
              <small>Nemotron keeps encoder and decoder caches within each Silero-bounded utterance.</small>
            </div>
            <footer v-if="streamingInferenceMs != null">
              LAST CHUNK · {{ streamingInferenceMs.toFixed(0) }}ms inference
              · RAW AUDIO
            </footer>
          </div>
          <div v-else-if="!segments.length" class="transcript-empty">
            <span aria-hidden="true">〰</span>
            <strong>Your words will appear here.</strong>
            <p>Click Start Listening and speak naturally. Each utterance appears after a short pause.</p>
          </div>
          <ol v-else class="transcript-list">
            <li v-for="segment in segments" :key="segment.id" :class="`segment--${segment.status}`">
              <div class="segment-meta">
                <span>{{ String(segment.id).padStart(2, "0") }}</span>
                <time>{{ segment.createdAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) }}</time>
                <small>{{ segment.durationSeconds.toFixed(1) }}s · {{ segment.modelName }}</small>
              </div>
              <div class="segment-content">
                <p v-if="segment.status === 'recognizing'"><i></i><i></i><i></i></p>
                <p v-else>{{ segment.text }}</p>
                <div v-if="segment.languages.length || segment.emotion || segment.events.length" class="speech-tags">
                  <span v-for="language in segment.languages" :key="language">🌐 {{ language }}</span>
                  <span v-if="segment.emotion">🙂 {{ segment.emotion }}</span>
                  <span v-for="event in segment.events" :key="event">🔊 {{ event }}</span>
                </div>
              </div>
              <small v-if="segment.inferenceMs != null">
                {{ segment.inferenceMs.toFixed(0) }}ms ASR
                <template v-if="segment.vadInferenceMs != null"> · {{ segment.vadInferenceMs.toFixed(0) }}ms VAD</template>
                <template v-if="segment.enhancementInferenceMs != null"> · {{ segment.enhancementInferenceMs.toFixed(0) }}ms DENOISE</template>
                <template v-if="segment.speechDurationSeconds != null"> · {{ segment.speechDurationSeconds.toFixed(1) }}s speech</template>
              </small>
            </li>
          </ol>
          <div ref="transcriptEnd"></div>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.transcription-page { padding-top:0; }
.transcription-hero { align-items:stretch; display:grid; gap:clamp(1.5rem,3vw,3.5rem); grid-template-columns:minmax(0,1fr) minmax(25rem,.72fr); padding:.65rem 0 1.4rem; }
.hero-copy { align-self:center; min-width:0; }
.transcription-hero h1 { font-size:clamp(2.4rem,5vw,4.6rem); letter-spacing:-.065em; line-height:.95; margin:.55rem 0 .8rem; white-space:nowrap; }
.transcription-hero h1 > span { display:inline-block; font-size:.58em; margin-right:.2em; transform:rotate(-8deg); }
.transcription-hero h1 em { color:transparent; font-style:normal; -webkit-text-stroke:1.5px var(--ink); }
.hero-copy > p:last-child { color:var(--muted); font-size:.82rem; line-height:1.5; max-width:42rem; }
.lifecycle-success { color:#75c763; font:.52rem var(--font-mono); }.lifecycle-error { color:#ff8066; font:.52rem var(--font-mono); }
.asr-model-picker { align-content:center; display:flex; flex-direction:column; gap:.55rem; }
.asr-model-picker article { align-items:center; background:#f7f5eb; border:1px solid var(--line); display:grid; gap:.8rem; grid-template-columns:minmax(0,1fr) minmax(10.5rem,.62fr); padding:.62rem .72rem; transition:border-color .2s,background .2s,transform .2s; }
.asr-model-picker article:hover { border-color:var(--ink); transform:translateX(-3px); }
.asr-model-picker article.selected { background:#c8ff4614; border-color:var(--ink); box-shadow:inset 3px 0 var(--signal); }
.asr-model-picker article.unavailable { opacity:.62; }
.model-identity { align-items:center; background:transparent; border:0; color:var(--ink); cursor:pointer; display:grid; gap:.7rem; grid-template-columns:2rem minmax(0,1fr) .55rem; padding:0; text-align:left; width:100%; }
.model-identity:disabled { cursor:not-allowed; }
.model-identity > span:first-child { font-size:1.35rem; text-align:center; }
.model-identity > span:nth-child(2) { display:flex; flex-direction:column; min-width:0; }
.model-identity small,.model-controls span { color:var(--muted); font:.46rem var(--font-mono); text-transform:uppercase; }
.model-identity strong { font-size:.9rem; line-height:1.15; }
.model-identity em { color:var(--muted); font-size:.58rem; font-style:normal; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.model-identity > i { background:#aaa; border-radius:50%; height:.48rem; width:.48rem; }
.asr-model-picker article.loaded .model-identity > i { background:var(--signal); box-shadow:0 0 8px #8ebd22; }
.model-controls { align-items:end; border-left:1px solid var(--line); display:grid; gap:.5rem; grid-template-columns:minmax(5.5rem,1fr) auto; margin:0; padding-left:.72rem; }
.model-controls label { display:flex; flex-direction:column; gap:.2rem; }
.model-controls select { background:transparent; border:0; color:var(--ink); font:600 .55rem var(--font-mono); min-width:0; outline:none; padding:0; text-transform:uppercase; width:100%; }
.model-controls > button { background:var(--ink); border:0; color:var(--paper); cursor:pointer; font:700 .5rem var(--font-mono); min-width:4.4rem; padding:.58rem .65rem; }
.model-controls > button:hover { background:var(--signal); color:var(--ink); }
.model-controls > button:disabled,.model-controls select:disabled { cursor:not-allowed; opacity:.4; }
.asr-model-picker article > small { grid-column:1/-1; margin-top:-.35rem; }
.transcription-studio { border:1px solid var(--ink); display:grid; grid-template-columns:minmax(17rem,.7fr) minmax(0,1.3fr); min-height:38rem; }
.recorder-panel { background:#141614; color:var(--paper); display:flex; flex-direction:column; padding:2rem; }
.recorder-state { align-items:center; display:flex; flex-direction:column; text-align:center; }
.mic-orbit { align-items:center; background:#20231f; border:1px solid #ffffff1f; border-radius:50%; display:flex; height:10rem; justify-content:center; margin:1rem 0 1.5rem; position:relative; width:10rem; }
.mic-orbit span { font-size:3.5rem; position:relative; z-index:2; }
.mic-orbit i { border:1px solid #c8ff4640; border-radius:50%; inset:-.7rem; opacity:0; position:absolute; }
.mic-orbit i:nth-of-type(2) { inset:-1.5rem; }.mic-orbit i:nth-of-type(3) { inset:-2.3rem; }
.mic-orbit.listening { border-color:#c8ff46; box-shadow:inset 0 0 40px #c8ff4615; }
.mic-orbit.listening i { animation:listen-pulse 2s ease-out infinite; opacity:1; }
.mic-orbit.listening i:nth-of-type(2) { animation-delay:.35s; }.mic-orbit.listening i:nth-of-type(3) { animation-delay:.7s; }
.mic-orbit.speaking { background:#c8ff46; transform:scale(1.04); transition:.15s; }
.recorder-state > strong { font:700 2.4rem var(--font-mono); }.recorder-state > strong small { color:#888; font-size:.55rem; }
.level-meter { align-items:end; display:flex; gap:.22rem; height:4.5rem; margin:1.2rem 0; }
.level-meter span { background:#ffffff16; flex:1; height:calc(18% + var(--bar,0%)); min-height:.35rem; transition:.08s; }
.level-meter span:nth-child(3n) { height:36%; }.level-meter span:nth-child(4n) { height:54%; }.level-meter span:nth-child(5n) { height:72%; }
.level-meter span.active { background:#8b9b6a; }.level-meter span.active.voice { background:#c8ff46; box-shadow:0 0 8px #c8ff4666; }
.vad-sensitivity { align-items:center; border-bottom:1px solid #ffffff1f; border-top:1px solid #ffffff1f; display:grid; gap:.8rem; grid-template-columns:1fr 7rem 2rem; padding:1rem 0; }
.vad-sensitivity span { display:flex; flex-direction:column; }.vad-sensitivity b,.vad-sensitivity output { font:.56rem var(--font-mono); }.vad-sensitivity small { color:#888; font-size:.65rem; margin-top:.2rem; }
.vad-sensitivity input { accent-color:#c8ff46; width:100%; }
.pipeline-components { border-bottom:1px solid #ffffff1f; display:grid; gap:.55rem; padding:.8rem 0; }
.pipeline-components article { align-items:start; display:grid; gap:.25rem .55rem; grid-template-columns:minmax(0,1fr) auto; }
.pipeline-components article > div { display:flex; flex-direction:column; min-width:0; }
.pipeline-components strong { font:.56rem var(--font-mono); }
.pipeline-components small,.pipeline-components em,.pipeline-components > p { color:#888; font:.5rem var(--font-mono); font-style:normal; line-height:1.4; margin:0; }
.pipeline-components article > em { grid-column:1/-1; }
.pipeline-components article > span { border:1px solid #ffffff2e; font:.45rem var(--font-mono); padding:.22rem .34rem; white-space:nowrap; }
.pipeline-components .component-loaded { border-color:#c8ff46; color:#c8ff46; }
.pipeline-components .component-not-loaded { color:#f0c966; }
.pipeline-components .component-not-downloaded { color:#999; }
.record-button { background:#c8ff46; border:0; cursor:pointer; font:700 .68rem var(--font-mono); margin-top:auto; padding:1.1rem; }
.record-button i { background:#111; border-radius:50%; display:inline-block; height:.55rem; margin-right:.6rem; width:.55rem; }
.record-button.recording { background:#ff7148; }.record-button.recording i { border-radius:1px; }
.record-button:disabled { cursor:not-allowed; filter:grayscale(1); opacity:.35; }
.recorder-note { color:#777; font:.5rem var(--font-mono); margin:.7rem 0 0; text-align:center; }
.transcript-panel { display:flex; flex-direction:column; min-width:0; }
.transcript-panel > header { align-items:center; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; padding:1.2rem 1.4rem; }
.transcript-panel > header > div:first-child span { color:var(--muted); font:.55rem var(--font-mono); }
.transcript-panel > header button { background:transparent; border:1px solid var(--line); cursor:pointer; font:.55rem var(--font-mono); margin-left:.4rem; padding:.55rem; }
.transcript-panel > header button:disabled { opacity:.3; }
.transcript-scroll { flex:1; max-height:39rem; overflow:auto; padding:1.5rem; }
.streaming-transcript { display:flex; flex-direction:column; min-height:31rem; }
.streaming-transcript > header { border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:.45rem; padding-bottom:.8rem; }
.streaming-transcript > header span { background:#e4e8dc; font:.5rem var(--font-mono); padding:.32rem .45rem; }
.streaming-transcript > p { font-size:clamp(1.4rem,2.6vw,2.25rem); letter-spacing:-.025em; line-height:1.5; margin:1.8rem 0; white-space:pre-wrap; }
.streaming-transcript > p i { animation:stream-cursor .8s steps(1) infinite; background:var(--signal); display:inline-block; height:1.15em; margin-left:.18rem; vertical-align:-.12em; width:.18em; }
.streaming-transcript > footer { border-top:1px solid var(--line); color:var(--muted); font:.5rem var(--font-mono); margin-top:auto; padding-top:.8rem; }
.streaming-placeholder { align-items:center; color:var(--muted); display:flex; flex:1; flex-direction:column; justify-content:center; text-align:center; }
.streaming-placeholder > span { color:var(--ink); font-size:5rem; }.streaming-placeholder strong { color:var(--ink); font-size:1.15rem; }.streaming-placeholder small { font-size:.68rem; margin-top:.5rem; max-width:26rem; }
.streaming-error { color:#cf3f27; }
.transcript-empty { align-items:center; color:var(--muted); display:flex; flex-direction:column; height:100%; justify-content:center; min-height:25rem; text-align:center; }
.transcript-empty > span { color:var(--ink); font-size:5rem; }.transcript-empty strong { color:var(--ink); font-size:1.35rem; }.transcript-empty p { font-size:.8rem; line-height:1.6; max-width:24rem; }
.transcript-list { list-style:none; margin:0; padding:0; }
.transcript-list li { display:grid; gap:1.2rem; grid-template-columns:6rem 1fr auto; padding:1.4rem 0; }
.transcript-list li + li { border-top:1px solid var(--line); }
.segment-meta { display:grid; font: .52rem var(--font-mono); gap:.3rem; grid-template-columns:1.5rem 1fr; }
.segment-meta > span { align-items:center; background:var(--ink); color:var(--paper); display:flex; grid-row:span 2; justify-content:center; }
.segment-meta small,.transcript-list li > small { color:var(--muted); }
.segment-content > p { font-size:1.15rem; line-height:1.55; margin:0; }
.speech-tags { display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.65rem; }
.speech-tags span { background:#e4e8dc; font:.52rem var(--font-mono); padding:.3rem .45rem; }
.segment--recognizing p i { animation:typing 1s infinite; background:var(--ink); border-radius:50%; display:inline-block; height:.4rem; margin:.25rem; width:.4rem; }
.segment--recognizing p i:nth-child(2) { animation-delay:.15s; }.segment--recognizing p i:nth-child(3) { animation-delay:.3s; }
.segment--error p { color:#cf3f27; }
@keyframes listen-pulse { from { opacity:.8; transform:scale(.85); } to { opacity:0; transform:scale(1.15); } }
@keyframes typing { 50% { opacity:.2; transform:translateY(-4px); } }
@keyframes stream-cursor { 50% { opacity:0; } }
@media (max-width:760px) {
  .transcription-page { padding-top:1rem; }
  .transcription-hero { align-items:stretch; gap:1.5rem; grid-template-columns:1fr; }
  .transcription-hero h1 { font-size:3rem; white-space:normal; }
  .asr-model-picker { width:100%; }
  .transcription-studio { grid-template-columns:1fr; }
  .recorder-panel { min-height:34rem; padding:1.3rem; }
  .transcript-scroll { max-height:none; min-height:30rem; }
  .transcript-list li { gap:.8rem; grid-template-columns:4.5rem 1fr; }
  .transcript-list li > small { display:none; }
  .transcript-panel > header { align-items:flex-start; gap:1rem; }
}
</style>
