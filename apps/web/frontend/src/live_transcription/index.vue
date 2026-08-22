<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue"

import {
  fetchAsrModels,
  fetchPipelineComponents,
  loadAsrModel,
  loadEnhancementModel,
  loadVadModel,
} from "./api"
import { errorMessage, unloadModel } from "@/modelLifecycle"
import type {
  AsrModel,
  LoadedAsrModel,
  PipelineComponent,
  StreamingPlaybackItem,
  TranscriptSegment,
} from "./types"
import { LiveTranscriptionSocket } from "./liveSocket"
import type { LiveEvent } from "./liveSocket"

const models = ref<AsrModel[]>([])
const selectedModelId = ref("")
const selectedVariantByModel = ref<Record<string, string>>({})
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
const useVad = ref(true)
const useEnhancement = ref(true)
const vadStreamConnected = ref(false)
const elapsedSeconds = ref(0)
const segments = ref<TranscriptSegment[]>([])
const pendingCount = ref(0)
const error = ref("")
const lifecycleMessage = ref<{ type: "error"; text: string } | null>(null)
const transcriptScroll = ref<HTMLElement | null>(null)
const streamingText = ref("")
const streamingCurrentText = ref("")
const streamingPlaybackItems = ref<StreamingPlaybackItem[]>([])
const streamingStatus = ref<"idle" | "streaming" | "processing" | "complete" | "error">("idle")
const streamingInferenceMs = ref<number | null>(null)
const streamingAudioSeconds = ref(0)
const streamingLanguage = ref<string | null>(null)
const streamingTokens = ref(0)

let mediaStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let sourceNode: MediaStreamAudioSourceNode | null = null
let processorNode: ScriptProcessorNode | null = null
let silentGain: GainNode | null = null
let sessionStartedAt = 0
let elapsedTimer: number | null = null
let liveSocket: LiveTranscriptionSocket | null = null
let transcriptScrollFrame: number | null = null
const segmentById = new Map<number, TranscriptSegment>()


const selectedModel = computed(() =>
  models.value.find((model) => model.model_id === selectedModelId.value) ?? null,
)
const variantForModel = (model: AsrModel) =>
  selectedVariantByModel.value[model.model_id] ?? model.variant
const runtimeKey = (modelId: string, variant: string, runtime: string) =>
  `${modelId}::${variant}::${runtime}`
const runtimeForModel = (model: AsrModel) =>
  selectedRuntimeByModel.value[model.model_id]
  ?? (model.resource.runtimes.some((item) => item.runtime === "coreml") ? "coreml" : model.runtime)
const runtimeAvailable = (model: AsrModel, runtime = runtimeForModel(model)) =>
  Boolean(
    model.variants.find((item) => item.name === variantForModel(model))
      ?.available_runtimes.includes(runtime),
  )
const loadedInstanceFor = (model: AsrModel) =>
  loadedModels.value[
    runtimeKey(model.model_id, variantForModel(model), runtimeForModel(model))
  ] ?? null
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
const selectedStreamingChunkSeconds = computed(() => {
  const model = selectedModel.value
  if (!model) return 2.24
  return model.variants.find((item) => item.name === variantForModel(model))
    ?.streaming_chunk_seconds ?? model.streaming_chunk_seconds ?? 2.24
})
const selectedArtifactAvailable = computed(() => {
  const model = selectedModel.value
  if (!model) return false
  return runtimeAvailable(model)
})
const threshold = computed(() => 0.75 - sensitivity.value * 0.5)
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
const componentEnabled = (component: PipelineComponent) => component.component_id === "vad"
  ? useVad.value
  : useEnhancement.value && !streamingMode.value
const componentApplicable = (component: PipelineComponent) =>
  component.component_id !== "enhancement" || !streamingMode.value
const componentDescription = (component: PipelineComponent) => component.component_id === "vad"
  ? "Neural speech-boundary detection"
  : "Core ML denoising before transcription"
const transcriptText = computed(() =>
  streamingMode.value
    ? streamingText.value
    : segments.value
    .filter((segment) => segment.status === "complete" && segment.text)
    .map((segment) => segment.text)
    .join(" "),
)

function scrollTranscriptToLatest(behavior: ScrollBehavior = "smooth") {
  void nextTick(() => {
    if (transcriptScrollFrame != null) cancelAnimationFrame(transcriptScrollFrame)
    transcriptScrollFrame = requestAnimationFrame(() => {
      const container = transcriptScroll.value
      if (container) container.scrollTo({ top: container.scrollHeight, behavior })
      transcriptScrollFrame = null
    })
  })
}

async function loadModels() {
  try {
    models.value = await fetchAsrModels()
    for (const model of models.value) {
      selectedVariantByModel.value[model.model_id] = model.variant
      selectedRuntimeByModel.value[model.model_id] = runtimeForModel(model)
      for (const ready of model.ready_instances ?? []) {
        loadedModels.value[runtimeKey(model.model_id, ready.variant, ready.runtime)] = {
          instance_id: ready.instance_id,
          model_id: model.model_id,
          variant: ready.variant,
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
    const variant = variantForModel(selectedModel.value)
    const loaded = await loadAsrModel(selectedModel.value.model_id, variant, runtime)
    loadedModels.value[runtimeKey(selectedModel.value.model_id, variant, runtime)] = loaded
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
    delete loadedModels.value[runtimeKey(instance.model_id, instance.variant, instance.runtime)]
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

function onVariantChange(model: AsrModel, event: Event) {
  if (listening.value || pendingCount.value || loadingModel.value) return
  selectedVariantByModel.value[model.model_id] = (event.target as HTMLSelectElement).value
  lifecycleMessage.value = null
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
  let instance: LoadedAsrModel
  try {
    instance = await ensureSelectedModelLoaded()
    if (useVad.value && !await ensureVadModelLoaded()) {
      throw new Error("Silero VAD is enabled but its model is unavailable")
    }
    if (!streamingMode.value && useEnhancement.value && !await ensureEnhancementModelLoaded()) {
      throw new Error("DeepFilterNet3 is enabled but its model is unavailable")
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: false, channelCount: 1 },
      video: false,
    })
    audioContext = new AudioContext()
    liveSocket = await LiveTranscriptionSocket.connect({
      modelId: instance.model_id,
      instanceId: instance.instance_id,
      sampleRate: audioContext.sampleRate,
      useVad: useVad.value,
      vadInstanceId: useVad.value ? loadedVadModel.value?.instance_id ?? null : null,
      useEnhancement: useEnhancement.value && !streamingMode.value,
      enhancementInstanceId: useEnhancement.value && !streamingMode.value
        ? loadedEnhancementModel.value?.instance_id ?? null : null,
      vadThreshold: threshold.value,
      sensitivity: sensitivity.value,
      streamingChunkSeconds: selectedStreamingChunkSeconds.value,
    }, handleLiveEvent, handleLiveAudio, handleLiveFailure)
    sourceNode = audioContext.createMediaStreamSource(mediaStream)
    processorNode = audioContext.createScriptProcessor(4096, 1, 1)
    silentGain = audioContext.createGain()
    silentGain.gain.value = 0
    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      let energy = 0
      for (const sample of input) energy += sample * sample
      inputLevel.value = Math.min(1, Math.sqrt(energy / input.length) / 0.12)
      liveSocket?.send(input)
    }
    sourceNode.connect(processorNode)
    processorNode.connect(silentGain)
    silentGain.connect(audioContext.destination)
    listening.value = true
    vadStreamConnected.value = true
    sessionStartedAt = performance.now()
    elapsedTimer = window.setInterval(() => {
      elapsedSeconds.value = (performance.now() - sessionStartedAt) / 1000
    }, 250)
    if (streamingMode.value) streamingStatus.value = "streaming"
  } catch (caught) {
    stopAudioGraph()
    error.value = caught instanceof DOMException && caught.name === "NotAllowedError"
      ? "Microphone permission was denied."
      : errorMessage(caught, "Failed to start live transcription")
  }
}

function handleLiveEvent(event: LiveEvent) {
  if (event.type === "speech_start" && event.utterance_id) {
    speaking.value = true
    const segment: TranscriptSegment = {
      id: event.utterance_id, createdAt: new Date(), durationSeconds: 0,
      status: "recognizing", text: "", inferenceMs: null,
      modelName: selectedModel.value?.short_name ?? "ASR", languages: [], emotion: null,
      events: [], speechDurationSeconds: null, vadInferenceMs: null,
      enhancementInferenceMs: null, inputAudioUrl: null,
    }
    segmentById.set(segment.id, segment)
    segments.value.push(segment)
    pendingCount.value += 1
    scrollTranscriptToLatest()
  } else if (event.type === "speech_end") {
    speaking.value = false
  } else if (event.type === "partial") {
    streamingCurrentText.value = event.text ?? ""
    streamingInferenceMs.value = event.inference_ms ?? null
    streamingStatus.value = "streaming"
    scrollTranscriptToLatest("auto")
  } else if (event.type === "transcript" && event.utterance_id) {
    const segment = segmentById.get(event.utterance_id)
    if (!segment) return
    segment.text = event.text || "No clear speech recognized"
    segment.durationSeconds = event.duration_seconds ?? 0
    segment.inferenceMs = event.inference_ms ?? null
    segment.vadInferenceMs = event.vad_inference_ms ?? null
    segment.enhancementInferenceMs = event.enhancement_inference_ms ?? null
    segment.speechDurationSeconds = event.speech_duration_seconds ?? null
    segment.languages = event.languages ?? []
    segment.emotion = event.emotion ?? null
    segment.events = event.events ?? []
    segment.status = "complete"
    streamingAudioSeconds.value += segment.durationSeconds
    streamingTokens.value += event.generated_tokens ?? 0
    streamingInferenceMs.value = event.inference_ms ?? null
    streamingLanguage.value = segment.languages[0] ?? streamingLanguage.value
    pendingCount.value = Math.max(0, pendingCount.value - 1)
    if (streamingMode.value) {
      streamingCurrentText.value = ""
      streamingText.value = [...segmentById.values()].filter((item) => item.status === "complete")
        .map((item) => item.text).join("\n")
      streamingStatus.value = "processing"
    }
    scrollTranscriptToLatest()
  } else if (event.type === "stopped") {
    liveSocket?.close()
    liveSocket = null
    if (streamingMode.value) streamingStatus.value = "complete"
  }
}

function handleLiveAudio(utteranceId: number, audio: Blob) {
  const segment = segmentById.get(utteranceId)
  if (!segment) return
  segment.inputAudioUrl = URL.createObjectURL(audio)
  if (streamingMode.value) streamingPlaybackItems.value.push({
    id: utteranceId, text: segment.text, durationSeconds: segment.durationSeconds,
    inferenceMs: segment.inferenceMs, inputAudioUrl: segment.inputAudioUrl,
  })
  scrollTranscriptToLatest()
}

function handleLiveFailure(caught: Error) {
  error.value = caught.message
  streamingStatus.value = "error"
  stopAudioGraph()
}

function stopListening() {
  stopCaptureGraph()
  liveSocket?.stop()
}

function stopCaptureGraph() {
  listening.value = false
  speaking.value = false
  inputLevel.value = 0
  if (elapsedTimer != null) window.clearInterval(elapsedTimer)
  elapsedTimer = null
  if (processorNode) processorNode.onaudioprocess = null
  processorNode?.disconnect(); sourceNode?.disconnect(); silentGain?.disconnect()
  mediaStream?.getTracks().forEach((track) => track.stop())
  void audioContext?.close()
  mediaStream = null; audioContext = null; sourceNode = null; processorNode = null; silentGain = null
}

function stopAudioGraph() {
  stopCaptureGraph()
  liveSocket?.close()
  liveSocket = null
  vadStreamConnected.value = false
}

async function copyTranscript() {
  if (transcriptText.value) await navigator.clipboard.writeText(transcriptText.value)
}

function clearTranscript() {
  for (const segment of segments.value) {
    if (segment.inputAudioUrl) URL.revokeObjectURL(segment.inputAudioUrl)
  }
  segments.value = []
  segmentById.clear()
  streamingPlaybackItems.value = []
  streamingText.value = ""
  streamingCurrentText.value = ""
  streamingStatus.value = "idle"
}

onMounted(loadModels)
onBeforeUnmount(() => {
  if (transcriptScrollFrame != null) cancelAnimationFrame(transcriptScrollFrame)
  stopAudioGraph()
  for (const segment of segments.value) {
    if (segment.inputAudioUrl) URL.revokeObjectURL(segment.inputAudioUrl)
  }
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
            <span class="model-copy">
              <small>{{ variantForModel(model) }} · {{ modelIsReady(model) ? "LOADED" : model.streaming ? "STREAMING ASR" : "ASR MODEL" }}</small>
              <strong>{{ model.display_name }}</strong>
              <em>{{ model.description }}</em>
            </span>
            <i aria-hidden="true"></i>
          </button>
          <div class="model-controls">
            <label>
              <span>VARIANT</span>
              <select
                :value="variantForModel(model)"
                :disabled="listening || pendingCount > 0 || loadingModel"
                @change="onVariantChange(model, $event)"
              >
                <option
                  v-for="variant in model.variants"
                  :key="variant.name"
                  :value="variant.name"
                >
                  {{ variant.display_name }}{{ variant.available ? "" : " · unavailable" }}
                </option>
              </select>
            </label>
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
            <small>{{ useVad && vadStreamConnected ? "Silero neural VAD" : "Backend energy gate" }}</small>
          </span>
          <input
            v-model.number="sensitivity"
            type="range"
            min="0"
            max="1"
            step="0.05"
            :disabled="listening"
          />
          <output>{{ Math.round(sensitivity * 100) }}%</output>
        </label>

        <div class="pipeline-components" aria-label="Optional audio components">
          <article
            v-for="component in pipelineComponents"
            :key="component.component_id"
            :class="{ 'component-off': !componentEnabled(component) }"
          >
            <label class="component-toggle">
              <input
                v-if="component.component_id === 'vad'"
                v-model="useVad"
                type="checkbox"
                :disabled="listening"
              />
              <input
                v-else
                type="checkbox"
                :checked="useEnhancement && !streamingMode"
                :disabled="listening || !componentApplicable(component)"
                @change="useEnhancement = ($event.target as HTMLInputElement).checked"
              />
              <span>
                <strong>{{ component.display_name }}</strong>
                <small>{{ componentDescription(component) }}</small>
              </span>
            </label>
            <span :class="`component-${componentState(component)}`">
              {{ componentStateLabel(component) }}
            </span>
            <em v-if="componentIssues[component.component_id]">
              {{ componentIssues[component.component_id] }}
            </em>
            <em v-else-if="!component.downloaded">Model resource is not installed</em>
            <em v-else-if="streamingMode && component.component_id === 'enhancement'">
              Not available for streaming ASR
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

        <div ref="transcriptScroll" class="transcript-scroll">
          <div v-if="streamingMode" class="streaming-transcript" :class="`streaming-${streamingStatus}`">
            <header>
              <span>STATEFUL RNN-T</span>
              <span v-if="streamingLanguage">LANGUAGE · {{ streamingLanguage }}</span>
              <span>CHUNK · {{ selectedStreamingChunkSeconds.toFixed(2) }}s</span>
            </header>
            <ol v-if="streamingPlaybackItems.length" class="streaming-playback-list">
              <li v-for="item in streamingPlaybackItems" :key="item.id">
                <header>
                  <span>UTTERANCE {{ String(item.id).padStart(2, "0") }}</span>
                  <small>{{ item.durationSeconds.toFixed(1) }}s</small>
                </header>
                <p>{{ item.text }}</p>
                <div class="input-audio-player">
                  <span>ACTUAL MODEL INPUT</span>
                  <audio :src="item.inputAudioUrl" controls preload="metadata"></audio>
                </div>
              </li>
            </ol>
            <p v-if="streamingCurrentText" class="streaming-current">
              {{ streamingCurrentText }}<i v-if="listening || streamingStatus === 'processing'" aria-hidden="true"></i>
            </p>
            <div v-else-if="!streamingPlaybackItems.length" class="streaming-placeholder">
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
              <div v-if="segment.inputAudioUrl" class="input-audio-player">
                <span>ACTUAL MODEL INPUT</span>
                <audio :src="segment.inputAudioUrl" controls preload="metadata"></audio>
              </div>
            </li>
          </ol>
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
.lifecycle-error { color:#ff8066; font:.52rem var(--font-mono); }
.asr-model-picker { align-content:center; display:flex; flex-direction:column; gap:.55rem; }
.asr-model-picker article { align-items:center; background:#f7f5eb; border:1px solid var(--line); display:grid; gap:.8rem; grid-template-columns:minmax(0,.8fr) minmax(18rem,1.2fr); min-width:0; padding:.62rem .72rem; transition:border-color .2s,background .2s,transform .2s; }
.asr-model-picker article:hover { border-color:var(--ink); transform:translateX(-3px); }
.asr-model-picker article.selected { background:#c8ff4614; border-color:var(--ink); box-shadow:inset 3px 0 var(--signal); }
.asr-model-picker article.unavailable { opacity:.62; }
.model-identity { align-items:center; background:transparent; border:0; color:var(--ink); cursor:pointer; display:grid; gap:.7rem; grid-template-columns:minmax(0,1fr) .55rem; padding:0; text-align:left; width:100%; }
.model-identity:disabled { cursor:not-allowed; }
.model-copy { display:flex; flex-direction:column; min-width:0; }
.model-identity small,.model-controls span { color:var(--muted); font:.46rem var(--font-mono); text-transform:uppercase; }
.model-identity strong { font-size:.9rem; line-height:1.15; }
.model-identity em { color:var(--muted); font-size:.58rem; font-style:normal; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.model-identity > i { background:#aaa; border-radius:50%; height:.48rem; width:.48rem; }
.asr-model-picker article.loaded .model-identity > i { background:var(--signal); box-shadow:0 0 8px #8ebd22; }
.model-controls { align-items:end; border-left:1px solid var(--line); display:grid; gap:.5rem; grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr) auto; margin:0; min-width:0; padding-left:.72rem; }
.model-controls label { display:flex; flex-direction:column; gap:.2rem; min-width:0; overflow:hidden; }
.model-controls select { background:transparent; border:0; color:var(--ink); font:600 .55rem var(--font-mono); max-width:100%; min-width:0; outline:none; overflow:hidden; padding:0; text-overflow:ellipsis; text-transform:uppercase; width:100%; }
.model-controls > button { background:var(--ink); border:0; color:var(--paper); cursor:pointer; font:700 .5rem var(--font-mono); min-width:4.4rem; padding:.58rem .65rem; }
.model-controls > button:hover { background:var(--signal); color:var(--ink); }
.model-controls > button:disabled,.model-controls select:disabled { cursor:not-allowed; opacity:.4; }
.asr-model-picker article > small { grid-column:1/-1; margin-top:-.35rem; }
.transcription-studio { border:0; display:grid; grid-template-columns:minmax(17rem,.7fr) minmax(0,1.3fr); min-height:38rem; }
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
.vad-sensitivity { align-items:center; border-bottom:1px solid #ffffff1f; border-top:1px solid #ffffff1f; display:grid; gap:.8rem; grid-template-columns:1fr 7rem 2rem; padding:1rem; }
.vad-sensitivity span { display:flex; flex-direction:column; }.vad-sensitivity b,.vad-sensitivity output { font:.56rem var(--font-mono); }.vad-sensitivity small { color:#888; font-size:.65rem; margin-top:.2rem; }
.vad-sensitivity input { accent-color:#c8ff46; width:100%; }
.pipeline-components { border-bottom:1px solid #ffffff1f; display:grid; gap:.55rem; padding:.8rem 0; }
.pipeline-components article { align-items:start; display:grid; gap:.25rem .55rem; grid-template-columns:minmax(0,1fr) auto; padding:.75rem; }
.component-toggle { align-items:center; cursor:pointer; display:flex; gap:.65rem; min-width:0; }
.component-toggle > input { accent-color:#c8ff46; flex:0 0 auto; height:.9rem; margin:0; width:.9rem; }
.component-toggle > span { display:flex; flex-direction:column; min-width:0; }
.component-toggle:has(input:disabled) { cursor:not-allowed; }
.pipeline-components strong { font:.56rem var(--font-mono); }
.pipeline-components small,.pipeline-components em,.pipeline-components > p { color:#888; font:.5rem var(--font-mono); font-style:normal; line-height:1.4; margin:0; }
.pipeline-components article > em { grid-column:1/-1; }
.pipeline-components article > span { border:1px solid #ffffff2e; font:.45rem var(--font-mono); padding:.22rem .34rem; white-space:nowrap; }
.pipeline-components .component-loaded { border-color:#c8ff46; color:#c8ff46; }
.pipeline-components .component-not-loaded { color:#f0c966; }
.pipeline-components .component-not-downloaded { color:#999; }
.pipeline-components article.component-off { opacity:.58; }
.record-button { background:#c8ff46; border:0; cursor:pointer; font:700 .68rem var(--font-mono); margin-top:auto; padding:1.1rem; }
.record-button i { background:#111; border-radius:50%; display:inline-block; height:.55rem; margin-right:.6rem; width:.55rem; }
.record-button.recording { background:#ff7148; }.record-button.recording i { border-radius:1px; }
.record-button:disabled { cursor:not-allowed; filter:grayscale(1); opacity:.35; }
.transcript-panel { display:flex; flex-direction:column; min-width:0; }
.transcript-panel > header { align-items:center; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; padding:1.2rem 1.4rem; }
.transcript-panel > header > div:first-child span { color:var(--muted); font:.55rem var(--font-mono); }
.transcript-panel > header button { background:transparent; border:1px solid var(--line); cursor:pointer; font:.55rem var(--font-mono); margin-left:.4rem; padding:.55rem; }
.transcript-panel > header button:disabled { opacity:.3; }
.transcript-scroll { flex:1; max-height:39rem; overflow:auto; overscroll-behavior:contain; padding:1.5rem; scroll-behavior:smooth; }
.streaming-transcript { display:flex; flex-direction:column; min-height:31rem; }
.streaming-transcript > header { border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:.45rem; padding-bottom:.8rem; }
.streaming-transcript > header span { background:#e4e8dc; font:.5rem var(--font-mono); padding:.32rem .45rem; }
.streaming-transcript > p { font-size:clamp(1.4rem,2.6vw,2.25rem); letter-spacing:-.025em; line-height:1.5; margin:1.8rem 0; white-space:pre-wrap; }
.streaming-transcript > p i { animation:stream-cursor .8s steps(1) infinite; background:var(--signal); display:inline-block; height:1.15em; margin-left:.18rem; vertical-align:-.12em; width:.18em; }
.streaming-transcript > footer { border-top:1px solid var(--line); color:var(--muted); font:.5rem var(--font-mono); margin-top:auto; padding-top:.8rem; }
.streaming-placeholder { align-items:center; color:var(--muted); display:flex; flex:1; flex-direction:column; justify-content:center; text-align:center; }
.streaming-placeholder > span { color:var(--ink); font-size:5rem; }.streaming-placeholder strong { color:var(--ink); font-size:1.15rem; }.streaming-placeholder small { font-size:.68rem; margin-top:.5rem; max-width:26rem; }
.streaming-error { color:#cf3f27; }
.streaming-playback-list { display:grid; gap:.8rem; list-style:none; margin:1rem 0; padding:0; }
.streaming-playback-list > li { background:color-mix(in srgb,var(--paper-deep) 62%,transparent); border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; }
.streaming-playback-list header { align-items:center; color:var(--muted); display:flex; font:.5rem var(--font-mono); justify-content:space-between; }
.streaming-playback-list p { font-size:1rem; line-height:1.55; margin:.55rem 0 .25rem; white-space:pre-wrap; }
.streaming-current { border-left:3px solid var(--signal); padding-left:1rem; }
.transcript-empty { align-items:center; color:var(--muted); display:flex; flex-direction:column; height:100%; justify-content:center; min-height:25rem; text-align:center; }
.transcript-empty > span { color:var(--ink); font-size:5rem; }.transcript-empty strong { color:var(--ink); font-size:1.35rem; }.transcript-empty p { font-size:.8rem; line-height:1.6; max-width:24rem; }
.transcript-list { list-style:none; margin:0; padding:0; }
.transcript-list li { display:grid; gap:.65rem 1.2rem; grid-template-columns:6rem minmax(0,1fr) auto; padding:1.15rem 0; }
.transcript-list li + li { border-top:1px solid var(--line); }
.segment-meta { display:grid; font: .52rem var(--font-mono); gap:.3rem; grid-template-columns:1.5rem 1fr; }
.segment-meta > span { align-items:center; background:var(--ink); color:var(--paper); display:flex; grid-row:span 2; justify-content:center; }
.segment-meta small,.transcript-list li > small { color:var(--muted); }
.segment-content > p { font-size:1.15rem; line-height:1.55; margin:0; }
.input-audio-player { align-items:center; border-top:1px solid color-mix(in srgb,var(--line) 72%,transparent); display:grid; gap:.75rem; grid-column:1/-1; grid-template-columns:auto minmax(0,1fr); margin-top:.15rem; min-width:0; padding:.5rem 0 0; }
.input-audio-player > span { color:var(--muted); font:700 .43rem var(--font-mono); letter-spacing:.09em; white-space:nowrap; }
.input-audio-player audio { background:rgb(118 118 128 / 8%); border-radius:999px; height:1.75rem; min-width:0; width:100%; }
.input-audio-player audio::-webkit-media-controls-enclosure { background:rgb(118 118 128 / 8%); border-radius:999px; }
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
  .asr-model-picker article { grid-template-columns:1fr; }
  .model-controls { border-left:0; border-top:1px solid var(--line); padding-left:0; padding-top:.65rem; }
  .transcription-studio { grid-template-columns:1fr; }
  .recorder-panel { min-height:34rem; padding:1.3rem; }
  .transcript-scroll { max-height:none; min-height:30rem; }
  .transcript-list li { gap:.8rem; grid-template-columns:4.5rem 1fr; }
  .transcript-list li > small { display:none; }
  .transcript-panel > header { align-items:flex-start; gap:1rem; }
}

/* Aurora Glass voice studio */
.transcription-hero h1 em { background:linear-gradient(90deg,#8f99aa,#d7dbe3);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-stroke:0; }
.asr-model-picker article {
  background:var(--surface);
  border:1px solid rgb(255 255 255 / 12%);
  border-radius:15px;
  box-shadow:0 8px 24px rgb(0 0 0 / 8%);
}
.asr-model-picker article:hover { border-color:rgb(10 132 255 / 28%);transform:translateX(-3px); }
.asr-model-picker article.selected { background:rgb(10 132 255 / 9%);border-color:rgb(10 132 255 / 35%);box-shadow:inset 3px 0 #0a84ff,0 10px 28px rgb(0 90 190 / 10%); }
.model-controls>button {
  background:linear-gradient(180deg,#218fff,#0878e8);
  border-radius:9px;
  color:white;
  font-family:var(--font-display);
}
.transcription-studio { gap:1rem; }
.recorder-panel,.transcript-panel {
  background:var(--surface);
  border:1px solid rgb(255 255 255 / 12%);
  border-radius:var(--radius-panel);
  box-shadow:var(--shadow-card);
  overflow:hidden;
  -webkit-backdrop-filter:blur(22px) saturate(145%);
  backdrop-filter:blur(22px) saturate(145%);
}
.recorder-panel { background:linear-gradient(150deg,rgb(31 37 49 / 94%),rgb(13 16 23 / 96%));color:#f5f5f7; }
.mic-orbit>span { filter:drop-shadow(0 10px 24px rgb(0 126 255 / 28%)); }
.mic-orbit.listening i { border-color:rgb(41 151 255 / 40%); }
.mic-orbit.speaking i { border-color:rgb(48 209 88 / 48%); }
.level-meter { background:rgb(255 255 255 / 5%);border:1px solid rgb(255 255 255 / 8%);border-radius:12px;padding:.65rem; }
.level-meter span { border-radius:999px; }
.level-meter span.active { background:#2997ff;box-shadow:0 0 8px rgb(41 151 255 / 32%); }
.level-meter span.voice { background:#30d158; }
.vad-sensitivity,.pipeline-components article { background:rgb(255 255 255 / 5%);border:1px solid rgb(255 255 255 / 8%);border-radius:13px; }
.record-button {
  background:linear-gradient(180deg,#2997ff,#0878e8);
  border-radius:14px;
  box-shadow:0 12px 28px rgb(0 105 220 / 28%);
  color:white;
  font-family:var(--font-display);
  transition:transform .18s var(--ease-spring),box-shadow .18s ease;
}
.record-button:hover:not(:disabled) { box-shadow:0 16px 36px rgb(0 121 255 / 38%);transform:translateY(-2px); }
.record-button.recording { background:linear-gradient(180deg,#ff6259,#e83c34);box-shadow:0 12px 28px rgb(255 69 58 / 24%); }
.record-button i { background:white; }
.transcript-panel>header { background:rgb(118 118 128 / 5%); }
.transcript-panel>header button { background:rgb(118 118 128 / 8%);border-radius:9px;font-family:var(--font-display);font-weight:650; }
.streaming-transcript>header span,.speech-tags span { background:rgb(10 132 255 / 9%);border:1px solid rgb(10 132 255 / 14%);border-radius:999px;color:#66b5ff; }
.streaming-transcript>p i { background:#2997ff;border-radius:99px; }
.segment-meta>span { background:linear-gradient(145deg,#2997ff,#5e5ce6);border-radius:8px;color:white; }
.transcript-list li { border-radius:15px;padding:1.1rem;transition:background .18s ease; }
.transcript-list li:hover { background:rgb(118 118 128 / 6%); }
</style>
