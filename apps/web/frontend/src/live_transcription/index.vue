<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue"

import {
  convertAsrCoreMl,
  downloadAsrWeights,
  fetchAsrModels,
  loadAsrModel,
  transcribeUtterance,
} from "./api"
import type {
  AsrModel,
  AsrResourceStatus,
  LoadedAsrModel,
  TranscriptSegment,
} from "./types"
import { encodeWave } from "./wav"

const SILENCE_SECONDS = 0.65
const MIN_UTTERANCE_SECONDS = 0.35
const MAX_UTTERANCE_SECONDS = 25
const PRE_ROLL_SECONDS = 0.18

const models = ref<AsrModel[]>([])
const selectedModelId = ref("")
const loadedModel = ref<LoadedAsrModel | null>(null)
const resourceBusy = ref(false)
const loadBusy = ref(false)
const listening = ref(false)
const speaking = ref(false)
const inputLevel = ref(0)
const sensitivity = ref(0.65)
const elapsedSeconds = ref(0)
const segments = ref<TranscriptSegment[]>([])
const pendingCount = ref(0)
const error = ref("")
const transcriptEnd = ref<HTMLElement | null>(null)

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
let sessionStartedAt = 0
let elapsedTimer: number | null = null
let segmentSequence = 0
let recognitionQueue = Promise.resolve()
const requests = new Set<AbortController>()

const selectedModel = computed(() =>
  models.value.find((model) => model.model_id === selectedModelId.value) ?? null,
)
const resource = computed(() => selectedModel.value?.resource ?? null)
const sourceArtifact = computed(() =>
  resource.value?.artifacts.find((artifact) => artifact.artifact_id === "source"),
)
const coremlArtifact = computed(() =>
  resource.value?.artifacts.find((artifact) => artifact.artifact_id === "coreml"),
)
const sourceReady = computed(() => Boolean(sourceArtifact.value?.available))
const coremlReady = computed(() => Boolean(coremlArtifact.value?.available))
const requiredArtifactReady = computed(() =>
  Boolean(
    resource.value?.artifacts.find(
      (artifact) => artifact.artifact_id === selectedModel.value?.required_artifact_id,
    )?.available,
  ),
)
const modelReady = computed(() =>
  Boolean(
    loadedModel.value
      && loadedModel.value.model_id === selectedModelId.value
      && loadedModel.value.state === "ready",
  ),
)
const threshold = computed(() => 0.036 - sensitivity.value * 0.026)
const transcriptText = computed(() =>
  segments.value
    .filter((segment) => segment.status === "complete" && segment.text)
    .map((segment) => segment.text)
    .join(" "),
)

function formatBytes(value: number | null | undefined): string {
  if (value == null) return "not downloaded"
  return `${(value / 1024 ** 2).toFixed(0)} MB`
}

async function loadModels() {
  try {
    models.value = await fetchAsrModels()
    selectedModelId.value ||= models.value[0]?.model_id ?? ""
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "ASR 模型列表读取失败"
  }
}

async function downloadModel() {
  if (!selectedModel.value) return
  resourceBusy.value = true
  error.value = ""
  try {
    selectedModel.value.resource = await downloadAsrWeights(selectedModel.value.model_id)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "ASR 模型下载失败"
  } finally {
    resourceBusy.value = false
  }
}

async function convertModel() {
  if (!selectedModel.value) return
  resourceBusy.value = true
  error.value = ""
  try {
    selectedModel.value.resource = await convertAsrCoreMl(selectedModel.value.model_id)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Audio8-ASR Core ML 转换失败"
  } finally {
    resourceBusy.value = false
  }
}

function selectModel(modelId: string) {
  if (listening.value || pendingCount.value) return
  stopAudioGraph()
  selectedModelId.value = modelId
  loadedModel.value = null
  error.value = ""
}

async function prepareModel() {
  if (!selectedModel.value || !requiredArtifactReady.value) return
  loadBusy.value = true
  error.value = ""
  try {
    loadedModel.value = await loadAsrModel(selectedModel.value.model_id)
  } catch (caught) {
    loadedModel.value = null
    error.value = caught instanceof Error ? caught.message : "ASR 模型加载失败"
  } finally {
    loadBusy.value = false
  }
}

async function startListening() {
  if (listening.value) return
  if (!modelReady.value) {
    error.value = "请先加载当前选择的 ASR 模型"
    return
  }
  error.value = ""
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true,
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
    listening.value = true
    sessionStartedAt = performance.now()
    elapsedTimer = window.setInterval(() => {
      elapsedSeconds.value = (performance.now() - sessionStartedAt) / 1000
    }, 250)
  } catch (caught) {
    stopAudioGraph()
    error.value = caught instanceof DOMException && caught.name === "NotAllowedError"
      ? "麦克风权限被拒绝，请在浏览器站点设置中允许访问"
      : caught instanceof Error ? caught.message : "麦克风启动失败"
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
  const voiced = rms >= threshold.value

  if (!speaking.value) {
    preRoll.push(chunk)
    preRollSamples += chunk.length
    const preRollLimit = audioContext.sampleRate * PRE_ROLL_SECONDS
    while (preRollSamples > preRollLimit && preRoll.length > 1) {
      preRollSamples -= preRoll.shift()!.length
    }
    if (!voiced) return
    speaking.value = true
    currentChunks = preRoll
    currentSamples = preRollSamples
    preRoll = []
    preRollSamples = 0
    silenceSamples = 0
  }

  currentChunks.push(chunk)
  currentSamples += chunk.length
  silenceSamples = voiced ? 0 : silenceSamples + chunk.length
  const duration = currentSamples / audioContext.sampleRate
  if (
    silenceSamples / audioContext.sampleRate >= SILENCE_SECONDS ||
    duration >= MAX_UTTERANCE_SECONDS
  ) {
    finishUtterance(audioContext.sampleRate)
  }
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

function enqueueRecognition(audio: Blob, durationSeconds: number) {
  const model = selectedModel.value
  const instance = loadedModel.value
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
        controller.signal,
      )
      const liveSegment = segments.value.find((item) => item.id === segment.id)
      if (!liveSegment) return
      liveSegment.text = response.text || "（未识别到清晰语音）"
      liveSegment.inferenceMs = response.inference_ms
      liveSegment.languages = response.languages
      liveSegment.emotion = response.emotion
      liveSegment.events = response.events
      liveSegment.status = "complete"
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      const liveSegment = segments.value.find((item) => item.id === segment.id)
      if (liveSegment) {
        liveSegment.text = caught instanceof Error ? caught.message : "识别失败"
        liveSegment.status = "error"
      }
    } finally {
      requests.delete(controller)
      pendingCount.value = Math.max(0, pendingCount.value - 1)
      void nextTick(() => transcriptEnd.value?.scrollIntoView({ behavior: "smooth" }))
    }
  })
}

function stopListening() {
  if (audioContext && currentSamples / audioContext.sampleRate >= MIN_UTTERANCE_SECONDS) {
    finishUtterance(audioContext.sampleRate)
  }
  stopAudioGraph()
}

function stopAudioGraph() {
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
}

async function copyTranscript() {
  if (transcriptText.value) await navigator.clipboard.writeText(transcriptText.value)
}

function clearTranscript() {
  segments.value = []
}

onMounted(loadModels)
onBeforeUnmount(() => {
  stopAudioGraph()
  requests.forEach((request) => request.abort())
})
</script>

<template>
  <div class="page inner-page transcription-page">
    <header class="transcription-hero">
      <div>
        <RouterLink class="back-link" to="/apps">← Neural Apps</RouterLink>
        <p class="kicker">MULTI-MODEL ASR · VAD-DRIVEN LIVE TRANSCRIPTION</p>
        <h1><span aria-hidden="true">🎙️</span> Voice,<br /><em>made visible.</em></h1>
        <p>选择模型、在本机加载，然后开始聆听。VAD 自动感知说话与停顿，每次停顿都会形成一句新的识别文本。</p>
      </div>
      <aside class="asr-model-card">
        <div class="asr-model-card__top"><span>SELECTED MODEL</span><i :class="{ ready: modelReady }"></i></div>
        <strong>{{ selectedModel?.short_name ?? "Loading…" }}</strong>
        <span>{{ selectedModel?.variant.toUpperCase() }} · {{ selectedModel?.runtime.toUpperCase() }}</span>
        <div class="model-download">
          <div>
            <b>{{ modelReady ? "LOADED" : requiredArtifactReady ? "READY TO LOAD" : sourceReady && selectedModel?.supports_coreml_conversion ? "CORE ML REQUIRED" : "WEIGHTS REQUIRED" }}</b>
            <small v-if="modelReady">{{ loadedModel?.device }} · instance ready</small>
            <small v-else>{{ requiredArtifactReady ? "assets prepared" : formatBytes(sourceArtifact?.size_bytes) }}</small>
          </div>
          <button v-if="!sourceReady" type="button" :disabled="resourceBusy || !resource" @click="downloadModel">
            {{ resourceBusy ? "DOWNLOADING…" : "DOWNLOAD" }}
          </button>
          <button v-else-if="selectedModel?.supports_coreml_conversion && !coremlReady" type="button" :disabled="resourceBusy" @click="convertModel">
            {{ resourceBusy ? "CONVERTING…" : "CONVERT" }}
          </button>
          <button v-else-if="!modelReady" type="button" :disabled="loadBusy || !requiredArtifactReady" @click="prepareModel">
            {{ loadBusy ? "LOADING…" : "LOAD MODEL" }}
          </button>
        </div>
      </aside>
    </header>

    <section class="asr-model-picker" aria-label="选择语音识别模型">
      <button
        v-for="model in models"
        :key="model.model_id"
        type="button"
        :class="{ selected: selectedModelId === model.model_id }"
        :disabled="listening || pendingCount > 0 || loadBusy"
        @click="selectModel(model.model_id)"
      >
        <span>{{ model.rich_understanding ? "🧠" : "⚡" }}</span>
        <div>
          <small>{{ model.runtime }} · {{ model.variant }}</small>
          <strong>{{ model.display_name }}</strong>
          <p>{{ model.description }}</p>
        </div>
        <i>{{ selectedModelId === model.model_id ? "SELECTED" : "SELECT →" }}</i>
      </button>
    </section>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>

    <section class="transcription-studio">
      <aside class="recorder-panel">
        <div class="recorder-state">
          <div class="mic-orbit" :class="{ listening, speaking }">
            <span>🎙️</span><i></i><i></i><i></i>
          </div>
          <p class="kicker">{{ listening ? (speaking ? "VOICE DETECTED" : "LISTENING FOR SPEECH") : "MICROPHONE IDLE" }}</p>
          <strong>{{ listening ? elapsedSeconds.toFixed(1) : "0.0" }}<small> SEC</small></strong>
        </div>

        <div class="level-meter" aria-label="麦克风输入音量">
          <span
            v-for="bar in 24"
            :key="bar"
            :class="{ active: inputLevel * 24 >= bar, voice: speaking }"
          ></span>
        </div>

        <label class="vad-sensitivity">
          <span><b>VAD SENSITIVITY</b><small>语音触发灵敏度</small></span>
          <input v-model.number="sensitivity" type="range" min="0" max="1" step="0.05" />
          <output>{{ Math.round(sensitivity * 100) }}%</output>
        </label>

        <button
          class="record-button"
          :class="{ recording: listening }"
          type="button"
          :disabled="!modelReady"
          @click="listening ? stopListening() : startListening()"
        >
          <i></i>{{ listening ? "STOP LISTENING" : "START LISTENING" }}
        </button>
        <p class="recorder-note">停顿 {{ Math.round(SILENCE_SECONDS * 1000) }}ms 自动断句 · 单句最长 {{ MAX_UTTERANCE_SECONDS }}s</p>
      </aside>

      <article class="transcript-panel">
        <header>
          <div>
            <p class="kicker">LIVE TRANSCRIPT</p>
            <span>{{ segments.length }} utterances · {{ pendingCount }} processing</span>
          </div>
          <div>
            <button type="button" :disabled="!transcriptText" @click="copyTranscript">COPY</button>
            <button type="button" :disabled="!segments.length" @click="clearTranscript">CLEAR</button>
          </div>
        </header>

        <div class="transcript-scroll">
          <div v-if="!segments.length" class="transcript-empty">
            <span aria-hidden="true">〰</span>
            <strong>Your words will appear here.</strong>
            <p>点击 Start Listening，然后自然说话。短暂停顿后，识别结果会逐句写入。</p>
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
              <small v-if="segment.inferenceMs != null">{{ segment.inferenceMs.toFixed(0) }}ms inference</small>
            </li>
          </ol>
          <div ref="transcriptEnd"></div>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.transcription-page { padding-top:2.2rem; }
.transcription-hero { align-items:end; display:grid; gap:3rem; grid-template-columns:1fr 19rem; padding:1rem 0 2.5rem; }
.transcription-hero h1 { font-size:clamp(3.8rem,7.5vw,8rem); letter-spacing:-.075em; line-height:.78; margin:1.2rem 0 1.8rem; }
.transcription-hero h1 > span { display:inline-block; font-size:.58em; margin-right:.2em; transform:rotate(-8deg); }
.transcription-hero h1 em { color:transparent; font-style:normal; -webkit-text-stroke:1.5px var(--ink); }
.transcription-hero > div > p:last-child { color:var(--muted); line-height:1.65; max-width:42rem; }
.asr-model-card { background:var(--ink); color:var(--paper); padding:1.35rem; }
.asr-model-card__top { align-items:center; border-bottom:1px solid #ffffff24; display:flex; font:.58rem var(--font-mono); justify-content:space-between; padding-bottom:.8rem; }
.asr-model-card__top i { background:#ff7148; border-radius:50%; height:.55rem; width:.55rem; }
.asr-model-card__top i.ready { background:var(--signal); box-shadow:0 0 12px var(--signal); }
.asr-model-card > strong { display:block; font-size:2.1rem; letter-spacing:-.05em; margin-top:1.1rem; }
.asr-model-card > span { color:#999; font:.58rem var(--font-mono); }
.model-download { align-items:center; border-top:1px solid #ffffff24; display:flex; justify-content:space-between; margin-top:1.2rem; padding-top:1rem; }
.model-download div { display:flex; flex-direction:column; }
.model-download b,.model-download small,.model-download button { font:.54rem var(--font-mono); }
.model-download small { color:#888; margin-top:.25rem; }
.model-download button { background:var(--signal); border:0; cursor:pointer; font-weight:700; padding:.65rem; }
.asr-model-picker { display:grid; gap:.8rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin-bottom:1.2rem; }
.asr-model-picker > button { align-items:center; background:transparent; border:1px solid var(--line); color:var(--ink); cursor:pointer; display:grid; gap:1rem; grid-template-columns:auto 1fr auto; padding:1rem; text-align:left; transition:.2s ease; }
.asr-model-picker > button:hover { border-color:var(--ink); transform:translateY(-2px); }
.asr-model-picker > button.selected { background:#c8ff4618; border-color:var(--ink); box-shadow:inset 0 -3px var(--signal); }
.asr-model-picker > button:disabled { cursor:not-allowed; opacity:.55; transform:none; }
.asr-model-picker > button > span { font-size:1.8rem; }
.asr-model-picker > button div { display:flex; flex-direction:column; gap:.22rem; }
.asr-model-picker small,.asr-model-picker i { color:var(--muted); font:.5rem var(--font-mono); text-transform:uppercase; }
.asr-model-picker strong { font-size:1rem; }
.asr-model-picker p { color:var(--muted); font-size:.67rem; line-height:1.4; margin:0; }
.asr-model-picker i { color:var(--ink); font-style:normal; }
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
@media (max-width:760px) {
  .transcription-page { padding-top:1rem; }
  .transcription-hero { align-items:stretch; gap:1.5rem; grid-template-columns:1fr; }
  .transcription-hero h1 { font-size:4rem; }
  .asr-model-card { padding:1rem; }
  .asr-model-picker { grid-template-columns:1fr; }
  .asr-model-picker > button { grid-template-columns:auto 1fr; }
  .asr-model-picker i { display:none; }
  .transcription-studio { grid-template-columns:1fr; }
  .recorder-panel { min-height:34rem; padding:1.3rem; }
  .transcript-scroll { max-height:none; min-height:30rem; }
  .transcript-list li { gap:.8rem; grid-template-columns:4.5rem 1fr; }
  .transcript-list li > small { display:none; }
  .transcript-panel > header { align-items:flex-start; gap:1rem; }
}
</style>
