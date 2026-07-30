<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue"

import {
  convertAsrCoreMl,
  downloadAsrWeights,
  fetchAsrResourceStatus,
  transcribeUtterance,
} from "./api"
import type {
  AsrResourceStatus,
  TranscriptSegment,
} from "./types"
import { encodeWave } from "./wav"

const SILENCE_SECONDS = 0.65
const MIN_UTTERANCE_SECONDS = 0.35
const MAX_UTTERANCE_SECONDS = 25
const PRE_ROLL_SECONDS = 0.18

const resource = ref<AsrResourceStatus | null>(null)
const resourceBusy = ref(false)
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

const sourceArtifact = computed(() =>
  resource.value?.artifacts.find((artifact) => artifact.artifact_id === "source"),
)
const coremlArtifact = computed(() =>
  resource.value?.artifacts.find((artifact) => artifact.artifact_id === "coreml"),
)
const sourceReady = computed(() => Boolean(sourceArtifact.value?.available))
const coremlReady = computed(() => Boolean(coremlArtifact.value?.available))
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

async function loadResource() {
  try {
    resource.value = await fetchAsrResourceStatus()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

async function downloadModel() {
  resourceBusy.value = true
  error.value = ""
  try {
    resource.value = await downloadAsrWeights()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Audio8-ASR 下载失败"
  } finally {
    resourceBusy.value = false
  }
}

async function convertModel() {
  resourceBusy.value = true
  error.value = ""
  try {
    resource.value = await convertAsrCoreMl()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Audio8-ASR Core ML 转换失败"
  } finally {
    resourceBusy.value = false
  }
}

async function startListening() {
  if (listening.value) return
  if (!coremlReady.value) {
    error.value = sourceReady.value
      ? "请先将 Audio8-ASR 转换为 Core ML"
      : "请先下载 Audio8-ASR 模型权重并转换为 Core ML"
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
  const segment: TranscriptSegment = {
    id: ++segmentSequence,
    createdAt: new Date(),
    durationSeconds,
    status: "recognizing",
    text: "",
    inferenceMs: null,
  }
  segments.value.push(segment)
  pendingCount.value += 1
  void nextTick(() => transcriptEnd.value?.scrollIntoView({ behavior: "smooth" }))
  recognitionQueue = recognitionQueue.then(async () => {
    const controller = new AbortController()
    requests.add(controller)
    try {
      const response = await transcribeUtterance(audio, controller.signal)
      const liveSegment = segments.value.find((item) => item.id === segment.id)
      if (!liveSegment) return
      liveSegment.text = response.text || "（未识别到清晰语音）"
      liveSegment.inferenceMs = response.inference_ms
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

onMounted(loadResource)
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
        <p class="kicker">AUDIO8-ASR · VAD-DRIVEN LIVE TRANSCRIPTION</p>
        <h1><span aria-hidden="true">🎙️</span> Voice,<br /><em>made visible.</em></h1>
        <p>声音留在本机。VAD 自动感知说话与停顿，每次停顿都会形成一句新的识别文本。</p>
      </div>
      <aside class="asr-model-card">
        <div class="asr-model-card__top"><span>LOCAL MODEL</span><i :class="{ ready: coremlReady }"></i></div>
        <strong>Audio8-ASR</strong>
        <span>0.1B · CORE ML / ANE + MPS</span>
        <div class="model-download">
          <div>
            <b>{{ coremlReady ? "READY" : sourceReady ? "CORE ML REQUIRED" : "WEIGHTS REQUIRED" }}</b>
            <small>{{ coremlReady ? formatBytes(coremlArtifact?.size_bytes) : formatBytes(sourceArtifact?.size_bytes) }}</small>
          </div>
          <button v-if="!sourceReady" type="button" :disabled="resourceBusy || !resource" @click="downloadModel">
            {{ resourceBusy ? "DOWNLOADING…" : "DOWNLOAD" }}
          </button>
          <button v-else-if="!coremlReady" type="button" :disabled="resourceBusy" @click="convertModel">
            {{ resourceBusy ? "CONVERTING…" : "CONVERT" }}
          </button>
        </div>
      </aside>
    </header>

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
          :disabled="!coremlReady"
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
                <small>{{ segment.durationSeconds.toFixed(1) }}s</small>
              </div>
              <p v-if="segment.status === 'recognizing'"><i></i><i></i><i></i></p>
              <p v-else>{{ segment.text }}</p>
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
.transcript-list li > p { font-size:1.15rem; line-height:1.55; margin:0; }
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
  .transcription-studio { grid-template-columns:1fr; }
  .recorder-panel { min-height:34rem; padding:1.3rem; }
  .transcript-scroll { max-height:none; min-height:30rem; }
  .transcript-list li { gap:.8rem; grid-template-columns:4.5rem 1fr; }
  .transcript-list li > small { display:none; }
  .transcript-panel > header { align-items:flex-start; gap:1rem; }
}
</style>
