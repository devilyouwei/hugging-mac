<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue"

import { fetchTtsModels, loadTtsModel, synthesizeSpeech } from "./api"
import type {
  GeneratedSpeech,
  LoadedTtsModel,
  TtsModel,
} from "./types"
import { encodeWave } from "@/live_transcription/wav"
import { errorMessage, unloadModel } from "@/modelLifecycle"

const LANGUAGE_NAMES: Record<string, string> = {
  a: "American English",
  b: "British English",
  e: "Spanish",
  f: "French",
  h: "Hindi",
  i: "Italian",
  p: "Portuguese",
  j: "Japanese",
  z: "Chinese",
  auto: "Automatic",
  chinese: "Chinese",
  english: "English",
  japanese: "Japanese",
  korean: "Korean",
  german: "German",
  french: "French",
  russian: "Russian",
  portuguese: "Portuguese",
  spanish: "Spanish",
  italian: "Italian",
  "en-us": "American English",
  "en-gb": "British English",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  it: "Italian",
  pt: "Portuguese",
  ja: "Japanese",
  zh: "Chinese",
}
const KOKORO_LANGUAGE_PREFIXES: Record<string, string> = {
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
const SAMPLE_TEXTS = [
  "Every voice carries a different texture. Today, the whole studio runs locally on this Mac.",
  "A warm breeze crosses the lake, and every voice gains texture and depth.",
  "A small model can still tell a beautiful story, one sentence at a time.",
]

const models = ref<TtsModel[]>([])
const selectedModelId = ref("")
const selectedRuntimeByModel = ref<Record<string, string>>({})
const loadedModels = ref<Record<string, LoadedTtsModel>>({})
const loadingModel = ref(false)
const text = ref(SAMPLE_TEXTS[0])
const voice = ref("af_heart")
const language = ref("en-us")
const speed = ref(1)
const voiceProfile = ref("speaker_a")
const referenceAudio = ref<File | null>(null)
const referenceAudioUrl = ref("")
const referenceText = ref("")
const referenceRecording = ref(false)
const useSavedProfile = ref(false)
const synthesisBusy = ref(false)
const error = ref("")
const results = ref<GeneratedSpeech[]>([])
let resultId = 0
let activeRequest: AbortController | null = null
let referenceStream: MediaStream | null = null
let referenceContext: AudioContext | null = null
let referenceSource: MediaStreamAudioSourceNode | null = null
let referenceProcessor: ScriptProcessorNode | null = null
let referenceGain: GainNode | null = null
let referenceChunks: Float32Array[] = []

const selectedModel = computed(() =>
  models.value.find((item) => item.model_id === selectedModelId.value) ?? null,
)
const runtimeKey = (modelId: string, runtime: string) => `${modelId}::${runtime}`
const runtimeForModel = (model: TtsModel) =>
  selectedRuntimeByModel.value[model.model_id] ?? model.runtime
const resourceRuntimes = (model: TtsModel) => model.resource.runtimes?.length
  ? model.resource.runtimes
  : model.resource.artifacts
    .filter((item) => item.runtime !== null)
    .map((item) => ({
      runtime: item.runtime!,
      available: item.available,
      size_bytes: item.size_bytes ?? 0,
      artifact_ids: [item.artifact_id],
    }))
const runtimeAvailable = (model: TtsModel, runtime = runtimeForModel(model)) =>
  Boolean(resourceRuntimes(model).find((item) => item.runtime === runtime)?.available)
const runtimeOptions = (model: TtsModel) => [...resourceRuntimes(model)].sort((left, right) =>
  left.runtime === "coreml" ? -1 : right.runtime === "coreml" ? 1 : 0,
)
const runtimeLabel = (runtime: string) => runtime === "pytorch-mps"
  ? "Torch MPS"
  : runtime === "pytorch" ? "PyTorch CPU" : runtime === "coreml" ? "Core ML" : runtime
const loadedInstanceFor = (model: TtsModel) =>
  loadedModels.value[runtimeKey(model.model_id, runtimeForModel(model))] ?? null
const modelIsReady = (model: TtsModel) => loadedInstanceFor(model)?.state === "ready"
const sourceArtifact = computed(() => {
  const model = selectedModel.value
  if (!model) return undefined
  return model.resource.artifacts.find((item) => item.runtime === runtimeForModel(model))
})
const loadedModel = computed(() =>
  selectedModel.value ? loadedInstanceFor(selectedModel.value) : null,
)
const modelReady = computed(() => loadedModel.value?.state === "ready")
const isKokoro = computed(
  () => Boolean(selectedModel.value?.voices.length) && !selectedModel.value?.requires_reference_voice,
)
const isAudio8Clone = computed(() => Boolean(selectedModel.value?.requires_reference_voice))
const isQwen3 = computed(() => Boolean(selectedModel.value?.requires_reference_audio))
const isQwen3CoreMl = computed(() =>
  isQwen3.value && selectedModel.value !== null && runtimeForModel(selectedModel.value) === "coreml",
)
const isQwen3Clone = computed(() => isQwen3.value && !isQwen3CoreMl.value)
const qwen3CoreMlLanguages = computed(() =>
  (selectedModel.value?.languages ?? []).filter((item) => item === "english" || item === "chinese"),
)
const availableVoices = computed(() => {
  const prefix = KOKORO_LANGUAGE_PREFIXES[language.value]
  const voices = selectedModel.value?.voices ?? []
  return prefix ? voices.filter((item) => item.startsWith(prefix)) : voices
})
const characterCount = computed(() => text.value.length)
const referenceReady = computed(() =>
  isQwen3Clone.value
    ? referenceAudio.value !== null && Boolean(referenceText.value.trim())
    : !isAudio8Clone.value
    || (Boolean(voiceProfile.value.trim())
    && (useSavedProfile.value
      || (referenceAudio.value !== null && Boolean(referenceText.value.trim())))),
)
const canGenerate = computed(() =>
  Boolean(text.value.trim()) && referenceReady.value && !synthesisBusy.value && !loadingModel.value
    && Boolean(sourceArtifact.value?.available),
)

watch([language, selectedModel], () => {
  if (isKokoro.value && !availableVoices.value.includes(voice.value)) {
    voice.value = availableVoices.value[0] ?? ""
  }
})

watch(isQwen3CoreMl, (enabled) => {
  if (enabled && !qwen3CoreMlLanguages.value.some((item) => item === language.value)) {
    language.value = qwen3CoreMlLanguages.value[0] ?? "english"
  }
})

async function loadModels() {
  try {
    models.value = await fetchTtsModels()
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
    selectedModelId.value =
      models.value.find((item) => item.model_id === "hexgrad/kokoro")?.model_id
      ?? models.value[0]?.model_id
      ?? ""
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "Failed to load the TTS model list"
  }
}

async function ensureSelectedModelLoaded(): Promise<LoadedTtsModel> {
  if (modelReady.value && loadedModel.value) return loadedModel.value
  if (!selectedModel.value || !sourceArtifact.value?.available) {
    throw new Error("模型资产不可用，请先在 Models 页面准备模型。")
  }
  loadingModel.value = true
  error.value = ""
  try {
    const runtime = runtimeForModel(selectedModel.value)
    const loaded = await loadTtsModel(selectedModel.value.model_id, runtime)
    loadedModels.value[runtimeKey(loaded.model_id, loaded.runtime)] = loaded
    return loaded
  } catch (caught) {
    const message = errorMessage(caught, "TTS 模型加载失败")
    error.value = message
    throw caught
  } finally {
    loadingModel.value = false
  }
}

function selectModel(modelId: string) {
  if (synthesisBusy.value) return
  selectedModelId.value = modelId
  const model = models.value.find((item) => item.model_id === modelId)
  if (model?.languages.length && !model.languages.includes(language.value)) {
    language.value = model.languages[0] ?? "en-us"
  }
  error.value = ""
}

function selectRuntime(model: TtsModel, runtime: string) {
  if (synthesisBusy.value || loadingModel.value) return
  selectedRuntimeByModel.value[model.model_id] = runtime
}

function onRuntimeChange(model: TtsModel, event: Event) {
  selectRuntime(model, (event.target as HTMLSelectElement).value)
}

async function toggleSelectedModel() {
  const current = loadedModel.value
  if (loadingModel.value) return
  if (!current) {
    try { await ensureSelectedModelLoaded() } catch { /* inline feedback */ }
    return
  }
  loadingModel.value = true
  error.value = ""
  try {
    await unloadModel(current.instance_id)
    delete loadedModels.value[runtimeKey(current.model_id, current.runtime)]
  } catch (caught) {
    error.value = errorMessage(caught, "模型卸载失败")
  } finally {
    loadingModel.value = false
  }
}

async function toggleModel(model: TtsModel) {
  selectModel(model.model_id)
  await toggleSelectedModel()
}

function selectReferenceAudio(event: Event) {
  const input = event.target as HTMLInputElement
  setReferenceAudio(input.files?.[0] ?? null)
  input.value = ""
}

function setReferenceAudio(file: File | null) {
  if (referenceAudioUrl.value) URL.revokeObjectURL(referenceAudioUrl.value)
  referenceAudio.value = file
  referenceAudioUrl.value = file ? URL.createObjectURL(file) : ""
}

async function startReferenceRecording() {
  if (referenceRecording.value) return
  error.value = ""
  try {
    referenceStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    })
    referenceContext = new AudioContext()
    referenceSource = referenceContext.createMediaStreamSource(referenceStream)
    referenceProcessor = referenceContext.createScriptProcessor(4096, 1, 1)
    referenceGain = referenceContext.createGain()
    referenceGain.gain.value = 0
    referenceChunks = []
    referenceProcessor.onaudioprocess = (event) => {
      referenceChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)))
    }
    referenceSource.connect(referenceProcessor)
    referenceProcessor.connect(referenceGain)
    referenceGain.connect(referenceContext.destination)
    referenceRecording.value = true
  } catch (caught) {
    stopReferenceCapture()
    error.value = caught instanceof Error ? caught.message : "Microphone access failed"
  }
}

async function stopReferenceRecording() {
  if (!referenceContext) return
  const wav = encodeWave(referenceChunks, referenceContext.sampleRate)
  setReferenceAudio(new File([wav], `reference-${Date.now()}.wav`, { type: "audio/wav" }))
  stopReferenceCapture()
}

function stopReferenceCapture() {
  referenceProcessor?.disconnect()
  referenceSource?.disconnect()
  referenceGain?.disconnect()
  referenceStream?.getTracks().forEach((track) => track.stop())
  void referenceContext?.close()
  referenceStream = null
  referenceContext = null
  referenceSource = null
  referenceProcessor = null
  referenceGain = null
  referenceRecording.value = false
}

async function generateSpeech() {
  const model = selectedModel.value
  if (!model || !text.value.trim() || synthesisBusy.value) return
  synthesisBusy.value = true
  error.value = ""
  activeRequest = new AbortController()
  try {
    const instance = await ensureSelectedModelLoaded()
    const response = await synthesizeSpeech(
      {
        model_id: model.model_id,
        instance_id: instance.instance_id,
        text: text.value.trim(),
        voice: isAudio8Clone.value
          ? voiceProfile.value.trim()
          : isKokoro.value ? voice.value : null,
        language: isKokoro.value || isQwen3.value ? language.value : null,
        speed: speed.value,
        referenceAudio: (isAudio8Clone.value && !useSavedProfile.value) || isQwen3Clone.value
          ? referenceAudio.value
          : null,
        referenceText: (isAudio8Clone.value && !useSavedProfile.value) || isQwen3Clone.value
          ? referenceText.value.trim()
          : null,
      },
      activeRequest.signal,
    )
    const url = URL.createObjectURL(response.blob)
    results.value.unshift({
      id: ++resultId,
      modelId: model.model_id,
      modelName: model.short_name,
      text: text.value.trim(),
      voice: isAudio8Clone.value
        ? voiceProfile.value.trim()
        : isKokoro.value ? voice.value : isQwen3Clone.value ? "reference" : null,
      runtime: response.headers.get("x-runtime") ?? instance.runtime,
      device: response.headers.get("x-device") ?? instance.device,
      durationSeconds: Number(response.headers.get("x-duration-seconds") ?? 0),
      inferenceMs: Number(response.headers.get("x-inference-ms") ?? 0),
      url,
      createdAt: new Date(),
    })
  } catch (caught) {
    if (!(caught instanceof DOMException && caught.name === "AbortError")) {
      error.value = caught instanceof Error ? caught.message : "Speech generation failed"
    }
  } finally {
    activeRequest = null
    synthesisBusy.value = false
  }
}

function useSample(index: number) {
  text.value = SAMPLE_TEXTS[index] ?? SAMPLE_TEXTS[0]
}

function removeResult(id: number) {
  const item = results.value.find((result) => result.id === id)
  if (item) URL.revokeObjectURL(item.url)
  results.value = results.value.filter((result) => result.id !== id)
}

onMounted(loadModels)
onBeforeUnmount(() => {
  activeRequest?.abort()
  stopReferenceCapture()
  if (referenceAudioUrl.value) URL.revokeObjectURL(referenceAudioUrl.value)
  results.value.forEach((item) => URL.revokeObjectURL(item.url))
})
</script>

<template>
  <div class="page inner-page tts-page">
    <header class="tts-hero">
      <div class="hero-copy">
        <RouterLink class="back-link" to="/apps">← Neural Apps</RouterLink>
        <p class="kicker">LOCAL TEXT-TO-SPEECH · MULTI-RUNTIME VOICE STUDIO</p>
        <h1>Type it. <em>Hear it.</em></h1>
        <p>Compare Audio8, Kokoro, and Qwen3-TTS in one local studio. Your text, reference voice, and generated audio stay on this Mac.</p>
      </div>
      <section class="tts-model-picker" aria-label="Select a text-to-speech model">
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
            :disabled="synthesisBusy"
            @click="selectModel(model.model_id)"
          >
            <span>♪</span>
            <span>
              <small>{{ model.variant }} · {{ modelIsReady(model) ? "LOADED" : "TTS MODEL" }}</small>
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
                :disabled="synthesisBusy || loadingModel"
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
              :disabled="(!modelIsReady(model) && !runtimeAvailable(model)) || loadingModel || synthesisBusy"
              @click="toggleModel(model)"
            >
              {{ loadingModel && selectedModelId === model.model_id ? "WAIT…" : modelIsReady(model) ? "UNLOAD" : "LOAD" }}
            </button>
          </div>
        </article>
      </section>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>

    <main class="tts-workbench">
      <section class="script-panel">
        <div class="panel-heading">
          <div>
            <p class="kicker">01 / SCRIPT</p>
            <h2>What should it say?</h2>
          </div>
          <span>{{ characterCount }} / 2000</span>
        </div>
        <textarea
          v-model="text"
          maxlength="2000"
          placeholder="Enter the text you want to hear…"
          aria-label="Text to synthesize"
        ></textarea>
        <div class="sample-row">
          <span>TRY A SAMPLE</span>
          <button v-for="(_, index) in SAMPLE_TEXTS" :key="index" type="button" @click="useSample(index)">
            {{ ["EN", "WARM", "STORY"][index] }}
          </button>
        </div>
      </section>

      <aside class="voice-panel">
        <div class="panel-heading">
          <div>
            <p class="kicker">02 / VOICE ENGINE</p>
            <h2>{{ selectedModel?.short_name ?? "Loading…" }}</h2>
          </div>
          <i :class="{ ready: modelReady }"></i>
        </div>

        <div v-if="isAudio8Clone" class="reference-controls">
          <label>
            <span>VOICE PROFILE</span>
            <input v-model="voiceProfile" maxlength="64" placeholder="speaker_a" />
          </label>
          <div class="profile-mode" role="group" aria-label="Voice profile source">
            <button
              type="button"
              :class="{ active: !useSavedProfile }"
              @click="useSavedProfile = false"
            >NEW REFERENCE</button>
            <button
              type="button"
              :class="{ active: useSavedProfile }"
              @click="useSavedProfile = true"
            >SAVED PROFILE</button>
          </div>
          <template v-if="!useSavedProfile">
            <label class="reference-file">
              <span>REFERENCE AUDIO · 0.5–30 SEC</span>
              <input
                type="file"
                accept=".wav,.flac,.mp3,.ogg,audio/wav,audio/flac,audio/mpeg,audio/ogg"
                @change="selectReferenceAudio"
              />
              <b>{{ referenceAudio?.name ?? "CHOOSE AUDIO" }}</b>
            </label>
            <div class="reference-actions">
              <button type="button" :disabled="referenceRecording" @click="startReferenceRecording">Record reference</button>
              <button v-if="referenceRecording" type="button" class="recording" @click="stopReferenceRecording">Stop recording</button>
            </div>
            <audio
              v-if="referenceAudioUrl"
              class="reference-player"
              :src="referenceAudioUrl"
              controls
              preload="metadata"
            ></audio>
            <label>
              <span>ACCURATE TRANSCRIPT</span>
              <textarea
                v-model="referenceText"
                class="reference-transcript"
                placeholder="Enter the exact transcript spoken in the reference audio…"
              ></textarea>
            </label>
            <p>The first synthesis saves this profile. Select Saved Profile to reuse it later.</p>
          </template>
          <p v-else>The locally saved <strong>{{ voiceProfile || "unnamed" }}</strong> profile will be used.</p>
        </div>
        <div v-else-if="isKokoro" class="voice-controls">
          <label>
            <span>LANGUAGE</span>
            <select v-model="language">
              <option v-for="item in selectedModel?.languages" :key="item" :value="item">
                {{ LANGUAGE_NAMES[item] ?? item }}
              </option>
            </select>
          </label>
          <label>
            <span>VOICE</span>
            <select v-model="voice">
              <option v-for="item in availableVoices" :key="item" :value="item">
                {{ item.replaceAll("_", " ") }}
              </option>
            </select>
          </label>
        </div>
        <div v-else-if="isQwen3CoreMl" class="voice-controls">
          <label>
            <span>LANGUAGE</span>
            <select v-model="language">
              <option v-for="item in qwen3CoreMlLanguages" :key="item" :value="item">
                {{ LANGUAGE_NAMES[item] ?? item }}
              </option>
            </select>
          </label>
          <div class="audio8-note">
            <strong>Bundled speaker</strong>
            <p>Core ML uses the model's built-in voice and supports English and Chinese.</p>
          </div>
        </div>
        <div v-else-if="isQwen3Clone" class="reference-controls">
          <label class="reference-file">
            <span>REFERENCE AUDIO · CLEAR SPEECH</span>
            <input
              type="file"
              accept=".wav,.flac,.mp3,.ogg,audio/wav,audio/flac,audio/mpeg,audio/ogg"
              @change="selectReferenceAudio"
            />
            <b>{{ referenceAudio?.name ?? "CHOOSE AUDIO" }}</b>
          </label>
          <div class="reference-actions">
            <button type="button" :disabled="referenceRecording" @click="startReferenceRecording">Record reference</button>
            <button v-if="referenceRecording" type="button" class="recording" @click="stopReferenceRecording">Stop recording</button>
          </div>
          <audio
            v-if="referenceAudioUrl"
            class="reference-player"
            :src="referenceAudioUrl"
            controls
            preload="metadata"
          ></audio>
          <label>
            <span>ACCURATE TRANSCRIPT</span>
            <textarea
              v-model="referenceText"
              class="reference-transcript"
              placeholder="Enter exactly what is spoken in the reference audio…"
            ></textarea>
          </label>
          <label class="reference-language">
            <span>LANGUAGE</span>
            <select v-model="language">
              <option v-for="item in selectedModel?.languages" :key="item" :value="item">
                {{ LANGUAGE_NAMES[item] ?? item }}
              </option>
            </select>
          </label>
          <p>Qwen3-TTS uses this reference for the current generation; it is not saved as a profile.</p>
        </div>
        <div v-else class="audio8-note">
          <strong>Natural multilingual mode</strong>
          <p>Audio8 detects the input language automatically and generates natural speech.</p>
        </div>

        <label class="speed-control">
          <span><b>SPEED</b><output>{{ speed.toFixed(2) }}×</output></span>
          <input v-model.number="speed" type="range" min="0.5" max="2" step="0.05" />
        </label>

        <button
          class="generate-button"
          type="button"
          :disabled="!canGenerate"
          @click="generateSpeech"
        >
          <span>{{ synthesisBusy ? "SYNTHESIZING" : "GENERATE SPEECH" }}</span>
          <i>{{ synthesisBusy ? "•••" : "▶" }}</i>
        </button>
      </aside>
    </main>

    <section class="output-section">
      <header>
        <div>
          <p class="kicker">03 / OUTPUT BENCH</p>
          <h2>Listen side by side.</h2>
        </div>
        <span>{{ results.length }} GENERATION{{ results.length === 1 ? "" : "S" }}</span>
      </header>
      <div v-if="!results.length" class="empty-output">
        <span>♪</span>
        <p>Generated audio appears here. Use the same text across models for a direct comparison.</p>
      </div>
      <article v-for="result in results" :key="result.id" class="audio-result">
        <div class="result-model">
          <span>{{ result.modelId.includes("kokoro") ? "K" : result.modelId.includes("qwen3-tts") ? "Q3" : "A8" }}</span>
          <div>
            <strong>{{ result.modelName }}</strong>
            <small>{{ result.runtime }} · {{ result.voice ?? "automatic voice" }}</small>
          </div>
        </div>
        <p>{{ result.text }}</p>
        <audio :src="result.url" controls preload="metadata"></audio>
        <div class="result-meta">
          <span>{{ result.durationSeconds.toFixed(1) }} SEC</span>
          <span>{{ result.inferenceMs.toFixed(0) }} MS</span>
          <a :href="result.url" :download="`hugging-mac-${result.id}.wav`">DOWNLOAD ↓</a>
          <button type="button" aria-label="Delete audio" @click="removeResult(result.id)">×</button>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.tts-page { padding-bottom: 6rem; }
.tts-hero { align-items: stretch; display: grid; gap: clamp(1.5rem, 3vw, 3.5rem); grid-template-columns: minmax(0, 1fr) minmax(25rem, .72fr); min-height: 0; padding: .65rem 0 1.4rem; }
.hero-copy { align-self: center; min-width: 0; }
.tts-hero h1 { font-size: clamp(2.4rem, 5vw, 4.6rem); font-weight: 600; letter-spacing: -.065em; line-height: .95; margin: .55rem 0 .8rem; white-space: nowrap; }
.tts-hero h1 em { color: transparent; font-style: normal; -webkit-text-stroke: 1.5px var(--ink); }
.hero-copy > p:last-child { color: var(--muted); font-size: .82rem; line-height: 1.5; max-width: 42rem; }
.tts-model-picker { align-content: center; display: flex; flex-direction: column; gap: .55rem; }
.tts-model-picker article { align-items: center; background: #f7f5eb; border: 1px solid var(--line); box-sizing: border-box; display: grid; gap: .8rem; grid-template-columns: minmax(0, 1fr) minmax(10.5rem, .62fr); height: 4.5rem; padding: .62rem .72rem; transition: border-color .2s, background .2s, transform .2s; }
.tts-model-picker article:hover { border-color: var(--ink); transform: translateX(-3px); }
.tts-model-picker article.selected { background: #c8ff4614; border-color: var(--ink); box-shadow: inset 3px 0 var(--signal); }
.tts-model-picker article.unavailable { opacity: .62; }
.model-identity { align-items: center; background: transparent; border: 0; color: var(--ink); cursor: pointer; display: grid; gap: .7rem; grid-template-columns: 2rem minmax(0, 1fr) .55rem; padding: 0; text-align: left; width: 100%; }
.model-identity:disabled { cursor: not-allowed; }
.model-identity > span:first-child { font-size: 1.35rem; text-align: center; }
.model-identity > span:nth-child(2) { display: flex; flex-direction: column; min-width: 0; }
.model-identity small, .model-controls span { color: var(--muted); font: .46rem var(--font-mono); text-transform: uppercase; }
.model-identity strong { font-size: .9rem; line-height: 1.15; }
.model-identity em { color: var(--muted); font-size: .58rem; font-style: normal; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.model-identity > i { background: #aaa; border-radius: 50%; height: .48rem; width: .48rem; }
.tts-model-picker article.loaded .model-identity > i { background: var(--signal); box-shadow: 0 0 8px #8ebd22; }
.model-controls { align-items: end; border-left: 1px solid var(--line); display: grid; gap: .5rem; grid-template-columns: minmax(5.5rem, 1fr) auto; padding-left: .72rem; }
.model-controls label { display: flex; flex-direction: column; gap: .2rem; }
.model-controls select { background: transparent; border: 0; color: var(--ink); font: 600 .55rem var(--font-mono); min-width: 0; outline: none; padding: 0; text-transform: uppercase; width: 100%; }
.model-controls > button { background: var(--ink); border: 0; color: var(--paper); cursor: pointer; font: 700 .5rem var(--font-mono); min-width: 4.4rem; padding: .58rem .65rem; }
.model-controls > button:hover { background: var(--signal); color: var(--ink); }
.model-controls > button:disabled, .model-controls select:disabled { cursor: not-allowed; opacity: .4; }
.tts-workbench { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(21rem, .65fr); margin-top: 1rem; }
.script-panel, .voice-panel { border: 1px solid var(--ink); min-height: 33rem; padding: clamp(1.5rem, 3vw, 2.6rem); }
.voice-panel { background: var(--ink); color: var(--paper); }
.panel-heading { align-items: start; display: flex; justify-content: space-between; }
.panel-heading h2, .output-section h2 { font-size: clamp(2rem, 4vw, 3.8rem); letter-spacing: -.06em; margin: .6rem 0 1.8rem; }
.panel-heading > span { font-family: var(--font-mono); font-size: .7rem; }
.panel-heading > i { background: var(--orange); border-radius: 50%; height: .7rem; margin-top: .5rem; width: .7rem; }
.panel-heading > i.ready { background: var(--signal); box-shadow: 0 0 0 5px rgb(184 242 60 / 15%); }
textarea { background: transparent; border: 0; border-bottom: 1px solid var(--line); color: var(--ink); font-size: clamp(1.5rem, 3vw, 2.5rem); font-weight: 500; height: 17rem; line-height: 1.35; outline: 0; padding: .5rem 0 2rem; resize: none; width: 100%; }
.sample-row { align-items: center; display: flex; gap: .5rem; margin-top: 1.3rem; }
.sample-row > span { color: var(--muted); font-family: var(--font-mono); font-size: .62rem; margin-right: auto; }
.sample-row button { background: transparent; border: 1px solid var(--line); cursor: pointer; font-family: var(--font-mono); font-size: .65rem; padding: .5rem .7rem; }
.voice-controls { display: grid; gap: 1rem; grid-template-columns: 1fr 1fr; }
.voice-controls label, .speed-control { display: grid; gap: .55rem; }
.voice-controls span, .speed-control span { color: #999b92; font-family: var(--font-mono); font-size: .64rem; letter-spacing: .08em; }
.voice-controls select { background: #292a26; border: 1px solid #4a4b46; color: var(--paper); min-height: 3.2rem; padding: 0 .8rem; text-transform: capitalize; }
.audio8-note { border: 1px solid #44453f; margin-bottom: 1rem; padding: 1rem; }
.audio8-note p { color: #a9aaa2; font-size: .83rem; line-height: 1.5; margin-bottom: 0; }
.reference-controls { display: grid; gap: .75rem; }
.reference-controls label { display: grid; gap: .5rem; }
.reference-controls label > span { color: #999b92; font-family: var(--font-mono); font-size: .64rem; letter-spacing: .08em; }
.reference-controls input[type="text"], .reference-controls label > input:not([type="file"]) { background: #292a26; border: 1px solid #4a4b46; color: var(--paper); min-height: 2.8rem; padding: 0 .8rem; }
.profile-mode { display: grid; grid-template-columns: 1fr 1fr; }
.profile-mode button { background: transparent; border: 1px solid #4a4b46; color: #999b92; cursor: pointer; font-family: var(--font-mono); font-size: .6rem; min-height: 2.5rem; }
.profile-mode button + button { border-left: 0; }
.profile-mode button.active { background: var(--signal); color: var(--ink); }
.reference-file { border: 1px dashed #5b5c55; cursor: pointer; padding: .75rem; }
.reference-file input { height: 1px; opacity: 0; position: absolute; width: 1px; }
.reference-file b { color: var(--signal); font-family: var(--font-mono); font-size: .65rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.reference-actions { display: flex; gap: .6rem; }
.reference-actions button { background: transparent; border: 1px solid #5b5c55; color: var(--signal); cursor: pointer; font: .62rem var(--font-mono); padding: .65rem .75rem; }
.reference-actions button.recording { border-color: var(--orange); color: var(--orange); }
.reference-actions button:disabled { cursor: not-allowed; opacity: .45; }
.reference-player { height: 2.5rem; width: 100%; }
.reference-transcript { background: #292a26; border: 1px solid #4a4b46; color: var(--paper); font-size: .8rem; height: 4.5rem; line-height: 1.4; padding: .65rem; resize: vertical; }
.reference-controls > p { color: #999b92; font-size: .72rem; line-height: 1.45; margin: 0; }
.reference-controls > p strong { color: var(--signal); }
.reference-language { display: grid; gap: .5rem; }
.reference-language select { background: #292a26; border: 1px solid #4a4b46; color: var(--paper); min-height: 2.8rem; padding: 0 .8rem; }
.speed-control { margin: 1.8rem 0; }
.speed-control span { display: flex; justify-content: space-between; }
.speed-control output { color: var(--signal); }
.speed-control input { accent-color: var(--signal); width: 100%; }
.generate-button { align-items: center; background: var(--signal); border: 0; cursor: pointer; display: flex; font-family: var(--font-mono); font-size: .75rem; font-weight: 800; justify-content: space-between; min-height: 4.5rem; padding: 0 1.2rem; width: 100%; }
.generate-button:disabled { cursor: not-allowed; opacity: .35; }
.generate-button i { font-size: 1.25rem; font-style: normal; }
.output-section { border-top: 1px solid var(--line); margin-top: 5rem; padding-top: 2.5rem; }
.output-section > header { align-items: end; display: flex; justify-content: space-between; }
.output-section > header > span { font-family: var(--font-mono); font-size: .66rem; margin-bottom: 2rem; }
.empty-output { align-items: center; border: 1px dashed var(--line); color: var(--muted); display: flex; gap: 1.2rem; justify-content: center; min-height: 10rem; }
.empty-output span { font-size: 2rem; }
.audio-result { align-items: center; border-top: 1px solid var(--line); display: grid; gap: 1.5rem; grid-template-columns: 13rem minmax(12rem, 1fr) minmax(16rem, .75fr) auto; padding: 1.3rem 0; }
.result-model { align-items: center; display: flex; gap: .8rem; }
.result-model > span { align-items: center; background: var(--ink); color: var(--signal); display: flex; font-family: var(--font-mono); height: 2.8rem; justify-content: center; width: 2.8rem; }
.result-model div { display: grid; gap: .25rem; }
.result-model small, .result-meta { color: var(--muted); font-family: var(--font-mono); font-size: .58rem; text-transform: uppercase; }
.audio-result > p { display: -webkit-box; font-size: .85rem; line-height: 1.45; margin: 0; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
.audio-result audio { height: 2.5rem; width: 100%; }
.result-meta { align-items: end; display: grid; gap: .2rem; justify-items: end; }
.result-meta a { color: var(--ink); font-weight: 800; }
.result-meta button { background: none; border: 0; cursor: pointer; font-size: 1.2rem; }
@media (max-width: 900px) {
  .tts-hero, .tts-workbench { grid-template-columns: 1fr; }
  .tts-model-picker { width: 100%; }
  .audio-result { grid-template-columns: 1fr; }
  .result-meta { display: flex; justify-content: start; }
}
@media (max-width: 600px) {
  .tts-hero { padding-top: .5rem; }
  .tts-hero h1 { font-size: 3rem; white-space: normal; }
  .voice-controls { grid-template-columns: 1fr; }
  .sample-row { flex-wrap: wrap; }
  .sample-row > span { width: 100%; }
}
</style>
