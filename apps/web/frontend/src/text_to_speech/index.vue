<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue"

import {
  downloadTtsWeights,
  fetchTtsModels,
  loadTtsModel,
  synthesizeSpeech,
} from "./api"
import type {
  GeneratedSpeech,
  LoadedTtsModel,
  TtsModel,
} from "./types"

const LANGUAGE_NAMES: Record<string, string> = {
  a: "American English",
  b: "British English",
  e: "Español",
  f: "Français",
  h: "हिन्दी",
  i: "Italiano",
  p: "Português",
  j: "日本語",
  z: "中文",
}
const SAMPLE_TEXTS = [
  "Every voice carries a different texture. Today, the whole studio runs locally on this Mac.",
  "春风吹过湖面，声音也可以拥有温度与层次。",
  "A small model can still tell a beautiful story, one sentence at a time.",
]

const models = ref<TtsModel[]>([])
const selectedModelId = ref("")
const loadedModels = ref<Record<string, LoadedTtsModel>>({})
const text = ref(SAMPLE_TEXTS[0])
const voice = ref("af_heart")
const language = ref("a")
const speed = ref(1)
const resourceBusy = ref(false)
const loadBusy = ref(false)
const synthesisBusy = ref(false)
const error = ref("")
const results = ref<GeneratedSpeech[]>([])
let resultId = 0
let activeRequest: AbortController | null = null

const selectedModel = computed(() =>
  models.value.find((item) => item.model_id === selectedModelId.value) ?? null,
)
const sourceArtifact = computed(() =>
  selectedModel.value?.resource.artifacts.find((item) => item.artifact_id === "source"),
)
const sourceReady = computed(() => Boolean(sourceArtifact.value?.available))
const requiredReady = computed(() =>
  Boolean(
    selectedModel.value?.resource.artifacts.find(
      (item) => item.artifact_id === selectedModel.value?.required_artifact_id,
    )?.available,
  ),
)
const loadedModel = computed(() =>
  selectedModel.value ? loadedModels.value[selectedModel.value.model_id] ?? null : null,
)
const modelReady = computed(() => loadedModel.value?.state === "ready")
const isKokoro = computed(() => selectedModel.value?.model_id === "hexgrad/kokoro-82m")
const characterCount = computed(() => text.value.length)

function formatBytes(value: number | null | undefined): string {
  if (value == null) return "not downloaded"
  return `${(value / 1024 ** 2).toFixed(0)} MB`
}

async function loadModels() {
  try {
    models.value = await fetchTtsModels()
    for (const model of models.value) {
      if (model.ready_instance_id) {
        loadedModels.value[model.model_id] = {
          instance_id: model.ready_instance_id,
          model_id: model.model_id,
          variant: model.variant,
          runtime: model.runtime,
          device: model.runtime,
          state: "ready",
        }
      }
    }
    selectedModelId.value =
      models.value.find((item) => item.model_id === "hexgrad/kokoro-82m")?.model_id
      ?? models.value[0]?.model_id
      ?? ""
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "TTS 模型列表读取失败"
  }
}

function selectModel(modelId: string) {
  if (synthesisBusy.value) return
  selectedModelId.value = modelId
  const model = models.value.find((item) => item.model_id === modelId)
  if (model?.voices.length && !model.voices.includes(voice.value)) {
    voice.value = model.voices[0] ?? "af_heart"
  }
  if (model?.languages.length && !model.languages.includes(language.value)) {
    language.value = model.languages[0] ?? "a"
  }
  error.value = ""
}

async function downloadModel() {
  if (!selectedModel.value) return
  resourceBusy.value = true
  error.value = ""
  try {
    selectedModel.value.resource = await downloadTtsWeights(selectedModel.value.model_id)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型下载失败"
  } finally {
    resourceBusy.value = false
  }
}

async function prepareModel() {
  if (!selectedModel.value || !requiredReady.value) return
  loadBusy.value = true
  error.value = ""
  try {
    loadedModels.value[selectedModel.value.model_id] =
      await loadTtsModel(selectedModel.value.model_id)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型加载失败"
  } finally {
    loadBusy.value = false
  }
}

async function generateSpeech() {
  const model = selectedModel.value
  const instance = loadedModel.value
  if (!model || !instance || !text.value.trim()) return
  synthesisBusy.value = true
  error.value = ""
  activeRequest = new AbortController()
  try {
    const response = await synthesizeSpeech(
      {
        model_id: model.model_id,
        instance_id: instance.instance_id,
        text: text.value.trim(),
        voice: isKokoro.value ? voice.value : null,
        language: isKokoro.value ? language.value : null,
        speed: speed.value,
      },
      activeRequest.signal,
    )
    const url = URL.createObjectURL(response.blob)
    results.value.unshift({
      id: ++resultId,
      modelId: model.model_id,
      modelName: model.short_name,
      text: text.value.trim(),
      voice: isKokoro.value ? voice.value : null,
      runtime: response.headers.get("x-runtime") ?? model.runtime,
      device: response.headers.get("x-device") ?? model.runtime,
      durationSeconds: Number(response.headers.get("x-duration-seconds") ?? 0),
      inferenceMs: Number(response.headers.get("x-inference-ms") ?? 0),
      url,
      createdAt: new Date(),
    })
  } catch (caught) {
    if (!(caught instanceof DOMException && caught.name === "AbortError")) {
      error.value = caught instanceof Error ? caught.message : "语音生成失败"
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
  results.value.forEach((item) => URL.revokeObjectURL(item.url))
})
</script>

<template>
  <div class="page inner-page tts-page">
    <header class="tts-hero">
      <div>
        <RouterLink class="back-link" to="/apps">← Neural Apps</RouterLink>
        <p class="kicker">LOCAL TEXT-TO-SPEECH · TWO ENGINES, ONE STUDIO</p>
        <h1>Type it.<br /><em>Hear it.</em></h1>
        <p>在同一个本地工作台里试听 Audio8 的表现力与 Kokoro 的速度。文字和生成音频都留在这台 Mac。</p>
      </div>
      <div class="sound-object" aria-hidden="true">
        <span v-for="bar in 18" :key="bar" :style="{ '--bar': bar }"></span>
        <strong>24 / 44.1</strong>
        <small>KHZ · LOCAL WAVEFORM</small>
      </div>
    </header>

    <section class="tts-model-switcher" aria-label="选择语音合成模型">
      <button
        v-for="(model, index) in models"
        :key="model.model_id"
        type="button"
        :class="{ selected: selectedModelId === model.model_id }"
        :disabled="synthesisBusy"
        @click="selectModel(model.model_id)"
      >
        <span class="model-index">0{{ index + 1 }}</span>
        <div>
          <small>{{ model.runtime }} · {{ model.variant }}</small>
          <strong>{{ model.display_name }}</strong>
          <p>{{ model.description }}</p>
        </div>
        <i>{{ selectedModelId === model.model_id ? "ACTIVE" : "COMPARE →" }}</i>
      </button>
    </section>

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
          placeholder="输入要转换成语音的文字…"
          aria-label="要合成的文字"
        ></textarea>
        <div class="sample-row">
          <span>TRY A SAMPLE</span>
          <button v-for="(_, index) in SAMPLE_TEXTS" :key="index" type="button" @click="useSample(index)">
            {{ ["EN", "中文", "STORY"][index] }}
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

        <div v-if="isKokoro" class="voice-controls">
          <label>
            <span>VOICE</span>
            <select v-model="voice">
              <option v-for="item in selectedModel?.voices" :key="item" :value="item">
                {{ item.replaceAll("_", " ") }}
              </option>
            </select>
          </label>
          <label>
            <span>LANGUAGE</span>
            <select v-model="language">
              <option v-for="item in selectedModel?.languages" :key="item" :value="item">
                {{ LANGUAGE_NAMES[item] ?? item }}
              </option>
            </select>
          </label>
        </div>
        <div v-else class="audio8-note">
          <strong>Natural multilingual mode</strong>
          <p>Audio8 会从输入文字自动判断语言并生成自然语音。</p>
        </div>

        <label class="speed-control">
          <span><b>SPEED</b><output>{{ speed.toFixed(2) }}×</output></span>
          <input v-model.number="speed" type="range" min="0.5" max="2" step="0.05" />
        </label>

        <div class="model-state">
          <div>
            <span>{{ modelReady ? "MODEL LOADED" : requiredReady ? "READY TO LOAD" : "SETUP REQUIRED" }}</span>
            <small v-if="modelReady">{{ loadedModel?.device }}</small>
            <small v-else>{{ formatBytes(sourceArtifact?.size_bytes) }}</small>
          </div>
          <button
            v-if="!sourceReady"
            type="button"
            :disabled="resourceBusy"
            @click="downloadModel"
          >{{ resourceBusy ? "DOWNLOADING…" : "DOWNLOAD WEIGHTS" }}</button>
          <button
            v-else-if="!modelReady"
            type="button"
            :disabled="loadBusy"
            @click="prepareModel"
          >{{ loadBusy ? "LOADING…" : "LOAD MODEL" }}</button>
        </div>

        <button
          class="generate-button"
          type="button"
          :disabled="!modelReady || !text.trim() || synthesisBusy"
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
        <p>生成的音频会出现在这里。切换模型并使用同一段文字，就能直接比较。</p>
      </div>
      <article v-for="result in results" :key="result.id" class="audio-result">
        <div class="result-model">
          <span>{{ result.modelId.includes("kokoro") ? "K" : "A8" }}</span>
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
          <button type="button" aria-label="删除音频" @click="removeResult(result.id)">×</button>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.tts-page { padding-bottom: 6rem; }
.tts-hero { align-items: end; border-bottom: 1px solid var(--line); display: grid; gap: 5vw; grid-template-columns: 1.4fr .6fr; min-height: 31rem; padding: 4rem 0 3rem; }
.tts-hero h1 { font-size: clamp(4.5rem, 10vw, 9rem); font-weight: 600; letter-spacing: -.08em; line-height: .78; margin: 1.4rem 0 2rem; }
.tts-hero h1 em { color: transparent; font-style: normal; -webkit-text-stroke: 1.5px var(--ink); }
.tts-hero > div:first-child > p:last-child { font-size: 1.08rem; line-height: 1.7; max-width: 42rem; }
.sound-object { align-items: center; background: var(--ink); color: var(--paper); display: flex; gap: .35rem; height: 19rem; justify-content: center; overflow: hidden; padding: 2rem; position: relative; }
.sound-object span { animation: sound-pulse 1.8s ease-in-out infinite alternate; animation-delay: calc(var(--bar) * -80ms); background: var(--signal); height: calc(18px + var(--bar) * 4px); opacity: .82; width: 5px; }
.sound-object strong { bottom: 1.8rem; font-family: var(--font-mono); font-size: 1.2rem; left: 1.8rem; position: absolute; }
.sound-object small { bottom: 1.8rem; color: #999b92; font-family: var(--font-mono); position: absolute; right: 1.8rem; }
@keyframes sound-pulse { to { transform: scaleY(.35); } }
.tts-model-switcher { display: grid; grid-template-columns: 1fr 1fr; margin: 2rem 0; }
.tts-model-switcher button { align-items: center; background: transparent; border: 1px solid var(--line); cursor: pointer; display: grid; gap: 1.2rem; grid-template-columns: auto 1fr auto; min-height: 10rem; padding: 1.4rem; text-align: left; }
.tts-model-switcher button + button { border-left: 0; }
.tts-model-switcher button.selected { background: var(--ink); color: var(--paper); }
.model-index { color: var(--muted); font-family: var(--font-mono); font-size: .72rem; }
.tts-model-switcher small, .tts-model-switcher i { color: var(--muted); font-family: var(--font-mono); font-size: .64rem; font-style: normal; text-transform: uppercase; }
.tts-model-switcher strong { display: block; font-size: 1.35rem; margin: .35rem 0; }
.tts-model-switcher p { font-size: .86rem; line-height: 1.45; margin: 0; max-width: 32rem; }
.tts-model-switcher .selected i { color: var(--signal); }
.tts-workbench { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(21rem, .65fr); margin-top: 2rem; }
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
.speed-control { margin: 1.8rem 0; }
.speed-control span { display: flex; justify-content: space-between; }
.speed-control output { color: var(--signal); }
.speed-control input { accent-color: var(--signal); width: 100%; }
.model-state { align-items: center; border-top: 1px solid #44453f; display: flex; justify-content: space-between; padding: 1.4rem 0; }
.model-state div { display: grid; gap: .3rem; }
.model-state span, .model-state small { font-family: var(--font-mono); font-size: .64rem; }
.model-state small { color: #999b92; }
.model-state button { background: transparent; border: 1px solid var(--signal); color: var(--signal); cursor: pointer; font-family: var(--font-mono); font-size: .63rem; padding: .7rem; }
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
  .sound-object { height: 12rem; }
  .tts-model-switcher { grid-template-columns: 1fr; }
  .tts-model-switcher button + button { border-left: 1px solid var(--line); border-top: 0; }
  .audio-result { grid-template-columns: 1fr; }
  .result-meta { display: flex; justify-content: start; }
}
@media (max-width: 600px) {
  .tts-hero { padding-top: 2rem; }
  .tts-hero h1 { font-size: 4.4rem; }
  .voice-controls { grid-template-columns: 1fr; }
  .sample-row { flex-wrap: wrap; }
  .sample-row > span { width: 100%; }
}
</style>
