<script setup lang="ts">
import type { Component } from "vue"
import { computed, onMounted, ref } from "vue"

import { errorMessage, loadSharedModel, unloadModel } from "@/modelLifecycle"
import type {
  ResourceStatus,
  RuntimeChoice,
  VisionInference,
  VisionOptions,
} from "../types"
import CameraInferenceTab from "./CameraInferenceTab.vue"
import ImageInferenceTab from "./ImageInferenceTab.vue"
import VideoInferenceTab from "./VideoInferenceTab.vue"

type InputMode = "image" | "camera" | "video"

const props = defineProps<{
  appNumber: string
  kicker: string
  titleTop: string
  titleBottom: string
  description: string
  modelLabel: string
  actionLabel: string
  actionNoun: string
  emptyCopy: string
  overlay: Component
  results: Component
  infer: VisionInference
  fetchStatus: (variant?: string) => Promise<ResourceStatus>
}>()

const activeMode = ref<InputMode>("camera")
const runtime = ref<RuntimeChoice>("auto")
const confidence = ref(0.5)
const iouThreshold = ref(0.7)
const maxDetections = ref(100)
const selectedVariant = ref("")
const error = ref("")
const resourceStatus = ref<ResourceStatus | null>(null)
const loadedInstances = ref<Record<string, string>>({})
const lifecycleBusy = ref(false)
const lifecycleMessage = ref<{ type: "success" | "error"; text: string } | null>(null)
const inferenceOptions = computed<VisionOptions>(() => ({
  runtime: runtime.value,
  variant: selectedVariant.value || resourceStatus.value?.variant || "n",
  confidence: confidence.value,
  iouThreshold: iouThreshold.value,
  maxDetections: maxDetections.value,
}))

const effectiveRuntime = computed(() =>
  runtime.value === "auto" ? resourceStatus.value?.default_runtime : runtime.value,
)
const runtimeArtifactReady = computed(() => {
  const selected = effectiveRuntime.value
  if (!selected || !resourceStatus.value) return false
  return resourceStatus.value.artifacts.some(
    (artifact) => artifact.runtime === selected && artifact.available,
  )
})
const modelKey = computed(() => `${selectedVariant.value}:${effectiveRuntime.value ?? "auto"}`)
const loadedInstanceId = computed(() => loadedInstances.value[modelKey.value] ?? null)

async function ensureModelLoaded(): Promise<string> {
  if (loadedInstanceId.value) return loadedInstanceId.value
  const status = resourceStatus.value
  const selectedRuntime = effectiveRuntime.value
  if (!status || !selectedRuntime || !runtimeArtifactReady.value) {
    throw new Error("当前模型资产不可用，请先在 Models 页面下载或转换。")
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    const loaded = await loadSharedModel(status.model_id, selectedVariant.value, selectedRuntime)
    loadedInstances.value[modelKey.value] = loaded.instance_id
    lifecycleMessage.value = { type: "success", text: "模型加载成功" }
    return loaded.instance_id
  } catch (caught) {
    const message = errorMessage(caught, "模型加载失败")
    lifecycleMessage.value = { type: "error", text: message }
    throw caught
  } finally {
    lifecycleBusy.value = false
  }
}

async function toggleModel() {
  if (lifecycleBusy.value) return
  const instanceId = loadedInstanceId.value
  if (!instanceId) {
    try { await ensureModelLoaded() } catch { /* feedback is displayed inline */ }
    return
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    await unloadModel(instanceId)
    delete loadedInstances.value[modelKey.value]
    lifecycleMessage.value = { type: "success", text: "模型卸载成功" }
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "模型卸载失败") }
  } finally {
    lifecycleBusy.value = false
  }
}

const managedInfer: VisionInference = async (file, options, requestOptions) => {
  await ensureModelLoaded()
  return props.infer(file, options, requestOptions)
}

async function refreshResources() {
  lifecycleMessage.value = null
  try {
    resourceStatus.value = await props.fetchStatus(selectedVariant.value || undefined)
    if (!selectedVariant.value) selectedVariant.value = resourceStatus.value.variant
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型资源状态读取失败"
  }
}

onMounted(refreshResources)
</script>

<template>
  <div class="page detection-page">
    <header class="app-hero">
      <div>
        <RouterLink class="back-link" to="/apps">← Apps</RouterLink>
        <p class="kicker">APP {{ appNumber }} / {{ kicker }}</p>
        <h1>{{ titleTop }} {{ titleBottom }}</h1>
      </div>
      <p>{{ description }}</p>
    </header>

    <div class="detection-workspace">
      <aside class="control-panel">
        <div class="control-panel__heading">
          <span>INPUT / CONFIG</span>
          <span>01—04</span>
        </div>
        <label class="field model-field">
          <span>{{ modelLabel }} variant</span>
          <div class="model-field__control">
            <select v-model="selectedVariant" :disabled="lifecycleBusy" @change="refreshResources">
              <option
                v-for="variant in resourceStatus?.variants ?? []"
                :key="variant.name"
                :value="variant.name"
              >
                {{ variant.display_name }} · {{ variant.description }}
              </option>
            </select>
            <button type="button" :disabled="lifecycleBusy || (!loadedInstanceId && !runtimeArtifactReady)" @click="toggleModel">
              {{ lifecycleBusy ? "WAIT…" : loadedInstanceId ? "UNLOAD" : "LOAD" }}
            </button>
          </div>
          <small v-if="lifecycleMessage" :class="`lifecycle-${lifecycleMessage.type}`">{{ lifecycleMessage.text }}</small>
        </label>

        <label class="field">
          <span>Runtime</span>
          <select v-model="runtime">
            <option value="auto">Auto · machine policy</option>
            <option value="coreml">Core ML</option>
            <option value="pytorch-mps">PyTorch MPS</option>
          </select>
        </label>

        <label class="field range-field">
          <span>Confidence <output>{{ confidence.toFixed(2) }}</output></span>
          <input v-model.number="confidence" type="range" min="0.05" max="0.95" step="0.05" />
        </label>

        <label class="field range-field">
          <span>IoU threshold <output>{{ iouThreshold.toFixed(2) }}</output></span>
          <input v-model.number="iouThreshold" type="range" min="0.1" max="0.9" step="0.05" />
        </label>

        <label class="field">
          <span>Maximum instances</span>
          <input
            v-model.number="maxDetections"
            class="number-input"
            type="number"
            min="1"
            max="1000"
          />
        </label>

        <p v-if="!runtimeArtifactReady" class="resource-note">
          当前 runtime 的模型资产尚未准备。请前往 Models 页面下载、转换并加载模型。
        </p>
        <p class="local-note">
          推理不会自动下载或转换模型；图片与视频帧也不会上传到第三方。
        </p>
      </aside>

      <section class="output-panel">
        <div class="output-panel__heading">
          <span>INPUT / LIVE OUTPUT</span>
          <span>{{ activeMode.toUpperCase() }}</span>
        </div>
        <div class="mode-tabs" role="tablist" aria-label="输入方式">
          <button
            type="button"
            role="tab"
            :aria-selected="activeMode === 'image'"
            :class="{ 'mode-tab--active': activeMode === 'image' }"
            @click="activeMode = 'image'"
          >
            <span>01</span> Image
          </button>
          <button
            type="button"
            role="tab"
            :aria-selected="activeMode === 'camera'"
            :class="{ 'mode-tab--active': activeMode === 'camera' }"
            @click="activeMode = 'camera'"
          >
            <span>02</span> Camera
          </button>
          <button
            type="button"
            role="tab"
            :aria-selected="activeMode === 'video'"
            :class="{ 'mode-tab--active': activeMode === 'video' }"
            @click="activeMode = 'video'"
          >
            <span>03</span> Video
          </button>
        </div>

        <div v-if="error" class="error-banner detection-error" role="alert">
          <div>
            <strong>Inference unavailable</strong>
            <span>{{ error }}</span>
          </div>
        </div>

        <ImageInferenceTab
          v-if="activeMode === 'image'"
          :options="inferenceOptions"
          :enabled="runtimeArtifactReady"
          :infer="managedInfer"
          :overlay="overlay"
          :results="results"
          :action-label="actionLabel"
          :empty-copy="emptyCopy"
        />
        <CameraInferenceTab
          v-else-if="activeMode === 'camera'"
          :options="inferenceOptions"
          :enabled="runtimeArtifactReady"
          :infer="managedInfer"
          :overlay="overlay"
          :results="results"
          :action-noun="actionNoun"
        />
        <VideoInferenceTab
          v-else
          :options="inferenceOptions"
          :enabled="runtimeArtifactReady"
          :infer="managedInfer"
          :overlay="overlay"
          :results="results"
        />
      </section>
    </div>
  </div>
</template>

<style scoped>
.model-field__control { display:flex; gap:.4rem; }
.model-field__control select { min-width:0; }
.model-field__control button { background:var(--ink); border:0; color:var(--paper); cursor:pointer; flex:0 0 auto; font:.55rem var(--font-mono); padding:0 .65rem; }
.model-field__control button:disabled { cursor:not-allowed; opacity:.4; }
.model-field small { font:.52rem var(--font-mono); }
.lifecycle-success { color:#3c8b2f; }
.lifecycle-error { color:#cf3f27; }
</style>
