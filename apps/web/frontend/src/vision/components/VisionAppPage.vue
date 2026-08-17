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
  secondaryModelLabel?: string
  fetchSecondaryStatus?: () => Promise<ResourceStatus>
  tertiaryModelLabel?: string
  fetchTertiaryStatus?: () => Promise<ResourceStatus>
  defaultConfidence?: number
  defaultSecondaryConfidence?: number
  defaultTertiaryConfidence?: number
}>()

const activeMode = ref<InputMode>("camera")
const runtime = ref<RuntimeChoice>("auto")
const confidence = ref(props.defaultConfidence ?? 0.5)
const faceConfidence = ref(props.defaultSecondaryConfidence ?? 0.5)
const handConfidence = ref(props.defaultTertiaryConfidence ?? 0.5)
const handLandmarksEnabled = ref(false)
const primaryEnabled = ref(true)
const secondaryEnabled = ref(true)
const tertiaryEnabled = ref(true)
const iouThreshold = ref(0.7)
const maxDetections = ref(100)
const selectedVariant = ref("")
const error = ref("")
const resourceStatus = ref<ResourceStatus | null>(null)
const secondaryResourceStatus = ref<ResourceStatus | null>(null)
const tertiaryResourceStatus = ref<ResourceStatus | null>(null)
const loadedInstances = ref<Record<string, string>>({})
const loadedRuntimes = ref<Record<string, string>>({})
const lifecycleBusy = ref(false)
const lifecycleMessage = ref<{ type: "success" | "error"; text: string } | null>(null)
const inferenceOptions = computed<VisionOptions>(() => ({
  runtime: runtime.value,
  variant: selectedVariant.value || resourceStatus.value?.variant || "n",
  confidence: confidence.value,
  iouThreshold: iouThreshold.value,
  maxDetections: maxDetections.value,
  ...(props.fetchSecondaryStatus
    ? {
        poseEnabled: primaryEnabled.value,
        faceEnabled: secondaryEnabled.value,
        faceConfidence: faceConfidence.value,
        ...(props.fetchTertiaryStatus
          ? {
              handEnabled: tertiaryEnabled.value,
              handConfidence: handConfidence.value,
              handLandmarksEnabled: handLandmarksEnabled.value,
              handLandmarkConfidence: 0.5,
            }
          : {}),
      }
    : {}),
}))

function artifactReady(status: ResourceStatus | null, selectedRuntime: string): boolean {
  if (!status) return false
  return status.artifacts.some(
    (artifact) => artifact.available
      && (selectedRuntime === "auto" || artifact.runtime === selectedRuntime),
  )
}

const runtimeArtifactReady = computed(() => {
  return artifactReady(resourceStatus.value, runtime.value)
})
const modelKey = computed(() => `${selectedVariant.value}:${runtime.value}`)
const loadedInstanceId = computed(() => loadedInstances.value[modelKey.value] ?? null)
const loadedRuntime = computed(() => loadedRuntimes.value[modelKey.value] ?? null)
const secondaryModelKey = computed(() =>
  `secondary:${secondaryResourceStatus.value?.variant ?? "default"}:auto`,
)
const secondaryLoadedInstanceId = computed(
  () => loadedInstances.value[secondaryModelKey.value] ?? null,
)
const secondaryLoadedRuntime = computed(
  () => loadedRuntimes.value[secondaryModelKey.value] ?? null,
)
const secondaryArtifactReady = computed(() => {
  return artifactReady(secondaryResourceStatus.value, "auto")
})
const tertiaryModelKey = computed(() =>
  `tertiary:${tertiaryResourceStatus.value?.variant ?? "default"}:auto`,
)
const tertiaryLoadedInstanceId = computed(
  () => loadedInstances.value[tertiaryModelKey.value] ?? null,
)
const tertiaryLoadedRuntime = computed(
  () => loadedRuntimes.value[tertiaryModelKey.value] ?? null,
)
const tertiaryArtifactReady = computed(() => {
  return artifactReady(tertiaryResourceStatus.value, "auto")
})
const inferenceReady = computed(() => {
  if (
    !primaryEnabled.value
    && (!props.fetchSecondaryStatus || !secondaryEnabled.value)
    && (!props.fetchTertiaryStatus || !tertiaryEnabled.value)
  ) return false
  return (!primaryEnabled.value || runtimeArtifactReady.value)
    && (!props.fetchSecondaryStatus || !secondaryEnabled.value || secondaryArtifactReady.value)
    && (!props.fetchTertiaryStatus || !tertiaryEnabled.value || tertiaryArtifactReady.value)
})

async function ensureModelLoaded(): Promise<string> {
  if (loadedInstanceId.value) return loadedInstanceId.value
  const status = resourceStatus.value
  if (!status || !runtimeArtifactReady.value) {
    throw new Error("当前模型资产不可用，请先在 Models 页面下载或转换。")
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    const loaded = await loadSharedModel(status.model_id, selectedVariant.value, runtime.value)
    loadedInstances.value[modelKey.value] = loaded.instance_id
    loadedRuntimes.value[modelKey.value] = loaded.runtime
    lifecycleMessage.value = { type: "success", text: `模型加载成功 · ${loaded.runtime}` }
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
    delete loadedRuntimes.value[modelKey.value]
    lifecycleMessage.value = { type: "success", text: "模型卸载成功" }
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "模型卸载失败") }
  } finally {
    lifecycleBusy.value = false
  }
}

async function ensureSecondaryModelLoaded(): Promise<string> {
  if (secondaryLoadedInstanceId.value) return secondaryLoadedInstanceId.value
  const status = secondaryResourceStatus.value
  if (!status || !secondaryArtifactReady.value) {
    throw new Error("RetinaFace 模型资产不可用，请先在 Models 页面下载。")
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    const loaded = await loadSharedModel(status.model_id, status.variant, "auto")
    loadedInstances.value[secondaryModelKey.value] = loaded.instance_id
    loadedRuntimes.value[secondaryModelKey.value] = loaded.runtime
    lifecycleMessage.value = { type: "success", text: `RetinaFace 加载成功 · ${loaded.runtime}` }
    return loaded.instance_id
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "RetinaFace 加载失败") }
    throw caught
  } finally {
    lifecycleBusy.value = false
  }
}

async function toggleSecondaryModel() {
  if (lifecycleBusy.value) return
  const instanceId = secondaryLoadedInstanceId.value
  if (!instanceId) {
    try { await ensureSecondaryModelLoaded() } catch { /* feedback is displayed inline */ }
    return
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    await unloadModel(instanceId)
    delete loadedInstances.value[secondaryModelKey.value]
    delete loadedRuntimes.value[secondaryModelKey.value]
    lifecycleMessage.value = { type: "success", text: "RetinaFace 卸载成功" }
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "RetinaFace 卸载失败") }
  } finally {
    lifecycleBusy.value = false
  }
}

async function ensureTertiaryModelLoaded(): Promise<string> {
  if (tertiaryLoadedInstanceId.value) return tertiaryLoadedInstanceId.value
  const status = tertiaryResourceStatus.value
  if (!status || !tertiaryArtifactReady.value) {
    throw new Error("MediaPipe Hand Detection 模型资产不可用，请先在 Models 页面下载或转换。")
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    const loaded = await loadSharedModel(status.model_id, status.variant, "auto")
    loadedInstances.value[tertiaryModelKey.value] = loaded.instance_id
    loadedRuntimes.value[tertiaryModelKey.value] = loaded.runtime
    lifecycleMessage.value = {
      type: "success",
      text: `MediaPipe Hand Detection 加载成功 · ${loaded.runtime}`,
    }
    return loaded.instance_id
  } catch (caught) {
    lifecycleMessage.value = {
      type: "error",
      text: errorMessage(caught, "MediaPipe Hand Detection 加载失败"),
    }
    throw caught
  } finally {
    lifecycleBusy.value = false
  }
}

async function toggleTertiaryModel() {
  if (lifecycleBusy.value) return
  const instanceId = tertiaryLoadedInstanceId.value
  if (!instanceId) {
    try { await ensureTertiaryModelLoaded() } catch { /* feedback is displayed inline */ }
    return
  }
  lifecycleBusy.value = true
  lifecycleMessage.value = null
  try {
    await unloadModel(instanceId)
    delete loadedInstances.value[tertiaryModelKey.value]
    delete loadedRuntimes.value[tertiaryModelKey.value]
    lifecycleMessage.value = { type: "success", text: "MediaPipe Hand Detection 卸载成功" }
  } catch (caught) {
    lifecycleMessage.value = {
      type: "error",
      text: errorMessage(caught, "MediaPipe Hand Detection 卸载失败"),
    }
  } finally {
    lifecycleBusy.value = false
  }
}

const managedInfer: VisionInference = async (file, options, requestOptions) => {
  if (primaryEnabled.value) await ensureModelLoaded()
  if (props.fetchSecondaryStatus && secondaryEnabled.value) await ensureSecondaryModelLoaded()
  if (props.fetchTertiaryStatus && tertiaryEnabled.value) await ensureTertiaryModelLoaded()
  return props.infer(file, options, requestOptions)
}

async function refreshResources() {
  lifecycleMessage.value = null
  try {
    resourceStatus.value = await props.fetchStatus(selectedVariant.value || undefined)
    if (!selectedVariant.value) selectedVariant.value = resourceStatus.value.variant
    if (props.fetchSecondaryStatus) {
      secondaryResourceStatus.value = await props.fetchSecondaryStatus()
    }
    if (props.fetchTertiaryStatus) {
      tertiaryResourceStatus.value = await props.fetchTertiaryStatus()
    }
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
          <span>
            <input v-if="fetchSecondaryStatus" v-model="primaryEnabled" type="checkbox" />
            {{ modelLabel }} variant
          </span>
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
          <small v-if="loadedRuntime" class="model-runtime">
            RUNTIME · {{ loadedRuntime.toUpperCase() }}
          </small>
          <small v-if="lifecycleMessage" :class="`lifecycle-${lifecycleMessage.type}`">{{ lifecycleMessage.text }}</small>
        </label>

        <label v-if="fetchSecondaryStatus" class="field model-field">
          <span><input v-model="secondaryEnabled" type="checkbox" /> {{ secondaryModelLabel }}</span>
          <div class="model-field__control">
            <select disabled>
              <option>{{ secondaryResourceStatus?.variants[0]?.display_name ?? "RetinaFace" }}</option>
            </select>
            <button
              type="button"
              :disabled="lifecycleBusy || (!secondaryLoadedInstanceId && !secondaryArtifactReady)"
              @click="toggleSecondaryModel"
            >
              {{ lifecycleBusy ? "WAIT…" : secondaryLoadedInstanceId ? "UNLOAD" : "LOAD" }}
            </button>
          </div>
          <small v-if="secondaryLoadedRuntime" class="model-runtime">
            RUNTIME · {{ secondaryLoadedRuntime.toUpperCase() }}
          </small>
        </label>

        <label v-if="fetchTertiaryStatus" class="field model-field">
          <span><input v-model="tertiaryEnabled" type="checkbox" /> {{ tertiaryModelLabel }}</span>
          <div class="model-field__control">
            <select disabled>
              <option>{{ tertiaryResourceStatus?.variants[0]?.display_name ?? "MediaPipe Hand" }}</option>
            </select>
            <button
              type="button"
              :disabled="lifecycleBusy || (!tertiaryLoadedInstanceId && !tertiaryArtifactReady)"
              @click="toggleTertiaryModel"
            >
              {{ lifecycleBusy ? "WAIT…" : tertiaryLoadedInstanceId ? "UNLOAD" : "LOAD" }}
            </button>
          </div>
          <small v-if="tertiaryLoadedRuntime" class="model-runtime">
            RUNTIME · {{ tertiaryLoadedRuntime.toUpperCase() }}
          </small>
        </label>

        <label class="field">
          <span>Runtime</span>
          <select v-model="runtime">
            <option value="auto">Auto · Core ML first</option>
            <option value="coreml">Core ML</option>
            <option value="pytorch-mps">PyTorch MPS</option>
          </select>
        </label>

        <label class="field range-field">
          <span>YOLO Pose confidence <output>{{ confidence.toFixed(2) }}</output></span>
          <input v-model.number="confidence" type="range" min="0.05" max="0.95" step="0.05" />
        </label>

        <label v-if="fetchSecondaryStatus" class="field range-field">
          <span>Face confidence <output>{{ faceConfidence.toFixed(2) }}</output></span>
          <input v-model.number="faceConfidence" type="range" min="0.05" max="0.95" step="0.05" />
        </label>

        <label v-if="fetchTertiaryStatus" class="field range-field">
          <span>Hand confidence <output>{{ handConfidence.toFixed(2) }}</output></span>
          <input v-model.number="handConfidence" type="range" min="0.05" max="0.95" step="0.05" />
        </label>

        <label v-if="fetchTertiaryStatus" class="field toggle-field">
          <span>
            <input v-model="handLandmarksEnabled" type="checkbox" />
            Hand landmarks · 21 points
          </span>
          <small>关闭时不会加载 landmark 子模型。</small>
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

        <p v-if="primaryEnabled && !runtimeArtifactReady" class="resource-note">
          当前 runtime 的模型资产尚未准备。请前往 Models 页面下载、转换并加载模型。
        </p>
        <p v-if="fetchSecondaryStatus && secondaryEnabled && !secondaryArtifactReady" class="resource-note">
          RetinaFace 模型资产尚未准备。请前往 Models 页面下载或转换模型。
        </p>
        <p v-if="fetchTertiaryStatus && tertiaryEnabled && !tertiaryArtifactReady" class="resource-note">
          MediaPipe Hand Detection 模型资产尚未准备。请前往 Models 页面下载或转换模型。
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
          :enabled="inferenceReady"
          :infer="managedInfer"
          :overlay="overlay"
          :results="results"
          :action-label="actionLabel"
          :empty-copy="emptyCopy"
        />
        <CameraInferenceTab
          v-else-if="activeMode === 'camera'"
          :options="inferenceOptions"
          :enabled="inferenceReady"
          :infer="managedInfer"
          :overlay="overlay"
          :results="results"
          :action-noun="actionNoun"
        />
        <VideoInferenceTab
          v-else
          :options="inferenceOptions"
          :enabled="inferenceReady"
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
.model-runtime { color:var(--muted); letter-spacing:.06em; }
.lifecycle-success { color:#3c8b2f; }
.lifecycle-error { color:#cf3f27; }
</style>
