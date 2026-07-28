<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import CameraDetectionTab from "./components/CameraDetectionTab.vue"
import ImageDetectionTab from "./components/ImageDetectionTab.vue"
import ResourceSetup from "./components/ResourceSetup.vue"
import VideoDetectionTab from "./components/VideoDetectionTab.vue"
import {
  convertCoreMl,
  downloadSource,
  fetchResourceStatus,
} from "./api"
import type {
  DetectOptions,
  ResourceStatus,
  RuntimeChoice,
} from "./types"

type DetectionMode = "image" | "camera" | "video"

const activeMode = ref<DetectionMode>("camera")
const runtime = ref<RuntimeChoice>("auto")
const confidence = ref(0.25)
const iouThreshold = ref(0.7)
const maxDetections = ref(100)
const selectedVariant = ref("")
const error = ref("")
const resourceStatus = ref<ResourceStatus | null>(null)
const resourceBusy = ref<"" | "download" | "convert">("")
const detectOptions = computed<DetectOptions>(() => ({
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

async function refreshResources() {
  try {
    resourceStatus.value = await fetchResourceStatus(selectedVariant.value || undefined)
    if (!selectedVariant.value) selectedVariant.value = resourceStatus.value.variant
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型资源状态读取失败"
  }
}

async function prepareResource(action: "download" | "convert") {
  if (resourceBusy.value) return
  resourceBusy.value = action
  error.value = ""
  try {
    resourceStatus.value =
      action === "download"
        ? await downloadSource(detectOptions.value.variant)
        : await convertCoreMl(detectOptions.value.variant)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型资源操作失败"
  } finally {
    resourceBusy.value = ""
  }
}

onMounted(refreshResources)
</script>

<template>
  <div class="page detection-page">
    <header class="app-hero">
      <div>
        <RouterLink class="back-link" to="/">← Studio</RouterLink>
        <p class="kicker">APP 01 / COMPUTER VISION</p>
        <h1>Object<br />Detection</h1>
      </div>
      <p>
        使用图片、浏览器摄像头或本地视频运行 YOLOv8。选择 Auto 时，平台会根据当前机器与依赖动态决定
        Core ML 或 PyTorch MPS；所有推理都发送到本机服务。
      </p>
    </header>

    <div class="detection-workspace">
      <aside class="control-panel">
        <div class="control-panel__heading">
          <span>INPUT / CONFIG</span>
          <span>01—04</span>
        </div>
        <label class="field">
          <span>YOLOv8 variant</span>
          <select v-model="selectedVariant" @change="refreshResources">
            <option
              v-for="variant in resourceStatus?.variants ?? []"
              :key="variant.name"
              :value="variant.name"
            >
              {{ variant.display_name }} · {{ variant.description }}
            </option>
          </select>
        </label>
        <ResourceSetup
          :status="resourceStatus"
          :busy="resourceBusy"
          @download="prepareResource('download')"
          @convert="prepareResource('convert')"
          @refresh="refreshResources"
        />

        <label class="field">
          <span>Runtime</span>
          <select v-model="runtime">
            <option value="auto">Auto · machine policy</option>
            <option value="coreml">Core ML</option>
            <option value="pytorch-mps">PyTorch MPS</option>
          </select>
        </label>

        <label class="field range-field">
          <span>
            Confidence
            <output>{{ confidence.toFixed(2) }}</output>
          </span>
          <input v-model.number="confidence" type="range" min="0.05" max="0.95" step="0.05" />
        </label>

        <label class="field range-field">
          <span>
            IoU threshold
            <output>{{ iouThreshold.toFixed(2) }}</output>
          </span>
          <input v-model.number="iouThreshold" type="range" min="0.1" max="0.9" step="0.05" />
        </label>

        <label class="field">
          <span>Maximum objects</span>
          <input
            v-model.number="maxDetections"
            class="number-input"
            type="number"
            min="1"
            max="1000"
          />
        </label>

        <p v-if="!runtimeArtifactReady" class="resource-note">
          当前 runtime 的模型资产尚未准备。请先执行上方显式下载或转换操作。
        </p>
        <p class="local-note">
          检测不会自动下载或转换模型；图片与视频帧也不会上传到第三方。
        </p>
      </aside>

      <section class="output-panel">
        <div class="output-panel__heading">
          <span>INPUT / LIVE OUTPUT</span>
          <span>{{ activeMode.toUpperCase() }}</span>
        </div>
        <div class="mode-tabs" role="tablist" aria-label="检测输入方式">
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
            <strong>Resource operation failed</strong>
            <span>{{ error }}</span>
          </div>
        </div>

        <ImageDetectionTab
          v-if="activeMode === 'image'"
          :options="detectOptions"
          :enabled="runtimeArtifactReady"
        />
        <CameraDetectionTab
          v-else-if="activeMode === 'camera'"
          :options="detectOptions"
          :enabled="runtimeArtifactReady"
        />
        <VideoDetectionTab
          v-else
          :options="detectOptions"
          :enabled="runtimeArtifactReady"
        />
      </section>
    </div>
  </div>
</template>
