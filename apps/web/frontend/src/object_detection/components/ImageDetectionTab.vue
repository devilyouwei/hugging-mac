<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue"

import { ApiError } from "@/api/client"
import { detectObjects } from "../api"
import type { DetectOptions, DetectionResult } from "../types"
import DetectionOverlay from "./DetectionOverlay.vue"
import ResultsPanel from "./ResultsPanel.vue"
import UploadDropzone from "./UploadDropzone.vue"

const props = defineProps<{
  options: DetectOptions
  enabled: boolean
}>()

const file = ref<File | null>(null)
const imageUrl = ref("")
const running = ref(false)
const result = ref<DetectionResult | null>(null)
const error = ref("")

function selectFile(selected: File) {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  file.value = selected
  imageUrl.value = URL.createObjectURL(selected)
  result.value = null
  error.value = ""
}

async function runDetection() {
  if (!file.value || running.value || !props.enabled) return
  running.value = true
  error.value = ""
  result.value = null
  try {
    result.value = await detectObjects(file.value, props.options)
  } catch (caught) {
    if (caught instanceof ApiError) {
      error.value =
        caught.code === "model_load_error"
          ? "模型首次加载失败。请检查模型资源和所选 runtime 后重试。"
          : caught.message
    } else {
      error.value = caught instanceof Error ? caught.message : "检测请求失败"
    }
  } finally {
    running.value = false
  }
}

onBeforeUnmount(() => {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
})
</script>

<template>
  <div class="mode-content">
    <UploadDropzone :filename="file?.name" @selected="selectFile" />
    <button
      class="button button--run"
      type="button"
      :disabled="!file || running || !enabled"
      @click="runDetection"
    >
      <span>{{ running ? "Running locally…" : "Detect objects" }}</span>
      <span aria-hidden="true">{{ running ? "◌" : "→" }}</span>
    </button>

    <div v-if="imageUrl" class="preview-frame mode-preview">
      <DetectionOverlay :image-url="imageUrl" :result="result" />
      <div v-if="running" class="scan-overlay" aria-label="正在执行检测">
        <span></span>
      </div>
    </div>
    <div v-else class="output-empty">
      <span class="output-empty__cross" aria-hidden="true">×</span>
      <strong>No image selected</strong>
      <p>选择图片后，检测结果会以 bounding box、类别和置信度叠加显示。</p>
    </div>

    <div v-if="error" class="error-banner detection-error" role="alert">
      <div>
        <strong>Detection failed</strong>
        <span>{{ error }}</span>
      </div>
    </div>
    <ResultsPanel v-if="result" :result="result" />
  </div>
</template>
