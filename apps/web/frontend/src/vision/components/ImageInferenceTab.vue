<script setup lang="ts">
import type { Component } from "vue"
import { onBeforeUnmount, ref } from "vue"

import { ApiError } from "@/api/client"
import UploadDropzone from "@/object_detection/components/UploadDropzone.vue"
import type {
  VisionInference,
  VisionOptions,
  VisionResultBase,
} from "../types"

const props = defineProps<{
  options: VisionOptions
  enabled: boolean
  infer: VisionInference
  overlay: Component
  results: Component
  actionLabel: string
  emptyCopy: string
}>()

const file = ref<File | null>(null)
const imageUrl = ref("")
const running = ref(false)
const result = ref<VisionResultBase | null>(null)
const error = ref("")

function selectFile(selected: File) {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  file.value = selected
  imageUrl.value = URL.createObjectURL(selected)
  result.value = null
  error.value = ""
}

async function runInference() {
  if (!file.value || running.value || !props.enabled) return
  running.value = true
  error.value = ""
  result.value = null
  try {
    result.value = await props.infer(file.value, props.options)
  } catch (caught) {
    if (caught instanceof ApiError) {
      error.value =
        caught.code === "model_load_error"
          ? "模型首次加载失败。请检查模型资源和所选 runtime 后重试。"
          : caught.message
    } else {
      error.value = caught instanceof Error ? caught.message : "推理请求失败"
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
      @click="runInference"
    >
      <span>{{ running ? "Running locally…" : actionLabel }}</span>
      <span aria-hidden="true">{{ running ? "◌" : "→" }}</span>
    </button>

    <div v-if="imageUrl" class="preview-frame mode-preview">
      <div class="vision-image-stage">
        <img :src="imageUrl" alt="待分析图片预览" />
        <component :is="overlay" :result="result" />
      </div>
      <div v-if="running" class="scan-overlay" aria-label="正在执行推理">
        <span></span>
      </div>
    </div>
    <div v-else class="output-empty">
      <span class="output-empty__cross" aria-hidden="true">×</span>
      <strong>No image selected</strong>
      <p>{{ emptyCopy }}</p>
    </div>

    <div v-if="error" class="error-banner detection-error" role="alert">
      <div>
        <strong>Inference failed</strong>
        <span>{{ error }}</span>
      </div>
    </div>
    <component :is="results" v-if="result" :result="result" />
  </div>
</template>

<style scoped>
.vision-image-stage {
  background:
    linear-gradient(45deg, #dad8cf 25%, transparent 25%),
    linear-gradient(-45deg, #dad8cf 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #dad8cf 75%),
    linear-gradient(-45deg, transparent 75%, #dad8cf 75%);
  background-position: 0 0, 0 8px, 8px -8px, -8px 0;
  background-size: 16px 16px;
  line-height: 0;
  overflow: hidden;
  position: relative;
}

img {
  display: block;
  height: auto;
  width: 100%;
}
</style>
