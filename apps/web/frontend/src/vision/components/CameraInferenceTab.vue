<script setup lang="ts">
import type { Component } from "vue"
import { computed, onBeforeUnmount, ref } from "vue"

import { ApiError } from "@/api/client"
import {
  captureVideoFrame,
  wait,
  waitForNextVideoFrame,
} from "@/object_detection/frame"
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
  actionNoun: string
}>()

const video = ref<HTMLVideoElement | null>(null)
const canvas = document.createElement("canvas")
const stream = ref<MediaStream | null>(null)
const cameraReady = ref(false)
const cameraAspectRatio = ref("16 / 9")
const detecting = ref(false)
type AnalysisRate = "max" | "1" | "2" | "5"

const analysisRate = ref<AnalysisRate>("max")
const result = ref<VisionResultBase | null>(null)
const error = ref("")
const frameNumber = ref(0)
const pipelineMs = ref(0)
const effectiveFps = computed(() => pipelineMs.value > 0 ? 1000 / pipelineMs.value : 0)
let activeRequest: AbortController | null = null
let loopGeneration = 0

async function startCamera() {
  if (stream.value || !video.value) return
  error.value = ""
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持摄像头访问，请使用 localhost 或 HTTPS 打开页面")
    }
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    })
    stream.value = mediaStream
    video.value.srcObject = mediaStream
    await video.value.play()
    cameraAspectRatio.value = `${video.value.videoWidth} / ${video.value.videoHeight}`
    cameraReady.value = true
  } catch (caught) {
    stopCamera()
    if (caught instanceof DOMException && caught.name === "NotAllowedError") {
      error.value = "摄像头权限被拒绝。请在浏览器站点设置中允许访问后重试。"
    } else {
      error.value = caught instanceof Error ? caught.message : "摄像头启动失败"
    }
  }
}

function stopInference() {
  loopGeneration += 1
  detecting.value = false
  activeRequest?.abort()
  activeRequest = null
}

function stopCamera() {
  stopInference()
  stream.value?.getTracks().forEach((track) => track.stop())
  stream.value = null
  cameraReady.value = false
  if (video.value) video.value.srcObject = null
}

async function runInferenceLoop() {
  if (!props.enabled || detecting.value) return
  if (!cameraReady.value) await startCamera()
  if (!cameraReady.value || !video.value) return

  const generation = ++loopGeneration
  let lastCapturedMediaTime = -1
  detecting.value = true
  error.value = ""
  while (detecting.value && generation === loopGeneration && video.value) {
    const startedAt = performance.now()
    try {
      if (video.value.currentTime <= lastCapturedMediaTime + 0.001) {
        await waitForNextVideoFrame(video.value)
      }
      lastCapturedMediaTime = video.value.currentTime
      const frame = await captureVideoFrame(video.value, canvas, "camera-latest-frame.jpg")
      const request = new AbortController()
      activeRequest = request
      const nextResult = await props.infer(frame, props.options, {
        signal: request.signal,
        cacheInput: false,
      })
      if (!detecting.value || generation !== loopGeneration) break
      result.value = nextResult
      frameNumber.value += 1
      pipelineMs.value = performance.now() - startedAt
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") break
      error.value =
        caught instanceof ApiError
          ? caught.message
          : caught instanceof Error
            ? caught.message
            : "实时推理失败"
      if (generation === loopGeneration) detecting.value = false
      break
    } finally {
      if (generation === loopGeneration) activeRequest = null
    }

    if (analysisRate.value !== "max") {
      const interval = 1000 / Number(analysisRate.value)
      await wait(Math.max(0, interval - (performance.now() - startedAt)))
    }
  }
}

onBeforeUnmount(stopCamera)
</script>

<template>
  <div class="mode-content">
    <div class="mode-toolbar">
      <div>
        <strong>Browser camera</strong>
        <span>MAX 模式完成一帧后立即读取最新画面；推理期间的中间帧直接丢弃。</span>
      </div>
      <label class="compact-field">
        <span>分析帧率</span>
        <select v-model="analysisRate" :disabled="detecting">
          <option value="max">MAX · latest frame</option>
          <option value="1">1 FPS</option>
          <option value="2">2 FPS</option>
          <option value="5">5 FPS</option>
        </select>
      </label>
    </div>

    <div
      class="media-stage camera-stage"
      :class="{ 'media-stage--empty': !cameraReady, 'camera-stage--ready': cameraReady }"
      :style="cameraReady ? { aspectRatio: cameraAspectRatio } : undefined"
    >
      <video ref="video" autoplay muted playsinline></video>
      <component :is="overlay" :result="cameraReady ? result : null" />
      <div v-if="detecting" class="live-indicator">
        <span></span>
        LIVE · {{ frameNumber }} frames ·
        {{ effectiveFps ? effectiveFps.toFixed(1) : "—" }} pipeline FPS
      </div>
      <div v-if="!cameraReady" class="media-placeholder">
        <span aria-hidden="true">◉</span>
        <strong>Camera is off</strong>
        <p>启动后浏览器会请求摄像头权限。</p>
      </div>
    </div>

    <div class="media-actions">
      <button v-if="!cameraReady" class="button" type="button" :disabled="!enabled" @click="startCamera">
        Open camera
      </button>
      <button
        v-else
        class="button button--run"
        type="button"
        :disabled="!enabled"
        @click="detecting ? stopInference() : runInferenceLoop()"
      >
        {{ detecting ? `Stop ${actionNoun}` : `Start live ${actionNoun}` }}
      </button>
      <button v-if="cameraReady" class="button button--quiet" type="button" @click="stopCamera">
        Close camera
      </button>
    </div>

    <div v-if="error" class="error-banner detection-error" role="alert">
      <div>
        <strong>Camera inference failed</strong>
        <span>{{ error }}</span>
      </div>
    </div>
    <component :is="results" v-if="result" :result="result" />
  </div>
</template>

<style scoped>
.camera-stage--ready {
  min-height: 0;
}

.camera-stage--ready video {
  height: 100%;
  inset: 0;
  max-height: none;
  object-fit: fill;
  position: absolute;
  width: 100%;
}
</style>
