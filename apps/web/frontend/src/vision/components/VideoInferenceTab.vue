<script setup lang="ts">
import type { Component } from "vue"
import { onBeforeUnmount, ref } from "vue"

import { ApiError } from "@/api/client"
import {
  captureVideoFrame,
  seekVideo,
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
}>()

const input = ref<HTMLInputElement | null>(null)
const video = ref<HTMLVideoElement | null>(null)
const canvas = document.createElement("canvas")
const videoUrl = ref("")
const filename = ref("")
const duration = ref(0)
type AnalysisRate = "max" | "1" | "2" | "5"

const analysisRate = ref<AnalysisRate>("max")
const analyzing = ref(false)
const completedFrames = ref(0)
const totalFrames = ref(0)
const currentTimestamp = ref(0)
const pipelineMs = ref(0)
const result = ref<VisionResultBase | null>(null)
const error = ref("")
let stopRequested = false
let activeRequest: AbortController | null = null
let analysisGeneration = 0

function selectVideo(event: Event) {
  const selected = (event.target as HTMLInputElement).files?.[0]
  if (!selected || !selected.type.startsWith("video/")) return
  stopAnalysis()
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  videoUrl.value = URL.createObjectURL(selected)
  filename.value = selected.name
  duration.value = 0
  completedFrames.value = 0
  totalFrames.value = 0
  currentTimestamp.value = 0
  result.value = null
  error.value = ""
}

function metadataReady() {
  duration.value = video.value && Number.isFinite(video.value.duration) ? video.value.duration : 0
}

function stopAnalysis() {
  analysisGeneration += 1
  stopRequested = true
  analyzing.value = false
  activeRequest?.abort()
  activeRequest = null
  video.value?.pause()
  result.value = null
  completedFrames.value = 0
  currentTimestamp.value = 0
  pipelineMs.value = 0
}

async function analyzeVideo() {
  if (!video.value || !duration.value || analyzing.value || !props.enabled) return

  const generation = ++analysisGeneration
  stopRequested = false
  analyzing.value = true
  completedFrames.value = 0
  totalFrames.value =
    analysisRate.value === "max"
      ? 0
      : Math.max(1, Math.ceil(duration.value * Number(analysisRate.value)))
  result.value = null
  error.value = ""

  try {
    if (analysisRate.value === "max") {
      await analyzeLatestFrames(generation)
    } else {
      await analyzeFixedRate(generation, Number(analysisRate.value))
    }
  } catch (caught) {
    if (!(caught instanceof DOMException && caught.name === "AbortError")) {
      error.value =
        caught instanceof ApiError
          ? caught.message
          : caught instanceof Error
            ? caught.message
            : "视频分析失败"
    }
  } finally {
    if (generation === analysisGeneration) {
      activeRequest = null
      analyzing.value = false
      video.value?.pause()
    }
  }
}

async function analyzeLatestFrames(generation: number) {
  if (!video.value) return
  if (video.value.ended || video.value.currentTime >= duration.value - 0.001) {
    video.value.currentTime = 0
  }
  video.value.muted = true
  await video.value.play()
  let lastCapturedMediaTime = -1

  while (!stopRequested && generation === analysisGeneration && video.value && !video.value.ended) {
    if (video.value.currentTime <= lastCapturedMediaTime + 0.001) {
      const hasFrame = await waitForNextVideoFrame(video.value)
      if (!hasFrame) break
    }
    if (stopRequested || generation !== analysisGeneration) break
    lastCapturedMediaTime = video.value.currentTime
    await inferCurrentFrame(video.value, generation)
  }
}

async function analyzeFixedRate(generation: number, fps: number) {
  if (!video.value) return
  video.value.pause()
  for (let index = 0; index < totalFrames.value; index += 1) {
    if (stopRequested || generation !== analysisGeneration || !video.value) break
    const timestamp = Math.min(index / fps, Math.max(0, duration.value - 0.001))
    await seekVideo(video.value, timestamp)
    if (stopRequested || generation !== analysisGeneration) break
    await inferCurrentFrame(video.value, generation)
  }
}

async function inferCurrentFrame(currentVideo: HTMLVideoElement, generation: number) {
  const startedAt = performance.now()
  const frame = await captureVideoFrame(currentVideo, canvas, "video-latest-frame.jpg")
  const request = new AbortController()
  activeRequest = request
  const nextResult = await props.infer(frame, props.options, {
    signal: request.signal,
    cacheInput: false,
  })
  if (generation !== analysisGeneration || stopRequested) return
  result.value = nextResult
  currentTimestamp.value = currentVideo.currentTime
  completedFrames.value += 1
  pipelineMs.value = performance.now() - startedAt
  activeRequest = null
}

onBeforeUnmount(() => {
  stopAnalysis()
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
})
</script>

<template>
  <div class="mode-content">
    <div class="mode-toolbar">
      <div>
        <strong>Local video</strong>
        <span>MAX 跟随视频播放时间轴，只处理推理结束时的最新解码帧，不建立帧队列。</span>
      </div>
      <label class="compact-field">
        <span>分析帧率</span>
        <select v-model="analysisRate" :disabled="analyzing">
          <option value="max">MAX · drop frames</option>
          <option value="1">1 FPS · exhaustive</option>
          <option value="2">2 FPS · exhaustive</option>
          <option value="5">5 FPS · exhaustive</option>
        </select>
      </label>
    </div>

    <input ref="input" type="file" accept="video/*" hidden @change="selectVideo" />
    <button v-if="!videoUrl" class="video-dropzone" type="button" @click="input?.click()">
      <span aria-hidden="true">＋</span>
      <strong>Select a local video</strong>
      <small>MP4 · MOV · WebM，具体格式取决于浏览器解码能力</small>
    </button>

    <template v-else>
      <div class="media-stage">
        <video
          ref="video"
          :src="videoUrl"
          controls
          playsinline
          preload="metadata"
          @loadedmetadata="metadataReady"
        ></video>
        <component :is="overlay" :result="result" />
        <div v-if="analyzing" class="live-indicator">
          <span></span>
          ANALYZING · {{ completedFrames }}{{ totalFrames ? `/${totalFrames}` : "" }}
          · {{ pipelineMs ? (1000 / pipelineMs).toFixed(1) : "—" }} pipeline FPS
        </div>
      </div>

      <div class="video-meta">
        <span>{{ filename }}</span>
        <span>{{ duration.toFixed(1) }} s</span>
        <span v-if="completedFrames">frame @ {{ currentTimestamp.toFixed(2) }} s</span>
      </div>
      <div v-if="totalFrames || (analyzing && duration)" class="analysis-progress">
        <span
          :style="{
            width: `${totalFrames
              ? (completedFrames / totalFrames) * 100
              : (currentTimestamp / duration) * 100}%`,
          }"
        ></span>
      </div>

      <div class="media-actions">
        <button
          class="button button--run"
          type="button"
          :disabled="!duration || !enabled"
          @click="analyzing ? stopAnalysis() : analyzeVideo()"
        >
          {{ analyzing ? "Stop analysis" : "Analyze sampled frames" }}
        </button>
        <button class="button button--quiet" type="button" :disabled="analyzing" @click="input?.click()">
          Replace video
        </button>
      </div>
    </template>

    <div v-if="error" class="error-banner detection-error" role="alert">
      <div>
        <strong>Video analysis failed</strong>
        <span>{{ error }}</span>
      </div>
    </div>
    <component :is="results" v-if="result" :result="result" />
  </div>
</template>
