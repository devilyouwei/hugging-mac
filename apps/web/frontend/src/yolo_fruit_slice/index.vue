<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from "vue"
import { useRouter } from "vue-router"

import { ApiError } from "@/api/client"
import { errorMessage, loadSharedModel, unloadModel } from "@/modelLifecycle"
import { captureVideoFrame, wait, waitForNextVideoFrame } from "@/object_detection/frame"
import { estimatePoses, fetchResourceStatus } from "@/pose_estimation/api"
import type { ResourceStatus, RuntimeChoice, VisionOptions } from "@/vision/types"

import { GameAudio } from "./audio"
import { FruitSliceEngine } from "./engine"
import { availableGameRuntimes } from "./modelRuntime"
import { PointerBladeTracker } from "./pointerBlade"
import { PoseBladeTracker } from "./poseBlade"
import { GameRenderer } from "./renderer"
import type { GamePhase, GameSnapshot, PoseFrame } from "./types"

const video = ref<HTMLVideoElement | null>(null)
const router = useRouter()
const stage = ref<HTMLElement | null>(null)
const gameCanvas = ref<HTMLCanvasElement | null>(null)
const stream = ref<MediaStream | null>(null)
const resource = ref<ResourceStatus | null>(null)
const modelInstanceId = ref<string | null>(null)
const selectedVariant = ref<"n" | "m">("n")
const loadedVariant = ref<"n" | "m" | null>(null)
const loadedRuntime = ref<"coreml" | "pytorch-mps" | null>(null)
const modelBusy = ref(false)
const cameraReady = ref(false)
const phase = ref<GamePhase>("lobby")
const countdown = ref<string | number>(3)
const error = ref("")
const poseFrame = shallowRef<PoseFrame | null>(null)
const snapshot = shallowRef<GameSnapshot | null>(null)
const finishReason = ref<"lives" | "bomb">("lives")
const inferenceState = ref<"idle" | "searching" | "tracking" | "error">("idle")
const pointerReady = ref(false)

const captureCanvas = document.createElement("canvas")
const audio = new GameAudio()
const tracker = new PoseBladeTracker()
const pointerTracker = new PointerBladeTracker()
let engine: FruitSliceEngine | null = null
let renderer: GameRenderer | null = null
let generation = 0
let animationFrame = 0
let lastFrameAt = 0
let inferenceRequest: AbortController | null = null
let resizeObserver: ResizeObserver | null = null
let countdownRunning = false
let calibrationDeadline = 0
let finishScheduled = false

const sourceReady = computed(() => resource.value?.artifacts.some((artifact) => artifact.runtime === "pytorch-mps" && artifact.available) ?? false)
const coremlReady = computed(() => resource.value?.artifacts.some((artifact) => artifact.runtime === "coreml" && artifact.available) ?? false)
const modelAvailable = computed(() => coremlReady.value || sourceReady.value)
const runtimeLabel = computed(() => loadedRuntime.value === "coreml" ? "CORE ML" : loadedRuntime.value === "pytorch-mps" ? "PYTORCH MPS" : "CORE ML → PYTORCH MPS")
const modelStatus = computed(() => modelBusy.value ? "LOADING" : modelInstanceId.value ? `${loadedVariant.value?.toUpperCase()} · ${runtimeLabel.value}` : modelAvailable.value ? "AVAILABLE" : "MISSING")
const poseStatus = computed(() => pointerReady.value ? "MOUSE BLADE" : inferenceState.value === "tracking" ? "BODY LOCKED" : inferenceState.value === "searching" ? "FINDING PLAYER" : inferenceState.value === "error" ? "POSE ERROR" : "STANDBY")

async function loadGame(): Promise<void> {
  try {
    resource.value = await fetchResourceStatus(selectedVariant.value)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "游戏资源加载失败"
  }
}

async function ensureModelLoaded(): Promise<boolean> {
  if (modelInstanceId.value && loadedVariant.value === selectedVariant.value) return true
  if (!resource.value || !modelAvailable.value) {
    error.value = `YOLOv8 Pose ${selectedVariant.value.toUpperCase()} 模型不可用，请先到 Models 页面准备模型`
    return false
  }
  modelBusy.value = true
  try {
    if (modelInstanceId.value) await unloadCurrentModel()
    const candidates = availableGameRuntimes(resource.value)
    let lastError: unknown
    for (const runtime of candidates) {
      try {
        const loaded = await loadSharedModel(resource.value.model_id, selectedVariant.value, runtime)
        modelInstanceId.value = loaded.instance_id
        loadedVariant.value = selectedVariant.value
        loadedRuntime.value = loaded.runtime === "coreml" ? "coreml" : "pytorch-mps"
        return true
      } catch (caught) {
        lastError = caught
      }
    }
    throw lastError ?? new Error("没有可用的本地推理 runtime")
  } catch (caught) {
    error.value = errorMessage(caught, "模型加载失败")
    return false
  } finally {
    modelBusy.value = false
  }
}

async function toggleModel(): Promise<void> {
  if (modelBusy.value) return
  if (!modelInstanceId.value) {
    await ensureModelLoaded()
    return
  }
  modelBusy.value = true
  try {
    await unloadCurrentModel()
  } catch (caught) {
    error.value = errorMessage(caught, "模型卸载失败")
  } finally {
    modelBusy.value = false
  }
}

async function unloadCurrentModel(): Promise<void> {
  if (modelInstanceId.value) await unloadModel(modelInstanceId.value)
  modelInstanceId.value = null
  loadedVariant.value = null
  loadedRuntime.value = null
}

async function selectModelVariant(variant: "n" | "m"): Promise<void> {
  if (variant === selectedVariant.value || modelBusy.value) return
  modelBusy.value = true
  error.value = ""
  try {
    await unloadCurrentModel()
    selectedVariant.value = variant
    resource.value = null
    resource.value = await fetchResourceStatus(variant)
  } catch (caught) {
    error.value = errorMessage(caught, "模型配置切换失败")
  } finally {
    modelBusy.value = false
  }
}

async function openCamera(): Promise<boolean> {
  if (cameraReady.value && stream.value) return true
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持摄像头，请使用 localhost 或 HTTPS")
    const nextStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    })
    stream.value = nextStream
    if (!video.value) await nextTick()
    if (!video.value) throw new Error("游戏视频舞台尚未就绪")
    video.value.srcObject = nextStream
    await video.value.play()
    cameraReady.value = true
    return true
  } catch (caught) {
    error.value = caught instanceof DOMException && caught.name === "NotAllowedError"
      ? "摄像头权限被拒绝，请允许访问后重试"
      : caught instanceof Error ? caught.message : "摄像头启动失败"
    closeCamera()
    return false
  }
}

async function startGame(): Promise<void> {
  error.value = ""
  await audio.unlock()
  if (!await ensureModelLoaded()) return
  phase.value = "calibrating"
  await nextTick()
  if (!await openCamera()) {
    phase.value = "lobby"
    return
  }
  stopLoops()
  tracker.reset()
  pointerTracker.reset()
  pointerReady.value = false
  poseFrame.value = null
  inferenceState.value = "searching"
  finishReason.value = "lives"
  finishScheduled = false
  countdownRunning = false
  calibrationDeadline = performance.now() + 12_000
  setupArena()
  engine!.reset()
  engine!.setPhase("calibrating")
  snapshot.value = { ...engine!.state }
  const currentGeneration = generation
  lastFrameAt = performance.now()
  animationFrame = requestAnimationFrame((now) => renderLoop(now, currentGeneration))
  void inferenceLoop(currentGeneration)
}

function setupArena(): void {
  if (!stage.value || !gameCanvas.value) return
  renderer = new GameRenderer(gameCanvas.value)
  const rect = stage.value.getBoundingClientRect()
  engine = new FruitSliceEngine(rect.width, rect.height)
  renderer.resize(rect.width, rect.height)
  resizeObserver?.disconnect()
  resizeObserver = new ResizeObserver(() => {
    if (!stage.value || !engine || !renderer) return
    const nextRect = stage.value.getBoundingClientRect()
    engine.resize(nextRect.width, nextRect.height)
    renderer.resize(nextRect.width, nextRect.height)
  })
  resizeObserver.observe(stage.value)
}

async function beginCountdown(currentGeneration: number): Promise<void> {
  if (countdownRunning || currentGeneration !== generation || !engine) return
  countdownRunning = true
  phase.value = "countdown"
  engine.setPhase("countdown")
  for (const value of [3, 2, 1]) {
    if (currentGeneration !== generation) return
    countdown.value = value
    audio.play("tick")
    await wait(720)
  }
  if (currentGeneration !== generation) return
  countdown.value = "SLICE!"
  audio.play("go")
  await wait(560)
  if (currentGeneration !== generation || !engine) return
  phase.value = "playing"
  engine.setPhase("playing")
}

function renderLoop(now: number, currentGeneration: number): void {
  if (currentGeneration !== generation || !engine || !renderer) return
  const deltaMs = Math.min(40, now - lastFrameAt)
  lastFrameAt = now
  const blades = [...(poseFrame.value?.blades ?? [])]
  const pointerBlade = pointerTracker.current(now)
  if (pointerBlade) blades.push(pointerBlade)
  engine.update(deltaMs, now, blades)
  for (const event of engine.drainEvents()) {
    if (event.type === "slice") audio.play(event.critical ? "critical" : "slice")
    if (event.type === "miss") audio.play("miss")
    if (event.type === "bomb") {
      audio.play("bomb")
      finishReason.value = "bomb"
    }
  }
  renderer.render(engine.state, poseFrame.value, deltaMs / 1000, blades)
  snapshot.value = { ...engine.state }
  if (engine.state.phase === "finished" && phase.value !== "finished") {
    inferenceRequest?.abort()
    if (finishReason.value === "bomb") {
      if (!finishScheduled) {
        finishScheduled = true
        window.setTimeout(() => {
          if (currentGeneration === generation) phase.value = "finished"
        }, 720)
      }
    } else {
      phase.value = "finished"
    }
  }
  animationFrame = requestAnimationFrame((nextNow) => renderLoop(nextNow, currentGeneration))
}

function handlePointerMove(event: PointerEvent): void {
  if (!stage.value || !["calibrating", "countdown", "playing"].includes(phase.value)) return
  const rect = stage.value.getBoundingClientRect()
  pointerTracker.update(
    { x: event.clientX - rect.left, y: event.clientY - rect.top },
    performance.now(), rect.width, rect.height,
  )
  pointerReady.value = true
  if (phase.value === "calibrating") void beginCountdown(generation)
}

function handlePointerLeave(): void {
  pointerTracker.reset()
  pointerReady.value = false
}

async function inferenceLoop(currentGeneration: number): Promise<void> {
  let lastMediaTime = -1
  while (currentGeneration === generation && video.value && phase.value !== "lobby" && phase.value !== "finished") {
    if (phase.value === "paused") {
      await wait(80)
      continue
    }
    try {
      if (video.value.currentTime <= lastMediaTime + 0.001) await waitForNextVideoFrame(video.value)
      lastMediaTime = video.value.currentTime
      const frame = await captureVideoFrame(video.value, captureCanvas, "fruit-slice-frame.jpg", { maxDimension: 640, quality: 0.76 })
      inferenceRequest = new AbortController()
      const inferenceOptions: VisionOptions & {
        poseEnabled: boolean
        faceEnabled: boolean
        handEnabled: boolean
      } = {
        runtime: (loadedRuntime.value ?? "coreml") as RuntimeChoice,
        variant: loadedVariant.value ?? selectedVariant.value,
        confidence: 0.25, iouThreshold: 0.7, maxDetections: 1,
        poseEnabled: true, faceEnabled: false, handEnabled: false,
      }
      const result = await estimatePoses(
        frame,
        inferenceOptions,
        { signal: inferenceRequest.signal, cacheInput: false },
      )
      if (currentGeneration !== generation || !stage.value) return
      const rect = stage.value.getBoundingClientRect()
      const person = result.poses.slice().sort((a, b) => b.confidence - a.confidence)[0]
      poseFrame.value = tracker.update(
        person, result.image_size.width, result.image_size.height, rect.width, rect.height,
        performance.now(), result.timings.inference_ms,
      )
      inferenceState.value = person ? "tracking" : "searching"
      if (phase.value === "calibrating" && person) void beginCountdown(currentGeneration)
      if (phase.value === "calibrating" && performance.now() > calibrationDeadline) {
        error.value = "没有检测到完整玩家，请后退一步并确保双臂进入画面"
        returnToLobby()
        return
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      inferenceState.value = "error"
      error.value = caught instanceof ApiError ? caught.message : caught instanceof Error ? caught.message : "实时姿态推理失败"
      returnToLobby()
      return
    }
  }
}

function togglePause(): void {
  if (!engine) return
  if (phase.value === "playing") {
    phase.value = "paused"
    engine.setPhase("paused")
  } else if (phase.value === "paused") {
    phase.value = "playing"
    engine.setPhase("playing")
    lastFrameAt = performance.now()
  }
}

function handleVisibility(): void {
  if (document.hidden && phase.value === "playing") togglePause()
}

async function toggleFullscreen(): Promise<void> {
  if (!document.fullscreenElement) await stage.value?.requestFullscreen()
  else await document.exitFullscreen()
}

function stopLoops(): void {
  generation += 1
  cancelAnimationFrame(animationFrame)
  animationFrame = 0
  inferenceRequest?.abort()
  inferenceRequest = null
  resizeObserver?.disconnect()
  resizeObserver = null
}

function closeCamera(): void {
  stream.value?.getTracks().forEach((track) => track.stop())
  stream.value = null
  cameraReady.value = false
  if (video.value) video.value.srcObject = null
}

function returnToLobby(): void {
  stopLoops()
  closeCamera()
  tracker.reset()
  pointerTracker.reset()
  engine = null
  renderer = null
  poseFrame.value = null
  snapshot.value = null
  inferenceState.value = "idle"
  pointerReady.value = false
  countdownRunning = false
  finishScheduled = false
  phase.value = "lobby"
}

function handleBack(): void {
  if (phase.value === "lobby") router.back()
  else returnToLobby()
}

onMounted(() => {
  document.body.classList.add("fruit-game-active")
  document.addEventListener("visibilitychange", handleVisibility)
  void loadGame()
})

onBeforeUnmount(() => {
  document.body.classList.remove("fruit-game-active")
  document.removeEventListener("visibilitychange", handleVisibility)
  stopLoops()
  closeCamera()
  void audio.close()
})
</script>

<template>
  <div class="fruit-game" :class="`fruit-game--${phase}`">
    <button type="button" class="game-back" aria-label="Back" @click="handleBack">←</button>

    <main v-if="phase === 'lobby'" class="fruit-lobby">
      <section class="fruit-lobby__copy">
        <p class="fruit-kicker">TWO ARMS · TWO BLADES · ONE CAMERA</p>
        <h1>Your body<br><em>is the blade.</em></h1>
        <p>在镜头前挥舞双臂，或直接快速滑动鼠标。前臂和鼠标都是刀刃——切开水果，避开炸弹，守住三条生命。</p>
        <div class="fruit-rules">
          <div><b>01</b><span>肘到腕或鼠标滑动<br>都可以作为刀刃</span></div>
          <div><b>02</b><span>快速挥动切水果<br>中心命中触发暴击</span></div>
          <div><b>03</b><span>漏掉三颗即结束<br>绝对不要碰炸弹</span></div>
        </div>
        <div class="fruit-model-picker" aria-label="选择 YOLO Pose 模型">
          <button type="button" :class="{ active: selectedVariant === 'n' }" :disabled="modelBusy" @click="selectModelVariant('n')">
            <strong>YOLO POSE N</strong><span>低延迟 · 推荐</span>
          </button>
          <button type="button" :class="{ active: selectedVariant === 'm' }" :disabled="modelBusy" @click="selectModelVariant('m')">
            <strong>YOLO POSE M</strong><span>高精度 · 较慢</span>
          </button>
        </div>
        <div class="fruit-model-row">
          <div><span :class="{ ready: modelInstanceId }"></span><b>YOLOv8 POSE {{ selectedVariant.toUpperCase() }}</b><small>{{ modelStatus }}</small></div>
          <button type="button" :disabled="modelBusy || (!modelInstanceId && !modelAvailable)" @click="toggleModel">
            {{ modelBusy ? "WAIT…" : modelInstanceId ? "UNLOAD" : "LOAD MODEL" }}
          </button>
        </div>
        <button class="fruit-start" type="button" :disabled="modelBusy || !modelAvailable" @click="startGame">
          <span>START CLASSIC</span><b>挥动双臂 →</b>
        </button>
        <p v-if="error" class="fruit-error" role="alert">{{ error }}</p>
      </section>
      <aside class="fruit-lobby__art" aria-hidden="true">
        <div class="preview-fruit preview-fruit--apple"></div>
        <div class="preview-fruit preview-fruit--orange"></div>
        <div class="preview-fruit preview-fruit--melon"></div>
        <div class="preview-slash preview-slash--one"></div>
        <div class="preview-slash preview-slash--two"></div>
        <strong>+30</strong><span>3 COMBO</span>
      </aside>
    </main>

    <main v-else ref="stage" class="fruit-stage" @pointermove="handlePointerMove" @pointerleave="handlePointerLeave">
      <video ref="video" autoplay muted playsinline></video>
      <canvas ref="gameCanvas"></canvas>
      <div class="fruit-vignette"></div>
      <div
        v-if="(snapshot?.flash ?? 0) > 0"
        class="bomb-screen-effect"
        :style="{ opacity: snapshot?.flash ?? 0 }"
      ></div>

      <div class="fruit-hud fruit-hud--score">
        <span>SCORE</span><strong>{{ (snapshot?.score ?? 0).toLocaleString() }}</strong>
        <small v-if="(snapshot?.combo ?? 0) >= 2">×{{ snapshot?.combo }} COMBO</small>
      </div>
      <div class="fruit-hud fruit-hud--lives" aria-label="剩余生命">
        <span v-for="life in 3" :key="life" :class="{ lost: life > (snapshot?.lives ?? 3) }">✕</span>
      </div>
      <div class="fruit-tracking"><i :class="{ ready: inferenceState === 'tracking' || pointerReady }"></i>{{ poseStatus }}<small v-if="poseFrame?.inferenceMs">{{ poseFrame.inferenceMs.toFixed(0) }} MS</small></div>
      <div class="fruit-actions">
        <button type="button" title="全屏" @click="toggleFullscreen">⛶</button>
        <button v-if="phase === 'playing' || phase === 'paused'" type="button" title="暂停" @click="togglePause">{{ phase === "paused" ? "▶" : "Ⅱ" }}</button>
      </div>

      <div v-if="phase === 'calibrating'" class="fruit-overlay fruit-overlay--calibrating">
        <div class="scan-figure"><i></i><i></i><i></i></div>
        <p>STEP INTO FRAME</p><strong>正在寻找玩家</strong><span>后退一步，让肩膀、双肘和双腕进入画面</span>
      </div>
      <div v-if="phase === 'countdown'" class="fruit-overlay fruit-overlay--countdown">
        <p>GET READY</p><strong :key="countdown">{{ countdown }}</strong><span>挥动前臂切开水果 · 避开炸弹</span>
      </div>
      <div v-if="phase === 'paused'" class="fruit-overlay fruit-overlay--paused">
        <p>GAME PAUSED</p><strong>暂停</strong><button type="button" @click="togglePause">继续游戏</button>
      </div>
      <div v-if="phase === 'finished'" class="fruit-overlay fruit-overlay--finished">
        <p>{{ finishReason === "bomb" ? "BOMB HIT" : "RUN COMPLETE" }}</p>
        <strong>{{ (snapshot?.score ?? 0).toLocaleString() }}</strong>
        <div class="result-stats">
          <span><b>{{ snapshot?.slicedCount ?? 0 }}</b> SLICED</span>
          <span><b>×{{ snapshot?.bestCombo ?? 0 }}</b> BEST COMBO</span>
          <span><b>{{ snapshot?.missedCount ?? 0 }}</b> MISSED</span>
        </div>
        <button type="button" @click="startGame">PLAY AGAIN</button>
        <button type="button" class="quiet" @click="returnToLobby">BACK TO MENU</button>
      </div>
    </main>
  </div>
</template>

<style scoped>
:global(body.fruit-game-active .site-header),:global(body.fruit-game-active .site-footer){display:none}
:global(body.fruit-game-active){background:#080a09;overflow:hidden}
.fruit-game{background:#080a09;color:#fff;min-height:100vh;overflow:hidden}.fruit-topbar{align-items:center;background:#090b0a;border-bottom:1px solid #ffffff1c;display:grid;font-family:var(--font-mono);grid-template-columns:1fr auto 1fr;height:3.4rem;padding:0 2rem;position:relative;z-index:30}.fruit-topbar__back{color:#969b98;font-size:.62rem;letter-spacing:.08em;text-decoration:none}.fruit-topbar__brand{font-size:.7rem;font-weight:800;letter-spacing:.12em}.fruit-topbar__brand i{background:#f34432;border-radius:50%;box-shadow:0 0 16px #f34432;display:inline-block;height:.5rem;margin-right:.55rem;width:.5rem}.fruit-topbar button{background:none;border:0;color:#999;cursor:pointer;font:.62rem var(--font-mono);justify-self:end}.fruit-lobby{display:grid;grid-template-columns:1.05fr .95fr;min-height:calc(100vh - 3.4rem)}.fruit-lobby__copy{align-self:center;max-width:900px;padding:3rem clamp(2rem,6vw,6rem)}.fruit-kicker{color:#ffcf48;font:700 .62rem var(--font-mono);letter-spacing:.14em}.fruit-lobby h1{font-size:clamp(3.2rem,6vw,6.4rem);letter-spacing:-.075em;line-height:.82;margin:1.1rem 0 1.8rem}.fruit-lobby h1 em{color:#ff4b35;font-style:normal}.fruit-lobby__copy>p:not(.fruit-kicker,.fruit-error,.fruit-message){color:#b1b6b2;line-height:1.75;max-width:39rem}.fruit-rules{border-bottom:1px solid #ffffff1a;border-top:1px solid #ffffff1a;display:grid;grid-template-columns:repeat(3,1fr);margin:2rem 0 1.3rem;padding:1rem 0}.fruit-rules div{align-items:flex-start;border-right:1px solid #ffffff18;display:flex;gap:.7rem;padding:.3rem .8rem}.fruit-rules div:first-child{padding-left:0}.fruit-rules div:last-child{border:0}.fruit-rules b{color:#ffcf48;font:.58rem var(--font-mono)}.fruit-rules span{color:#8f9691;font-size:.7rem;line-height:1.5}.fruit-model-row{align-items:center;background:#121513;border:1px solid #ffffff1d;display:flex;justify-content:space-between;padding:.85rem}.fruit-model-row>div{align-items:center;display:flex;gap:.55rem}.fruit-model-row>div>span{background:#c05242;border-radius:50%;height:.5rem;width:.5rem}.fruit-model-row>div>span.ready{background:#7cda43;box-shadow:0 0 10px #7cda43}.fruit-model-row b{font:.62rem var(--font-mono)}.fruit-model-row small{color:#7d837f;font:.52rem var(--font-mono)}.fruit-model-row button{background:#252a27;border:0;color:#fff;cursor:pointer;font:700 .55rem var(--font-mono);padding:.65rem .85rem}.fruit-model-row button:disabled{cursor:not-allowed;opacity:.35}.fruit-start{align-items:center;background:#ffcf48;border:0;color:#15120a;cursor:pointer;display:flex;font:800 .7rem var(--font-mono);justify-content:space-between;margin-top:.7rem;padding:1.1rem 1.2rem;width:100%}.fruit-start:disabled{cursor:not-allowed;filter:grayscale(1);opacity:.35}.fruit-start b{font-size:.58rem}.fruit-message{color:#79d25c;font:.55rem var(--font-mono);margin:.7rem 0}.fruit-error{color:#ff745f;font:.6rem var(--font-mono);margin-top:.8rem}.fruit-lobby__art{background:radial-gradient(circle at 45% 45%,#82382955,transparent 30%),radial-gradient(circle at 70% 30%,#ffcf4815,transparent 24%),#111411;overflow:hidden;position:relative}.preview-fruit{animation:fruit-float 3s ease-in-out infinite;border-radius:50%;box-shadow:inset -18px -22px 32px #0006,0 22px 40px #0008;position:absolute}.preview-fruit::before{background:#4aaa4a;border-radius:100% 0;content:"";height:18%;position:absolute;right:6%;top:-10%;transform:rotate(-25deg);width:38%}.preview-fruit--apple{background:#d9343b;height:16vw;left:18%;max-height:230px;max-width:230px;top:30%;width:16vw}.preview-fruit--orange{animation-delay:-.8s;background:#ff941f;height:10vw;max-height:150px;max-width:150px;right:15%;top:17%;width:10vw}.preview-fruit--melon{animation-delay:-1.5s;background:repeating-linear-gradient(90deg,#188345 0 13%,#3eae5e 13% 25%);bottom:12%;height:13vw;max-height:190px;max-width:190px;right:23%;width:13vw}.preview-slash{background:#fff;border-radius:50%;box-shadow:0 0 18px #fff,0 0 36px #67e8ff;height:3px;position:absolute;transform-origin:left;width:52%}.preview-slash--one{left:11%;top:63%;transform:rotate(-43deg)}.preview-slash--two{left:38%;top:66%;transform:rotate(-71deg)}.fruit-lobby__art strong{color:#ffcf48;font:900 5rem var(--font-mono);position:absolute;right:8%;top:48%;transform:rotate(7deg)}.fruit-lobby__art>span{color:#fff;font:800 .7rem var(--font-mono);position:absolute;right:11%;top:61%}.fruit-stage{background:#080a09;height:calc(100vh - 3.4rem);overflow:hidden;position:relative}.fruit-stage video,.fruit-stage canvas,.fruit-vignette{height:100%;inset:0;position:absolute;width:100%}.fruit-stage video{object-fit:cover;transform:scaleX(-1)}.fruit-stage canvas{z-index:2}.fruit-vignette{background:radial-gradient(circle,transparent 48%,#0008 115%),linear-gradient(180deg,#0007,transparent 20%,transparent 70%,#0008);pointer-events:none;z-index:3}.fruit-hud,.fruit-tracking,.fruit-actions{position:absolute;z-index:5}.fruit-hud--score{left:1.6rem;top:1.35rem}.fruit-hud--score span{color:#ffcf48;font:700 .55rem var(--font-mono);letter-spacing:.15em}.fruit-hud--score strong{display:block;font:900 clamp(2.4rem,5vw,4.6rem)/.9 var(--font-mono);letter-spacing:-.08em;text-shadow:0 4px 12px #000}.fruit-hud--score small{color:#fff;font:700 .65rem var(--font-mono)}.fruit-hud--lives{display:flex;gap:.45rem;right:1.5rem;top:1.35rem}.fruit-hud--lives span{color:#ff4a36;font:900 1.9rem var(--font-mono);text-shadow:0 0 14px #ff3b2e}.fruit-hud--lives span.lost{color:#565a57;text-shadow:none}.fruit-tracking{align-items:center;bottom:1.25rem;display:flex;font:700 .53rem var(--font-mono);gap:.45rem;left:1.5rem;letter-spacing:.08em}.fruit-tracking i{background:#f0a43a;border-radius:50%;height:.45rem;width:.45rem}.fruit-tracking i.ready{background:#7ee757;box-shadow:0 0 12px #7ee757}.fruit-tracking small{color:#8b918d;margin-left:.2rem}.fruit-actions{bottom:1rem;display:flex;gap:.4rem;right:1.25rem}.fruit-actions button{backdrop-filter:blur(8px);background:#080a09aa;border:1px solid #ffffff42;color:#fff;cursor:pointer;font-size:1rem;height:2.2rem;width:2.2rem}.fruit-overlay{align-items:center;background:#060806d8;display:flex;flex-direction:column;inset:0;justify-content:center;position:absolute;text-align:center;z-index:10}.fruit-overlay p{color:#ffcf48;font:700 .65rem var(--font-mono);letter-spacing:.2em}.fruit-overlay>strong{font-size:clamp(2.4rem,6vw,6rem);letter-spacing:-.05em}.fruit-overlay>span{color:#9ba19d;margin-top:.6rem}.scan-figure{animation:scan-pulse 1.5s infinite;border:1px solid #70e8ff55;height:14rem;margin-bottom:2rem;position:relative;width:9rem}.scan-figure::before{animation:scan 1.8s linear infinite;background:#70e8ff;box-shadow:0 0 16px #70e8ff;content:"";height:1px;left:-1rem;position:absolute;right:-1rem;top:0}.scan-figure i{background:#70e8ff;border-radius:50%;height:1.1rem;left:calc(50% - .55rem);position:absolute;top:15%;width:1.1rem}.scan-figure i:nth-child(2){border-radius:3rem;height:5.5rem;left:calc(50% - 1.9rem);top:30%;width:3.8rem}.scan-figure i:nth-child(3){border-radius:0;height:6rem;left:18%;top:41%;transform:rotate(18deg);width:.35rem}.fruit-overlay--countdown{background:#080a0948}.fruit-overlay--countdown strong{animation:count-slam .7s ease-out;color:#fff;font:900 min(32vw,17rem)/1 var(--font-mono);text-shadow:0 0 45px #ffcf48}.fruit-overlay--paused button,.fruit-overlay--finished>button{background:#ffcf48;border:0;color:#15120a;cursor:pointer;font:800 .65rem var(--font-mono);margin-top:1.5rem;min-width:14rem;padding:1rem}.fruit-overlay--finished{background:#070907e8}.fruit-overlay--finished>strong{font:900 clamp(5rem,16vw,14rem)/.85 var(--font-mono)}.result-stats{display:flex;gap:3rem;margin-top:1.5rem}.result-stats span{color:#8e958f;font:.58rem var(--font-mono)}.result-stats b{color:#fff;display:block;font-size:1.4rem;margin-bottom:.3rem}.fruit-overlay--finished>button.quiet{background:transparent;border:1px solid #ffffff35;color:#fff;margin-top:.45rem}@keyframes fruit-float{50%{transform:translateY(-18px) rotate(7deg)}}@keyframes scan{to{top:100%}}@keyframes scan-pulse{50%{box-shadow:0 0 40px #70e8ff22}}@keyframes count-slam{from{opacity:0;transform:scale(1.8) rotate(-5deg)}}@media(max-width:800px){.fruit-topbar{grid-template-columns:1fr auto;padding:0 1rem}.fruit-topbar button{display:none}.fruit-lobby{grid-template-columns:1fr}.fruit-lobby__copy{padding:2.4rem 1.2rem}.fruit-lobby h1{font-size:3.7rem}.fruit-rules{grid-template-columns:1fr}.fruit-rules div{border-bottom:1px solid #ffffff14;border-right:0;padding:.6rem 0}.fruit-lobby__art{display:none}.fruit-hud--score{left:1rem}.fruit-hud--lives{right:1rem}.result-stats{gap:1.3rem}.fruit-stage{height:calc(100svh - 3.4rem)}}
.fruit-model-picker{display:grid;gap:.45rem;grid-template-columns:1fr 1fr;margin-bottom:.45rem}.fruit-model-picker button{background:#101311;border:1px solid #ffffff1d;color:#858b87;cursor:pointer;display:flex;justify-content:space-between;padding:.7rem .8rem;text-align:left}.fruit-model-picker button.active{background:#ffcf480d;border-color:#ffcf48;color:#fff;box-shadow:inset 0 -2px #ffcf48}.fruit-model-picker button:disabled{cursor:not-allowed;opacity:.55}.fruit-model-picker strong{font:.58rem var(--font-mono)}.fruit-model-picker span{font:.5rem var(--font-mono)}
.fruit-stage{cursor:crosshair}
.bomb-screen-effect{background:radial-gradient(circle,transparent 42%,rgba(255,190,35,.12) 72%,rgba(255,43,20,.52) 100%);border:10px solid #ffcf38;box-shadow:inset 0 0 34px 9px #ffd338,inset 0 0 105px 30px #ff2d18,0 0 40px #ff4a1f;inset:0;pointer-events:none;position:absolute;z-index:9}

/* Apple Arcade-inspired finish */
.fruit-game{background:radial-gradient(circle at 72% 18%,#17332f 0,transparent 35rem),linear-gradient(145deg,#080b10,#0d1215);}
.fruit-topbar{background:rgb(16 19 24 / 68%);border:1px solid rgb(255 255 255 / 10%);border-radius:17px;height:3.2rem;margin:10px 12px 0;padding:0 1rem;position:relative;top:auto;-webkit-backdrop-filter:blur(24px) saturate(150%);backdrop-filter:blur(24px) saturate(150%);}
.fruit-lobby{min-height:calc(100vh - 4.2rem);padding-top:0;}
.fruit-lobby__copy{padding-top:2rem;}
.fruit-lobby h1{background:linear-gradient(145deg,#fff 30%,#ffd569 70%,#ff745e);-webkit-background-clip:text;background-clip:text;color:transparent;}
.fruit-lobby h1 em{color:inherit;}
.fruit-rules{background:rgb(255 255 255 / 4%);border:1px solid rgb(255 255 255 / 8%);border-radius:16px;padding:.8rem;}
.fruit-model-picker button,.fruit-model-row{background:rgb(255 255 255 / 5%);border-color:rgb(255 255 255 / 9%);border-radius:12px;}
.fruit-model-picker button.active{background:rgb(255 207 72 / 10%);border-color:rgb(255 207 72 / 34%);box-shadow:0 8px 24px rgb(0 0 0 / 16%);}
.fruit-model-row button{background:rgb(255 255 255 / 9%);border-radius:9px;}
.fruit-start{background:linear-gradient(180deg,#ffd85d,#f5b82f);border-radius:14px;box-shadow:0 14px 34px rgb(255 183 34 / 22%);font-family:var(--font-display);transition:transform .18s var(--ease-spring),box-shadow .18s ease;}
.fruit-start:hover:not(:disabled){box-shadow:0 18px 44px rgb(255 183 34 / 32%);transform:translateY(-2px);}
.fruit-lobby__art{border:1px solid rgb(255 255 255 / 8%);border-radius:28px 0 0 28px;margin:1rem 0 1rem 1rem;}
.fruit-stage{height:calc(100vh - 4.2rem);}
.fruit-hud--score,.fruit-hud--lives,.fruit-tracking,.fruit-actions{background:rgb(18 22 27 / 55%);border:1px solid rgb(255 255 255 / 12%);border-radius:16px;box-shadow:0 12px 34px rgb(0 0 0 / 22%);padding:.7rem .9rem;-webkit-backdrop-filter:blur(20px) saturate(150%);backdrop-filter:blur(20px) saturate(150%);}
.fruit-hud--lives{padding:.45rem .8rem;}
.fruit-actions{padding:.35rem;}
.fruit-actions button{background:rgb(255 255 255 / 8%);border-color:rgb(255 255 255 / 12%);border-radius:11px;}
.fruit-overlay{background:rgb(6 9 12 / 74%);-webkit-backdrop-filter:blur(24px) saturate(130%);backdrop-filter:blur(24px) saturate(130%);}
.fruit-overlay--paused button,.fruit-overlay--finished>button{background:linear-gradient(180deg,#ffd85d,#f2b52c);border-radius:13px;font-family:var(--font-display);}
.fruit-overlay--finished>button.quiet{border-radius:13px;}
.fruit-game{height:100svh;min-height:0;overflow:hidden;position:relative}.game-back{align-items:center;background:rgb(10 13 12 / 68%);border:1px solid rgb(255 255 255 / 18%);border-radius:50%;color:#fff;cursor:pointer;display:flex;font:400 1.35rem/1 system-ui;height:2.7rem;justify-content:center;position:fixed;left:1rem;top:1rem;transition:.2s ease;width:2.7rem;z-index:100}.game-back:hover{background:#ffffff18;border-color:#ffcf48;box-shadow:0 0 24px #ffcf4855;transform:translateX(-2px)}.fruit-lobby,.fruit-stage{height:100svh;min-height:0}.fruit-lobby{min-height:0}@media(max-height:850px) and (min-width:801px){.fruit-lobby__copy{padding:clamp(1rem,3vh,2rem) clamp(2rem,5vw,5rem)}.fruit-lobby h1{font-size:clamp(2.8rem,5vw,5rem);margin:.55rem 0 .85rem}.fruit-lobby__copy>p:not(.fruit-kicker,.fruit-error,.fruit-message){font-size:.88rem;line-height:1.5}.fruit-rules{margin:.8rem 0 .65rem;padding:.55rem}.fruit-model-picker button{padding:.55rem .7rem}.fruit-model-row{padding:.6rem}.fruit-start{padding:.8rem 1rem}}
@media(max-width:800px){.fruit-lobby__copy{box-sizing:border-box;height:100%;overflow:hidden;padding:3.6rem 1rem .8rem}.fruit-lobby h1{font-size:clamp(2.5rem,10vw,3.6rem);margin:.35rem 0 .65rem}.fruit-lobby__copy>p:not(.fruit-kicker,.fruit-error,.fruit-message){font-size:.76rem;line-height:1.35}.fruit-rules{grid-template-columns:repeat(3,1fr);margin:.6rem 0 .4rem;padding:.4rem}.fruit-rules div{border-bottom:0;border-right:1px solid #ffffff14;padding:.35rem}.fruit-rules span{font-size:.56rem}.fruit-model-picker button,.fruit-model-row{padding:.5rem}.fruit-model-picker span{display:none}.fruit-start{padding:.7rem}.game-back{left:.7rem;top:.7rem}}
</style>
