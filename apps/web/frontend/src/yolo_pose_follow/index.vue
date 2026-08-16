<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue"

import { ApiError } from "@/api/client"
import { captureVideoFrame, waitForNextVideoFrame } from "@/object_detection/frame"
import { estimatePoses, fetchResourceStatus } from "@/pose_estimation/api"
import PoseSkeletonLayer from "@/pose_estimation/components/PoseSkeletonLayer.vue"
import type { PoseResult } from "@/pose_estimation/types"
import type { ResourceStatus } from "@/vision/types"
import { errorMessage, loadSharedModel, unloadModel } from "@/modelLifecycle"

import { fetchPoseTemplates, matchPose } from "./api"
import PoseFigure from "./PoseFigure.vue"
import type { PoseMatch, PoseTemplate } from "./types"

type GamePhase = "lobby" | "preparing" | "playing" | "round-result" | "finished"
type Difficulty = "easy" | "normal" | "hard" | "expert"

const LEVELS: Record<Difficulty, {
  label: string
  rounds: number
  startMs: number
  endMs: number
  maxPoseDifficulty: number
}> = {
  easy: { label: "Easy", rounds: 8, startMs: 3000, endMs: 2200, maxPoseDifficulty: 2 },
  normal: { label: "Normal", rounds: 12, startMs: 2600, endMs: 1700, maxPoseDifficulty: 3 },
  hard: { label: "Hard", rounds: 16, startMs: 2200, endMs: 1250, maxPoseDifficulty: 4 },
  expert: { label: "Expert", rounds: 24, startMs: 1800, endMs: 900, maxPoseDifficulty: 4 },
}

const video = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)
const cameraReady = ref(false)
const templates = ref<PoseTemplate[]>([])
const resource = ref<ResourceStatus | null>(null)
const phase = ref<GamePhase>("lobby")
const difficulty = ref<Difficulty>("normal")
const prepCount = ref(3)
const roundIndex = ref(0)
const target = ref<PoseTemplate | null>(null)
const result = ref<PoseResult | null>(null)
const poseMatch = ref<PoseMatch | null>(null)
const score = ref(0)
const combo = ref(0)
const bestCombo = ref(0)
const successCount = ref(0)
const timeLeftMs = ref(0)
const roundOutcome = ref<"success" | "fail" | null>(null)
const error = ref("")
const modelInstanceId = ref<string | null>(null)
const modelBusy = ref(false)
const lifecycleMessage = ref<{ type: "success" | "error"; text: string } | null>(null)
const canvas = document.createElement("canvas")

let gameGeneration = 0
let roundDeadline = 0
let ticker: number | null = null
let inferenceRequest: AbortController | null = null
let matchRequest: AbortController | null = null
let matchedFrames = 0
let audioContext: AudioContext | null = null
let previousTemplateId = ""

const selectedLevel = computed(() => LEVELS[difficulty.value])
const sourceReady = computed(() =>
  resource.value?.artifacts.some((artifact) => artifact.artifact_id === "source" && artifact.available),
)
const progress = computed(() => roundIndex.value / selectedLevel.value.rounds)
const timerProgress = computed(() => {
  const duration = currentRoundDuration()
  return duration ? Math.max(0, Math.min(1, timeLeftMs.value / duration)) : 0
})
const scorePercent = computed(() => Math.round((poseMatch.value?.score ?? 0) * 100))

function currentRoundDuration(): number {
  const level = selectedLevel.value
  if (level.rounds <= 1) return level.endMs
  const position = Math.max(0, roundIndex.value - 1) / (level.rounds - 1)
  return Math.round(level.startMs + (level.endMs - level.startMs) * position)
}

function playTone(kind: "tick" | "go" | "success" | "fail") {
  const AudioCtor = window.AudioContext
  audioContext ??= new AudioCtor()
  const now = audioContext.currentTime
  const frequencies = kind === "success" ? [520, 720, 960] : kind === "fail" ? [180, 120] : [kind === "go" ? 720 : 360]
  frequencies.forEach((frequency, index) => {
    const oscillator = audioContext!.createOscillator()
    const gain = audioContext!.createGain()
    oscillator.type = kind === "fail" ? "sawtooth" : "sine"
    oscillator.frequency.value = frequency
    gain.gain.setValueAtTime(0.0001, now + index * 0.07)
    gain.gain.exponentialRampToValueAtTime(0.12, now + index * 0.07 + 0.01)
    gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.07 + 0.13)
    oscillator.connect(gain).connect(audioContext!.destination)
    oscillator.start(now + index * 0.07)
    oscillator.stop(now + index * 0.07 + 0.15)
  })
}

async function loadGame() {
  try {
    const [nextTemplates, nextResource] = await Promise.all([
      fetchPoseTemplates(),
      fetchResourceStatus("m"),
    ])
    templates.value = nextTemplates
    resource.value = nextResource
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "游戏资源加载失败"
  }
}

async function openCamera(): Promise<boolean> {
  if (cameraReady.value) return true
  error.value = ""
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持摄像头，请使用 localhost 或 HTTPS")
    }
    const nextStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    })
    stream.value = nextStream
    if (!video.value) return false
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

async function ensureModelLoaded(): Promise<boolean> {
  if (modelInstanceId.value) return true
  if (!resource.value || !sourceReady.value) return false
  modelBusy.value = true
  lifecycleMessage.value = null
  try {
    const loadedModel = await loadSharedModel(resource.value.model_id, "m", "auto")
    modelInstanceId.value = loadedModel.instance_id
    lifecycleMessage.value = { type: "success", text: "模型加载成功" }
    return true
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "模型加载失败") }
    return false
  } finally {
    modelBusy.value = false
  }
}

async function toggleModel() {
  if (modelBusy.value) return
  if (!modelInstanceId.value) {
    await ensureModelLoaded()
    return
  }
  modelBusy.value = true
  lifecycleMessage.value = null
  try {
    await unloadModel(modelInstanceId.value)
    modelInstanceId.value = null
    lifecycleMessage.value = { type: "success", text: "模型卸载成功" }
  } catch (caught) {
    lifecycleMessage.value = { type: "error", text: errorMessage(caught, "模型卸载失败") }
  } finally {
    modelBusy.value = false
  }
}

function closeCamera() {
  stopActiveWork()
  stream.value?.getTracks().forEach((track) => track.stop())
  stream.value = null
  cameraReady.value = false
  if (video.value) video.value.srcObject = null
}

function stopActiveWork() {
  gameGeneration += 1
  if (ticker != null) window.clearInterval(ticker)
  ticker = null
  inferenceRequest?.abort()
  matchRequest?.abort()
  inferenceRequest = null
  matchRequest = null
}

async function startGame() {
  if (!sourceReady.value) {
    error.value = "模型不可用，请前往 Models 页面下载、转换并加载 YOLOv8 Pose M"
    return
  }
  if (!await ensureModelLoaded()) return
  phase.value = "preparing"
  await nextTick()
  if (!(await openCamera())) {
    phase.value = "lobby"
    return
  }
  stopActiveWork()
  const generation = gameGeneration
  score.value = 0
  combo.value = 0
  bestCombo.value = 0
  successCount.value = 0
  roundIndex.value = 0
  result.value = null
  poseMatch.value = null
  prepCount.value = 3
  for (let count = 3; count > 0; count -= 1) {
    if (generation !== gameGeneration) return
    prepCount.value = count
    playTone("tick")
    await new Promise((resolve) => window.setTimeout(resolve, 700))
  }
  if (generation !== gameGeneration) return
  playTone("go")
  startNextRound(generation)
  void inferenceLoop(generation)
}

function chooseTemplate(): PoseTemplate {
  const pool = templates.value.filter(
    (item) => item.difficulty <= selectedLevel.value.maxPoseDifficulty,
  )
  const alternatives = pool.filter((item) => item.template_id !== previousTemplateId)
  const selected = alternatives[Math.floor(Math.random() * alternatives.length)] ?? pool[0]!
  previousTemplateId = selected.template_id
  return selected
}

function startNextRound(generation: number) {
  if (generation !== gameGeneration) return
  if (roundIndex.value >= selectedLevel.value.rounds) {
    finishGame()
    return
  }
  roundIndex.value += 1
  target.value = chooseTemplate()
  poseMatch.value = null
  matchedFrames = 0
  roundOutcome.value = null
  phase.value = "playing"
  timeLeftMs.value = currentRoundDuration()
  roundDeadline = performance.now() + timeLeftMs.value
  if (ticker != null) window.clearInterval(ticker)
  ticker = window.setInterval(() => {
    if (phase.value !== "playing") return
    timeLeftMs.value = Math.max(0, roundDeadline - performance.now())
    if (timeLeftMs.value <= 0) completeRound(false, generation)
  }, 40)
}

function completeRound(success: boolean, generation: number) {
  if (generation !== gameGeneration || phase.value !== "playing") return
  phase.value = "round-result"
  roundOutcome.value = success ? "success" : "fail"
  if (ticker != null) window.clearInterval(ticker)
  ticker = null
  if (success) {
    const speedBonus = Math.round(timeLeftMs.value / 20)
    combo.value += 1
    bestCombo.value = Math.max(bestCombo.value, combo.value)
    successCount.value += 1
    score.value += 100 + speedBonus + Math.min(combo.value, 10) * 15
    playTone("success")
  } else {
    combo.value = 0
    playTone("fail")
  }
  window.setTimeout(() => startNextRound(generation), success ? 620 : 520)
}

async function inferenceLoop(generation: number) {
  let lastMediaTime = -1
  while (generation === gameGeneration && phase.value !== "finished" && video.value) {
    if (phase.value !== "playing" || !target.value) {
      await new Promise((resolve) => window.setTimeout(resolve, 50))
      continue
    }
    try {
      if (video.value.currentTime <= lastMediaTime + 0.001) {
        await waitForNextVideoFrame(video.value)
      }
      lastMediaTime = video.value.currentTime
      const frame = await captureVideoFrame(video.value, canvas, "pose-follow-frame.jpg")
      inferenceRequest = new AbortController()
      const nextResult = await estimatePoses(frame, {
        runtime: "auto",
        variant: "m",
        confidence: 0.28,
        iouThreshold: 0.7,
        maxDetections: 1,
      }, { signal: inferenceRequest.signal, cacheInput: false })
      if (generation !== gameGeneration) return
      result.value = nextResult
      const person = nextResult.poses[0]
      if (!person || !target.value || phase.value !== "playing") {
        poseMatch.value = null
        matchedFrames = 0
        continue
      }
      matchRequest = new AbortController()
      const nextMatch = await matchPose(target.value.template_id, person, matchRequest.signal)
      if (generation !== gameGeneration || phase.value !== "playing") continue
      poseMatch.value = nextMatch
      matchedFrames = nextMatch.matched ? matchedFrames + 1 : 0
      if (matchedFrames >= 2) completeRound(true, generation)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return
      error.value = caught instanceof ApiError ? caught.message : caught instanceof Error
        ? caught.message : "实时姿态推理失败"
      stopActiveWork()
      phase.value = "lobby"
      return
    }
  }
}

function finishGame() {
  phase.value = "finished"
  if (ticker != null) window.clearInterval(ticker)
  ticker = null
}

function returnToLobby() {
  closeCamera()
  phase.value = "lobby"
  target.value = null
  poseMatch.value = null
  result.value = null
  roundOutcome.value = null
  timeLeftMs.value = 0
}

onMounted(() => {
  document.body.classList.add("pose-game-active")
  void loadGame()
})
onBeforeUnmount(() => {
  document.body.classList.remove("pose-game-active")
  closeCamera()
  void audioContext?.close()
})
</script>

<template>
  <div class="pose-game" :class="[`pose-game--${phase}`, { 'pose-game--success': roundOutcome === 'success', 'pose-game--fail': roundOutcome === 'fail' }]">
    <header class="pose-game__topbar">
      <RouterLink to="/games" class="pose-game__back">← Neural Games</RouterLink>
      <div class="pose-game__brand"><span>🕺</span> Yolo Pose Follow</div>
      <button v-if="phase !== 'lobby'" type="button" @click="returnToLobby">EXIT</button>
    </header>

    <main v-if="phase === 'lobby'" class="game-lobby">
      <section class="game-lobby__intro">
        <p class="kicker">NEURAL CAMERA GAME · YOLOV8 POSE M</p>
        <h1>Follow the pose. <em>Beat the clock.</em></h1>
        <p>镜头中的你就是控制器。模仿骨架、保持姿势、连续得分，节奏会越来越快。</p>
        <div class="difficulty-picker" aria-label="选择难度">
          <button
            v-for="(level, key) in LEVELS"
            :key="key"
            type="button"
            :class="{ active: difficulty === key }"
            @click="difficulty = key as Difficulty"
          >
            <strong>{{ level.label }}</strong>
            <span>{{ level.rounds }} poses · {{ (level.endMs / 1000).toFixed(1) }}s finish</span>
          </button>
        </div>
        <div v-if="!sourceReady" class="game-model-setup">
          <div><span></span><strong>YOLOv8 Pose M weights required</strong></div>
          <p class="game-error">模型不可用，请前往 Models 页面管理 YOLOv8 Pose M。</p>
        </div>
        <div class="game-model-action">
          <span>YOLOv8 Pose M · AUTO</span>
          <button type="button" :disabled="modelBusy || (!modelInstanceId && !sourceReady)" @click="toggleModel">
            {{ modelBusy ? "WAIT…" : modelInstanceId ? "UNLOAD" : "LOAD" }}
          </button>
        </div>
        <small v-if="lifecycleMessage" :class="`lifecycle-${lifecycleMessage.type}`">{{ lifecycleMessage.text }}</small>
        <button class="game-start" type="button" :disabled="modelBusy || !sourceReady || !templates.length" @click="startGame">
          <span>START {{ selectedLevel.label.toUpperCase() }}</span><b>→</b>
        </button>
        <p v-if="error" class="game-error" role="alert">{{ error }}</p>
      </section>
      <aside class="game-lobby__preview">
        <div class="preview-orbit preview-orbit--one"></div>
        <div class="preview-orbit preview-orbit--two"></div>
        <PoseFigure v-if="templates[16]" :pose="templates[16]" />
        <span class="preview-score">+100</span>
      </aside>
    </main>

    <main v-else class="game-arena">
      <section class="camera-stage">
        <video ref="video" autoplay muted playsinline></video>
        <PoseSkeletonLayer :result="result" />
        <div class="camera-vignette"></div>
        <div class="camera-label"><i></i> YOU · LIVE</div>
        <div v-if="poseMatch" class="match-meter">
          <span :style="{ width: `${scorePercent}%` }"></span>
          <b>{{ scorePercent }}% MATCH</b>
        </div>
      </section>

      <section class="target-stage">
        <div class="target-stage__label">TARGET POSE · {{ String(roundIndex).padStart(2, "0") }}</div>
        <PoseFigure v-if="target" :pose="target" />
        <div v-if="target" class="target-cue">
          <strong>{{ target.name }}</strong><span>{{ target.cue }}</span>
        </div>
      </section>

      <div class="game-hud">
        <div><span>SCORE</span><strong>{{ score.toLocaleString() }}</strong></div>
        <div><span>COMBO</span><strong>×{{ combo }}</strong></div>
        <div><span>POSE</span><strong>{{ roundIndex }}/{{ selectedLevel.rounds }}</strong></div>
      </div>

      <div class="round-timer" :style="{ '--timer': timerProgress }">
        <div><strong>{{ (timeLeftMs / 1000).toFixed(1) }}</strong><span>SEC</span></div>
      </div>

      <div class="round-progress"><span :style="{ width: `${progress * 100}%` }"></span></div>

      <div v-if="phase === 'preparing'" class="game-overlay game-overlay--prep">
        <p>GET READY</p><strong>{{ prepCount }}</strong><span>站到镜头中央，让全身进入画面</span>
      </div>
      <div v-if="phase === 'round-result'" class="game-overlay game-overlay--result">
        <strong>{{ roundOutcome === "success" ? "NICE!" : "MISS" }}</strong>
        <span>{{ roundOutcome === "success" ? `+${100 + Math.round(timeLeftMs / 20)}` : "Next pose…" }}</span>
        <i v-for="particle in 18" :key="particle" :style="{ '--i': particle }"></i>
      </div>
      <div v-if="phase === 'finished'" class="game-overlay game-overlay--finished">
        <p>SESSION COMPLETE</p><strong>{{ score.toLocaleString() }}</strong>
        <div><span>{{ successCount }}/{{ selectedLevel.rounds }} cleared</span><span>Best combo ×{{ bestCombo }}</span></div>
        <button type="button" @click="startGame">PLAY AGAIN</button>
        <button type="button" class="quiet" @click="returnToLobby">CHANGE LEVEL</button>
      </div>
    </main>
  </div>
</template>

<style scoped>
:global(body.pose-game-active .site-header),
:global(body.pose-game-active .site-footer) { display:none; }
.pose-game { background:#0c0d0c; color:#f4f1e8; min-height:100vh; overflow:hidden; }
.pose-game__topbar { align-items:center; border-bottom:1px solid #ffffff24; display:grid; font-family:var(--font-mono); grid-template-columns:1fr auto 1fr; height:3.3rem; padding:0 2.1rem; position:relative; z-index:20; }
.pose-game__back { color:#aaa; font-size:.68rem; text-decoration:none; text-transform:uppercase; }
.pose-game__brand { font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
.pose-game__brand span { font-size:1.15rem; margin-right:.45rem; }
.pose-game__topbar button { background:none; border:0; color:#aaa; cursor:pointer; font:inherit; font-size:.65rem; justify-self:end; }
.game-lobby { display:grid; grid-template-columns:1.08fr .92fr; min-height:calc(100vh - 3.3rem); }
.game-lobby__intro { align-self:start; max-width:900px; padding:2rem clamp(2rem,5vw,5rem); }
.game-lobby__intro .kicker { color:#c8ff46; }
.game-lobby h1 { font-size:clamp(2.3rem,4.5vw,4.8rem); letter-spacing:-.065em; line-height:.95; margin:.65rem 0 1rem; }
.game-lobby h1 em { color:transparent; display:block; font-style:normal; -webkit-text-stroke:1.5px #f4f1e8; }
.game-lobby__intro > p:not(.kicker,.game-error) { color:#aaa; font-size:1rem; line-height:1.6; max-width:36rem; }
.difficulty-picker { display:grid; gap:.5rem; grid-template-columns:repeat(4,1fr); margin:2rem 0 1rem; }
.difficulty-picker button { background:#171917; border:1px solid #ffffff22; color:#aaa; cursor:pointer; padding:.8rem; text-align:left; transition:.2s ease; }
.difficulty-picker button:hover,.difficulty-picker button.active { border-color:#c8ff46; color:#fff; transform:translateY(-2px); }
.difficulty-picker button.active { background:#c8ff4612; box-shadow:inset 0 -2px #c8ff46; }
.difficulty-picker strong,.difficulty-picker span { display:block; }
.difficulty-picker strong { font-size:.78rem; text-transform:uppercase; }
.difficulty-picker span { font: .52rem/1.4 var(--font-mono); margin-top:.3rem; }
.game-model-setup { align-items:center; background:#211d13; border:1px solid #f6bf4f55; display:flex; justify-content:space-between; margin:1rem 0; padding:.8rem; }
.game-model-setup div { align-items:center; display:flex; font-size:.72rem; gap:.55rem; }
.game-model-setup div span { background:#f6bf4f; border-radius:50%; height:.5rem; width:.5rem; }
.game-model-setup button { background:#f6bf4f; border:0; cursor:pointer; font:700 .58rem var(--font-mono); padding:.6rem .8rem; }
.game-start { align-items:center; background:#c8ff46; border:0; color:#0c0d0c; cursor:pointer; display:flex; font:700 .72rem var(--font-mono); justify-content:space-between; margin-top:1rem; padding:1.1rem 1.2rem; width:100%; }
.game-start b { font-size:1.3rem; }
.game-start:disabled { cursor:not-allowed; filter:grayscale(1); opacity:.35; }
.game-model-action { align-items:center; display:flex; font:.56rem var(--font-mono); justify-content:space-between; margin-top:1rem; }.game-model-action button { background:#0c0d0c; border:0; color:#fff; cursor:pointer; font:inherit; padding:.55rem .7rem; }.game-model-action button:disabled { cursor:not-allowed; opacity:.4; }
.lifecycle-success { color:#3c8b2f; font:.54rem var(--font-mono); }.lifecycle-error { color:#cf3f27; font:.54rem var(--font-mono); }
.game-error { color:#ff8066!important; font: .65rem var(--font-mono); margin-top:1rem; }
.game-lobby__preview { align-items:center; background:radial-gradient(circle at center,#c8ff4628 0,transparent 48%),linear-gradient(135deg,#181b17,#0d0e0d); display:flex; justify-content:center; min-height:520px; overflow:hidden; position:relative; }
.game-lobby__preview :deep(.target-figure) { height:min(68vh,680px); position:relative; width:72%; z-index:2; }
.preview-orbit { border:1px solid #c8ff4635; border-radius:50%; position:absolute; }
.preview-orbit--one { animation:spin 14s linear infinite; height:72%; width:72%; }
.preview-orbit--two { animation:spin 9s linear infinite reverse; border-style:dashed; height:52%; width:52%; }
.preview-score { animation:float 2s ease-in-out infinite; color:#c8ff46; font:700 2rem var(--font-mono); position:absolute; right:14%; top:24%; }
.game-arena { display:grid; grid-template-columns:1.55fr .75fr; height:calc(100vh - 4.2rem); position:relative; }
.camera-stage { background:#111; overflow:hidden; position:relative; }
.camera-stage video { height:100%; object-fit:cover; transform:scaleX(-1); width:100%; }
.camera-stage :deep(.pose-layer) { transform:scaleX(-1); }
.camera-vignette { background:linear-gradient(90deg,#0005,transparent 30%,transparent 70%,#0007),linear-gradient(0deg,#0009,transparent 28%); inset:0; pointer-events:none; position:absolute; }
.camera-label,.target-stage__label { font: .62rem var(--font-mono); left:1.4rem; letter-spacing:.09em; position:absolute; top:1.3rem; z-index:3; }
.camera-label i { animation:pulse 1s infinite; background:#c8ff46; border-radius:50%; display:inline-block; height:.45rem; margin-right:.4rem; width:.45rem; }
.match-meter { bottom:1.5rem; left:1.5rem; position:absolute; width:min(20rem,45%); z-index:4; }
.match-meter::before { background:#ffffff1e; content:""; inset:0; position:absolute; }
.match-meter span { background:#c8ff46; display:block; height:.32rem; transition:width .2s; }
.match-meter b { display:block; font: .55rem var(--font-mono); margin-top:.45rem; }
.target-stage { align-items:center; background:radial-gradient(circle,#c8ff4615,transparent 60%),#151715; border-left:1px solid #ffffff20; display:flex; flex-direction:column; justify-content:center; padding:4rem 2rem 2rem; position:relative; }
.target-cue { text-align:center; }
.target-cue strong,.target-cue span { display:block; }
.target-cue strong { font-size:1.6rem; }
.target-cue span { color:#aaa; margin-top:.4rem; }
.game-hud { display:flex; gap:2.5rem; left:1.5rem; position:absolute; top:1.2rem; z-index:5; }
.game-hud div { display:flex; flex-direction:column; }
.game-hud span { color:#aaa; font: .52rem var(--font-mono); }
.game-hud strong { font-size:1.25rem; }
.round-timer { background:conic-gradient(#c8ff46 calc(var(--timer) * 1turn),#ffffff1a 0); border-radius:50%; padding:4px; position:absolute; right:1.5rem; top:1.1rem; z-index:6; }
.round-timer > div { align-items:center; background:#111; border-radius:50%; display:flex; flex-direction:column; height:4.6rem; justify-content:center; width:4.6rem; }
.round-timer strong { font:700 1.25rem var(--font-mono); }
.round-timer span { color:#888; font:.45rem var(--font-mono); }
.round-progress { background:#ffffff14; bottom:0; height:.35rem; left:0; position:absolute; right:0; z-index:8; }
.round-progress span { background:#c8ff46; display:block; height:100%; transition:width .35s; }
.game-overlay { align-items:center; background:#080908e8; display:flex; flex-direction:column; inset:0; justify-content:center; position:absolute; z-index:15; }
.game-overlay--prep p,.game-overlay--finished p { color:#c8ff46; font:.7rem var(--font-mono); letter-spacing:.15em; }
.game-overlay--prep strong { animation:count .7s ease-out; font-size:min(35vw,18rem); line-height:1; }
.game-overlay--prep span { color:#aaa; }
.game-overlay--result { background:#0c0d0c88; pointer-events:none; }
.game-overlay--result strong { animation:slam .35s ease-out; color:#c8ff46; font-size:clamp(5rem,15vw,14rem); letter-spacing:-.08em; }
.pose-game--fail .game-overlay--result strong { color:#ff7148; }
.game-overlay--result > span { font:700 1rem var(--font-mono); }
.game-overlay--result i { --angle:calc(var(--i) * 20deg); animation:burst .6s ease-out forwards; background:#c8ff46; height:.5rem; left:50%; position:absolute; top:50%; transform:rotate(var(--angle)) translateY(-2rem); width:.5rem; }
.pose-game--fail .game-overlay--result i { display:none; }
.pose-game--fail .game-arena { animation:shake .35s; }
.game-overlay--finished strong { font-size:clamp(5rem,13vw,12rem); letter-spacing:-.07em; line-height:1; }
.game-overlay--finished > div { display:flex; gap:2rem; margin:1rem 0 2rem; }
.game-overlay--finished > div span { color:#aaa; font:.7rem var(--font-mono); text-transform:uppercase; }
.game-overlay--finished button { background:#c8ff46; border:0; cursor:pointer; font:700 .7rem var(--font-mono); margin:.25rem; min-width:13rem; padding:1rem; }
.game-overlay--finished button.quiet { background:transparent; border:1px solid #ffffff3a; color:#fff; }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes float { 50% { transform:translateY(-12px) rotate(-4deg); } }
@keyframes pulse { 50% { box-shadow:0 0 0 7px #c8ff4600; opacity:.5; } }
@keyframes count { from { opacity:0; transform:scale(1.8); } }
@keyframes slam { from { opacity:0; transform:scale(1.8) rotate(-5deg); } }
@keyframes burst { to { opacity:0; transform:rotate(var(--angle)) translateY(-15rem) scale(.2); } }
@keyframes shake { 25% { transform:translateX(-8px); } 50% { transform:translateX(8px); } 75% { transform:translateX(-4px); } }
@media (max-width:800px) {
  .pose-game { overflow:auto; }
  .pose-game__topbar { grid-template-columns:1fr auto; padding:0 1rem; }
  .pose-game__brand { justify-self:end; }
  .pose-game__topbar button { display:none; }
  .game-lobby { grid-template-columns:1fr; }
  .game-lobby__intro { padding:2.4rem 1.2rem; }
  .game-lobby h1 { font-size:3rem; white-space:normal; }
  .difficulty-picker { grid-template-columns:repeat(2,1fr); }
  .game-lobby__preview { min-height:380px; }
  .game-arena { grid-template-columns:1fr; grid-template-rows:62% 38%; height:calc(100svh - 4.2rem); }
  .target-stage { border-left:0; border-top:1px solid #ffffff20; padding:.5rem 7rem .5rem 1rem; }
  .target-stage :deep(.target-figure) { height:100%; width:50%; }
  .target-cue { position:absolute; right:1rem; width:44%; }
  .target-cue strong { font-size:1.05rem; }
  .game-hud { gap:1rem; left:1rem; }
  .round-timer { right:1rem; }
  .camera-label,.target-stage__label { display:none; }
}
</style>
