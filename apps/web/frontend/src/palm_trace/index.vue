<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from "vue"
import { useRouter } from "vue-router"
import { errorMessage, loadSharedModel, unloadModel } from "@/modelLifecycle"
import { captureVideoFrame, wait, waitForNextVideoFrame } from "@/object_detection/frame"
import { estimatePoses, fetchHandResourceStatus } from "@/pose_estimation/api"
import type { ResourceStatus, VisionOptions } from "@/vision/types"
import { TraceAudio } from "./audio"
import { DIFFICULTIES, HAND_MODEL_ID, selectPalmRuntime } from "./config"
import { PalmTraceEngine } from "./engine"
import { TraceRenderer } from "./renderer"
import { PalmTracker } from "./tracker"
import type { Difficulty, GamePhase, GameSnapshot, TrackedPalm } from "./types"

const router = useRouter(), phase = ref<GamePhase>("lobby"), difficulty = ref<Difficulty>("arcade")
const resource = ref<ResourceStatus | null>(null), modelInstanceId = ref<string | null>(null), modelRuntime = ref(""), modelBusy = ref(false)
const error = ref(""), countdown = ref<string | number>(3), inferenceMs = ref<number | null>(null), soundEnabled = ref(true)
const video = ref<HTMLVideoElement | null>(null), stage = ref<HTMLElement | null>(null), canvas = ref<HTMLCanvasElement | null>(null), stream = ref<MediaStream | null>(null)
const palms = shallowRef<TrackedPalm[]>([]), snapshot = shallowRef<GameSnapshot | null>(null)
const captureCanvas = document.createElement("canvas"), tracker = new PalmTracker(), audio = new TraceAudio()
let engine: PalmTraceEngine | null = null, renderer: TraceRenderer | null = null, frameId = 0, generation = 0, lastAt = 0, aborter: AbortController | null = null, observer: ResizeObserver | null = null

const availableRuntime = computed(() => selectPalmRuntime(resource.value))
const modelStatus = computed(() => modelBusy.value ? "LOADING" : modelInstanceId.value ? `${modelRuntime.value.toUpperCase()} READY` : availableRuntime.value ? `${availableRuntime.value.toUpperCase()} AVAILABLE` : "MODEL MISSING")
const activePalms = computed(() => palms.value.filter((palm) => palm.active).length)
const timeSeconds = computed(() => ((snapshot.value?.remainingMs ?? 0) / 1000).toFixed(1))

async function loadResource(): Promise<void> { try { resource.value = await fetchHandResourceStatus() } catch (caught) { error.value = errorMessage(caught, "Could not read the palm model status") } }
async function ensureModel(): Promise<boolean> {
  if (modelInstanceId.value) return true
  if (!availableRuntime.value) { error.value = "The palm detection model is not ready. Prepare it on the Models page first."; return false }
  modelBusy.value = true
  try { const loaded = await loadSharedModel(HAND_MODEL_ID, "float", availableRuntime.value); modelInstanceId.value = loaded.instance_id; modelRuntime.value = loaded.runtime; return true }
  catch (caught) { error.value = errorMessage(caught, "Palm model loading failed"); return false }
  finally { modelBusy.value = false }
}
async function toggleModel(): Promise<void> { if (modelInstanceId.value) { await unloadModel(modelInstanceId.value); modelInstanceId.value = null; modelRuntime.value = "" } else await ensureModel() }
async function openCamera(): Promise<boolean> {
  try { stream.value = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } }); await nextTick(); if (!video.value) throw new Error("The game stage is not ready"); video.value.srcObject = stream.value; await video.value.play(); return true }
  catch (caught) { error.value = caught instanceof DOMException && caught.name === "NotAllowedError" ? "Camera permission was denied." : errorMessage(caught, "Could not start the camera"); closeCamera(); return false }
}
function setup(): void {
  if (!stage.value || !canvas.value) return
  const rect = stage.value.getBoundingClientRect(); engine = new PalmTraceEngine(difficulty.value); renderer = new TraceRenderer(canvas.value); renderer.resize(rect.width, rect.height); snapshot.value = engine.state
  observer = new ResizeObserver(() => { if (!stage.value || !renderer) return; const next = stage.value.getBoundingClientRect(); renderer.resize(next.width, next.height) }); observer.observe(stage.value)
}
async function start(): Promise<void> {
  error.value = ""; await audio.unlock(); if (!await ensureModel()) return
  phase.value = "calibrating"; await nextTick(); if (!await openCamera()) { phase.value = "lobby"; return }
  tracker.reset(); palms.value = []; setup(); const token = ++generation; lastAt = performance.now(); frameId = requestAnimationFrame((now) => renderLoop(now, token)); void inferLoop(token)
  const deadline = performance.now() + 12_000
  while (token === generation && activePalms.value < 1 && performance.now() < deadline) await wait(80)
  if (token !== generation) return
  if (activePalms.value < 1) { error.value = "No palm detected. Keep one open hand fully inside the camera frame."; exit(); return }
  phase.value = "countdown"; engine?.setPhase("countdown")
  for (const value of [3, 2, 1]) { countdown.value = value; audio.play("tick"); await wait(650); if (token !== generation) return }
  countdown.value = "WIPE"; await wait(450); if (token !== generation) return
  phase.value = "playing"; engine?.setPhase("playing"); lastAt = performance.now()
}
function renderLoop(now: number, token: number): void {
  if (token !== generation || !engine || !renderer) return
  if (phase.value === "playing") engine.update(now - lastAt, palms.value)
  lastAt = now; while (engine.events.length) audio.play(engine.events.shift()!)
  snapshot.value = engine.state; renderer.render(engine.state, palms.value)
  if (engine.state.phase === "won" || engine.state.phase === "lost") phase.value = engine.state.phase
  frameId = requestAnimationFrame((next) => renderLoop(next, token))
}
async function inferLoop(token: number): Promise<void> {
  let mediaTime = -1
  while (token === generation && video.value && phase.value !== "lobby") {
    if (phase.value === "paused") { await wait(80); continue }
    try {
      if (video.value.currentTime <= mediaTime + .001) await waitForNextVideoFrame(video.value); mediaTime = video.value.currentTime
      const file = await captureVideoFrame(video.value, captureCanvas, "palm-trace-frame.jpg", { maxDimension: 640, quality: .72, mirror: true }); aborter = new AbortController()
      const options: VisionOptions = { runtime: "auto", variant: "float", confidence: .6, iouThreshold: .45, maxDetections: 2, poseEnabled: false, faceEnabled: false, handEnabled: true, handConfidence: .55, handLandmarksEnabled: false, handInputMirrored: true }
      const result = await estimatePoses(file, options, { signal: aborter.signal, cacheInput: false }); if (token !== generation) return
      palms.value = tracker.update(result.hands, result.image_size.width, result.image_size.height, performance.now()); inferenceMs.value = result.parallel_timings.hand_ms
    } catch (caught) { if (caught instanceof DOMException && caught.name === "AbortError") return; error.value = errorMessage(caught, "Live palm tracking failed"); exit(); return }
  }
}
function togglePause(): void { if (!engine) return; phase.value = phase.value === "paused" ? "playing" : "paused"; engine.setPhase(phase.value); lastAt = performance.now() }
function toggleSound(): void { soundEnabled.value = !soundEnabled.value; audio.enabled = soundEnabled.value }
function closeCamera(): void { stream.value?.getTracks().forEach((track) => track.stop()); stream.value = null; if (video.value) video.value.srcObject = null }
function exit(): void { generation++; cancelAnimationFrame(frameId); aborter?.abort(); observer?.disconnect(); closeCamera(); tracker.reset(); palms.value = []; snapshot.value = null; engine = null; renderer = null; phase.value = "lobby" }
function handleBack(): void { if (phase.value === "lobby") router.back(); else exit() }
onMounted(() => { document.body.classList.add("trace-active"); void loadResource() })
onBeforeUnmount(() => { document.body.classList.remove("trace-active"); exit(); void audio.close() })
</script>

<template>
  <div class="trace-game">
    <button class="game-back" type="button" aria-label="Back" @click="handleBack">←</button>
    <main v-if="phase === 'lobby'" class="trace-lobby">
      <section class="lobby-copy">
        <p class="eyebrow">TESTING PREVIEW · ON-DEVICE PALM ARCADE</p><h1>PALM<br><em>TRACE</em></h1>
        <p class="lead">Your screen is dirty. Follow every glowing trail with your palm, wipe it clean, and bring both hands in when the geometry splits.</p>
        <div class="rules"><span><b>01</b>ENTER THE GLOWING START</span><span><b>02</b>STAY INSIDE THE WIDE TRAIL</span><span><b>03</b>WIPE TWO SIDES TOGETHER</span></div>
        <div class="difficulty"><button v-for="(item, key) in DIFFICULTIES" :key="key" :class="{ active: difficulty === key }" @click="difficulty = key"><b>{{ item.label }}</b><small>{{ key === 'calm' ? 'WIDE TOLERANCE' : key === 'rush' ? 'FAST CLOCK' : 'TRUE RHYTHM' }}</small></button></div>
        <div class="model"><i :class="{ ready: modelInstanceId }"></i><span><b>MEDIAPIPE PALM DETECTOR</b><small>NO LANDMARKS · {{ modelStatus }}</small></span><button :disabled="modelBusy || (!availableRuntime && !modelInstanceId)" @click="toggleModel">{{ modelInstanceId ? "UNLOAD" : "LOAD" }}</button></div>
        <button class="start" :disabled="modelBusy || !availableRuntime" @click="start"><span>OPEN CAMERA</span><b>START WIPING →</b></button><p v-if="error" class="error">{{ error }}</p>
      </section>
      <aside aria-hidden="true"><div class="dirty-orbit"><i></i><i></i><i></i><span>✋</span></div><strong>12<br>TRACES</strong><small>ONE SCREEN · TWO PALMS</small></aside>
    </main>
    <main v-else ref="stage" class="trace-arena"><video ref="video" autoplay muted playsinline></video><div class="camera-wash"></div><canvas ref="canvas"></canvas>
      <header class="hud"><div><small>LEVEL</small><b>{{ snapshot?.level ?? 1 }}<i>/12</i></b></div><div><small>SCORE</small><b>{{ (snapshot?.score ?? 0).toLocaleString() }}</b></div><div><small>COMBO</small><b>×{{ snapshot?.combo ?? 0 }}</b></div><div class="clock"><small>TIME</small><b>{{ timeSeconds }}</b></div><div><small>LIVES</small><b>{{ "●".repeat(snapshot?.lives ?? 3) }}<i>{{ "○".repeat(3 - (snapshot?.lives ?? 3)) }}</i></b></div></header>
      <div class="status"><i></i>{{ activePalms }} PALM{{ activePalms === 1 ? "" : "S" }} TRACKED <small v-if="inferenceMs">{{ inferenceMs.toFixed(0) }} MS · {{ modelRuntime.toUpperCase() }}</small></div>
      <div class="message">{{ snapshot?.message }}</div><div class="arena-actions"><button @click="toggleSound">{{ soundEnabled ? "♪" : "×" }}</button><button v-if="phase === 'playing' || phase === 'paused'" @click="togglePause">{{ phase === "paused" ? "▶" : "Ⅱ" }}</button></div>
      <div v-if="phase === 'calibrating'" class="overlay"><div class="palm-icon">✋</div><p>PALM CALIBRATION · {{ activePalms }}/2</p><h2>SHOW AT LEAST ONE PALM</h2><span>Keep both hands visible now to prepare for split trails later.</span></div>
      <div v-if="phase === 'countdown'" class="overlay clear"><p>SCREEN READY</p><h2 class="count">{{ countdown }}</h2><span>FIND THE GLOW · FOLLOW THE TRAIL · WIPE IT CLEAN</span></div>
      <div v-if="phase === 'paused'" class="overlay"><p>BREAK TIME</p><h2>WIPE PAUSED</h2><button @click="togglePause">RESUME</button></div>
      <div v-if="phase === 'won' || phase === 'lost'" class="overlay result"><p>{{ phase === "won" ? "SCREEN PERFECT" : "WIPE FAILED" }}</p><h2>{{ phase === "won" ? "SPOTLESS!" : "TOO MUCH DIRT" }}</h2><b>{{ (snapshot?.score ?? 0).toLocaleString() }}</b><span>FINAL SCORE</span><button @click="start">PLAY AGAIN</button><button class="quiet" @click="exit">RETURN TO LOBBY</button></div>
    </main>
  </div>
</template>

<style scoped>
:global(body.trace-active){height:100%;overflow:hidden!important;background:#061211}:global(body.trace-active .site-header),:global(body.trace-active .site-footer){display:none}.trace-game{height:100svh;overflow:hidden;background:#061211;color:#f5fff9;font-family:Inter,system-ui,sans-serif}.game-back{position:fixed;z-index:100;left:1rem;top:1rem;width:2.7rem;height:2.7rem;border-radius:50%;border:1px solid #ffffff2b;background:#071a17bb;color:#fff;font-size:1.35rem;cursor:pointer}.game-back:hover{border-color:#54f7e3;transform:translateX(-2px)}.trace-lobby{height:100%;display:grid;grid-template-columns:minmax(0,1.12fr) minmax(22rem,.88fr);background:radial-gradient(circle at 78% 40%,#23534d 0,#0b211d 24%,#061211 60%)}.lobby-copy{padding:clamp(3.5rem,6vw,6rem);display:flex;flex-direction:column;justify-content:center;max-width:900px;z-index:2}.eyebrow,.hud small,.status,.message{font:800 .62rem ui-monospace,monospace;letter-spacing:.16em}.eyebrow{color:#54f7e3}.lobby-copy h1{font:950 clamp(4rem,8vw,8.2rem)/.72 ui-monospace,monospace;letter-spacing:-.09em;margin:1.1rem 0 2rem}.lobby-copy h1 em{font-style:normal;color:#ff5cc8;text-shadow:0 0 45px #ff5cc855}.lead{max-width:45rem;color:#a9c4bc;line-height:1.75}.rules{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid #ffffff1c;border-radius:14px;margin:1.6rem 0 .8rem;background:#ffffff08}.rules span{padding:1rem;border-right:1px solid #ffffff17;font:700 .66rem/1.5 ui-monospace}.rules span:last-child{border:0}.rules b{display:block;color:#54f7e3}.difficulty{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem}.difficulty button{padding:.75rem;background:#ffffff08;border:1px solid #ffffff1a;border-radius:10px;color:#8facaa;text-align:left}.difficulty button.active{border-color:#54f7e3;background:#54f7e319;color:#fff}.difficulty b,.difficulty small{display:block}.difficulty small{font-size:.48rem;margin-top:.2rem}.model{display:flex;align-items:center;gap:.7rem;margin-top:.55rem;padding:.7rem .85rem;border-radius:11px;background:#020b0a;border:1px solid #ffffff1a}.model i{width:.55rem;height:.55rem;border-radius:50%;background:#e45569}.model i.ready{background:#64ff92;box-shadow:0 0 12px #64ff92}.model span{margin-right:auto}.model b,.model small{display:block;font:700 .55rem ui-monospace}.model small{color:#68867f;margin-top:.18rem}.model button{background:#ffffff10;border:0;border-radius:7px;color:#fff;padding:.5rem}.start{margin-top:.55rem;padding:1rem;border:0;border-radius:12px;background:linear-gradient(100deg,#54f7e3,#dbff70);color:#06110f;display:flex;justify-content:space-between;font-weight:950}.start:disabled{filter:grayscale(1);opacity:.35}.error{color:#ff7185;font:700 .58rem ui-monospace}.trace-lobby aside{position:relative;overflow:hidden}.dirty-orbit{position:absolute;width:min(34vw,31rem);aspect-ratio:1;left:48%;top:46%;transform:translate(-50%,-50%);border:4rem solid #ffffff09;border-radius:50%;box-shadow:0 0 0 2px #54f7e344,0 0 80px #54f7e322}.dirty-orbit i{position:absolute;inset:-2rem;border:2px dashed #ff5cc866;border-radius:50%;animation:spin 15s linear infinite}.dirty-orbit i:nth-child(2){inset:3rem;border-color:#dfff7066;animation-direction:reverse}.dirty-orbit i:nth-child(3){inset:7rem;border-style:solid;border-color:#ffffff24}.dirty-orbit span{position:absolute;inset:0;display:grid;place-items:center;font-size:8rem;filter:drop-shadow(0 0 24px #54f7e3);animation:float 2.4s ease-in-out infinite}.trace-lobby aside strong{position:absolute;right:8%;bottom:12%;font:950 4rem/.8 ui-monospace;color:#ffffff20}.trace-lobby aside small{position:absolute;right:8%;bottom:7%;font:700 .52rem ui-monospace;color:#718d86}.trace-arena{position:relative;width:100%;height:100%;overflow:hidden;background:#061211}.trace-arena video,.trace-arena canvas,.camera-wash{position:absolute;inset:0;width:100%;height:100%}.trace-arena video{object-fit:cover;transform:scaleX(-1);filter:saturate(.65) contrast(1.08)}.camera-wash{z-index:1;background:linear-gradient(#061a17aa,transparent 25%,transparent 72%,#03100ddd),radial-gradient(circle,transparent 25%,#02100d99 105%)}.trace-arena canvas{z-index:2}.hud{position:absolute;z-index:5;left:50%;top:1rem;transform:translateX(-50%);display:flex;background:#03110ed9;border:1px solid #ffffff20;border-radius:14px;backdrop-filter:blur(12px);overflow:hidden}.hud>div{min-width:6.5rem;padding:.55rem .9rem;border-right:1px solid #ffffff17}.hud>div:last-child{border:0}.hud small{display:block;color:#70938a;font-size:.48rem}.hud b{font:900 1.2rem ui-monospace}.hud b i{font-style:normal;color:#66867e;font-size:.7rem}.hud .clock b{color:#dbff70}.status,.message,.arena-actions{position:absolute;z-index:6}.status{left:1rem;bottom:1rem;background:#03110ed9;padding:.6rem .8rem;border-radius:9px}.status>i{display:inline-block;width:.45rem;height:.45rem;background:#63ff90;border-radius:50%;box-shadow:0 0 10px #63ff90;margin-right:.5rem}.status small{color:#719088;margin-left:.7rem}.message{bottom:1rem;left:50%;transform:translateX(-50%);color:#fff;background:#03110ed9;padding:.62rem 1rem;border-radius:9px}.arena-actions{right:1rem;bottom:1rem;display:flex;gap:.4rem}.arena-actions button,.overlay button{border:1px solid #ffffff31;background:#ffffff12;color:#fff;border-radius:9px;padding:.65rem 1rem}.overlay{position:absolute;z-index:10;inset:0;background:#03110eea;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}.overlay.clear{background:#03110e66}.overlay p{color:#54f7e3;font:800 .65rem ui-monospace;letter-spacing:.2em}.overlay h2{font:950 clamp(2.8rem,7vw,6.5rem)/.9 ui-monospace;margin:.6rem}.overlay>span{color:#9ebbb3}.palm-icon{font-size:7rem;filter:drop-shadow(0 0 24px #54f7e3);animation:float 1.5s infinite}.count{font-size:min(25vw,15rem)!important}.result>b{font:950 3rem ui-monospace;color:#dbff70}.result>span{font:700 .55rem ui-monospace;margin-bottom:1rem}.result button{min-width:13rem;margin-top:.5rem;background:#54f7e3;color:#06110f;font-weight:900}.result .quiet{background:transparent;color:#fff}@keyframes spin{to{transform:rotate(360deg)}}@keyframes float{50%{transform:translateY(-12px)}}
@media(max-height:850px) and (min-width:761px){.lobby-copy{padding:2rem 5vw}.lobby-copy h1{font-size:clamp(3.5rem,7vw,6.4rem);margin:.6rem 0 1rem}.lead{line-height:1.4;font-size:.85rem}.rules{margin:.8rem 0 .5rem}.rules span{padding:.65rem}.difficulty button{padding:.55rem}.start{padding:.75rem}}
@media(max-width:760px){.trace-lobby{display:block}.lobby-copy{box-sizing:border-box;height:100%;padding:3.7rem 1rem .8rem}.trace-lobby aside{display:none}.lobby-copy h1{font-size:3.9rem;margin:.45rem 0 .8rem}.lead{font-size:.75rem;line-height:1.4}.rules{margin:.7rem 0 .45rem}.rules span{font-size:.52rem;padding:.5rem}.model{padding:.5rem}.hud{top:.65rem;width:calc(100% - 5rem)}.hud>div{min-width:0;flex:1;padding:.45rem .35rem}.hud b{font-size:.85rem}.hud>div:nth-child(3){display:none}.message{bottom:3.6rem;white-space:nowrap}.status small{display:none}}
</style>
