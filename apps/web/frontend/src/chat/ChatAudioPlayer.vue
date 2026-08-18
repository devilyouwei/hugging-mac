<script setup lang="ts">
import { computed, ref, watch } from "vue"

const props = defineProps<{
  src: string | null
  status: "generating" | "ready"
  durationSeconds: number
}>()
const emit = defineEmits<{ play: [] }>()

const audio = ref<HTMLAudioElement | null>(null)
const playing = ref(false)
const currentTime = ref(0)
const metadataDuration = ref(0)
const duration = computed(() => metadataDuration.value || props.durationSeconds || 0)
const progress = computed(() => duration.value ? Math.min(1, currentTime.value / duration.value) : 0)

watch(() => props.src, () => {
  playing.value = false
  currentTime.value = 0
  metadataDuration.value = 0
  audio.value?.load()
})

async function togglePlayback() {
  const element = audio.value
  if (!element || !props.src) return
  if (!element.paused) {
    element.pause()
    return
  }
  emit("play")
  try { await element.play() } catch { playing.value = false }
}

function seek(event: Event) {
  const element = audio.value
  if (!element || !duration.value) return
  element.currentTime = Number((event.target as HTMLInputElement).value) * duration.value
}

</script>

<template>
  <div class="voice-player" :class="{ generating: status === 'generating' }">
    <audio
      ref="audio"
      class="voice-player-audio"
      :src="src ?? undefined"
      preload="metadata"
      @loadedmetadata="metadataDuration = audio?.duration ?? 0"
      @timeupdate="currentTime = audio?.currentTime ?? 0"
      @play="playing = true"
      @pause="playing = false"
      @ended="playing = false; currentTime = 0"
    ></audio>
    <button type="button" :disabled="!src" :aria-label="playing ? '暂停语音' : '播放语音'" @click="togglePlayback">
      <span v-if="playing" class="pause-icon"><i></i><i></i></span>
      <span v-else class="play-icon"></span>
    </button>
    <div class="voice-progress">
      <span><i :style="{ width: `${progress * 100}%` }"></i></span>
      <input aria-label="音频进度" type="range" min="0" max="1" step="0.001" :value="progress" @input="seek" />
    </div>
  </div>
</template>

<style scoped>
.voice-player { align-items:center; background:#0f120d; border:1px solid #353a30; border-radius:.7rem; display:grid; gap:.5rem; grid-template-columns:1.55rem minmax(8rem,1fr); margin-top:.65rem; max-width:24rem; padding:.24rem .55rem .24rem .28rem; }
.voice-player-audio { display:none; }
button { align-items:center; background:var(--signal); border:0; border-radius:50%; color:#10120e; cursor:pointer; display:flex; height:1.55rem; justify-content:center; width:1.55rem; }
button:disabled { background:#343831; cursor:wait; }
.play-icon { border-bottom:.25rem solid transparent; border-left:.4rem solid currentColor; border-top:.25rem solid transparent; margin-left:.1rem; }
.pause-icon { display:flex; gap:.12rem; }.pause-icon i { background:currentColor; border-radius:.04rem; height:.55rem; width:.14rem; }
.voice-progress { height:.9rem; position:relative; }
.voice-progress>span { background:#343a31; border-radius:1rem; height:2px; left:0; overflow:hidden; position:absolute; right:0; top:calc(50% - 1px); }
.voice-progress>span i { background:var(--signal); border-radius:inherit; display:block; height:100%; transition:width .1s linear; }
.voice-progress input { cursor:pointer; height:100%; inset:0; margin:0; opacity:0; position:absolute; width:100%; }
.generating .voice-progress>span i { animation:voice-loading 1s ease-in-out infinite; background:linear-gradient(90deg,transparent,var(--signal),transparent); width:38%!important; }
@keyframes voice-loading { from { transform:translateX(-110%); } to { transform:translateX(280%); } }
</style>
