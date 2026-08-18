<script setup lang="ts">
import type { GestureTarget } from "./types"

defineProps<{ target: GestureTarget }>()

const labels = {
  "open-palm": { emoji: "✋", name: "Open Palm", cue: "张开手掌" },
  fist: { emoji: "✊", name: "Fist", cue: "握紧拳头" },
  victory: { emoji: "✌️", name: "Victory", cue: "做出 V 手势" },
  point: { emoji: "☝️", name: "Point", cue: "伸出食指" },
} as const
</script>

<template>
  <div class="gesture-target" :class="`gesture-target--${target.hand}`">
    <span class="gesture-target__side">{{ target.hand === "left" ? "LEFT HAND · 左手" : "RIGHT HAND · 右手" }}</span>
    <b aria-hidden="true">{{ labels[target.gesture].emoji }}</b>
    <strong>{{ labels[target.gesture].name }}</strong>
    <small>{{ labels[target.gesture].cue }}</small>
  </div>
</template>

<style scoped>
.gesture-target { align-items:center; background:#c8ff4610; border:1px solid #c8ff4650; display:flex; flex-direction:column; padding:.8rem 1.1rem; text-align:center; }
.gesture-target__side { color:#c8ff46; font:700 .5rem var(--font-mono); letter-spacing:.12em; }
.gesture-target b { font-size:3.2rem; line-height:1.15; margin:.25rem 0; }
.gesture-target--right b { transform:scaleX(-1); }
.gesture-target strong { font-size:.85rem; }
.gesture-target small { color:#999; font-size:.62rem; margin-top:.15rem; }
</style>
