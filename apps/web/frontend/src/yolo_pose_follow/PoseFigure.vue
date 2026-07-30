<script setup lang="ts">
import type { PoseTemplate } from "./types"

defineProps<{
  pose: PoseTemplate
  compact?: boolean
}>()

const skeleton: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
  [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
  [13, 15], [12, 14], [14, 16],
]
</script>

<template>
  <svg
    class="target-figure"
    :class="{ 'target-figure--compact': compact }"
    viewBox="0 0 100 110"
    role="img"
    :aria-label="pose.cue"
  >
    <defs>
      <filter id="pose-glow">
        <feGaussianBlur stdDeviation="1.5" result="blur" />
        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>
    <g filter="url(#pose-glow)" transform="translate(0 2)">
      <line
        v-for="[start, end] in skeleton"
        :key="`${start}-${end}`"
        :x1="pose.points[start]!.x * 100"
        :y1="pose.points[start]!.y * 100"
        :x2="pose.points[end]!.x * 100"
        :y2="pose.points[end]!.y * 100"
      />
      <circle
        v-for="(point, index) in pose.points"
        :key="index"
        :cx="point.x * 100"
        :cy="point.y * 100"
        r="1.8"
      />
    </g>
  </svg>
</template>

<style scoped>
.target-figure {
  height: min(44vh, 420px);
  overflow: visible;
  width: 100%;
}

line {
  stroke: #c8ff46;
  stroke-linecap: round;
  stroke-width: 3.2;
}

circle {
  fill: #f8f5ed;
  stroke: #111;
  stroke-width: 0.7;
}

.target-figure--compact {
  height: 13rem;
}
</style>
