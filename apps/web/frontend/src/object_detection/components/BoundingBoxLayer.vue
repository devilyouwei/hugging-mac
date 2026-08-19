<script setup lang="ts">
import type { DetectionResult } from "../types"

defineProps<{
  result: DetectionResult | null
}>()

const colors = ["#30d158", "#ff453a", "#64d2ff", "#ffd60a", "#bf5af2"]

function percentage(value: number, size: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(size) || size <= 0) return "0%"
  return `${Math.min(100, Math.max(0, (value / size) * 100))}%`
}
</script>

<template>
  <div v-if="result" class="bbox-layer" aria-live="polite">
    <div
      v-for="(detection, index) in result.detections"
      :key="`${detection.label}-${index}`"
      class="detection-box"
      :style="{
        left: percentage(detection.box.x1, result.image_size.width),
        top: percentage(detection.box.y1, result.image_size.height),
        width: percentage(
          Math.max(0, Math.min(result.image_size.width, detection.box.x2) - Math.max(0, detection.box.x1)),
          result.image_size.width,
        ),
        height: percentage(
          Math.max(0, Math.min(result.image_size.height, detection.box.y2) - Math.max(0, detection.box.y1)),
          result.image_size.height,
        ),
        borderColor: colors[index % colors.length],
      }"
    >
      <span :style="{ backgroundColor: colors[index % colors.length] }">
        {{ detection.label }} {{ Math.round(detection.confidence * 100) }}%
      </span>
    </div>
  </div>
</template>

<style scoped>
.bbox-layer {
  inset: 0;
  pointer-events: none;
  position: absolute;
}

.detection-box {
  border: 2px solid;
  position: absolute;
}

.detection-box span {
  color: #111;
  display: inline-block;
  font-family: var(--font-mono);
  font-size: clamp(0.55rem, 1.2vw, 0.72rem);
  font-weight: 700;
  left: -2px;
  line-height: 1;
  padding: 0.3rem 0.42rem;
  position: absolute;
  top: 0;
  transform: translateY(-100%);
  white-space: nowrap;
}
</style>
