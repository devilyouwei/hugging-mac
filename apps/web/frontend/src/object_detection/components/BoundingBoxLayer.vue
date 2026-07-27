<script setup lang="ts">
import type { DetectionResult } from "../types"

defineProps<{
  result: DetectionResult | null
}>()

const colors = ["#c8ff46", "#ff7148", "#7de2ff", "#f6bf4f", "#cb9cff"]
</script>

<template>
  <div v-if="result" class="bbox-layer" aria-live="polite">
    <div
      v-for="(detection, index) in result.detections"
      :key="`${detection.label}-${index}`"
      class="detection-box"
      :style="{
        left: `${(detection.box.x1 / result.image_size.width) * 100}%`,
        top: `${(detection.box.y1 / result.image_size.height) * 100}%`,
        width: `${((detection.box.x2 - detection.box.x1) / result.image_size.width) * 100}%`,
        height: `${((detection.box.y2 - detection.box.y1) / result.image_size.height) * 100}%`,
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
