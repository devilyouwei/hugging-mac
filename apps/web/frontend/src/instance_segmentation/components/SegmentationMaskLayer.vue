<script setup lang="ts">
import type { SegmentationResult } from "../types"

defineProps<{
  result: SegmentationResult | null
}>()

const colors = ["#30d158", "#ff453a", "#64d2ff", "#ffd60a", "#bf5af2", "#63e6be"]

function polygonPoints(
  polygon: SegmentationResult["segments"][number]["polygons"][number],
): string {
  return polygon.map((point) => `${point.x},${point.y}`).join(" ")
}
</script>

<template>
  <div v-if="result" class="mask-layer" aria-label="Instance segmentation mask overlay">
    <svg
      :viewBox="`0 0 ${result.image_size.width} ${result.image_size.height}`"
      preserveAspectRatio="none"
    >
      <g v-for="(segment, segmentIndex) in result.segments" :key="segmentIndex">
        <polygon
          v-for="(polygon, polygonIndex) in segment.polygons"
          :key="polygonIndex"
          :points="polygonPoints(polygon)"
          :fill="colors[segmentIndex % colors.length]"
          :stroke="colors[segmentIndex % colors.length]"
        />
      </g>
    </svg>
    <span
      v-for="(segment, index) in result.segments"
      :key="`${segment.label}-${index}`"
      class="mask-label"
      :style="{
        left: `${(segment.box.x1 / result.image_size.width) * 100}%`,
        top: `${(segment.box.y1 / result.image_size.height) * 100}%`,
        backgroundColor: colors[index % colors.length],
      }"
    >
      {{ segment.label }} {{ Math.round(segment.confidence * 100) }}%
    </span>
  </div>
</template>

<style scoped>
.mask-layer,
svg {
  inset: 0;
  pointer-events: none;
  position: absolute;
}

svg {
  height: 100%;
  width: 100%;
}

polygon {
  fill-opacity: 0.42;
  stroke-opacity: 0.95;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.mask-label {
  color: #111;
  font-family: var(--font-mono);
  font-size: clamp(0.55rem, 1.2vw, 0.72rem);
  font-weight: 700;
  line-height: 1;
  padding: 0.3rem 0.42rem;
  position: absolute;
  transform: translateY(-100%);
  white-space: nowrap;
}
</style>
