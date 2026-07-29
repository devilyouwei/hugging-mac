<script setup lang="ts">
import type { PoseResult } from "../types"

defineProps<{
  result: PoseResult | null
}>()

const skeleton: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
]
const colors = ["#c8ff46", "#ff7148", "#7de2ff", "#f6bf4f", "#cb9cff"]

function keypointVisible(
  pose: PoseResult["poses"][number],
  index: number,
): boolean {
  const point = pose.keypoints[index]
  return Boolean(point && (point.confidence == null || point.confidence >= 0.25))
}

function boneVisible(
  pose: PoseResult["poses"][number],
  start: number,
  end: number,
): boolean {
  return keypointVisible(pose, start) && keypointVisible(pose, end)
}
</script>

<template>
  <svg
    v-if="result"
    class="pose-layer"
    :viewBox="`0 0 ${result.image_size.width} ${result.image_size.height}`"
    preserveAspectRatio="none"
    aria-label="Pose skeleton overlay"
  >
    <g v-for="(pose, poseIndex) in result.poses" :key="poseIndex">
      <line
        v-for="[start, end] in skeleton"
        v-show="boneVisible(pose, start, end)"
        :key="`${start}-${end}`"
        :x1="pose.keypoints[start]?.x"
        :y1="pose.keypoints[start]?.y"
        :x2="pose.keypoints[end]?.x"
        :y2="pose.keypoints[end]?.y"
        :stroke="colors[poseIndex % colors.length]"
        class="pose-bone"
      />
      <circle
        v-for="(point, pointIndex) in pose.keypoints"
        v-show="keypointVisible(pose, pointIndex)"
        :key="pointIndex"
        :cx="point.x"
        :cy="point.y"
        :r="Math.max(3, result.image_size.width / 220)"
        :fill="colors[poseIndex % colors.length]"
        class="pose-point"
      />
      <rect
        :x="pose.box.x1"
        :y="pose.box.y1"
        :width="pose.box.x2 - pose.box.x1"
        :height="pose.box.y2 - pose.box.y1"
        :stroke="colors[poseIndex % colors.length]"
        class="pose-box"
      />
    </g>
  </svg>
</template>

<style scoped>
.pose-layer {
  inset: 0;
  overflow: visible;
  pointer-events: none;
  position: absolute;
}

.pose-bone {
  filter: drop-shadow(0 1px 1px rgb(0 0 0 / 0.7));
  stroke-linecap: round;
  stroke-width: 4;
  vector-effect: non-scaling-stroke;
}

.pose-point {
  stroke: #111;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}

.pose-box {
  fill: none;
  opacity: 0.55;
  stroke-dasharray: 7 5;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
</style>
