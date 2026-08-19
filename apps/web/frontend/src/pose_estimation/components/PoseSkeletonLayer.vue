<script setup lang="ts">
import type { PoseResult } from "../types"

const props = withDefaults(defineProps<{
  result: PoseResult | null
  fit?: "fill" | "cover"
}>(), {
  fit: "fill",
})

const skeleton: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
]
const colors = ["#30d158", "#ff453a", "#64d2ff", "#ffd60a", "#bf5af2"]
const facePointNames = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"] as const
const handSkeleton: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]

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
    :preserveAspectRatio="props.fit === 'cover' ? 'xMidYMid slice' : 'none'"
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
    <g v-for="(face, faceIndex) in result.faces" :key="`face-${faceIndex}`">
      <rect
        :x="face.box.x1"
        :y="face.box.y1"
        :width="face.box.x2 - face.box.x1"
        :height="face.box.y2 - face.box.y1"
        class="face-box"
      />
      <circle
        v-for="name in facePointNames"
        :key="name"
        :cx="face.landmarks[name].x"
        :cy="face.landmarks[name].y"
        :r="Math.max(3, result.image_size.width / 260)"
        class="face-point"
      />
      <text :x="face.box.x1" :y="Math.max(14, face.box.y1 - 6)" class="face-label">
        face {{ Math.round(face.confidence * 100) }}%
      </text>
    </g>
    <g v-for="(hand, handIndex) in result.hands" :key="`hand-${handIndex}`">
      <line
        v-for="[start, end] in handSkeleton"
        v-show="hand.landmarks.length === 21"
        :key="`hand-${handIndex}-${start}-${end}`"
        :x1="hand.landmarks[start]?.x"
        :y1="hand.landmarks[start]?.y"
        :x2="hand.landmarks[end]?.x"
        :y2="hand.landmarks[end]?.y"
        class="hand-bone"
      />
      <circle
        v-for="(point, pointIndex) in hand.landmarks"
        :key="`hand-${handIndex}-point-${pointIndex}`"
        :cx="point.x"
        :cy="point.y"
        :r="Math.max(2.5, result.image_size.width / 280)"
        class="hand-point"
      />
      <rect
        :x="hand.box.x1"
        :y="hand.box.y1"
        :width="hand.box.x2 - hand.box.x1"
        :height="hand.box.y2 - hand.box.y1"
        class="hand-box"
      />
      <text :x="hand.box.x1" :y="Math.max(14, hand.box.y1 - 6)" class="hand-label">
        {{ hand.handedness ?? "hand" }} {{ Math.round(hand.confidence * 100) }}%
      </text>
    </g>
  </svg>
</template>

<style scoped>
.pose-layer {
  height: 100%;
  inset: 0;
  overflow: visible;
  pointer-events: none;
  position: absolute;
  width: 100%;
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

.face-box {
  fill: none;
  stroke: #ff4fd8;
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}

.face-point {
  fill: #4dfff3;
  stroke: #111;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}

.face-label {
  fill: #ff4fd8;
  font: 600 13px ui-monospace, monospace;
  paint-order: stroke;
  stroke: #111;
  stroke-width: 3px;
}

.hand-box {
  fill: none;
  stroke: #ffb629;
  stroke-width: 2.5;
  vector-effect: non-scaling-stroke;
}

.hand-bone {
  filter: drop-shadow(0 1px 1px rgb(0 0 0 / 0.7));
  stroke: #ffb629;
  stroke-linecap: round;
  stroke-width: 3;
  vector-effect: non-scaling-stroke;
}

.hand-point {
  fill: #fff27a;
  stroke: #111;
  stroke-width: 1.25;
  vector-effect: non-scaling-stroke;
}

.hand-label {
  fill: #ffb629;
  font: 600 13px ui-monospace, monospace;
  paint-order: stroke;
  stroke: #111;
  stroke-width: 3px;
}
</style>
