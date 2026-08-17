<script setup lang="ts">
import type { PoseResult } from "../types"

defineProps<{ result: PoseResult }>()

function visibleKeypoints(pose: PoseResult["poses"][number]): number {
  return pose.keypoints.filter(
    (point) => point.confidence == null || point.confidence >= 0.25,
  ).length
}
</script>

<template>
  <section class="results-panel">
    <div class="result-summary" role="table" aria-label="各模型检测数量与耗时">
      <div class="result-summary__row result-summary__head" role="row">
        <span role="columnheader">MODEL</span>
        <span role="columnheader">DETECTED</span>
        <span role="columnheader">TIME</span>
      </div>
      <div class="result-summary__row" role="row">
        <strong role="cell">YOLO POSE</strong>
        <span role="cell">{{ result.poses.length }} poses</span>
        <span role="cell">{{ result.parallel_timings.pose_ms?.toFixed(1) ?? "—" }} ms</span>
      </div>
      <div class="result-summary__row" role="row">
        <strong role="cell">RETINAFACE</strong>
        <span role="cell">{{ result.faces.length }} faces</span>
        <span role="cell">{{ result.parallel_timings.face_ms?.toFixed(1) ?? "—" }} ms</span>
      </div>
      <div class="result-summary__row" role="row">
        <strong role="cell">MEDIAPIPE HAND</strong>
        <span role="cell">{{ result.hands.length }} hands</span>
        <span role="cell">{{ result.parallel_timings.hand_ms?.toFixed(1) ?? "—" }} ms</span>
      </div>
      <div class="result-summary__row result-summary__totals" role="row">
        <strong role="cell">TOTAL</strong>
        <span role="cell">SUM · {{ result.parallel_timings.sum_ms.toFixed(1) }} ms</span>
        <span role="cell">ROUND · {{ result.parallel_timings.round_ms.toFixed(1) }} ms</span>
      </div>
    </div>
    <div class="result-list">
      <div v-for="(pose, index) in result.poses" :key="index" class="result-row">
        <span class="result-row__index">{{ String(index + 1).padStart(2, "0") }}</span>
        <strong>{{ pose.label }}</strong>
        <span>{{ visibleKeypoints(pose) }}/{{ pose.keypoints.length }} keypoints</span>
        <span>{{ (pose.confidence * 100).toFixed(1) }}%</span>
      </div>
      <div v-for="(face, index) in result.faces" :key="`face-${index}`" class="result-row">
        <span class="result-row__index">F{{ String(index + 1).padStart(2, "0") }}</span>
        <strong>face</strong>
        <span>5 landmarks</span>
        <span>{{ (face.confidence * 100).toFixed(1) }}%</span>
      </div>
      <div v-for="(hand, index) in result.hands" :key="`hand-${index}`" class="result-row">
        <span class="result-row__index">H{{ String(index + 1).padStart(2, "0") }}</span>
        <strong>{{ hand.handedness ?? hand.label }}</strong>
        <span>{{ hand.landmarks.length ? `${hand.landmarks.length} landmarks` : "bounding box" }}</span>
        <span>{{ (hand.confidence * 100).toFixed(1) }}%</span>
      </div>
      <p v-if="!result.poses.length && !result.faces.length && !result.hands.length" class="no-results">
        当前阈值下没有识别到人体姿态、人脸或手部，可以分别降低对应 confidence 后重试。
      </p>
    </div>
  </section>
</template>

<style scoped>
.results-panel {
  border-top: 1px solid var(--line);
  margin-top: 1.5rem;
  padding-top: 1.5rem;
}

.result-summary {
  background: var(--ink);
  color: var(--paper);
}

.result-summary__row {
  align-items: center;
  border-bottom: 1px solid #353530;
  display: grid;
  gap: 1rem;
  grid-template-columns: minmax(9rem, 1.4fr) minmax(7rem, 1fr) minmax(6rem, 1fr);
  padding: 0.75rem 1rem;
}

.result-summary__row:last-child {
  border-bottom: 0;
}

.result-summary__head {
  padding-bottom: 0.55rem;
  padding-top: 0.55rem;
}

.result-summary__totals {
  background: #24241f;
}

.result-summary span {
  color: #aaa9a2;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.08em;
}

.result-summary strong {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  letter-spacing: 0.04em;
}

.result-summary__totals span {
  color: var(--paper);
}

@media (max-width: 620px) {
  .result-summary__row {
    gap: 0.6rem;
    grid-template-columns: minmax(7rem, 1.3fr) minmax(5rem, 1fr) minmax(5rem, 1fr);
    padding-inline: 0.7rem;
  }
}

.result-list {
  margin-top: 0.75rem;
}

.result-row {
  align-items: center;
  border-bottom: 1px solid var(--line);
  display: grid;
  font-size: 0.85rem;
  gap: 0.8rem;
  grid-template-columns: 2rem minmax(5rem, 1fr) minmax(8rem, 2fr) 3.5rem;
  padding: 0.78rem 0;
}

.result-row__index,
.result-row > span {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.72rem;
}

.no-results {
  color: var(--muted);
  padding: 1.5rem 0;
}
</style>
