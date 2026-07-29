<script setup lang="ts">
import { computed } from "vue"

import type { SegmentationResult } from "../types"

const props = defineProps<{ result: SegmentationResult }>()

const totalMs = computed(() =>
  [
    props.result.timings.preprocess_ms,
    props.result.timings.inference_ms,
    props.result.timings.postprocess_ms,
  ].reduce<number>((sum, value) => sum + (value ?? 0), 0),
)
</script>

<template>
  <section class="results-panel">
    <div class="result-summary">
      <div><span>MASKS</span><strong>{{ result.segments.length }}</strong></div>
      <div><span>RUNTIME</span><strong>{{ result.runtime }}</strong></div>
      <div><span>TOTAL</span><strong>{{ totalMs.toFixed(1) }} ms</strong></div>
    </div>
    <div class="result-list">
      <div v-for="(segment, index) in result.segments" :key="index" class="result-row">
        <span class="result-row__index">{{ String(index + 1).padStart(2, "0") }}</span>
        <strong>{{ segment.label }}</strong>
        <span>{{ segment.polygons.length }} mask polygon{{ segment.polygons.length === 1 ? "" : "s" }}</span>
        <span>{{ (segment.confidence * 100).toFixed(1) }}%</span>
      </div>
      <p v-if="!result.segments.length" class="no-results">
        当前阈值下没有分割出主体。可以降低 confidence 后重试。
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
  display: grid;
  gap: 1px;
  grid-template-columns: repeat(3, 1fr);
}

.result-summary > div {
  background: var(--ink);
  color: var(--paper);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
}

.result-summary span {
  color: #aaa9a2;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.08em;
}

.result-summary strong {
  font-family: var(--font-display);
  font-size: clamp(1rem, 2vw, 1.35rem);
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
