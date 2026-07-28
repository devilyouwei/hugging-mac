<script setup lang="ts">
import { computed } from "vue"

import type { ResourceStatus } from "../types"

const props = defineProps<{
  status: ResourceStatus | null
  busy: "" | "download" | "convert"
}>()

defineEmits<{
  download: []
  convert: []
  refresh: []
}>()

const source = computed(() =>
  props.status?.artifacts.find((artifact) => artifact.artifact_id === "source"),
)
const coreml = computed(() =>
  props.status?.artifacts.find((artifact) => artifact.artifact_id === "coreml"),
)

function formatSize(size: number | null | undefined): string {
  if (size == null) return "not prepared"
  return `${(size / 1024 ** 2).toFixed(1)} MB`
}
</script>

<template>
  <section class="resource-setup">
    <div class="resource-setup__heading">
      <span>MODEL ASSETS</span>
      <button type="button" :disabled="Boolean(busy)" @click="$emit('refresh')">
        Refresh
      </button>
    </div>

    <div class="asset-row">
      <span class="asset-state" :class="{ 'asset-state--ready': source?.available }"></span>
      <div>
        <strong>YOLOv8{{ status?.variant ?? "" }} source</strong>
        <small>PyTorch · {{ formatSize(source?.size_bytes) }}</small>
      </div>
      <button
        v-if="!source?.available"
        type="button"
        :disabled="Boolean(busy)"
        @click="$emit('download')"
      >
        {{ busy === "download" ? "Downloading…" : "Download" }}
      </button>
      <span v-else class="asset-ready">READY</span>
    </div>

    <div class="asset-row">
      <span class="asset-state" :class="{ 'asset-state--ready': coreml?.available }"></span>
      <div>
        <strong>Core ML package</strong>
        <small>Apple runtime · {{ formatSize(coreml?.size_bytes) }}</small>
      </div>
      <button
        v-if="!coreml?.available"
        type="button"
        :disabled="Boolean(busy) || !source?.available"
        @click="$emit('convert')"
      >
        {{ busy === "convert" ? "Converting…" : "Convert" }}
      </button>
      <span v-else class="asset-ready">READY</span>
    </div>
  </section>
</template>

<style scoped>
.resource-setup {
  border: 1px solid var(--ink);
  margin-bottom: 1.3rem;
}

.resource-setup__heading {
  align-items: center;
  background: var(--ink);
  color: var(--paper);
  display: flex;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  justify-content: space-between;
  letter-spacing: 0.08em;
  padding: 0.65rem 0.75rem;
}

button {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  padding: 0;
  text-decoration: underline;
  text-transform: uppercase;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.asset-row {
  align-items: center;
  display: grid;
  gap: 0.7rem;
  grid-template-columns: auto 1fr auto;
  padding: 0.85rem 0.75rem;
}

.asset-row + .asset-row {
  border-top: 1px solid var(--line);
}

.asset-row > div {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.asset-row strong {
  font-size: 0.78rem;
}

.asset-row small {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.58rem;
}

.asset-state {
  background: var(--orange);
  border-radius: 50%;
  height: 0.55rem;
  width: 0.55rem;
}

.asset-state--ready {
  background: var(--signal);
}

.asset-ready {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 0.08em;
}
</style>
