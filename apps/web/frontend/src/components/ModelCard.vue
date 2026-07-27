<script setup lang="ts">
import type { ModelSummary } from "@/api/types"
import StatusPill from "./StatusPill.vue"

defineProps<{ model: ModelSummary }>()
</script>

<template>
  <article class="catalog-card model-card">
    <div class="card-topline">
      <span class="eyebrow">{{ model.family }}</span>
      <StatusPill
        :label="model.ready_count ? `${model.ready_count} ready` : 'not loaded'"
        :tone="model.ready_count ? 'ready' : 'idle'"
      />
    </div>
    <div>
      <h3>{{ model.name }}</h3>
      <p>{{ model.description }}</p>
    </div>
    <div class="runtime-list">
      <span
        v-for="runtime in model.runtimes"
        :key="runtime.name"
        class="runtime-chip"
        :class="{ 'runtime-chip--off': !runtime.available }"
        :title="runtime.unavailable_reason ?? runtime.name"
      >
        {{ runtime.name }}
      </span>
    </div>
    <div class="card-footer">
      <span>{{ model.capabilities.join(" · ") }}</span>
      <RouterLink to="/models">Inspect →</RouterLink>
    </div>
  </article>
</template>
