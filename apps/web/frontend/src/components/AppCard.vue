<script setup lang="ts">
import type { AppSummary } from "@/api/types"
import StatusPill from "./StatusPill.vue"

defineProps<{ app: AppSummary; index: number }>()
</script>

<template>
  <RouterLink
    class="app-card"
    :class="{ 'app-card--disabled': app.status === 'unavailable' }"
    :to="app.status === 'unavailable' ? '#' : app.manifest.frontend_route"
  >
    <span class="app-number">{{ String(index + 1).padStart(2, "0") }}</span>
    <div class="app-card__content">
      <div class="card-topline">
        <span class="eyebrow">{{ app.manifest.tags.slice(0, 2).join(" / ") }}</span>
        <StatusPill
          :label="app.status"
          :tone="app.status === 'available' ? 'ready' : 'warning'"
        />
      </div>
      <h3>{{ app.manifest.name }}</h3>
      <p>{{ app.manifest.description }}</p>
      <span class="app-card__action">
        {{ app.status === "unavailable" ? app.unavailable_reason : "Open workspace →" }}
      </span>
    </div>
  </RouterLink>
</template>
