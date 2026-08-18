<script setup lang="ts">
import { computed } from "vue"

import type { AppSummary } from "@/api/types"
import StatusPill from "./StatusPill.vue"

const props = defineProps<{ app: AppSummary; index: number }>()

const icon = computed(() => {
  const icons: Record<string, string> = {
    "object-detection": "🔎",
    "pose-estimation": "🕺",
    "instance-segmentation": "🎨",
    "yolo-pose-follow": "🕺",
    "yolo-fruit-slice": "🍉",
    "live-transcription": "🎙️",
    "text-to-speech": "🔊",
    chat: "💬",
  }
  return icons[props.app.manifest.app_id] ?? "🧩"
})
</script>

<template>
  <RouterLink
    class="app-card"
    :class="{ 'app-card--disabled': app.status === 'unavailable' }"
    :to="app.status === 'unavailable' ? '#' : app.manifest.frontend_route"
  >
    <div class="app-card__marker" aria-hidden="true">
      <span class="app-icon">{{ icon }}</span>
      <span class="app-number">{{ String(index + 1).padStart(2, "0") }}</span>
    </div>
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
