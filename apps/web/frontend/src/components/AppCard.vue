<script setup lang="ts">
import { computed } from "vue"

import type { AppSummary } from "@/api/types"
import StatusPill from "./StatusPill.vue"

const props = defineProps<{ app: AppSummary }>()

const icon = computed(() => {
  const icons: Record<string, string> = {
    "object-detection": "🔎",
    "pose-estimation": "🕺",
    "instance-segmentation": "🎨",
    "yolo-pose-follow": "🕺",
    "yolo-fruit-slice": "🍉",
    "palm-thunder": "🛩️",
    "palm-trace": "🧽",
    "live-transcription": "🎙️",
    "text-to-speech": "🔊",
    "document-parser": "📄",
    chat: "💬",
  }
  return icons[props.app.manifest.app_id] ?? "🧩"
})

const category = computed(() =>
  props.app.manifest.tags.slice(0, 2).join(" · ") || "Local Intelligence",
)

const capabilities = computed(() => [
  ...new Set(
    props.app.manifest.required_models.flatMap(
      (requirement) => requirement.capabilities,
    ),
  ),
])
</script>

<template>
  <RouterLink
    class="app-card"
    :class="[
      `app-card--${app.manifest.app_id}`,
      { 'app-card--disabled': app.status === 'unavailable' },
    ]"
    :to="app.status === 'unavailable' ? '#' : app.manifest.frontend_route"
  >
    <div class="app-card__marker" aria-hidden="true">
      <span class="app-icon">{{ icon }}</span>
    </div>
    <div class="app-card__content">
      <div class="card-topline">
        <span class="eyebrow">{{ category }}</span>
        <StatusPill
          :label="app.status"
          :tone="app.status === 'available' ? 'ready' : 'warning'"
        />
      </div>
      <h3>{{ app.manifest.name }}</h3>
      <p>{{ app.manifest.description }}</p>
      <div
        v-if="capabilities.length"
        class="app-capabilities"
        aria-label="Capabilities"
      >
        <span
          v-for="capability in capabilities"
          :key="capability"
          class="app-capability-badge"
        >
          {{ capability }}
        </span>
      </div>
    </div>
    <div class="app-card__trailing">
      <span class="app-card__action">
        {{ app.status === "unavailable" ? "Unavailable" : "Open" }}
      </span>
      <small>Runs locally</small>
    </div>
  </RouterLink>
</template>
