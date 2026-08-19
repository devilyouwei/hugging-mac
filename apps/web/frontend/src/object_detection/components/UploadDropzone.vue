<script setup lang="ts">
import { ref } from "vue"

const emit = defineEmits<{
  selected: [file: File]
}>()

defineProps<{
  filename?: string
}>()

const dragging = ref(false)
const input = ref<HTMLInputElement | null>(null)

function accept(file?: File) {
  if (!file || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) return
  emit("selected", file)
}

function handleDrop(event: DragEvent) {
  dragging.value = false
  accept(event.dataTransfer?.files[0])
}

function handleInput(event: Event) {
  accept((event.target as HTMLInputElement).files?.[0])
}
</script>

<template>
  <button
    type="button"
    class="dropzone"
    :class="{ 'dropzone--active': dragging }"
    @click="input?.click()"
    @dragenter.prevent="dragging = true"
    @dragover.prevent="dragging = true"
    @dragleave.prevent="dragging = false"
    @drop.prevent="handleDrop"
  >
    <input
      ref="input"
      type="file"
      accept="image/jpeg,image/png,image/webp"
      hidden
      @change="handleInput"
    />
    <span class="dropzone__mark" aria-hidden="true">+</span>
    <span class="dropzone__title">
      {{ filename ? "Replace image" : "Drop an image here" }}
    </span>
    <span class="dropzone__meta">
      {{ filename || "JPEG · PNG · WEBP · up to 50 MB" }}
    </span>
  </button>
</template>

<style scoped>
.dropzone {
  align-items: center;
  background: color-mix(in srgb, var(--paper) 78%, transparent);
  border: 1px dashed var(--line);
  color: var(--ink);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 13rem;
  padding: 2rem;
  transition:
    background 160ms ease,
    transform 160ms ease;
  width: 100%;
}

.dropzone:hover,
.dropzone--active {
  background: var(--signal-soft);
  border-color: color-mix(in srgb, var(--accent) 42%, var(--line));
  transform: translateY(-2px);
}

.dropzone__mark {
  align-items: center;
  border: 1px solid currentColor;
  border-radius: 50%;
  display: flex;
  font-size: 1.8rem;
  height: 3rem;
  justify-content: center;
  margin-bottom: 1rem;
  width: 3rem;
}

.dropzone__title {
  font-family: var(--font-display);
  font-size: 1.25rem;
  font-weight: 700;
}

.dropzone__meta {
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  margin-top: 0.45rem;
}
</style>
