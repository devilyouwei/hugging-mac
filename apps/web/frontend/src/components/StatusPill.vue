<script setup lang="ts">
defineProps<{
  label: string
  tone?: "ready" | "idle" | "warning"
}>()
</script>

<template>
  <span class="status-pill" :class="`status-pill--${tone ?? 'idle'}`">
    <span class="status-pill__dot" aria-hidden="true"></span>
    {{ label }}
  </span>
</template>

<style scoped>
.status-pill {
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 999px;
  display: inline-flex;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  gap: 0.42rem;
  letter-spacing: 0.06em;
  padding: 0.36rem 0.6rem;
  text-transform: uppercase;
}

.status-pill__dot {
  background: var(--muted);
  border-radius: 50%;
  height: 0.42rem;
  width: 0.42rem;
}

.status-pill--ready .status-pill__dot {
  background: var(--signal);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--signal) 20%, transparent);
}

.status-pill--warning .status-pill__dot {
  background: var(--orange);
}
</style>
