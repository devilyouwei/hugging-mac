<script setup lang="ts">
import { AlertTriangle, Trash2, X } from "@lucide/vue"

defineProps<{ name: string; busy: boolean }>()
defineEmits<{ cancel: []; confirm: [] }>()
</script>

<template>
  <Teleport to="body">
    <div class="confirm-backdrop" role="presentation" @click.self="$emit('cancel')">
      <section class="confirm-card" role="alertdialog" aria-modal="true" aria-labelledby="delete-title">
        <button class="close" aria-label="Close" @click="$emit('cancel')"><X :size="17" /></button>
        <span class="warning-icon"><AlertTriangle :size="24" /></span>
        <h2 id="delete-title">Delete document?</h2>
        <p><strong>{{ name }}</strong> and its locally cached results will be removed.</p>
        <div><button @click="$emit('cancel')">Cancel</button><button class="danger" :disabled="busy" @click="$emit('confirm')"><Trash2 :size="15" />{{ busy ? "Deleting…" : "Delete" }}</button></div>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.confirm-backdrop{align-items:center;background:rgb(5 8 15 / 62%);backdrop-filter:blur(18px);display:flex;inset:0;justify-content:center;padding:1rem;position:fixed;z-index:1200}.confirm-card{background:color-mix(in srgb,var(--surface-solid) 90%,transparent);border:1px solid color-mix(in srgb,var(--line) 75%,white);border-radius:24px;box-shadow:0 28px 90px #0006,inset 0 1px 0 #ffffff80;max-width:410px;padding:1.6rem;position:relative;text-align:center;width:100%}.warning-icon{align-items:center;background:color-mix(in srgb,var(--danger) 12%,transparent);border-radius:14px;color:var(--danger);display:inline-flex;height:50px;justify-content:center;width:50px}.confirm-card h2{font-size:1.15rem;margin:.9rem 0 .4rem}.confirm-card p{color:var(--muted);font-size:.75rem;line-height:1.6;margin:0 auto 1.25rem;max-width:310px;overflow-wrap:anywhere}.confirm-card>div{display:flex;gap:.55rem;justify-content:flex-end}.confirm-card button{align-items:center;background:var(--surface);border:1px solid var(--line);border-radius:11px;color:var(--ink);display:inline-flex;font-weight:700;gap:.35rem;justify-content:center;min-height:39px;padding:.5rem .85rem}.confirm-card .danger{background:var(--danger);border-color:var(--danger);color:white}.confirm-card .close{height:32px;min-height:32px;padding:0;position:absolute;right:.8rem;top:.8rem;width:32px}
</style>
