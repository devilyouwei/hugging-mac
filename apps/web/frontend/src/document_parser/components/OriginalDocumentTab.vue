<script setup lang="ts">
import { ChevronLeft, ChevronRight, Maximize2, Minus, Plus, X } from "@lucide/vue"
import { onBeforeUnmount, onMounted, ref } from "vue"
import { pageImageUrl } from "../api"
import type { DocumentRecord } from "../types"

const props = defineProps<{ document: DocumentRecord }>()
const page = defineModel<number>("page", { required: true })
const modalOpen = ref(false)
const zoom = ref(1)
function setPage(value: number) { page.value = Math.max(0, Math.min(props.document.page_count - 1, value)) }
function zoomBy(value: number) { zoom.value = Math.min(3, Math.max(.4, Math.round((zoom.value + value) * 10) / 10)) }
function open() { zoom.value = 1; modalOpen.value = true }
function wheel(event: WheelEvent) { if (event.ctrlKey || event.metaKey) { event.preventDefault(); zoomBy(event.deltaY < 0 ? .1 : -.1) } }
function keydown(event: KeyboardEvent) { if (!modalOpen.value) return; if (event.key === "Escape") modalOpen.value = false; else if (event.key === "+" || event.key === "=") zoomBy(.1); else if (event.key === "-") zoomBy(-.1); else if (event.key === "0") zoom.value = 1 }
onMounted(() => window.addEventListener("keydown", keydown))
onBeforeUnmount(() => window.removeEventListener("keydown", keydown))
</script>

<template>
  <section class="original-tab">
    <header><div><p>ORIGINAL</p><h2>Document Preview</h2></div><span>Page {{ page + 1 }} of {{ document.page_count }}</span></header>
    <button class="page-preview" title="Open full-size preview" @click="open"><img :src="pageImageUrl(document.id, page)" :alt="`Page ${page + 1}`"><span><Maximize2 :size="15" />Open preview</span></button>
    <nav><button :disabled="page === 0" @click="setPage(page - 1)"><ChevronLeft :size="16" />Previous</button><label>Page <input :value="page + 1" type="number" min="1" :max="document.page_count" @change="setPage(Number(($event.target as HTMLInputElement).value) - 1)"> of {{ document.page_count }}</label><button :disabled="page >= document.page_count - 1" @click="setPage(page + 1)">Next<ChevronRight :size="16" /></button></nav>
    <Teleport to="body"><div v-if="modalOpen" class="preview-modal" role="dialog" aria-modal="true" @click.self="modalOpen = false"><header><div><strong>{{ document.name }}</strong><small>Page {{ page + 1 }} of {{ document.page_count }}</small></div><div><button @click="zoomBy(-.1)"><Minus :size="17" /></button><button @click="zoom = 1">{{ Math.round(zoom * 100) }}%</button><button @click="zoomBy(.1)"><Plus :size="17" /></button><button @click="modalOpen = false"><X :size="18" /></button></div></header><main><button class="arrow left" :disabled="page === 0" @click="setPage(page - 1)"><ChevronLeft /></button><div class="scroll" @wheel="wheel"><img :src="pageImageUrl(document.id, page)" :style="{ width: `${zoom * 100}%` }" :alt="`Full preview of page ${page + 1}`"></div><button class="arrow right" :disabled="page >= document.page_count - 1" @click="setPage(page + 1)"><ChevronRight /></button></main></div></Teleport>
  </section>
</template>

<style scoped>
.original-tab{box-sizing:border-box;display:flex;flex-direction:column;height:100%;max-width:100%;min-height:650px;min-width:0;overflow:hidden;padding:1rem;width:100%}.original-tab>header{align-items:center;display:flex;justify-content:space-between;min-width:0}.original-tab h2{font-size:1rem;margin:.15rem 0}.original-tab header p{color:var(--accent);font-size:.58rem;font-weight:800;letter-spacing:.14em;margin:0}.original-tab header span{color:var(--muted);font-size:.68rem}.page-preview{background:color-mix(in srgb,var(--surface-solid) 60%,transparent);border:1px solid var(--line);border-radius:18px;box-sizing:border-box;cursor:zoom-in;flex:1;margin:.8rem 0;max-width:100%;min-height:480px;min-width:0;overflow:hidden;padding:1rem;position:relative;width:100%}.page-preview img{height:100%;max-width:100%;object-fit:contain;width:100%}.page-preview span{align-items:center;background:#10131ccc;border-radius:999px;bottom:1.2rem;color:#fff;display:flex;font-size:.65rem;gap:.35rem;left:50%;opacity:0;padding:.45rem .7rem;position:absolute;transform:translateX(-50%);transition:opacity .18s}.page-preview:hover span{opacity:1}.original-tab nav{align-items:center;display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);min-width:0}.original-tab nav button{justify-self:start}.original-tab nav button:last-child{justify-self:end}.original-tab nav label{color:var(--muted);font-size:.66rem}.original-tab nav input{background:var(--surface);border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:.3rem;text-align:center;width:48px}.preview-modal{background:rgb(5 7 12 / 90%);backdrop-filter:blur(20px);display:grid;grid-template-rows:auto minmax(0,1fr);inset:0;padding:1rem;position:fixed;z-index:1100}.preview-modal>header{align-items:center;color:#fff;display:flex;justify-content:space-between;padding:0 .3rem .7rem}.preview-modal header small{color:#ffffff99;display:block;font-size:.65rem}.preview-modal header>div:last-child{display:flex;gap:.35rem}.preview-modal button{align-items:center;background:#ffffff14;border:1px solid #ffffff20;border-radius:10px;color:white;display:flex;justify-content:center;min-height:36px;padding:.4rem .65rem}.preview-modal main{min-height:0;min-width:0;position:relative}.scroll{background:#0d1017;border:1px solid #ffffff18;border-radius:18px;box-sizing:border-box;height:100%;max-width:100%;overflow:auto;padding:1rem}.scroll img{background:white;border-radius:9px;box-shadow:0 20px 70px #000a;display:block;height:auto;margin:auto;max-width:none;min-width:300px}.arrow{position:absolute!important;top:50%;transform:translateY(-50%);z-index:2}.arrow.left{left:.7rem}.arrow.right{right:.7rem}
</style>
