<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted } from "vue"

import type { ModelStructure, TensorStructure } from "@/api/types"

const props = defineProps<{
  modelName: string
  structure: ModelStructure | null
  loading: boolean
  error: string
}>()

const emit = defineEmits<{ close: [] }>()

const totalNodes = computed(() => {
  const values = props.structure?.components
    .map((component) => component.node_count)
    .filter((value): value is number => value !== null) ?? []
  return values.length ? values.reduce((total, value) => total + value, 0) : null
})

const totalParameters = computed(() => {
  const values = props.structure?.components
    .map((component) => component.parameter_count)
    .filter((value): value is number => value !== null) ?? []
  return values.length ? values.reduce((total, value) => total + value, 0) : null
})

const totalLayers = computed(() =>
  props.structure?.components.reduce((total, component) => total + component.layers.length, 0) ?? 0,
)

function formatCount(value: number | null): string {
  if (value === null) return "Unknown"
  return new Intl.NumberFormat("en", { notation: value >= 1_000_000 ? "compact" : "standard", maximumFractionDigits: 2 }).format(value)
}

function tensorShape(tensor: TensorStructure): string {
  return tensor.shape.length ? `[${tensor.shape.join(", ")}]` : "shape unavailable"
}

function visibleTensors(tensors: TensorStructure[]): TensorStructure[] {
  return tensors.slice(0, 24)
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") emit("close")
}

onMounted(() => window.addEventListener("keydown", handleKeydown))
onBeforeUnmount(() => window.removeEventListener("keydown", handleKeydown))
</script>

<template>
  <Teleport to="body">
    <div class="structure-overlay" role="presentation" @click.self="emit('close')">
      <section class="structure-modal" role="dialog" aria-modal="true" aria-labelledby="structure-title">
        <header class="structure-header">
          <div>
            <span>MODEL STRUCTURE</span>
            <h2 id="structure-title">{{ modelName }}</h2>
            <code v-if="structure">{{ structure.variant }} · {{ structure.runtime }} · {{ structure.artifact_id }}</code>
          </div>
          <button type="button" aria-label="Close model structure" @click="emit('close')">×</button>
        </header>

        <div v-if="loading" class="structure-state">Inspecting model artifact…</div>
        <div v-else-if="error" class="structure-state structure-state--error">{{ error }}</div>
        <div v-else-if="structure" class="structure-content">
          <div class="structure-summary">
            <div><span>Components</span><strong>{{ structure.components.length }}</strong></div>
            <div><span>{{ totalLayers ? "Layers" : "Graph nodes" }}</span><strong>{{ formatCount(totalLayers || totalNodes) }}</strong></div>
            <div><span>Parameters</span><strong>{{ formatCount(totalParameters) }}</strong></div>
            <div><span>Format</span><strong>{{ structure.format }}</strong></div>
          </div>

          <article v-for="component in structure.components" :key="component.name" class="structure-component">
            <div class="component-heading">
              <div><span>{{ component.model_type }}</span><h3>{{ component.name }}</h3></div>
              <small>{{ formatCount(component.node_count) }} nodes · {{ formatCount(component.parameter_count) }} parameters</small>
            </div>

            <div v-if="component.inputs.length || component.outputs.length" class="tensor-columns">
              <section v-if="component.inputs.length">
                <h4>Inputs <span>{{ component.inputs.length }}</span></h4>
                <div v-for="tensor in visibleTensors(component.inputs)" :key="tensor.name" class="tensor-row">
                  <code>{{ tensor.name }}</code><span>{{ tensor.dtype ?? "unknown" }} · {{ tensorShape(tensor) }}</span>
                </div>
                <small v-if="component.inputs.length > 24">+ {{ component.inputs.length - 24 }} more inputs</small>
              </section>
              <section v-if="component.outputs.length">
                <h4>{{ ['safetensors', 'pytorch-state-dict'].includes(component.model_type) ? 'Tensors' : 'Outputs' }} <span>{{ component.outputs.length }}</span></h4>
                <div v-for="tensor in visibleTensors(component.outputs)" :key="tensor.name" class="tensor-row">
                  <code>{{ tensor.name }}</code><span>{{ tensor.dtype ?? "unknown" }} · {{ tensorShape(tensor) }}</span>
                </div>
                <small v-if="component.outputs.length > 24">+ {{ component.outputs.length - 24 }} more tensors</small>
              </section>
            </div>

            <section v-if="Object.keys(component.operator_counts).length" class="operator-section">
              <h4>Top operators</h4>
              <div class="operator-list">
                <span v-for="([operator, count]) in Object.entries(component.operator_counts).slice(0, 16)" :key="operator"><code>{{ operator }}</code>{{ count }}</span>
              </div>
            </section>

            <details v-if="component.layers.length" class="layer-section">
              <summary>Layer details <span>{{ component.layers.length }}</span></summary>
              <div class="layer-table">
                <div v-for="layer in component.layers" :key="layer.name" class="layer-row">
                  <code :title="layer.name">{{ layer.name }}</code>
                  <span>{{ layer.layer_type }}</span>
                  <span>{{ formatCount(layer.parameter_count) }} params</span>
                </div>
              </div>
            </details>
          </article>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.structure-overlay { align-items:center; background:rgb(3 7 14 / 78%); display:flex; inset:0; justify-content:center; padding:24px; position:fixed; z-index:1000; }
.structure-modal { background:var(--surface, #111722); border:1px solid var(--line, #2c3442); border-radius:20px; box-shadow:0 24px 80px rgb(0 0 0 / 45%); color:var(--ink, #f5f7fb); max-height:min(82vh, 900px); max-width:980px; overflow:auto; width:100%; }
.structure-header { align-items:flex-start; background:inherit; border-bottom:1px solid var(--line, #2c3442); display:flex; justify-content:space-between; padding:22px 24px; position:sticky; top:0; z-index:1; }
.structure-header span { color:var(--accent, #2997ff); font-size:10px; font-weight:800; letter-spacing:.14em; }
.structure-header h2 { font-size:26px; margin:5px 0 4px; }
.structure-header code { color:var(--muted, #9ba6b8); font-size:11px; }
.structure-header button { background:transparent; border:1px solid var(--line, #2c3442); border-radius:999px; color:inherit; cursor:pointer; font-size:22px; height:36px; line-height:1; width:36px; }
.structure-content { padding:20px 24px 28px; }
.structure-state { color:var(--muted, #9ba6b8); padding:48px 24px; text-align:center; }
.structure-state--error { color:#ff7e87; }
.structure-summary { display:grid; gap:10px; grid-template-columns:repeat(4, 1fr); margin-bottom:18px; }
.structure-summary div { background:rgb(255 255 255 / 4%); border:1px solid var(--line, #2c3442); border-radius:12px; padding:12px; }
.structure-summary span { color:var(--muted, #9ba6b8); display:block; font-size:10px; text-transform:uppercase; }
.structure-summary strong { display:block; font-size:20px; margin-top:4px; }
.structure-component { border:1px solid var(--line, #2c3442); border-radius:14px; padding:16px; }
.structure-component + .structure-component { margin-top:12px; }
.component-heading { align-items:end; display:flex; gap:16px; justify-content:space-between; }
.component-heading span { color:var(--accent, #2997ff); font-size:10px; text-transform:uppercase; }
.component-heading h3 { font-size:17px; margin:3px 0 0; }
.component-heading small, .tensor-columns > section > small { color:var(--muted, #9ba6b8); }
.tensor-columns { display:grid; gap:12px; grid-template-columns:repeat(2, minmax(0, 1fr)); margin-top:14px; }
h4 { font-size:11px; letter-spacing:.08em; margin:0 0 7px; text-transform:uppercase; }
h4 span { color:var(--muted, #9ba6b8); }
.tensor-row { align-items:center; border-top:1px solid rgb(255 255 255 / 7%); display:flex; gap:10px; justify-content:space-between; padding:7px 0; }
.tensor-row code { font-size:11px; max-width:58%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tensor-row span { color:var(--muted, #9ba6b8); font-size:10px; text-align:right; }
.operator-section { margin-top:14px; }
.operator-list { display:flex; flex-wrap:wrap; gap:6px; }
.operator-list span { align-items:center; background:rgb(255 255 255 / 5%); border-radius:7px; color:var(--muted, #9ba6b8); display:flex; font-size:10px; gap:7px; padding:5px 7px; }
.operator-list code { color:var(--ink, #f5f7fb); }
.layer-section { border-top:1px solid rgb(255 255 255 / 7%); margin-top:14px; padding-top:12px; }
.layer-section summary { color:var(--ink, #f5f7fb); cursor:pointer; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.layer-section summary span { color:var(--muted, #9ba6b8); margin-left:5px; }
.layer-table { margin-top:9px; max-height:310px; overflow:auto; }
.layer-row { align-items:center; border-top:1px solid rgb(255 255 255 / 7%); display:grid; font-size:10px; gap:12px; grid-template-columns:minmax(0, 1fr) 150px 90px; padding:7px 0; }
.layer-row code { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.layer-row span { color:var(--muted, #9ba6b8); }
.layer-row span:last-child { text-align:right; }
@media (max-width:700px) { .structure-overlay { padding:10px; } .structure-summary { grid-template-columns:repeat(2, 1fr); } .tensor-columns { grid-template-columns:1fr; } .component-heading { align-items:start; flex-direction:column; gap:5px; } }
</style>
