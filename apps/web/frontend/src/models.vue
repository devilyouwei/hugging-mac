<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import {
  convertModelResources,
  deleteModelArtifact,
  downloadModelResource,
  fetchModelInventory,
  fetchModelStructure,
  fetchModels,
  loadModel,
  unloadModel,
} from "@/api/catalog"
import type {
  ArtifactInventoryItem,
  InstanceSummary,
  ModelInventory,
  ModelStructure,
  ModelSummary,
  ResourceOption,
  RuntimeSummary,
} from "@/api/types"
import StatusPill from "@/components/StatusPill.vue"
import ModelStructureModal from "@/components/ModelStructureModal.vue"

type Filter = "all" | "ready" | "setup" | "loaded"

const models = ref<ModelSummary[]>([])
const inventories = ref<Record<string, ModelInventory>>({})
const selectedVariants = ref<Record<string, string>>({})
const pending = ref<Record<string, boolean>>({})
const notices = ref<Record<string, { tone: "success" | "error"; text: string }>>({})
const loading = ref(true)
const pageError = ref("")
const query = ref("")
const filter = ref<Filter>("all")
const expandedModels = ref<Set<string>>(new Set())
const inspectedStructure = ref<ModelStructure | null>(null)
const inspectedModelName = ref("")
const inspectionLoading = ref(false)
const inspectionError = ref("")
const inspectionOpen = ref(false)

const visibleModels = computed(() => {
  const term = query.value.trim().toLowerCase()
  return models.value.filter((model) => {
    const matchesSearch = !term || [
      model.name,
      model.model_id,
      model.family,
      ...model.capabilities,
      ...model.tags,
    ].some((value) => value.toLowerCase().includes(term))
    if (!matchesSearch) return false
    if (filter.value === "loaded") return model.instance_count > 0
    const ready = selectedVariantHasReadyRuntime(model)
    if (filter.value === "ready") return ready
    if (filter.value === "setup") return !ready
    return true
  })
})

const totalLocalBytes = computed(() => Object.values(inventories.value).reduce(
  (total, inventory) => total + inventory.artifacts.reduce(
    (sum, artifact) => sum + (artifact.size_bytes ?? 0),
    0,
  ),
  0,
))
const readyModelCount = computed(() => models.value.filter(modelHasReadyRuntime).length)
const loadedInstanceCount = computed(() => models.value.reduce(
  (count, model) => count + model.instance_count,
  0,
))

function operationKey(action: string, modelId: string, item = ""): string {
  return `${action}:${modelId}:${item}`
}

function artifactKey(artifact: ArtifactInventoryItem): string {
  return `${artifact.variant}:${artifact.runtime}:${artifact.artifact_id}`
}

function isPending(key: string): boolean {
  return Boolean(pending.value[key])
}

function isExpanded(modelId: string): boolean {
  return expandedModels.value.has(modelId)
}

function toggleModel(modelId: string): void {
  const next = new Set(expandedModels.value)
  if (next.has(modelId)) next.delete(modelId)
  else next.add(modelId)
  expandedModels.value = next
}

function selectedVariant(model: ModelSummary): string {
  return selectedVariants.value[model.model_id] ?? model.default_variant
}

function selectVariant(model: ModelSummary, variant: string): void {
  selectedVariants.value[model.model_id] = variant
  delete notices.value[model.model_id]
}

function sourceResources(model: ModelSummary): ResourceOption[] {
  return (inventories.value[model.model_id]?.resources ?? []).filter(
    (resource) => resource.shared || resource.variant === selectedVariant(model),
  )
}

function resourceArtifact(model: ModelSummary, resource: ResourceOption): ArtifactInventoryItem | undefined {
  return (inventories.value[model.model_id]?.artifacts ?? []).find(
    (artifact) => artifact.variant === resource.variant
      && artifact.runtime === resource.runtime
      && artifact.artifact_id === resource.artifact_id,
  )
}

function variantArtifacts(model: ModelSummary): ArtifactInventoryItem[] {
  return (inventories.value[model.model_id]?.artifacts ?? []).filter(
    (artifact) => !artifact.shared && artifact.variant === selectedVariant(model),
  )
}

function sharedArtifacts(model: ModelSummary): ArtifactInventoryItem[] {
  return (inventories.value[model.model_id]?.artifacts ?? []).filter(
    (artifact) => artifact.shared,
  )
}

function resourceReady(model: ModelSummary, resource: ResourceOption): boolean {
  if (resource.shared) return resource.available
  return resource.available && sharedArtifacts(model).every((artifact) => artifact.available)
}

function conversionTargets(model: ModelSummary): ArtifactInventoryItem[] {
  return variantArtifacts(model).filter((artifact) => artifact.convertible)
}

function artifactInspectable(artifact: ArtifactInventoryItem): boolean {
  return ["coreml", "onnx", "pytorch", "torchscript", "mlx", "safetensors"].includes(
    artifact.format,
  )
}

function runtimeArchitectureArtifact(model: ModelSummary, runtime: string): ArtifactInventoryItem | undefined {
  return runtimeArtifacts(model, runtime).find(
    (artifact) => artifact.available && artifactInspectable(artifact),
  )
}

function runtimeArtifacts(
  model: ModelSummary,
  runtime: string,
  variant = selectedVariant(model),
): ArtifactInventoryItem[] {
  return (inventories.value[model.model_id]?.artifacts ?? []).filter(
    (artifact) => !artifact.shared
      && artifact.variant === variant
      && artifact.runtime === runtime,
  )
}

function runtimeReady(model: ModelSummary, runtime: string): boolean {
  const artifacts = [...runtimeArtifacts(model, runtime), ...sharedArtifacts(model)]
  return artifacts.length > 0 && artifacts.every((artifact) => artifact.available)
}

function runtimeRequiredArtifacts(model: ModelSummary, runtime: string): ArtifactInventoryItem[] {
  return [...runtimeArtifacts(model, runtime), ...sharedArtifacts(model)]
}

function modelHasReadyRuntime(model: ModelSummary): boolean {
  return model.variants.some((variant) => model.runtimes.some(
    (runtime) => runtime.available
      && runtimeArtifacts(model, runtime.name, variant.name).length > 0
      && [...runtimeArtifacts(model, runtime.name, variant.name), ...sharedArtifacts(model)].every(
        (artifact) => artifact.available,
      ),
  ))
}

function selectedVariantHasReadyRuntime(model: ModelSummary): boolean {
  return model.runtimes.some(
    (runtime) => runtime.available && runtimeReady(model, runtime.name),
  )
}

function matchingInstances(model: ModelSummary, runtime: string): InstanceSummary[] {
  return model.instances.filter(
    (instance) => instance.variant === selectedVariant(model) && instance.runtime === runtime,
  )
}

function variantHasInstances(model: ModelSummary): boolean {
  return model.instances.some((instance) => instance.variant === selectedVariant(model))
}

function artifactHasInstances(model: ModelSummary, artifact: ArtifactInventoryItem): boolean {
  if (artifact.shared) return model.instances.length > 0
  return model.instances.some(
    (instance) => instance.variant === artifact.variant && instance.runtime === artifact.runtime,
  )
}

function resourceRemoveDisabled(model: ModelSummary, resource: ResourceOption): boolean {
  const artifact = resourceArtifact(model, resource)
  return !artifact?.available || artifactHasInstances(model, artifact)
}

function sourceReady(model: ModelSummary): boolean {
  return variantArtifacts(model).some(
    (artifact) => artifact.available && !artifact.convertible,
  )
}

function sourceLabel(resource: ResourceOption): string {
  if (resource.source.kind === "composite") return "Bundled model resources"
  if (resource.source.repo_id) {
    return resource.source.filename
      ? `${resource.source.repo_id} / ${resource.source.filename}`
      : resource.source.repo_id
  }
  return resource.source.url ?? resource.source.kind
}

function formatName(value: string): string {
  const names: Record<string, string> = {
    ane: "ANE",
    cpu: "CPU",
    coreml: "Core ML",
    "coreml-int8": "Core ML INT8",
    gpu: "GPU",
    mlx: "MLX",
    "mlx-4bit": "MLX 4-bit",
    "mlx-bf16": "MLX BF16",
    "mlx-optiq-4bit": "MLX OptiQ 4-bit",
    mps: "MPS",
    onnx: "ONNX",
    pytorch: "PyTorch",
    "pytorch-mps": "PyTorch MPS",
    safetensors: "SafeTensors",
    tokenizer: "Tokenizer",
  }
  return names[value] ?? value.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatBytes(value: number | null): string {
  if (value === null) return "Not installed"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  return `${amount.toFixed(index < 2 ? 0 : 1)} ${units[index]}`
}

function localVariantBytes(model: ModelSummary): number {
  return [...variantArtifacts(model), ...sharedArtifacts(model)].reduce(
    (total, artifact) => total + (artifact.size_bytes ?? 0),
    0,
  )
}

function variantLabel(model: ModelSummary): string {
  return model.variants.find((variant) => variant.name === selectedVariant(model))
    ?.display_name ?? selectedVariant(model)
}

function runtimeBlockReason(model: ModelSummary, runtime: RuntimeSummary): string | null {
  if (!runtime.available) return runtime.unavailable_reason ?? "Runtime unavailable on this system"
  if (!runtimeReady(model, runtime.name)) return "Install the required files first"
  if (matchingInstances(model, runtime.name).length) return "An instance is already loaded"
  return null
}

async function refresh(): Promise<void> {
  models.value = await fetchModels()
  for (const model of models.value) {
    selectedVariants.value[model.model_id] ||= model.default_variant
  }
  const entries = await Promise.all(models.value.map(async (model) => [
    model.model_id,
    await fetchModelInventory(model.model_id),
  ] as const))
  inventories.value = Object.fromEntries(entries)
}

async function runOperation(
  model: ModelSummary,
  key: string,
  success: string,
  action: () => Promise<void>,
): Promise<void> {
  pending.value[key] = true
  delete notices.value[model.model_id]
  try {
    await action()
    await refresh()
    notices.value[model.model_id] = { tone: "success", text: success }
  } catch (caught) {
    notices.value[model.model_id] = {
      tone: "error",
      text: caught instanceof Error ? caught.message : "The operation could not be completed.",
    }
  } finally {
    pending.value[key] = false
  }
}

async function handleDownload(model: ModelSummary, resource: ResourceOption): Promise<void> {
  const key = operationKey("download", model.model_id, resource.resource_id)
  await runOperation(model, key, "Resource installed.", async () => {
    inventories.value[model.model_id] = await downloadModelResource(
      model.model_id,
      resource,
      resourceReady(model, resource),
    )
  })
}

async function handleConvert(model: ModelSummary, artifact: ArtifactInventoryItem): Promise<void> {
  const key = operationKey("convert", model.model_id, artifactKey(artifact))
  await runOperation(model, key, `${formatName(artifact.format)} build completed.`, async () => {
    await convertModelResources(model.model_id, artifact.format, artifact.variant)
  })
}

async function handleDelete(model: ModelSummary, artifact: ArtifactInventoryItem): Promise<void> {
  const confirmed = window.confirm(
    `Remove ${variantLabel(model)} / ${formatName(artifact.artifact_id)} from local storage?`,
  )
  if (!confirmed) return
  const key = operationKey("delete", model.model_id, artifactKey(artifact))
  await runOperation(model, key, "Local artifact removed.", async () => {
    inventories.value[model.model_id] = await deleteModelArtifact(model.model_id, artifact)
  })
}

async function handleDeleteResource(model: ModelSummary, resource: ResourceOption): Promise<void> {
  const artifact = resourceArtifact(model, resource)
  if (artifact) await handleDelete(model, artifact)
}

async function handleLoad(model: ModelSummary, runtime: RuntimeSummary): Promise<void> {
  const item = `${selectedVariant(model)}:${runtime.name}`
  const key = operationKey("load", model.model_id, item)
  await runOperation(model, key, `${formatName(runtime.name)} instance loaded.`, async () => {
    await loadModel(model.model_id, runtime.name, { variant: selectedVariant(model) })
  })
}

async function handleUnload(model: ModelSummary, instance: InstanceSummary): Promise<void> {
  if (instance.reference_count > 0 && !window.confirm(
    `This instance is used by ${instance.reference_count} active ${instance.reference_count === 1 ? "session" : "sessions"}. Unload it anyway?`,
  )) return
  const key = operationKey("unload", model.model_id, instance.instance_id)
  await runOperation(model, key, "Instance unloaded.", async () => {
    await unloadModel(instance.instance_id, instance.reference_count > 0)
  })
}

async function handleInspect(model: ModelSummary, artifact: ArtifactInventoryItem): Promise<void> {
  inspectionOpen.value = true
  inspectionLoading.value = true
  inspectionError.value = ""
  inspectedStructure.value = null
  inspectedModelName.value = model.name
  try {
    inspectedStructure.value = await fetchModelStructure(model.model_id, artifact)
  } catch (caught) {
    inspectionError.value = caught instanceof Error
      ? caught.message
      : "The model architecture could not be inspected."
  } finally {
    inspectionLoading.value = false
  }
}

async function handleInspectRuntime(model: ModelSummary, runtime: string): Promise<void> {
  const artifact = runtimeArchitectureArtifact(model, runtime)
  if (artifact) await handleInspect(model, artifact)
}

function closeInspection(): void {
  inspectionOpen.value = false
}

onMounted(async () => {
  try {
    await refresh()
  } catch (caught) {
    pageError.value = caught instanceof Error ? caught.message : "The model catalog could not be loaded."
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="page inner-page models-directory-page">
    <header class="models-hero">
      <div>
        <RouterLink class="back-link" to="/">← Studio</RouterLink>
        <p class="kicker">MODEL LIBRARY</p>
        <h1>Models</h1>
        <p>Install model files, build optimized artifacts, and manage local runtime instances.</p>
      </div>
      <div class="models-summary" aria-label="Model library summary">
        <div><strong>{{ readyModelCount }}</strong><span>Models ready</span></div>
        <div><strong>{{ loadedInstanceCount }}</strong><span>Instances loaded</span></div>
        <div><strong>{{ formatBytes(totalLocalBytes) }}</strong><span>Local storage</span></div>
      </div>
    </header>

    <section class="models-toolbar" aria-label="Model filters">
      <label class="model-search">
        <span>Search</span>
        <input v-model="query" type="search" placeholder="Model, family, or capability" />
      </label>
      <div class="filter-tabs" role="group" aria-label="Filter models">
        <button
          v-for="option in ([['all', 'All'], ['ready', 'Ready'], ['setup', 'Needs setup'], ['loaded', 'Loaded']] as const)"
          :key="option[0]"
          :class="{ active: filter === option[0] }"
          type="button"
          @click="filter = option[0]"
        >
          {{ option[1] }}
        </button>
      </div>
    </section>

    <div v-if="pageError" class="error-banner" role="alert">{{ pageError }}</div>
    <div v-if="loading" class="model-workspace skeleton-card"></div>
    <p v-else-if="!visibleModels.length" class="models-empty">No models match this filter.</p>

    <article
      v-for="model in visibleModels"
      :key="model.model_id"
      class="model-workspace"
      :class="{
        'model-workspace--ready': modelHasReadyRuntime(model),
        'model-workspace--loaded': model.instance_count,
        'model-workspace--expanded': isExpanded(model.model_id),
      }"
    >
      <header class="model-workspace__header" @click="toggleModel(model.model_id)">
        <div class="model-identity">
          <div class="model-identity__topline">
            <span class="eyebrow">{{ model.family }}</span>
            <StatusPill
              :label="model.instance_count ? `${model.instance_count} loaded` : modelHasReadyRuntime(model) ? 'Ready' : 'Needs setup'"
              :tone="model.instance_count || modelHasReadyRuntime(model) ? 'ready' : 'warning'"
            />
          </div>
          <h2>{{ model.name }}</h2>
          <code>{{ model.model_id }}</code>
          <div class="model-tags">
            <span v-for="capability in model.capabilities" :key="capability">
              {{ formatName(capability) }}
            </span>
          </div>
        </div>
        <div v-if="isExpanded(model.model_id)" class="variant-control" @click.stop>
          <span class="column-label">VARIANT</span>
          <div class="variant-tabs" role="group" :aria-label="`${model.name} variants`">
            <button
              v-for="variant in model.variants"
              :key="variant.name"
              :class="{ active: selectedVariant(model) === variant.name }"
              type="button"
              @click="selectVariant(model, variant.name)"
            >
              {{ variant.display_name }}
              <i v-if="variant.default">Default</i>
            </button>
          </div>
          <small>{{ formatBytes(localVariantBytes(model)) }} installed for this variant</small>
        </div>
        <div v-else class="model-collapsed-summary">
          <span>{{ model.variants.length }} {{ model.variants.length === 1 ? 'variant' : 'variants' }}</span>
          <span>{{ formatBytes(localVariantBytes(model)) }} installed</span>
        </div>
        <button
          class="model-expand-button"
          type="button"
          :aria-expanded="isExpanded(model.model_id)"
          :aria-controls="`model-details-${model.model_id}`"
          :aria-label="`${isExpanded(model.model_id) ? 'Collapse' : 'Expand'} ${model.name}`"
          @click.stop="toggleModel(model.model_id)"
        >
          <span aria-hidden="true">⌄</span>
        </button>
      </header>

      <Transition name="model-details">
        <div v-if="isExpanded(model.model_id)" class="model-workspace__details">
          <div class="model-workspace__details-inner">
            <div
              v-if="notices[model.model_id]"
              class="model-notice"
              :class="`model-notice--${notices[model.model_id].tone}`"
              role="status"
            >
              {{ notices[model.model_id].text }}
            </div>

      <div :id="`model-details-${model.model_id}`" class="model-workspace__body">
        <section class="workflow-panel">
          <div class="workflow-panel__heading">
            <span class="step-number">01</span>
            <div><h3>Source files</h3><p>Install each ready-to-use artifact and its required files.</p></div>
          </div>
          <div class="workflow-list">
            <div v-for="resource in sourceResources(model)" :key="resource.resource_id" class="workflow-item">
              <div class="workflow-item__icon">↓</div>
              <div class="workflow-item__content">
                <strong>{{ formatName(resource.artifact_id) }}</strong>
                <small :title="sourceLabel(resource)">{{ sourceLabel(resource) }}</small>
              </div>
              <div class="workflow-item__meta">
                <span>{{ formatBytes(resource.size_bytes) }}</span>
                <StatusPill
                  :label="resourceReady(model, resource) ? 'Installed' : resource.available ? 'Shared files missing' : 'Required'"
                  :tone="resourceReady(model, resource) ? 'ready' : 'warning'"
                />
              </div>
              <div class="workflow-item__actions">
                <button
                  class="button button--compact"
                  :disabled="isPending(operationKey('download', model.model_id, resource.resource_id))"
                  type="button"
                  @click="handleDownload(model, resource)"
                >
                  {{ isPending(operationKey('download', model.model_id, resource.resource_id)) ? 'Installing…' : resourceReady(model, resource) ? 'Reinstall' : resource.available ? 'Install shared files' : 'Install' }}
                </button>
                <button
                  class="text-action text-action--danger"
                  :disabled="resourceRemoveDisabled(model, resource) || isPending(operationKey('delete', model.model_id, resource.resource_id))"
                  :title="resourceRemoveDisabled(model, resource) ? 'Install the artifact and unload matching instances before removing it' : 'Remove from local storage'"
                  type="button"
                  @click="handleDeleteResource(model, resource)"
                >
                  {{ isPending(operationKey('delete', model.model_id, resource.resource_id)) ? 'Removing…' : 'Remove' }}
                </button>
              </div>
            </div>
            <p v-if="!sourceResources(model).length" class="workflow-empty">No downloadable resources are declared.</p>
          </div>
        </section>

        <section class="workflow-panel">
          <div class="workflow-panel__heading">
            <span class="step-number">02</span>
            <div><h3>Optimized builds</h3><p>Create artifacts declared by the model package.</p></div>
          </div>
          <div class="workflow-list">
            <div v-for="artifact in conversionTargets(model)" :key="artifactKey(artifact)" class="workflow-item">
              <div class="workflow-item__icon">◇</div>
              <div class="workflow-item__content">
                <strong>{{ formatName(artifact.format) }}</strong>
                <small>{{ formatName(artifact.runtime) }} runtime</small>
              </div>
              <div class="workflow-item__meta">
                <span>{{ formatBytes(artifact.size_bytes) }}</span>
                <StatusPill :label="artifact.available ? 'Built' : sourceReady(model) ? 'Available' : 'Source required'" :tone="artifact.available ? 'ready' : sourceReady(model) ? 'idle' : 'warning'" />
              </div>
              <div class="workflow-item__actions">
                <button
                  class="button button--compact"
                  :disabled="variantHasInstances(model) || !sourceReady(model) || isPending(operationKey('convert', model.model_id, artifactKey(artifact)))"
                  type="button"
                  @click="handleConvert(model, artifact)"
                >
                  {{ isPending(operationKey('convert', model.model_id, artifactKey(artifact))) ? 'Building…' : artifact.available ? 'Rebuild' : 'Build' }}
                </button>
                <button
                  class="text-action text-action--danger"
                  :disabled="!artifact.available || artifactHasInstances(model, artifact) || isPending(operationKey('delete', model.model_id, artifactKey(artifact)))"
                  type="button"
                  @click="handleDelete(model, artifact)"
                >
                  {{ isPending(operationKey('delete', model.model_id, artifactKey(artifact))) ? 'Removing…' : 'Remove' }}
                </button>
              </div>
            </div>
            <p v-if="!conversionTargets(model).length" class="workflow-empty">This model ships ready-to-use artifacts. No build step is required.</p>
          </div>
        </section>

        <section class="workflow-panel workflow-panel--runtime">
          <div class="workflow-panel__heading">
            <span class="step-number">03</span>
            <div><h3>Runtime</h3><p>Load only when every required artifact is installed.</p></div>
          </div>
          <div class="runtime-options">
            <div v-for="runtime in model.runtimes" :key="runtime.name" class="runtime-option">
              <div class="runtime-option__header">
                <div><strong>{{ formatName(runtime.name) }}</strong><small>{{ runtime.devices.map(formatName).join(' · ') }}</small></div>
                <StatusPill
                  :label="matchingInstances(model, runtime.name).length ? 'Loaded' : runtimeReady(model, runtime.name) ? 'Ready' : 'Files missing'"
                  :tone="matchingInstances(model, runtime.name).length || runtimeReady(model, runtime.name) ? 'ready' : 'warning'"
                />
              </div>
              <div class="runtime-requirements">
                <span v-for="artifact in runtimeRequiredArtifacts(model, runtime.name)" :key="artifactKey(artifact)" :class="{ ready: artifact.available }">
                  {{ artifact.available ? '✓' : '○' }} {{ formatName(artifact.artifact_id) }}{{ artifact.shared ? ' · shared' : '' }}
                </span>
              </div>
              <div class="runtime-option__actions">
                <button
                  class="button button--compact runtime-load-button"
                  :title="runtimeBlockReason(model, runtime) ?? 'Load a local instance'"
                  :disabled="Boolean(runtimeBlockReason(model, runtime)) || isPending(operationKey('load', model.model_id, `${selectedVariant(model)}:${runtime.name}`))"
                  type="button"
                  @click="handleLoad(model, runtime)"
                >
                  {{ isPending(operationKey('load', model.model_id, `${selectedVariant(model)}:${runtime.name}`)) ? 'Loading…' : matchingInstances(model, runtime.name).length ? 'Loaded' : 'Load instance' }}
                </button>
                <button
                  v-if="runtimeArchitectureArtifact(model, runtime.name)"
                  class="text-action"
                  type="button"
                  @click="handleInspectRuntime(model, runtime.name)"
                >
                  Architecture
                </button>
              </div>
              <div v-for="instance in matchingInstances(model, runtime.name)" :key="instance.instance_id" class="loaded-instance">
                <code>{{ instance.instance_id.slice(0, 8) }}</code>
                <span>{{ instance.reference_count }} references</span>
                <button
                  class="text-action text-action--danger"
                  :disabled="isPending(operationKey('unload', model.model_id, instance.instance_id))"
                  type="button"
                  @click="handleUnload(model, instance)"
                >
                  {{ isPending(operationKey('unload', model.model_id, instance.instance_id)) ? 'Unloading…' : 'Unload' }}
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>

          </div>
        </div>
      </Transition>
    </article>
    <ModelStructureModal
      v-if="inspectionOpen"
      :error="inspectionError"
      :loading="inspectionLoading"
      :model-name="inspectedModelName"
      :structure="inspectedStructure"
      @close="closeInspection"
    />
  </div>
</template>
