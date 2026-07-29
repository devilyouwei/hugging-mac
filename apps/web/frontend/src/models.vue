<script setup lang="ts">
import { onMounted, ref } from "vue"

import {
  convertModelResources,
  deleteModelResources,
  downloadModelResources,
  fetchModelResources,
  fetchModels,
  loadModel,
  unloadModel,
} from "@/api/catalog"
import type {
  ConversionTargetStatus,
  InstanceSummary,
  ModelResourceStatus,
  ModelSummary,
  RuntimeSummary,
  VariantSummary,
} from "@/api/types"
import StatusPill from "@/components/StatusPill.vue"

const models = ref<ModelSummary[]>([])
const loading = ref(true)
const error = ref("")
const pending = ref<Record<string, boolean>>({})
const resources = ref<Record<string, ModelResourceStatus>>({})
const resourceErrors = ref<Record<string, string>>({})
const selectedVariants = ref<Record<string, string>>({})

async function refreshModels(): Promise<void> {
  models.value = await fetchModels()
  for (const model of models.value) {
    if (!selectedVariants.value[model.model_id]) {
      selectedVariants.value[model.model_id] = model.default_variant
    }
  }
  await Promise.all(
    models.value.map(async (model) => {
      try {
        const variant = selectedVariant(model)
        resources.value[resourceKey(model.model_id, variant)] =
          await fetchModelResources(model.model_id, variant)
        delete resourceErrors.value[resourceKey(model.model_id, variant)]
      } catch (caught) {
        resourceErrors.value[resourceKey(model.model_id, selectedVariant(model))] =
          caught instanceof Error ? caught.message : "资源状态不可用"
      }
    }),
  )
}

onMounted(async () => {
  try {
    await refreshModels()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型目录加载失败"
  } finally {
    loading.value = false
  }
})

function operationKey(action: string, id: string): string {
  return `${action}:${id}`
}

function resourceKey(modelId: string, variant: string): string {
  return `${modelId}:${variant}`
}

function selectedVariant(model: ModelSummary): string {
  return selectedVariants.value[model.model_id] ?? model.default_variant
}

function selectedVariantSummary(model: ModelSummary): VariantSummary {
  return (
    model.variants.find((item) => item.name === selectedVariant(model)) ??
    model.variants[0]
  )
}

function modelResources(model: ModelSummary): ModelResourceStatus | undefined {
  return resources.value[resourceKey(model.model_id, selectedVariant(model))]
}

async function selectVariant(model: ModelSummary): Promise<void> {
  const variant = selectedVariant(model)
  try {
    resources.value[resourceKey(model.model_id, variant)] = await fetchModelResources(
      model.model_id,
      variant,
    )
    delete resourceErrors.value[resourceKey(model.model_id, variant)]
  } catch (caught) {
    resourceErrors.value[resourceKey(model.model_id, variant)] =
      caught instanceof Error ? caught.message : "资源状态不可用"
  }
}

function isPending(action: string, id: string): boolean {
  return Boolean(pending.value[operationKey(action, id)])
}

async function runOperation(key: string, operation: () => Promise<void>): Promise<void> {
  pending.value[key] = true
  error.value = ""
  try {
    await operation()
    await refreshModels()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型操作失败"
  } finally {
    pending.value[key] = false
  }
}

async function handleLoad(model: ModelSummary, runtime: RuntimeSummary): Promise<void> {
  const variant = selectedVariant(model)
  const key = operationKey("load", `${model.model_id}:${variant}:${runtime.name}`)
  await runOperation(key, async () => {
    await loadModel(model.model_id, runtime.name, { variant })
  })
}

async function handleUnload(instance: InstanceSummary): Promise<void> {
  const key = operationKey("unload", instance.instance_id)
  await runOperation(key, async () => {
    await unloadModel(instance.instance_id)
  })
}

async function handleDownload(model: ModelSummary): Promise<void> {
  const variant = selectedVariant(model)
  const key = operationKey("download", `${model.model_id}:${variant}`)
  await runOperation(key, async () => {
    resources.value[resourceKey(model.model_id, variant)] = await downloadModelResources(
      model.model_id,
      variant,
    )
  })
}

async function handleConvert(
  model: ModelSummary,
  target: ConversionTargetStatus,
): Promise<void> {
  const variant = selectedVariant(model)
  const key = operationKey(
    "convert",
    `${model.model_id}:${variant}:${target.target_format}`,
  )
  await runOperation(key, async () => {
    resources.value[resourceKey(model.model_id, variant)] = await convertModelResources(
      model.model_id,
      target.target_format,
      variant,
    )
  })
}

async function handleDelete(model: ModelSummary): Promise<void> {
  const variant = selectedVariant(model)
  const confirmed = window.confirm(
    `Delete local ${selectedVariantSummary(model).display_name} weights? You can download them again later.`,
  )
  if (!confirmed) return
  const key = operationKey("delete", `${model.model_id}:${variant}`)
  await runOperation(key, async () => {
    resources.value[resourceKey(model.model_id, variant)] = await deleteModelResources(
      model.model_id,
      variant,
    )
  })
}

function runtimeSize(model: ModelSummary, runtime: string): number | null {
  return (
    modelResources(model)?.runtimes.find((item) => item.runtime === runtime)
      ?.size_bytes ?? null
  )
}

function sourceAvailable(model: ModelSummary): boolean {
  return Boolean(
    modelResources(model)?.artifacts.find(
      (artifact) => artifact.artifact_id === "source",
    )?.available,
  )
}

function formatPlatformName(format: string): string {
  const names: Record<string, string> = {
    coreml: "Core ML",
    mlx: "MLX",
    onnx: "ONNX",
    openvino: "OpenVINO",
    rknn: "RKNN",
    tflite: "TensorFlow Lite",
    torchscript: "TorchScript",
  }
  return names[format] ?? format.toUpperCase()
}

function formatDuration(value: number): string {
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`
}

function formatBytes(value: number | null): string {
  if (value === null) return "unavailable"
  const units = ["B", "KB", "MB", "GB"]
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  return `${amount.toFixed(index < 2 ? 0 : 1)} ${units[index]}`
}
</script>

<template>
  <div class="page inner-page models-directory-page">
    <header class="page-title">
      <RouterLink class="back-link" to="/">← Studio</RouterLink>
      <p class="kicker">MODEL REGISTRY / LIVE INSTANCES</p>
      <h1>Models</h1>
      <p>模型定义与当前进程中的实例快照。浏览目录不会触发下载、转换或加载。</p>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
    <div v-if="loading" class="model-row skeleton-card"></div>

    <section v-for="model in models" :key="model.model_id" class="model-row">
      <div class="model-row__identity">
        <span class="eyebrow">{{ model.family }}</span>
        <h2>{{ model.name }}</h2>
        <code>{{ model.model_id }}</code>
        <p>{{ model.description }}</p>
        <div class="model-config-grid">
          <label class="variant-picker">
            <span>WEIGHT VARIANT</span>
            <select
              v-model="selectedVariants[model.model_id]"
              @change="selectVariant(model)"
            >
              <option v-for="variant in model.variants" :key="variant.name" :value="variant.name">
                {{ variant.display_name }}{{ variant.default ? " · default" : "" }}
              </option>
            </select>
            <small>{{ selectedVariantSummary(model).description }}</small>
          </label>
          <div class="model-resource-summary">
            <span>
              {{ selectedVariantSummary(model).display_name }} · local size
              <strong>{{ formatBytes(modelResources(model)?.total_size_bytes ?? 0) }}</strong>
            </span>
            <div class="model-resource-actions">
              <button
                class="button button--compact"
                :disabled="
                  selectedVariantSummary(model).instance_count > 0 ||
                  isPending('download', `${model.model_id}:${selectedVariant(model)}`)
                "
                type="button"
                @click="handleDownload(model)"
              >
                {{
                  isPending("download", `${model.model_id}:${selectedVariant(model)}`)
                    ? "Downloading…"
                    : sourceAvailable(model)
                      ? "Re-download"
                      : "Download"
                }}
              </button>
              <button
                v-for="target in modelResources(model)?.conversion_targets ?? []"
                :key="target.target_format"
                class="button button--compact"
                :disabled="
                  selectedVariantSummary(model).instance_count > 0 ||
                  !sourceAvailable(model) ||
                  isPending(
                    'convert',
                    `${model.model_id}:${selectedVariant(model)}:${target.target_format}`,
                  )
                "
                type="button"
                @click="handleConvert(model, target)"
              >
                {{
                  isPending(
                    "convert",
                    `${model.model_id}:${selectedVariant(model)}:${target.target_format}`,
                  )
                    ? `Converting to ${formatPlatformName(target.target_format)}…`
                    : target.available
                      ? `Re-convert to ${formatPlatformName(target.target_format)}`
                      : `Convert to ${formatPlatformName(target.target_format)}`
                }}
              </button>
              <button
                class="button button--compact button--danger"
                :disabled="
                  selectedVariantSummary(model).instance_count > 0 ||
                  !modelResources(model)?.total_size_bytes ||
                  isPending('delete', `${model.model_id}:${selectedVariant(model)}`)
                "
                type="button"
                @click="handleDelete(model)"
              >
                {{
                  isPending("delete", `${model.model_id}:${selectedVariant(model)}`)
                    ? "Deleting…"
                    : "Delete weights"
                }}
              </button>
            </div>
            <small v-if="selectedVariantSummary(model).instance_count > 0">
              Unload this variant's instances before changing its weights.
            </small>
            <small v-if="resourceErrors[resourceKey(model.model_id, selectedVariant(model))]">
              {{ resourceErrors[resourceKey(model.model_id, selectedVariant(model))] }}
            </small>
          </div>
        </div>
      </div>
      <div class="model-row__operations">
        <div class="model-row__runtime">
          <p class="column-label">RUNTIMES</p>
          <div class="runtime-grid">
            <div v-for="runtime in model.runtimes" :key="runtime.name" class="runtime-row">
              <StatusPill
                :label="runtime.available ? runtime.name : `${runtime.name} unavailable`"
                :tone="runtime.available ? 'ready' : 'warning'"
              />
              <span>{{ runtime.devices.join(" · ") }}</span>
              <span>
                Local files · {{ formatBytes(runtimeSize(model, runtime.name)) }}
              </span>
              <button
                class="button button--compact"
                :disabled="
                  !runtime.available ||
                  isPending('load', `${model.model_id}:${selectedVariant(model)}:${runtime.name}`)
                "
                type="button"
                @click="handleLoad(model, runtime)"
              >
                {{
                  isPending(
                    "load",
                    `${model.model_id}:${selectedVariant(model)}:${runtime.name}`,
                  )
                    ? "Loading…"
                    : "Load instance"
                }}
              </button>
            </div>
          </div>
        </div>
        <div class="model-row__instances">
          <p class="column-label">INSTANCES</p>
          <div class="instance-count">
            <strong>{{ model.instance_count }}</strong>
            <div>
              <span>{{ model.ready_count }} ready</span>
              <small v-if="!model.instances.length">Not instantiated</small>
            </div>
          </div>
          <template v-if="model.instances.length">
            <div class="instance-list">
              <article
                v-for="instance in model.instances"
                :key="instance.instance_id"
                class="instance-panel"
              >
                <div class="instance-panel__header">
                  <StatusPill :label="instance.state" tone="ready" />
                  <code>{{ instance.variant }} · {{ instance.runtime }}</code>
                </div>
                <small>{{ instance.instance_id.slice(0, 8) }} · refs {{ instance.reference_count }}</small>
                <div v-if="instance.load_metrics" class="lifecycle-metric">
                  <span>Load {{ formatDuration(instance.load_metrics.duration_ms) }}</span>
                  <span>
                    RSS allocated {{ formatBytes(instance.load_metrics.memory_allocated_bytes) }}
                  </span>
                  <span>
                    Process RSS {{ formatBytes(instance.load_metrics.process_rss_after_bytes) }}
                  </span>
                </div>
                <button
                  class="button button--compact button--danger"
                  :disabled="
                    instance.reference_count > 0 ||
                    isPending('unload', instance.instance_id)
                  "
                  type="button"
                  @click="handleUnload(instance)"
                >
                  {{ isPending("unload", instance.instance_id) ? "Unloading…" : "Unload" }}
                </button>
              </article>
            </div>
          </template>
        </div>
      </div>
    </section>
    <p class="measurement-note">
      Memory values are process RSS observations around each operation. Concurrent work,
      allocator caches, memory mapping, and GPU/ANE allocations can affect the delta.
    </p>
  </div>
</template>
