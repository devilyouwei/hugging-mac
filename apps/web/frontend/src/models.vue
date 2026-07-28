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
  UnloadResult,
} from "@/api/types"
import StatusPill from "@/components/StatusPill.vue"

const models = ref<ModelSummary[]>([])
const loading = ref(true)
const error = ref("")
const pending = ref<Record<string, boolean>>({})
const lastUnloads = ref<Record<string, UnloadResult>>({})
const resources = ref<Record<string, ModelResourceStatus>>({})
const resourceErrors = ref<Record<string, string>>({})

async function refreshModels(): Promise<void> {
  models.value = await fetchModels()
  await Promise.all(
    models.value.map(async (model) => {
      try {
        resources.value[model.model_id] = await fetchModelResources(model.model_id)
        delete resourceErrors.value[model.model_id]
      } catch (caught) {
        resourceErrors.value[model.model_id] =
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
  const key = operationKey("load", `${model.model_id}:${runtime.name}`)
  await runOperation(key, async () => {
    await loadModel(model.model_id, runtime.name)
  })
}

async function handleUnload(instance: InstanceSummary): Promise<void> {
  const key = operationKey("unload", instance.instance_id)
  await runOperation(key, async () => {
    const result = await unloadModel(instance.instance_id)
    lastUnloads.value[`${result.model_id}:${result.runtime}`] = result
  })
}

async function handleDownload(model: ModelSummary): Promise<void> {
  const key = operationKey("download", model.model_id)
  await runOperation(key, async () => {
    resources.value[model.model_id] = await downloadModelResources(model.model_id)
  })
}

async function handleConvert(
  model: ModelSummary,
  target: ConversionTargetStatus,
): Promise<void> {
  const key = operationKey(
    "convert",
    `${model.model_id}:${target.target_format}`,
  )
  await runOperation(key, async () => {
    resources.value[model.model_id] = await convertModelResources(
      model.model_id,
      target.target_format,
    )
  })
}

async function handleDelete(model: ModelSummary): Promise<void> {
  const confirmed = window.confirm(
    `Delete all local weights for ${model.name}? You can download them again later.`,
  )
  if (!confirmed) return
  const key = operationKey("delete", model.model_id)
  await runOperation(key, async () => {
    resources.value[model.model_id] = await deleteModelResources(model.model_id)
  })
}

function lastUnload(modelId: string, runtime: string): UnloadResult | undefined {
  return lastUnloads.value[`${modelId}:${runtime}`]
}

function runtimeSize(modelId: string, runtime: string): number | null {
  return (
    resources.value[modelId]?.runtimes.find((item) => item.runtime === runtime)
      ?.size_bytes ?? null
  )
}

function sourceAvailable(modelId: string): boolean {
  return Boolean(
    resources.value[modelId]?.artifacts.find(
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
  <div class="page inner-page">
    <header class="page-title">
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
        <div class="model-resource-summary">
          <span>
            Local size
            <strong>{{ formatBytes(resources[model.model_id]?.total_size_bytes ?? 0) }}</strong>
          </span>
          <div class="model-resource-actions">
            <button
              class="button button--compact"
              :disabled="
                model.instance_count > 0 ||
                isPending('download', model.model_id)
              "
              type="button"
              @click="handleDownload(model)"
            >
              {{
                isPending("download", model.model_id)
                  ? "Downloading…"
                  : sourceAvailable(model.model_id)
                    ? "Re-download"
                    : "Download"
              }}
            </button>
            <button
              v-for="target in resources[model.model_id]?.conversion_targets ?? []"
              :key="target.target_format"
              class="button button--compact"
              :disabled="
                model.instance_count > 0 ||
                !sourceAvailable(model.model_id) ||
                isPending(
                  'convert',
                  `${model.model_id}:${target.target_format}`,
                )
              "
              type="button"
              @click="handleConvert(model, target)"
            >
              {{
                isPending(
                  "convert",
                  `${model.model_id}:${target.target_format}`,
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
                model.instance_count > 0 ||
                !resources[model.model_id]?.total_size_bytes ||
                isPending('delete', model.model_id)
              "
              type="button"
              @click="handleDelete(model)"
            >
              {{ isPending("delete", model.model_id) ? "Deleting…" : "Delete weights" }}
            </button>
          </div>
          <small v-if="model.instance_count > 0">
            Unload all instances before changing weights.
          </small>
          <small v-if="resourceErrors[model.model_id]">
            {{ resourceErrors[model.model_id] }}
          </small>
        </div>
      </div>
      <div class="model-row__runtime">
        <p class="column-label">RUNTIMES</p>
        <div v-for="runtime in model.runtimes" :key="runtime.name" class="runtime-row">
          <StatusPill
            :label="runtime.available ? runtime.name : `${runtime.name} unavailable`"
            :tone="runtime.available ? 'ready' : 'warning'"
          />
          <span>{{ runtime.devices.join(" · ") }}</span>
          <span>
            Local files · {{ formatBytes(runtimeSize(model.model_id, runtime.name)) }}
          </span>
          <button
            class="button button--compact"
            :disabled="
              !runtime.available ||
              isPending('load', `${model.model_id}:${runtime.name}`)
            "
            type="button"
            @click="handleLoad(model, runtime)"
          >
            {{
              isPending("load", `${model.model_id}:${runtime.name}`)
                ? "Loading…"
                : "Load instance"
            }}
          </button>
          <div
            v-if="lastUnload(model.model_id, runtime.name)"
            class="lifecycle-metric lifecycle-metric--released"
          >
            <strong>Last unload</strong>
            <span>
              {{ formatDuration(lastUnload(model.model_id, runtime.name)!.metrics.duration_ms) }}
            </span>
            <span>
              RSS released
              {{
                formatBytes(
                  lastUnload(model.model_id, runtime.name)!.metrics.memory_released_bytes,
                )
              }}
            </span>
          </div>
        </div>
      </div>
      <div class="model-row__instances">
        <p class="column-label">INSTANCES</p>
        <strong>{{ model.instance_count }}</strong>
        <span>{{ model.ready_count }} ready</span>
        <small v-if="!model.instances.length">Not instantiated</small>
        <article
          v-for="instance in model.instances"
          v-else
          :key="instance.instance_id"
          class="instance-panel"
        >
          <div class="instance-panel__header">
            <StatusPill :label="instance.state" tone="ready" />
            <code>{{ instance.runtime }}</code>
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
    </section>
    <p class="measurement-note">
      Memory values are process RSS observations around each operation. Concurrent work,
      allocator caches, memory mapping, and GPU/ANE allocations can affect the delta.
    </p>
  </div>
</template>
