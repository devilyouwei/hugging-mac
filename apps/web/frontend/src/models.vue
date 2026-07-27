<script setup lang="ts">
import { onMounted, ref } from "vue"

import { fetchModels } from "@/api/catalog"
import type { ModelSummary } from "@/api/types"
import StatusPill from "@/components/StatusPill.vue"

const models = ref<ModelSummary[]>([])
const loading = ref(true)
const error = ref("")

onMounted(async () => {
  try {
    models.value = await fetchModels()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型目录加载失败"
  } finally {
    loading.value = false
  }
})
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
      </div>
      <div class="model-row__runtime">
        <p class="column-label">RUNTIMES</p>
        <div v-for="runtime in model.runtimes" :key="runtime.name" class="runtime-row">
          <StatusPill
            :label="runtime.available ? runtime.name : `${runtime.name} unavailable`"
            :tone="runtime.available ? 'ready' : 'warning'"
          />
          <span>{{ runtime.devices.join(" · ") }}</span>
        </div>
      </div>
      <div class="model-row__instances">
        <p class="column-label">INSTANCES</p>
        <strong>{{ model.instance_count }}</strong>
        <span>{{ model.ready_count }} ready</span>
        <small v-if="!model.instances.length">Not instantiated</small>
        <small v-for="instance in model.instances" v-else :key="instance.instance_id">
          {{ instance.runtime }} / {{ instance.state }} /
          refs {{ instance.reference_count }}
        </small>
      </div>
    </section>
  </div>
</template>
