<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import { fetchCatalog, fetchSystemInfo } from "@/api/catalog"
import type { AppSummary, ModelSummary, SystemInfo } from "@/api/types"
import AppCard from "@/components/AppCard.vue"
import ModelCard from "@/components/ModelCard.vue"

const models = ref<ModelSummary[]>([])
const apps = ref<AppSummary[]>([])
const system = ref<SystemInfo | null>(null)
const loading = ref(true)
const error = ref("")

const readyInstances = computed(() =>
  models.value.reduce((total, model) => total + model.ready_count, 0),
)

async function loadHome() {
  loading.value = true
  error.value = ""
  try {
    const [catalog, machine] = await Promise.all([fetchCatalog(), fetchSystemInfo()])
    models.value = catalog.models
    apps.value = catalog.apps
    system.value = machine
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "无法连接本地平台"
  } finally {
    loading.value = false
  }
}

onMounted(loadHome)
</script>

<template>
  <div class="page home-page">
    <section class="hero">
      <div class="hero-copy">
        <p class="kicker">LOCAL MODEL STUDIO · APPLE SILICON</p>
        <h1>
          One Mac.<br />
          <span>Many minds.</span>
        </h1>
        <p class="hero-intro">
          从经典视觉网络到现代多模态模型，在统一 SDK 上加载、观察和运行。
          数据留在本机，runtime 为当前机器动态选择。
        </p>
        <div class="hero-actions">
          <RouterLink class="button button--primary" to="/apps/object-detection">
            Run first demo
          </RouterLink>
          <RouterLink class="button button--ghost" to="/models">
            Browse models
          </RouterLink>
        </div>
      </div>
      <aside class="machine-panel" aria-label="本机运行环境">
        <div class="machine-panel__header">
          <span>THIS MACHINE</span>
          <span class="live-indicator">LIVE</span>
        </div>
        <template v-if="system">
          <strong>{{ system.machine }}</strong>
          <dl>
            <div>
              <dt>macOS</dt>
              <dd>{{ system.os_version || "Apple Silicon" }}</dd>
            </div>
            <div>
              <dt>Memory free</dt>
              <dd>
                {{ (system.memory_available_bytes / 1024 ** 3).toFixed(1) }} GB
              </dd>
            </div>
            <div>
              <dt>Python</dt>
              <dd>{{ system.python_version }}</dd>
            </div>
          </dl>
        </template>
        <div v-else class="machine-skeleton">Reading local system…</div>
        <div class="machine-grid" aria-hidden="true">
          <span v-for="cell in 24" :key="cell"></span>
        </div>
      </aside>
    </section>

    <div v-if="error" class="error-banner" role="alert">
      <div>
        <strong>Backend unavailable</strong>
        <span>{{ error }}</span>
      </div>
      <button type="button" @click="loadHome">Retry</button>
    </div>

    <section class="section-block">
      <div class="section-heading">
        <div>
          <span class="section-index">01</span>
          <div>
            <p class="kicker">MODEL LAYER</p>
            <h2>Available models</h2>
          </div>
        </div>
        <p>
          {{ models.length }} registered · {{ readyInstances }} ready instance{{
            readyInstances === 1 ? "" : "s"
          }}
        </p>
      </div>
      <div v-if="loading" class="card-grid">
        <div v-for="item in 2" :key="item" class="catalog-card skeleton-card"></div>
      </div>
      <div v-else class="card-grid">
        <ModelCard v-for="model in models" :key="model.model_id" :model="model" />
      </div>
    </section>

    <section class="section-block apps-block">
      <div class="section-heading">
        <div>
          <span class="section-index">02</span>
          <div>
            <p class="kicker">APPLICATION LAYER</p>
            <h2>Model-powered apps</h2>
          </div>
        </div>
        <p>Focused tools, shared model SDK.</p>
      </div>
      <div class="app-list">
        <AppCard
          v-for="(app, appIndex) in apps"
          :key="app.manifest.app_id"
          :app="app"
          :index="appIndex"
        />
        <div v-if="!loading && !apps.length" class="empty-state">
          No applications are registered.
        </div>
      </div>
    </section>
  </div>
</template>
