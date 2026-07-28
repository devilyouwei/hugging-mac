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

const chipLabel = computed(() => {
  const value = system.value?.chip_name
  return value ? value.replace(/^Apple\s+/i, "") : system.value?.machine || "Mac"
})

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
          <span class="hero-title-line">
            <span class="hero-icon" aria-hidden="true">
              <img src="/hugging-mac-icon.png" alt="" />
            </span>
            <span class="hero-title-solid">One Mac.</span>
          </span>
          <span class="hero-title-outline">Many minds.</span>
        </h1>
        <p class="hero-intro">
          Squeeze every last drop of performance from your Mac's Apple silicon.
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
          <p class="machine-panel__platform">Apple silicon</p>
          <strong>{{ chipLabel }}</strong>
          <div class="machine-spec-grid">
            <div>
              <span>CPU</span>
              <strong>
                {{ system.cpu_physical_count ?? system.cpu_logical_count ?? "—" }}
              </strong>
              <small
                v-if="
                  system.cpu_performance_cores != null &&
                  system.cpu_efficiency_cores != null
                "
              >
                {{ system.cpu_performance_cores }}P ·
                {{ system.cpu_efficiency_cores }}E
              </small>
              <small v-else>cores</small>
            </div>
            <div>
              <span>GPU</span>
              <strong>{{ system.gpu_cores ?? "—" }}</strong>
              <small>cores</small>
            </div>
            <div>
              <span>Neural</span>
              <strong>{{ system.neural_engine_cores ?? "—" }}</strong>
              <small>cores</small>
            </div>
          </div>
          <dl>
            <div>
              <dt>macOS</dt>
              <dd>{{ system.os_version || "Apple Silicon" }}</dd>
            </div>
            <div>
              <dt>Memory free</dt>
              <dd>
                {{ (system.memory_available_bytes / 1024 ** 3).toFixed(1) }} /
                {{ (system.memory_total_bytes / 1024 ** 3).toFixed(0) }} GB
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
