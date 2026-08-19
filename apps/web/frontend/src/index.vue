<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import { fetchCatalog, fetchSystemInfo } from "@/api/catalog"
import type { AppSummary, ModelSummary, SystemInfo } from "@/api/types"
import AppCard from "@/components/AppCard.vue"
import ModelCard from "@/components/ModelCard.vue"

const models = ref<ModelSummary[]>([])
const apps = ref<AppSummary[]>([])
const games = ref<AppSummary[]>([])
const system = ref<SystemInfo | null>(null)
const loading = ref(true)
const error = ref("")

const readyInstances = computed(() =>
  models.value.reduce((total, model) => total + model.ready_count, 0),
)
const featuredModels = computed(() => models.value.slice(0, 2))
const featuredApps = computed(() => apps.value.slice(0, 3))
const featuredGames = computed(() => games.value.slice(0, 3))

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
    games.value = catalog.games
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
        <p class="kicker hero-kicker"><span></span> BUILT FOR APPLE SILICON</p>
        <h1>
          <span class="hero-title-line">
            <span class="hero-icon" aria-hidden="true">
              <img src="/hugging-mac-icon.png" alt="" />
            </span>
            <span class="hero-title-solid">One Mac,</span>
          </span>
          <span class="hero-title-outline">Many minds.</span>
        </h1>
        <p class="hero-intro">
          Your private AI studio, accelerated by Apple silicon. Explore, create,
          and run powerful models entirely on your Mac.
        </p>
        <div class="hero-actions">
          <RouterLink class="button button--primary" to="/apps/object-detection">
            <span>Run first demo</span><span aria-hidden="true">↗</span>
          </RouterLink>
          <RouterLink class="button button--glass" to="/models">
            Explore models
          </RouterLink>
        </div>
        <div class="hero-privacy">
          <span aria-hidden="true">⌁</span>
          <span>Private by design</span>
          <i></i>
          <span>Zero cloud required</span>
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

    <div class="home-catalog-layout">
      <section class="section-block home-catalog-column">
        <div class="section-heading">
          <div>
            <div>
              <p class="kicker">NEURAL MODEL LAYER</p>
              <h2>Neural Models</h2>
            </div>
          </div>
          <RouterLink class="section-heading__link" to="/models">
            全部 <span aria-hidden="true">→</span>
          </RouterLink>
        </div>
        <p class="home-catalog-column__meta">
          {{ models.length }} registered · {{ readyInstances }} ready
        </p>
        <div v-if="loading" class="card-grid">
          <div v-for="item in 2" :key="item" class="catalog-card skeleton-card"></div>
        </div>
        <div v-else class="card-grid">
          <ModelCard
            v-for="model in featuredModels"
            :key="model.model_id"
            :model="model"
          />
        </div>
      </section>

      <div class="home-catalog-stack">
      <section class="section-block home-catalog-column home-catalog-column--apps">
        <div class="section-heading">
          <div>
            <div>
              <p class="kicker">NEURAL APPLICATION LAYER</p>
              <h2>Neural Apps</h2>
            </div>
          </div>
          <RouterLink class="section-heading__link" to="/apps">
            全部 <span aria-hidden="true">→</span>
          </RouterLink>
        </div>
        <p class="home-catalog-column__meta">
          {{ apps.length }} registered · local neural tools, shared SDK
        </p>
        <div class="app-list">
          <AppCard
            v-for="(app, appIndex) in featuredApps"
            :key="app.manifest.app_id"
            :app="app"
            :index="appIndex"
          />
          <div v-if="!loading && !apps.length" class="empty-state">
            No applications are registered.
          </div>
        </div>
      </section>

      <section class="section-block home-catalog-column home-catalog-column--games">
        <div class="section-heading">
          <div>
            <div>
              <p class="kicker">NEURAL GAME LAYER</p>
              <h2>Neural Games</h2>
            </div>
          </div>
          <RouterLink class="section-heading__link" to="/games">
            全部 <span aria-hidden="true">→</span>
          </RouterLink>
        </div>
        <p class="home-catalog-column__meta">
          {{ games.length }} registered · interactive worlds powered by local models
        </p>
        <div v-if="featuredGames.length" class="app-list home-games-list">
          <AppCard
            v-for="(game, gameIndex) in featuredGames"
            :key="game.manifest.app_id"
            :app="game"
            :index="gameIndex"
          />
        </div>
        <div v-else class="game-empty-card">
          <span aria-hidden="true">🕹️</span>
          <div>
            <strong>Neural games are next.</strong>
            <p>Our first local-model game is in development. This space is ready for it.</p>
          </div>
        </div>
      </section>
      </div>
    </div>
  </div>
</template>
