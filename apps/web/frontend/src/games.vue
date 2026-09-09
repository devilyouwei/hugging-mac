<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import { fetchGames } from "@/api/catalog"
import type { AppSummary } from "@/api/types"
import AppCard from "@/components/AppCard.vue"

const games = ref<AppSummary[]>([])
const loading = ref(true)
const error = ref("")
const featured = computed(() => games.value[0] ?? null)

onMounted(async () => {
  try {
    games.value = await fetchGames()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "游戏目录加载失败"
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="page inner-page games-directory-page store-page store-page--games">
    <header class="store-titlebar">
      <div>
        <RouterLink class="back-link" to="/">← Studio</RouterLink>
        <p class="kicker">ON-DEVICE ARCADE</p>
        <h1>Play</h1>
      </div>
      <p>身体、镜头和本地神经网络组成的新型游戏控制器。你的动作只在这台 Mac 上处理。</p>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
    <div v-else-if="loading" class="game-empty-card skeleton-card"></div>
    <template v-else-if="games.length">
      <RouterLink v-if="featured" class="store-feature store-feature--games" :to="featured.manifest.frontend_route">
        <div class="store-feature__copy">
          <span>GAME OF THE MOMENT</span>
          <h2>{{ featured.manifest.name }}</h2>
          <p>{{ featured.manifest.description }}</p>
          <b>Play now</b>
        </div>
        <div class="store-feature__game-art" aria-hidden="true">
          <span>🍉</span><span>✨</span><span>🕺</span>
        </div>
      </RouterLink>
      <section class="store-section">
        <header><h2>Games we love</h2><span>All games</span></header>
        <div class="store-list-grid">
          <AppCard v-for="game in games" :key="game.manifest.app_id" :app="game" />
        </div>
      </section>
    </template>
    <section v-else class="games-empty-state">
      <span aria-hidden="true">🕹️</span>
      <p class="kicker">COMING SOON</p>
      <h2>The first neural game is incubating.</h2>
      <p>Games built with local models will appear here when they are ready to play.</p>
    </section>
  </div>
</template>
