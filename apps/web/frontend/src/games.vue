<script setup lang="ts">
import { onMounted, ref } from "vue"

import { fetchGames } from "@/api/catalog"
import type { AppSummary } from "@/api/types"
import AppCard from "@/components/AppCard.vue"

const games = ref<AppSummary[]>([])
const loading = ref(true)
const error = ref("")

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
  <div class="page inner-page games-directory-page">
    <header class="page-title">
      <RouterLink class="back-link" to="/">← Studio</RouterLink>
      <p class="kicker">NEURAL GAME REGISTRY / LOCAL PLAYGROUND</p>
      <h1>Neural Games</h1>
      <p>面向本地神经模型的互动游戏实验场。未来的游戏会在这里运行、观察并持续进化。</p>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
    <div v-else-if="loading" class="game-empty-card skeleton-card"></div>
    <div v-else-if="games.length" class="app-list apps-directory-list">
      <AppCard
        v-for="(game, gameIndex) in games"
        :key="game.manifest.app_id"
        :app="game"
        :index="gameIndex"
      />
    </div>
    <section v-else class="games-empty-state">
      <span aria-hidden="true">🕹️</span>
      <p class="kicker">COMING SOON</p>
      <h2>The first neural game is incubating.</h2>
      <p>Games built with local models will appear here when they are ready to play.</p>
    </section>
  </div>
</template>
