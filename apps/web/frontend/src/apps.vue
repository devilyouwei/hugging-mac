<script setup lang="ts">
import { onMounted, ref } from "vue"

import { fetchApps } from "@/api/catalog"
import type { AppSummary } from "@/api/types"
import AppCard from "@/components/AppCard.vue"

const apps = ref<AppSummary[]>([])
const loading = ref(true)
const error = ref("")

onMounted(async () => {
  try {
    apps.value = await fetchApps()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "应用目录加载失败"
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="page inner-page apps-directory-page">
    <header class="page-title">
      <RouterLink class="back-link" to="/">← Studio</RouterLink>
      <p class="kicker">APPLICATION REGISTRY / LOCAL EXPERIENCES</p>
      <h1>Apps</h1>
      <p>浏览所有基于本地模型能力构建的应用，选择一个项目进入完整体验。</p>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
    <div v-if="loading" class="app-list apps-directory-list">
      <div v-for="item in 3" :key="item" class="app-card skeleton-card"></div>
    </div>
    <div v-else class="app-list apps-directory-list">
      <AppCard
        v-for="(app, appIndex) in apps"
        :key="app.manifest.app_id"
        :app="app"
        :index="appIndex"
      />
      <div v-if="!apps.length" class="empty-state">No applications are registered.</div>
    </div>
  </div>
</template>
