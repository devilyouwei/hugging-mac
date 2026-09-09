<script setup lang="ts">
import { computed, onMounted, ref } from "vue"

import { fetchApps } from "@/api/catalog"
import type { AppSummary } from "@/api/types"
import AppCard from "@/components/AppCard.vue"

const apps = ref<AppSummary[]>([])
const loading = ref(true)
const error = ref("")
const featured = computed(() => apps.value[0] ?? null)

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
  <div class="page inner-page apps-directory-page store-page">
    <header class="store-titlebar">
      <div>
        <RouterLink class="back-link" to="/">← Studio</RouterLink>
        <p class="kicker">LOCAL APP COLLECTION</p>
        <h1>Discover</h1>
      </div>
      <p>为 Apple silicon 打造的本地智能工具。无需云端，打开即可在这台 Mac 上运行。</p>
    </header>

    <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
    <div v-if="loading" class="store-list-grid">
      <div v-for="item in 3" :key="item" class="app-card skeleton-card"></div>
    </div>
    <template v-else>
      <RouterLink v-if="featured" class="store-feature store-feature--apps" :to="featured.manifest.frontend_route">
        <div class="store-feature__copy">
          <span>EDITOR’S CHOICE</span>
          <h2>{{ featured.manifest.name }}</h2>
          <p>{{ featured.manifest.description }}</p>
          <b>Explore on this Mac</b>
        </div>
        <div class="store-feature__art" aria-hidden="true">
          <span>✦</span><span>◉</span><span>⌁</span>
        </div>
      </RouterLink>

      <section class="store-section">
        <header><h2>Essential local apps</h2><span>{{ apps.length }} apps</span></header>
        <div class="store-list-grid">
          <AppCard v-for="app in apps" :key="app.manifest.app_id" :app="app" />
        </div>
        <div v-if="!apps.length" class="empty-state">No neural applications are registered.</div>
      </section>
    </template>
  </div>
</template>
