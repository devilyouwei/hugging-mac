<script setup lang="ts">
import { computed } from "vue"
import { RouterLink, RouterView } from "vue-router"
import { useRoute } from "vue-router"

const route = useRoute()
const isChatPage = computed(() => route.name === "chat")
</script>

<template>
  <div class="app-shell" :class="{ 'app-shell--fixed': isChatPage }">
    <div class="ambient-canvas" aria-hidden="true">
      <span class="ambient-orb ambient-orb--blue"></span>
      <span class="ambient-orb ambient-orb--violet"></span>
      <span class="ambient-orb ambient-orb--cyan"></span>
      <span class="ambient-noise"></span>
    </div>

    <header class="site-header" :class="{ 'site-header--chat': isChatPage }">
      <div class="site-toolbar">
        <RouterLink class="brand" to="/" aria-label="Hugging Mac home">
          <span class="brand-icon-wrap">
            <img
              class="brand-mark"
              src="/hugging-mac-icon.png"
              alt=""
              aria-hidden="true"
            />
          </span>
          <span class="brand-copy">
            <strong>Hugging Mac</strong>
            <small>Local Intelligence</small>
          </span>
        </RouterLink>

        <nav class="main-nav" aria-label="Primary navigation">
          <RouterLink to="/models">Models</RouterLink>
          <RouterLink to="/apps">Apps</RouterLink>
          <RouterLink to="/games">Games</RouterLink>
        </nav>

        <div class="local-badge" aria-label="本地运行">
          <span class="status-dot"></span>
          <span>On device</span>
        </div>
      </div>
    </header>

    <main>
      <RouterView />
    </main>

    <footer v-if="!isChatPage" class="site-footer">
      <span>hugging-mac / Apple Silicon model studio</span>
      <span>Data stays on this machine</span>
    </footer>
  </div>
</template>
