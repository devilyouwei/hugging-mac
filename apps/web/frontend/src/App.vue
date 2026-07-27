<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router"
</script>

<template>
  <div class="app-shell">
    <header class="site-header">
      <RouterLink class="brand" to="/" aria-label="hugging-mac 首页">
        <img
          class="brand-mark"
          src="/hugging-mac-icon.png"
          alt=""
          aria-hidden="true"
        />
        <span>Hugging Mac</span>
      </RouterLink>
      <nav class="main-nav" aria-label="主导航">
        <RouterLink to="/">Studio</RouterLink>
        <RouterLink to="/models">Models</RouterLink>
        <RouterLink to="/apps/object-detection">Object Detection</RouterLink>
      </nav>
      <div class="local-badge">
        <span class="status-dot" aria-hidden="true"></span>
        Local runtime
      </div>
    </header>

    <main>
      <RouterView />
    </main>

    <footer class="site-footer">
      <span>hugging-mac / Apple Silicon model studio</span>
      <span>Data stays on this machine</span>
    </footer>
  </div>
</template>
