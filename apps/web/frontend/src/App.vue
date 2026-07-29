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
