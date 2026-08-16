<script setup lang="ts">
import { computed } from "vue"
import { RouterLink, RouterView } from "vue-router"
import { useRoute } from "vue-router"

const route = useRoute()
const isChatPage = computed(() => route.name === "chat")
</script>

<template>
  <div class="app-shell" :class="{ 'app-shell--fixed': isChatPage }">
    <header class="site-header">
      <RouterLink class="brand" to="/" aria-label="Hugging Mac home">
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

    <footer v-if="!isChatPage" class="site-footer">
      <span>hugging-mac / Apple Silicon model studio</span>
      <span>Data stays on this machine</span>
    </footer>
  </div>
</template>
