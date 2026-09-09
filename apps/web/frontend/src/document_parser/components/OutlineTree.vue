<script setup lang="ts">
import type { OutlineItem } from "../types"
defineProps<{ items: OutlineItem[] }>()
</script>

<template>
  <ol class="outline-tree">
    <li v-for="item in items" :key="item.id">
      <div><span>{{ item.title }}</span><small>Page {{ item.page }}</small></div>
      <OutlineTree v-if="item.children.length" :items="item.children" />
    </li>
  </ol>
</template>

<style scoped>
.outline-tree{display:grid;gap:.38rem;list-style:none;margin:.45rem 0 0;padding:0}.outline-tree .outline-tree{border-left:1px solid var(--line);margin-left:.6rem;padding-left:.85rem}.outline-tree div{align-items:baseline;display:flex;gap:1rem;justify-content:space-between}.outline-tree span{font-size:.76rem;line-height:1.45}.outline-tree small{color:var(--muted);flex:0 0 auto;font-size:.58rem}
</style>
