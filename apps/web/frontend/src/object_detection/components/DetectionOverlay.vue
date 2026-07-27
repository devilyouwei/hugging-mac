<script setup lang="ts">
import BoundingBoxLayer from "./BoundingBoxLayer.vue"
import type { DetectionResult } from "../types"

defineProps<{
  imageUrl: string
  result: DetectionResult | null
}>()

</script>

<template>
  <div class="detection-stage">
    <img :src="imageUrl" alt="待检测图片预览" />
    <BoundingBoxLayer :result="result" />
  </div>
</template>

<style scoped>
.detection-stage {
  background:
    linear-gradient(45deg, #dad8cf 25%, transparent 25%),
    linear-gradient(-45deg, #dad8cf 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #dad8cf 75%),
    linear-gradient(-45deg, transparent 75%, #dad8cf 75%);
  background-position:
    0 0,
    0 8px,
    8px -8px,
    -8px 0;
  background-size: 16px 16px;
  line-height: 0;
  overflow: hidden;
  position: relative;
}

img {
  display: block;
  height: auto;
  width: 100%;
}

</style>
