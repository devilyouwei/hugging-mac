<script setup lang="ts">
import VisionAppPage from "@/vision/components/VisionAppPage.vue"

import { detectObjects, fetchResourceStatus } from "./api"
import DetectionOverlay from "./components/DetectionOverlay.vue"
import DetectionResults from "./components/ResultsPanel.vue"
</script>

<template>
  <VisionAppPage
    app-number="01"
    kicker="COMPUTER VISION"
    title-top="Object"
    title-bottom="Detection"
    description="使用图片、浏览器摄像头或本地视频运行 YOLOv8。所有推理均在本机完成。"
    model-label="YOLOv8"
    action-label="Detect objects"
    action-noun="detection"
    empty-copy="选择图片后，检测框、类别和置信度会叠加显示。"
    :overlay="DetectionOverlay"
    :results="DetectionResults"
    :infer="detectObjects"
    :fetch-status="fetchResourceStatus"
  />
</template>
