<script setup lang="ts">
import VisionAppPage from "@/vision/components/VisionAppPage.vue"
import {
  convertCoreMl,
  downloadSource,
  fetchResourceStatus,
  segmentInstances,
} from "./api"
import SegmentationMaskLayer from "./components/SegmentationMaskLayer.vue"
import SegmentationResultsPanel from "./components/SegmentationResultsPanel.vue"
</script>

<template>
  <VisionAppPage
    app-number="03"
    kicker="INSTANCE MASKS"
    title-top="Instance"
    title-bottom="Segmentation"
    description="使用图片、浏览器摄像头或本地视频运行 YOLOv8 Seg。每个主体的轮廓会以独立颜色 Mask 叠加在本机画面上。"
    model-label="YOLOv8 Seg"
    action-label="Segment instances"
    action-noun="segmentation"
    empty-copy="选择图片后，每个主体的轮廓、彩色 Mask 和置信度会叠加显示。"
    :overlay="SegmentationMaskLayer"
    :results="SegmentationResultsPanel"
    :infer="segmentInstances"
    :fetch-status="fetchResourceStatus"
    :download-source="downloadSource"
    :convert-core-ml="convertCoreMl"
  />
</template>
