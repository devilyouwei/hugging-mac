<script setup lang="ts">
import VisionAppPage from "@/vision/components/VisionAppPage.vue"
import {
  convertCoreMl,
  downloadSource,
  estimatePoses,
  fetchResourceStatus,
} from "./api"
import PoseResultsPanel from "./components/PoseResultsPanel.vue"
import PoseSkeletonLayer from "./components/PoseSkeletonLayer.vue"
</script>

<template>
  <VisionAppPage
    app-number="02"
    kicker="HUMAN POSE"
    title-top="Pose"
    title-bottom="Estimation"
    description="使用图片、浏览器摄像头或本地视频运行 YOLOv8 Pose。17 点人体骨架会直接叠加在本机推理画面上。"
    model-label="YOLOv8 Pose"
    action-label="Estimate poses"
    action-noun="pose estimation"
    empty-copy="选择图片后，人体关键点、骨架连接和置信度会叠加显示。"
    :overlay="PoseSkeletonLayer"
    :results="PoseResultsPanel"
    :infer="estimatePoses"
    :fetch-status="fetchResourceStatus"
    :download-source="downloadSource"
    :convert-core-ml="convertCoreMl"
  />
</template>
