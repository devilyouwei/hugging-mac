<script setup lang="ts">
import VisionAppPage from "@/vision/components/VisionAppPage.vue"
import {
  estimatePoses,
  fetchFaceResourceStatus,
  fetchHandResourceStatus,
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
    description="使用图片、浏览器摄像头或本地视频并行运行 YOLOv8 Pose、RetinaFace 与 MediaPipe Hand Detection，叠加人体骨架、人脸与手部框。"
    model-label="YOLOv8 Pose"
    action-label="Estimate poses"
    action-noun="pose estimation"
    empty-copy="选择图片后，人体关键点、骨架连接和置信度会叠加显示。"
    :overlay="PoseSkeletonLayer"
    :results="PoseResultsPanel"
    :infer="estimatePoses"
    :fetch-status="fetchResourceStatus"
    secondary-model-label="RetinaFace"
    :fetch-secondary-status="fetchFaceResourceStatus"
    tertiary-model-label="MediaPipe Hand Detection"
    :fetch-tertiary-status="fetchHandResourceStatus"
    :default-confidence="0.65"
    :default-secondary-confidence="0.9"
    :default-tertiary-confidence="0.85"
  />
</template>
