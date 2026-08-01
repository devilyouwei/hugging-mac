import { createRouter, createWebHistory } from "vue-router"

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "home",
      component: () => import("./index.vue"),
    },
    {
      path: "/models",
      name: "models",
      component: () => import("./models.vue"),
    },
    {
      path: "/apps",
      name: "apps",
      component: () => import("./apps.vue"),
    },
    {
      path: "/games",
      name: "games",
      component: () => import("./games.vue"),
    },
    {
      path: "/games/yolo-pose-follow",
      name: "yolo-pose-follow",
      component: () => import("./yolo_pose_follow/index.vue"),
    },
    {
      path: "/apps/chat",
      name: "chat",
      component: () => import("./chat/index.vue"),
    },
    {
      path: "/apps/object-detection",
      name: "object-detection",
      component: () => import("./object_detection/index.vue"),
    },
    {
      path: "/apps/pose-estimation",
      name: "pose-estimation",
      component: () => import("./pose_estimation/index.vue"),
    },
    {
      path: "/apps/instance-segmentation",
      name: "instance-segmentation",
      component: () => import("./instance_segmentation/index.vue"),
    },
    {
      path: "/apps/live-transcription",
      name: "live-transcription",
      component: () => import("./live_transcription/index.vue"),
    },
    {
      path: "/apps/text-to-speech",
      name: "text-to-speech",
      component: () => import("./text_to_speech/index.vue"),
    },
    {
      path: "/:pathMatch(.*)*",
      redirect: "/",
    },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
