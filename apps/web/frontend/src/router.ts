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
      path: "/apps/object-detection",
      name: "object-detection",
      component: () => import("./object_detection/index.vue"),
    },
    {
      path: "/:pathMatch(.*)*",
      redirect: "/",
    },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
