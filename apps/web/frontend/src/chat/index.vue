<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue"

import {
  deleteChatModel,
  downloadChatModel,
  fetchChatModels,
  fetchLoadedChatModel,
  loadChatModel,
  streamChatMessage,
  unloadChatModel,
} from "./api"
import type { ChatModel, ConversationMessage, LoadedChatModel } from "./types"

const model = ref<LoadedChatModel | null>(null)
const models = ref<ChatModel[]>([])
const selectedModelId = ref("")
const messages = ref<ConversationMessage[]>([])
const prompt = ref("")
const images = ref<Array<{ file: File; url: string }>>([])
const loadingModel = ref(false)
const changingResources = ref(false)
const sending = ref(false)
const error = ref("")
const maxTokens = ref(512)
const temperature = ref(0)
const enableThinking = ref(false)
const conversation = ref<HTMLElement | null>(null)
const autoFollow = ref(true)
let activeRequest: AbortController | null = null
let nextId = 0

const ready = computed(() => model.value?.state === "ready")
const selectedProfile = computed(() =>
  models.value.find((item) => item.model_id === selectedModelId.value),
)
const resourcesAvailable = computed(() => {
  const profile = selectedProfile.value
  return Boolean(
    profile?.resource.artifacts.find(
      (item) => item.artifact_id === profile.required_artifact_id,
    )?.available,
  )
})
const supportsImages = computed(() => selectedProfile.value?.supports_images ?? false)
const canSend = computed(
  () => ready.value && Boolean(prompt.value.trim()) && !sending.value,
)

async function refreshModel() {
  try {
    models.value = await fetchChatModels()
    if (!selectedModelId.value) selectedModelId.value = models.value[0]?.model_id ?? ""
    model.value = selectedModelId.value
      ? await fetchLoadedChatModel(selectedModelId.value)
      : null
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

async function prepareModel() {
  if (!selectedModelId.value) return
  loadingModel.value = true
  error.value = ""
  try {
    model.value = await loadChatModel(selectedModelId.value)
    await refreshModel()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型加载失败"
  } finally {
    loadingModel.value = false
  }
}

async function selectModel() {
  resetConversation()
  images.value.forEach((item) => URL.revokeObjectURL(item.url))
  images.value = []
  error.value = ""
  try {
    model.value = await fetchLoadedChatModel(selectedModelId.value)
  } catch (caught) {
    model.value = null
    error.value = caught instanceof Error ? caught.message : "模型状态读取失败"
  }
}

async function downloadSelectedModel() {
  if (!selectedModelId.value) return
  changingResources.value = true
  error.value = ""
  try {
    await downloadChatModel(selectedModelId.value)
    await refreshModel()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型下载失败"
  } finally {
    changingResources.value = false
  }
}

async function deleteSelectedModel() {
  if (!selectedModelId.value || !window.confirm("删除这个模型的本地权重？")) return
  changingResources.value = true
  error.value = ""
  try {
    await deleteChatModel(selectedModelId.value)
    model.value = null
    await refreshModel()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型删除失败"
  } finally {
    changingResources.value = false
  }
}

async function unloadSelectedModel() {
  if (!selectedModelId.value) return
  loadingModel.value = true
  error.value = ""
  try {
    await unloadChatModel(selectedModelId.value)
    model.value = null
    await refreshModel()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : "模型卸载失败"
  } finally {
    loadingModel.value = false
  }
}

function selectImages(event: Event) {
  if (!supportsImages.value) return
  const input = event.target as HTMLInputElement
  for (const file of Array.from(input.files ?? [])) {
    if (images.value.length >= 4) break
    images.value.push({ file, url: URL.createObjectURL(file) })
  }
  input.value = ""
}

function removeImage(index: number) {
  const item = images.value[index]
  if (item) URL.revokeObjectURL(item.url)
  images.value.splice(index, 1)
}

function resetConversation() {
  activeRequest?.abort()
  messages.value.forEach((message) =>
    message.images.forEach((image) => URL.revokeObjectURL(image.url)),
  )
  messages.value = []
  error.value = ""
  autoFollow.value = true
}

async function scrollToBottom(behavior: ScrollBehavior = "smooth") {
  await nextTick()
  const target = conversation.value
  if (!target) return
  target.scrollTo({ top: target.scrollHeight, behavior })
}

function handleConversationScroll() {
  const target = conversation.value
  if (!target) return
  autoFollow.value = target.scrollHeight - target.scrollTop - target.clientHeight < 96
}

function resumeAutoFollow() {
  autoFollow.value = true
  void scrollToBottom()
}

async function followLatestContent() {
  if (autoFollow.value) await scrollToBottom("auto")
}

async function submit() {
  if (!canSend.value || !model.value) return
  const text = prompt.value.trim()
  const attachments = images.value.splice(0)
  const history = messages.value.map(({ role, content }) => ({ role, content }))
  messages.value.push({
    id: ++nextId,
    role: "user",
    content: text,
    images: attachments.map((item) => ({ name: item.file.name, url: item.url })),
  })
  const assistantMessage = reactive<ConversationMessage>({
    id: ++nextId,
    role: "assistant",
    content: "",
    images: [],
  })
  messages.value.push(assistantMessage)
  prompt.value = ""
  sending.value = true
  autoFollow.value = true
  error.value = ""
  await scrollToBottom()
  activeRequest = new AbortController()
  try {
    const finalEvent = await streamChatMessage(
      {
        instanceId: model.value.instance_id,
        prompt: text,
        history,
        images: attachments.map((item) => item.file),
        maxTokens: maxTokens.value,
        temperature: temperature.value,
        enableThinking: enableThinking.value,
      },
      (event) => {
        if (event.delta) assistantMessage.content += event.delta
        if (event.finish_reason) {
          assistantMessage.meta = `${event.runtime.toUpperCase()} · ${event.generated_tokens ?? "—"} tokens · ${Math.round(event.inference_ms ?? 0)} ms`
        }
        void followLatestContent()
      },
      activeRequest.signal,
    )
    if (!assistantMessage.meta) {
      assistantMessage.meta = `${finalEvent.runtime.toUpperCase()} · ${finalEvent.generated_tokens ?? "—"} tokens`
    }
  } catch (caught) {
    if (!(caught instanceof DOMException && caught.name === "AbortError")) {
      error.value = caught instanceof Error ? caught.message : "消息发送失败"
      if (!assistantMessage.content) {
        messages.value = messages.value.filter((item) => item.id !== assistantMessage.id)
      }
    }
  } finally {
    activeRequest = null
    sending.value = false
    await followLatestContent()
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    void submit()
  }
}

onMounted(refreshModel)
onBeforeUnmount(() => {
  activeRequest?.abort()
  images.value.forEach((item) => URL.revokeObjectURL(item.url))
  messages.value.forEach((message) =>
    message.images.forEach((image) => URL.revokeObjectURL(image.url)),
  )
})
</script>

<template>
  <div class="chat-page">
    <aside class="chat-sidebar">
      <RouterLink class="chat-brand" to="/apps">← Neural Apps</RouterLink>
      <div>
        <p class="chat-label">LOCAL CHAT MODEL</p>
        <h1>{{ selectedProfile?.short_name ?? "Qwen 3.5" }}<br /><em>MLX</em></h1>
        <p class="chat-copy">{{ selectedProfile?.description ?? "选择一个本地模型开始对话。" }}</p>
      </div>

      <label class="chat-model-select">
        MODEL
        <select v-model="selectedModelId" :disabled="sending || loadingModel" @change="selectModel">
          <option v-for="item in models" :key="item.model_id" :value="item.model_id">
            {{ item.display_name }}
          </option>
        </select>
      </label>

      <div class="chat-status" :class="{ ready }">
        <span></span>
        <div>
          <b>{{ ready ? "Model ready" : "Model offline" }}</b>
          <small>{{ model ? `${model.variant} · ${model.device}` : `${((selectedProfile?.disk_size_bytes ?? 0) / 1e9).toFixed(1)} GB weights` }}</small>
        </div>
      </div>
      <button v-if="resourcesAvailable && !ready" class="chat-primary" :disabled="loadingModel" @click="prepareModel">
        {{ loadingModel ? "Loading into memory…" : "Load model" }}
      </button>
      <button v-if="!resourcesAvailable" class="chat-primary" :disabled="changingResources" @click="downloadSelectedModel">
        {{ changingResources ? "Downloading…" : "Download weights" }}
      </button>
      <button v-if="resourcesAvailable && !ready" class="chat-delete" :disabled="changingResources" @click="deleteSelectedModel">
        {{ changingResources ? "Working…" : "Delete local weights" }}
      </button>
      <button v-if="ready" class="chat-delete" :disabled="loadingModel || sending" @click="unloadSelectedModel">
        Unload model
      </button>

      <details class="chat-settings">
        <summary>Generation settings</summary>
        <label>Max tokens <output>{{ maxTokens }}</output></label>
        <input v-model.number="maxTokens" type="range" min="64" max="2048" step="64" />
        <label>Temperature <output>{{ temperature.toFixed(1) }}</output></label>
        <input v-model.number="temperature" type="range" min="0" max="1.5" step="0.1" />
        <label class="chat-toggle"><input v-model="enableThinking" type="checkbox" /> Thinking mode</label>
      </details>
      <button class="chat-clear" :disabled="!messages.length || sending" @click="resetConversation">
        ＋ New conversation
      </button>
    </aside>

    <main class="chat-workspace">
      <header class="chat-topbar">
        <div><span class="status-dot"></span> ON-DEVICE SESSION</div>
        <span>{{ messages.length }} MESSAGES</span>
      </header>

      <section ref="conversation" class="chat-conversation" aria-live="polite" @scroll.passive="handleConversationScroll">
        <div v-if="!messages.length" class="chat-empty">
          <span>Q / 35</span>
          <h2>What can we<br /><i>explore?</i></h2>
          <p>Ask a question, continue a thought<span v-if="supportsImages">, or attach an image for visual understanding</span>.</p>
          <div class="chat-suggestions">
            <button @click="prompt = '解释一下 Transformer 的注意力机制。'">解释一个复杂概念</button>
            <button v-if="supportsImages" @click="prompt = '请分析我上传的图片，并指出关键细节。'">分析一张图片</button>
            <button @click="prompt = '帮我设计一个清晰的项目实施计划。'">制定项目计划</button>
          </div>
        </div>

        <article v-for="message in messages" :key="message.id" :class="['chat-message', message.role]">
          <div class="chat-avatar">{{ message.role === "user" ? "YOU" : "Q" }}</div>
          <div class="chat-bubble">
            <div v-if="message.images.length" class="chat-message-images">
              <img v-for="image in message.images" :key="image.url" :src="image.url" :alt="image.name" />
            </div>
            <p v-if="message.content">{{ message.content }}</p>
            <div v-else-if="message.role === 'assistant'" class="chat-thinking"><i></i><i></i><i></i></div>
            <small v-if="message.meta">{{ message.meta }}</small>
          </div>
        </article>
      </section>

      <button v-if="!autoFollow" class="chat-scroll-latest" type="button" @click="resumeAutoFollow">
        ↓ Latest
      </button>

      <div v-if="error" class="chat-error" role="alert">{{ error }}</div>
      <footer class="chat-composer">
        <div v-if="images.length" class="chat-previews">
          <figure v-for="(image, index) in images" :key="image.url">
            <img :src="image.url" :alt="image.file.name" />
            <button aria-label="移除图片" @click="removeImage(index)">×</button>
          </figure>
        </div>
        <div class="chat-input-row">
          <label v-if="supportsImages" class="chat-attach" :class="{ disabled: !ready || images.length >= 4 }">
            <input type="file" accept="image/png,image/jpeg,image/webp" multiple :disabled="!ready || images.length >= 4" @change="selectImages" />
            <span>＋</span><small>IMAGE</small>
          </label>
          <textarea v-model="prompt" rows="1" maxlength="32000" :disabled="!ready" :placeholder="ready ? 'Message Qwen…' : 'Download and load the model to begin'" @keydown="handleKeydown"></textarea>
          <button class="chat-send" :disabled="!canSend" @click="submit">
            <span>{{ sending ? "···" : "↑" }}</span>
          </button>
        </div>
        <small class="chat-hint">ENTER TO SEND · SHIFT + ENTER FOR NEW LINE<span v-if="supportsImages"> · UP TO 4 IMAGES</span></small>
      </footer>
    </main>
  </div>
</template>

<style scoped>
.chat-page { background:#0d0e0d; color:#f4f2e9; display:grid; grid-template-columns:19rem minmax(0,1fr); height:100%; min-height:0; overflow:hidden; }
.chat-sidebar { border-right:1px solid #343630; display:flex; flex-direction:column; gap:1.5rem; min-height:0; overflow:auto; padding:2rem; }
.chat-brand,.chat-label,.chat-status small,.chat-model-link,.chat-settings,.chat-clear,.chat-topbar,.chat-hint,.chat-bubble small { font-family:var(--font-mono); }
.chat-brand { color:#aaa99f; font-size:.68rem; letter-spacing:.06em; }
.chat-label { color:#85877f; font-size:.62rem; letter-spacing:.1em; margin:0 0 .7rem; }
.chat-sidebar h1 { font-size:2.5rem; letter-spacing:-.06em; line-height:.9; margin:0; }
.chat-sidebar h1 em { color:var(--signal); font-style:normal; }
.chat-copy { color:#aaa99f; font-size:.82rem; line-height:1.55; }
.chat-status { align-items:center; border:1px solid #343630; display:flex; gap:.8rem; padding:.85rem; }
.chat-status>span { background:#6c6d66; border-radius:50%; height:.55rem; width:.55rem; }
.chat-status.ready>span { background:var(--signal); box-shadow:0 0 0 4px #b8f23c20; }
.chat-status div { display:flex; flex-direction:column; gap:.2rem; }
.chat-status b { font-size:.75rem; }.chat-status small { color:#85877f; font-size:.56rem; }
.chat-primary { background:var(--signal); border:0; color:#0d0e0d; cursor:pointer; font-weight:800; padding:.9rem; }.chat-primary:disabled { opacity:.5; }
.chat-model-link { color:var(--signal); font-size:.6rem; text-align:center; }
.chat-model-select { color:#85877f; display:flex; flex-direction:column; font:.56rem var(--font-mono); gap:.45rem; }.chat-model-select select { background:#191b18; border:1px solid #343630; color:#f4f2e9; font:inherit; padding:.7rem; width:100%; }
.chat-delete { background:none; border:1px solid #5d3934; color:#e7a59d; cursor:pointer; font:.58rem var(--font-mono); padding:.7rem; }.chat-delete:disabled { opacity:.5; }
.chat-settings { border-top:1px solid #343630; color:#aaa99f; font-size:.62rem; padding-top:1rem; }.chat-settings summary { cursor:pointer; margin-bottom:1rem; }.chat-settings label { display:flex; justify-content:space-between; margin-top:.7rem; }.chat-settings input[type=range] { accent-color:var(--signal); width:100%; }.chat-toggle { justify-content:flex-start!important; gap:.5rem; }
.chat-clear { background:none; border:1px solid #343630; color:#f4f2e9; cursor:pointer; font-size:.62rem; margin-top:auto; padding:.8rem; }
.chat-workspace { display:grid; grid-template-rows:auto minmax(0,1fr) auto auto; height:100%; min-height:0; min-width:0; overflow:hidden; position:relative; }
.chat-topbar { border-bottom:1px solid #343630; color:#85877f; display:flex; font-size:.58rem; justify-content:space-between; letter-spacing:.08em; padding:1rem 2rem; }.chat-topbar div { align-items:center; display:flex; gap:.6rem; }
.chat-conversation { min-height:0; overscroll-behavior:contain; overflow-x:hidden; overflow-y:auto; padding:3rem max(4vw,2rem); scrollbar-gutter:stable; }
.chat-empty { margin:7vh auto 0; max-width:43rem; }.chat-empty>span { color:var(--signal); font:700 .7rem var(--font-mono); }.chat-empty h2 { font-size:clamp(3rem,6vw,6rem); letter-spacing:-.07em; line-height:.83; margin:1rem 0 1.5rem; }.chat-empty h2 i { color:#a6a79e; font-weight:400; }.chat-empty>p { color:#999a92; line-height:1.6; max-width:34rem; }
.chat-suggestions { display:grid; gap:.6rem; grid-template-columns:repeat(3,1fr); margin-top:2rem; }.chat-suggestions button { background:#151714; border:1px solid #343630; color:#d7d6cf; cursor:pointer; font-size:.7rem; padding:1rem; text-align:left; }.chat-suggestions button:hover { border-color:var(--signal); }
.chat-message { display:grid; gap:1rem; grid-template-columns:2.5rem minmax(0,1fr); margin:0 auto 2rem; max-width:52rem; }.chat-avatar { align-items:center; background:#292b27; border-radius:50%; display:flex; font:700 .55rem var(--font-mono); height:2.5rem; justify-content:center; }.chat-message.assistant .chat-avatar { background:var(--signal); color:#111; font-size:1rem; }.chat-bubble { min-width:0; }.chat-bubble p { font-size:.95rem; line-height:1.7; margin:.35rem 0; white-space:pre-wrap; }.chat-message.user .chat-bubble { background:#1d201b; border-radius:0 1rem 1rem 1rem; padding:.8rem 1rem; }.chat-bubble small { color:#72746c; font-size:.54rem; text-transform:uppercase; }
.chat-message-images { display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:.8rem; }.chat-message-images img { border-radius:.5rem; height:8rem; object-fit:cover; width:10rem; }
.chat-thinking { display:flex; gap:.3rem; padding-top:.8rem; }.chat-thinking i { animation:pulse 1s infinite alternate; background:var(--signal); border-radius:50%; height:.4rem; width:.4rem; }.chat-thinking i:nth-child(2){animation-delay:.2s}.chat-thinking i:nth-child(3){animation-delay:.4s}@keyframes pulse{to{opacity:.2;transform:translateY(-.2rem)}}
.chat-error { background:#5d211b; color:#ffd9d3; font-size:.72rem; margin:0 max(4vw,2rem) 1rem; padding:.75rem 1rem; }
.chat-scroll-latest { background:#262923; border:1px solid #52554c; border-radius:2rem; bottom:7.4rem; color:#f4f2e9; cursor:pointer; font:600 .62rem var(--font-mono); left:50%; padding:.65rem 1rem; position:absolute; transform:translateX(-50%); z-index:3; }
.chat-composer { border-top:1px solid #343630; padding:1rem max(4vw,2rem) 1.4rem; }.chat-input-row { align-items:end; background:#191b18; border:1px solid #3b3d37; display:grid; grid-template-columns:auto minmax(0,1fr) auto; padding:.6rem; }.chat-input-row:focus-within { border-color:#77796f; }.chat-attach { align-items:center; cursor:pointer; display:flex; gap:.35rem; padding:.6rem; }.chat-attach input { display:none; }.chat-attach span { font-size:1.4rem; }.chat-attach small { color:#8c8d85; font:.52rem var(--font-mono); }.chat-attach.disabled { opacity:.35; }.chat-input-row textarea { background:none; border:0; color:#f4f2e9; font:1rem var(--font-display); max-height:10rem; min-height:2.6rem; outline:0; padding:.7rem; resize:vertical; }.chat-send { align-items:center; background:var(--signal); border:0; border-radius:50%; color:#111; cursor:pointer; display:flex; font-size:1.4rem; height:2.7rem; justify-content:center; width:2.7rem; }.chat-send:disabled { background:#42443e; color:#85877f; cursor:default; }.chat-hint { color:#62645d; display:block; font-size:.5rem; letter-spacing:.08em; margin-top:.55rem; text-align:center; }
.chat-previews { display:flex; gap:.6rem; margin-bottom:.7rem; }.chat-previews figure { height:4.5rem; margin:0; position:relative; width:4.5rem; }.chat-previews img { border-radius:.4rem; height:100%; object-fit:cover; width:100%; }.chat-previews button { background:#111; border:1px solid #555; border-radius:50%; color:white; cursor:pointer; height:1.3rem; position:absolute; right:-.3rem; top:-.3rem; width:1.3rem; }
@media(max-width:800px){.chat-page{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr)}.chat-sidebar{border-bottom:1px solid #343630;border-right:0;display:grid;grid-template-columns:1fr auto;height:auto;overflow:visible;padding:1rem 1.2rem}.chat-sidebar>div:nth-of-type(2),.chat-copy,.chat-settings,.chat-clear{display:none}.chat-status{grid-column:2;grid-row:1 / span 2}.chat-brand{align-self:center}.chat-workspace{height:100%;min-height:0}.chat-suggestions{grid-template-columns:1fr}.chat-conversation{padding:2rem 1rem}.chat-composer{padding:1rem}}
</style>
