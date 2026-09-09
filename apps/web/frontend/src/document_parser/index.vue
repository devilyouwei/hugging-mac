<script setup lang="ts">
import { CloudUpload, FileImage, FileText, MoreHorizontal, Trash2, X } from "@lucide/vue"
import { computed, onBeforeUnmount, onMounted, ref } from "vue"

import { cancelJob, createJob, deleteDocument, fetchDocument, fetchDocuments, fetchJobs, fetchJobSnapshot, fetchLayoutModels, fetchOutput, fetchParserModels, jobEventsUrl, uploadDocument } from "./api"
import ConfirmDeleteModal from "./components/ConfirmDeleteModal.vue"
import MarkdownDocumentTab from "./components/MarkdownDocumentTab.vue"
import OriginalDocumentTab from "./components/OriginalDocumentTab.vue"
import StructuredDocumentTab from "./components/StructuredDocumentTab.vue"
import { preferredParserModelId } from "./modelSelection"
import type { DocumentOutput, DocumentRecord, Job, JobKind, OutputStreamEvent, PageLayout, ParserModel, StructuredDocument } from "./types"

type Tab = "original" | "markdown" | "structure"
const documents = ref<DocumentRecord[]>([])
const selected = ref<DocumentRecord | null>(null)
const selectedTab = ref<Tab>("original")
const previewPage = ref(0)
const files = ref<File[]>([])
const busy = ref(false)
const dragging = ref(false)
const error = ref("")
const openMenu = ref<string | null>(null)
const deleteTarget = ref<DocumentRecord | null>(null)
const deleting = ref(false)
const outputs = ref<Record<string, Partial<Record<JobKind, DocumentOutput>>>>({})
const draftPages = ref<Record<string, Record<string, string>>>({})
const draftStructures = ref<Record<string, StructuredDocument>>({})
const parserModels = ref<ParserModel[]>([])
const layoutModels = ref<ParserModel[]>([])
const selectedModels = ref<Record<string, string>>({})
const selectedLayoutModels = ref<Record<string, string>>({})
const currentLayouts = ref<Record<string, PageLayout>>({})
const streams = new Map<string, EventSource>()
const MODEL_STORAGE_KEY = "document-parser:ocr-model"
const LAYOUT_MODEL_STORAGE_KEY = "document-parser:layout-model"

const activeJob = computed(() => {
  const jobs = selected.value?.jobs
  return jobs ? Object.values(jobs).find((job) => job && isActive(job)) ?? null : null
})
const markdownJob = computed(() => selected.value?.jobs?.markdown ?? null)
const structureJob = computed(() => selected.value?.jobs?.structure ?? null)
const markdownOutput = computed(() => selected.value ? outputs.value[selected.value.id]?.markdown ?? null : null)
const structureOutput = computed(() => selected.value ? outputs.value[selected.value.id]?.structure?.structured ?? null : null)
const markdownDraft = computed(() => {
  const job = markdownJob.value
  if (!job || !isActive(job)) return ""
  const pages = draftPages.value[job.id] ?? {}
  return Object.keys(pages).map(Number).sort((a, b) => a - b).filter((page) => pages[String(page)]?.trim()).map((page) => `<!-- Page ${page + 1} -->\n\n${pages[String(page)]}`).join("\n\n")
})
const structureDraft = computed(() => {
  const job = structureJob.value
  return job && isActive(job) ? draftStructures.value[job.id] ?? null : null
})
const selectedModelId = computed({
  get: () => selected.value ? selectedModels.value[selected.value.id] ?? "" : "",
  set: (modelId: string) => {
    if (!selected.value) return
    selectedModels.value = { ...selectedModels.value, [selected.value.id]: modelId }
    localStorage.setItem(MODEL_STORAGE_KEY, modelId)
  },
})
const selectedLayoutModelId = computed({
  get: () => selected.value ? selectedLayoutModels.value[selected.value.id] ?? "" : "",
  set: (modelId: string) => {
    if (!selected.value) return
    selectedLayoutModels.value = { ...selectedLayoutModels.value, [selected.value.id]: modelId }
    localStorage.setItem(LAYOUT_MODEL_STORAGE_KEY, modelId)
  },
})
const currentLayout = computed(() => {
  const job = markdownJob.value
  return job ? currentLayouts.value[job.id] ?? null : null
})

function isActive(job: Job) { return ["queued", "waiting_for_model", "running"].includes(job.state) }
function acceptFiles(value: File[]) { if (!value.length) return; const pdf = value.find((file) => file.name.toLowerCase().endsWith(".pdf")); files.value = pdf ? [pdf] : value }
function choose(event: Event) { acceptFiles(Array.from((event.target as HTMLInputElement).files ?? [])) }
function drop(event: DragEvent) { dragging.value = false; acceptFiles(Array.from(event.dataTransfer?.files ?? [])) }

async function upload() {
  if (!files.value.length) return
  busy.value = true
  error.value = ""
  try { const created = await uploadDocument(files.value); files.value = []; await refreshDocuments(); await selectDocument(created.document.id) }
  catch (caught) { error.value = caught instanceof Error ? caught.message : "Upload failed" }
  finally { busy.value = false }
}
async function refreshDocuments() { documents.value = await fetchDocuments() }
async function refreshParserModels() { parserModels.value = await fetchParserModels() }
async function refreshLayoutModels() { layoutModels.value = await fetchLayoutModels() }
function ensureSelectedModel(document: DocumentRecord) {
  if (!selectedModels.value[document.id]) {
    const remembered = localStorage.getItem(MODEL_STORAGE_KEY) ?? undefined
    selectedModels.value = {
      ...selectedModels.value,
      [document.id]: preferredParserModelId(parserModels.value, document, remembered),
    }
  }
  if (!selectedLayoutModels.value[document.id]) {
    const rememberedLayout = localStorage.getItem(LAYOUT_MODEL_STORAGE_KEY)
    const latestLayout = document.jobs?.markdown?.layout_model_id
    const available = layoutModels.value.find((model) => model.ready)?.model_id ?? ""
    const layoutId = [latestLayout, rememberedLayout, available].find(
      (candidate) => candidate && layoutModels.value.some((model) => model.model_id === candidate),
    ) ?? ""
    selectedLayoutModels.value = { ...selectedLayoutModels.value, [document.id]: layoutId }
  }
}
async function loadOutputs(id: string) {
  const [markdown, structure] = await Promise.all([fetchOutput(id, "markdown"), fetchOutput(id, "structure")])
  outputs.value = { ...outputs.value, [id]: { markdown, structure } }
}
async function selectDocument(id: string) {
  openMenu.value = null
  try {
    selected.value = await fetchDocument(id)
    ensureSelectedModel(selected.value)
    previewPage.value = 0
    await loadOutputs(id)
    for (const job of Object.values(selected.value.jobs ?? {})) if (job && isActive(job)) await resumeJob(job)
  } catch (caught) { error.value = caught instanceof Error ? caught.message : "Could not load document" }
}
function updateJob(job: Job) {
  if (selected.value?.id === job.document_id && (job.kind === "markdown" || job.kind === "structure")) {
    selected.value = { ...selected.value, jobs: { ...selected.value.jobs, [job.kind]: job }, job: isActive(job) ? job : selected.value.job }
  }
}
async function start(kind: JobKind) {
  if (!selected.value || activeJob.value) return
  error.value = ""
  try { const job = await createJob(selected.value.id, kind, kind === "markdown" ? selectedModelId.value : undefined, kind === "markdown" ? selectedLayoutModelId.value : undefined); updateJob(job); await resumeJob(job); await refreshDocuments() }
  catch (caught) { error.value = caught instanceof Error ? caught.message.replace("markdown_required", "Generate Markdown first") : "Could not start task" }
}
async function stop(job: Job | null) {
  if (!job || !isActive(job)) return
  try { updateJob(await cancelJob(job.id)) } catch (caught) { error.value = caught instanceof Error ? caught.message : "Could not stop task" }
}
async function resumeJob(job: Job) {
  if (!isActive(job) || streams.has(job.id)) return
  const snapshot = await fetchJobSnapshot(job.id).catch(() => null)
  if (snapshot) {
    updateJob(snapshot.job)
    if (snapshot.draft.pages) draftPages.value = { ...draftPages.value, [job.id]: snapshot.draft.pages }
    if (snapshot.draft.structured) draftStructures.value = { ...draftStructures.value, [job.id]: snapshot.draft.structured }
    if (snapshot.draft.current_layout) currentLayouts.value = { ...currentLayouts.value, [job.id]: snapshot.draft.current_layout }
  }
  connect(job, snapshot?.event_sequence ?? job.event_sequence)
}
function connect(job: Job, after: number) {
  if (streams.has(job.id)) return
  const stream = new EventSource(jobEventsUrl(job.id, after))
  streams.set(job.id, stream)
  const names = ["output_reset", "output_delta", "output_commit", "progress", "page_layout_started", "page_layout", "page_layout_failed", "layout_region_started", "layout_region_completed", "structure_field_started", "structure_snapshot", "output_complete", "waiting_for_model", "completed", "completed_with_warnings", "failed", "cancelled", "superseded"]
  names.forEach((name) => stream.addEventListener(name, (event) => handleEvent(job, event as MessageEvent<string>)))
  stream.onerror = () => undefined
}
function handleEvent(job: Job, event: MessageEvent<string>) {
  try {
    const payload = JSON.parse(event.data)
    if (["output_reset", "output_delta", "output_commit"].includes(event.type) && job.kind === "markdown") updateMarkdownDraft(job.id, event.type, payload as OutputStreamEvent)
    if (event.type === "structure_snapshot" && payload.structured) draftStructures.value = { ...draftStructures.value, [job.id]: payload.structured }
    if (event.type === "page_layout_started") currentLayouts.value = { ...currentLayouts.value, [job.id]: { ...payload, analyzing: true } as PageLayout }
    if (event.type === "page_layout") currentLayouts.value = { ...currentLayouts.value, [job.id]: payload as PageLayout }
    if (event.type === "page_layout_failed") currentLayouts.value = { ...currentLayouts.value, [job.id]: { ...payload, model_id: job.layout_model_id, error: payload.message } as PageLayout }
    if (event.type === "layout_region_started") currentLayouts.value = { ...currentLayouts.value, [job.id]: { ...(currentLayouts.value[job.id] ?? payload), current_region: { index: payload.region, total: payload.region_count, label: payload.label } } as PageLayout }
    if (event.type === "layout_region_completed") currentLayouts.value = { ...currentLayouts.value, [job.id]: { ...(currentLayouts.value[job.id] ?? payload), current_region: payload.region < payload.region_count ? { index: payload.region + 1, total: payload.region_count, label: "next region" } : undefined } as PageLayout }
    if (event.type === "progress") updateJob({ ...job, stage: payload.stage ?? job.stage, completed_units: payload.completed ?? job.completed_units, total_units: payload.total ?? job.total_units, state: "running" })
  } catch { /* Persisted snapshots remain authoritative. */ }
  if (["completed", "completed_with_warnings", "failed", "cancelled", "superseded"].includes(event.type)) { closeStream(job.id); void refreshAfterTerminal(job.document_id) }
}
function updateMarkdownDraft(jobId: string, event: string, payload: OutputStreamEvent) {
  if (payload.page < 0) { draftPages.value = { ...draftPages.value, [jobId]: {} }; return }
  const pages = draftPages.value[jobId] ?? {}
  const key = String(payload.page)
  const current = pages[key] ?? ""
  const next = event === "output_reset" ? "" : event === "output_commit" ? payload.markdown ?? current : current + (payload.delta ?? "")
  draftPages.value = { ...draftPages.value, [jobId]: { ...pages, [key]: next } }
}
async function refreshAfterTerminal(documentId: string) {
  await refreshDocuments().catch(() => undefined)
  if (selected.value?.id === documentId) {
    selected.value = await fetchDocument(documentId).catch(() => null)
    if (selected.value) await loadOutputs(documentId).catch(() => undefined)
  }
}
function closeStream(id: string) { streams.get(id)?.close(); streams.delete(id) }
async function confirmDelete() {
  if (!deleteTarget.value) return
  deleting.value = true
  const target = deleteTarget.value
  const index = documents.value.findIndex((item) => item.id === target.id)
  try {
    await deleteDocument(target.id)
    for (const job of Object.values(target.jobs ?? {})) if (job) closeStream(job.id)
    deleteTarget.value = null
    await refreshDocuments()
    if (selected.value?.id === target.id) { const next = documents.value[Math.min(index, documents.value.length - 1)]; selected.value = null; if (next) await selectDocument(next.id) }
  } catch (caught) { error.value = caught instanceof Error ? caught.message : "Delete failed" }
  finally { deleting.value = false }
}
function formatDate(value: string) { const date = new Date(value); return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric" }).format(date) }

onMounted(async () => {
  await Promise.all([refreshDocuments(), refreshParserModels(), refreshLayoutModels()]).catch((caught) => { error.value = String(caught) })
  const jobs = await fetchJobs(true).catch(() => [])
  await Promise.all(jobs.filter((job) => job.kind !== ("titles" as JobKind)).map(resumeJob))
  if (documents.value[0]) await selectDocument(documents.value[0].id)
})
onBeforeUnmount(() => { streams.forEach((stream) => stream.close()); streams.clear() })
</script>

<template>
  <div class="page inner-page parser-page">
    <header class="app-hero parser-hero"><div><RouterLink class="back-link" to="/apps">← Apps</RouterLink><p class="kicker">APP 07 / DOCUMENT INTELLIGENCE</p><h1>Document <span>Parser</span></h1></div><p>Local MLX OCR · Streamed and resumable</p></header>
    <div v-if="error" class="notice error-banner">{{ error }}<button aria-label="Dismiss" @click="error = ''"><X :size="15" /></button></div>
    <main class="parser-shell">
      <aside class="library glass-panel">
        <label class="upload-zone" :class="{ dragging }" @dragenter.prevent="dragging = true" @dragover.prevent="dragging = true" @dragleave.prevent="dragging = false" @drop.prevent="drop"><CloudUpload :size="24" /><strong>Drop a document here</strong><span>PDF or ordered page images</span><b>Choose files</b><small v-if="files.length">{{ files.length === 1 ? files[0].name : `${files.length} files selected` }}</small><input type="file" multiple accept="application/pdf,image/jpeg,image/png,image/webp" @change="choose"></label>
        <button class="upload-button" :disabled="busy || !files.length" @click="upload"><CloudUpload :size="16" />{{ busy ? "Uploading…" : "Upload" }}</button>
        <div class="library-heading"><h2>Documents</h2><span>{{ documents.length }}</span></div>
        <div class="document-list">
          <div v-for="document in documents" :key="document.id" class="document-item" :class="{ selected: selected?.id === document.id }">
            <button class="document-select" @click="selectDocument(document.id)"><span class="file-icon"><FileText v-if="document.source_kind === 'pdf'" :size="14" /><FileImage v-else :size="14" /></span><span class="document-copy"><strong :title="document.name">{{ document.name }}</strong><small>{{ document.page_count }} pages · {{ formatDate(document.created_at) }}</small></span></button>
            <button class="operation" aria-label="Document operations" @click.stop="openMenu = openMenu === document.id ? null : document.id"><MoreHorizontal :size="17" /></button>
            <div v-if="openMenu === document.id" class="operation-menu"><button @click.stop="deleteTarget = document; openMenu = null"><Trash2 :size="14" />Delete</button></div>
          </div>
          <p v-if="!documents.length">No documents yet.</p>
        </div>
      </aside>
      <section v-if="selected" class="workspace glass-panel">
        <nav class="tabs" role="tablist"><button :class="{ active: selectedTab === 'original' }" @click="selectedTab = 'original'">Original</button><button :class="{ active: selectedTab === 'markdown' }" @click="selectedTab = 'markdown'">Markdown</button><button :class="{ active: selectedTab === 'structure' }" @click="selectedTab = 'structure'">Structured Information</button></nav>
        <div class="tab-content">
          <OriginalDocumentTab v-if="selectedTab === 'original'" v-model:page="previewPage" :document="selected" />
          <MarkdownDocumentTab v-else-if="selectedTab === 'markdown'" v-model:model-id="selectedModelId" v-model:layout-model-id="selectedLayoutModelId" :document="selected" :output="markdownOutput" :draft="markdownDraft" :job="markdownJob" :models="parserModels" :layout-models="layoutModels" :current-layout="currentLayout" :blocked="Boolean(activeJob && activeJob.kind !== 'markdown')" @generate="start('markdown')" @stop="stop(markdownJob)" />
          <StructuredDocumentTab v-else :value="structureOutput" :draft="structureDraft" :job="structureJob" :has-markdown="Boolean(markdownOutput?.content)" :blocked="Boolean(activeJob && activeJob.kind !== 'structure')" @generate="start('structure')" @stop="stop(structureJob)" @open-markdown="selectedTab = 'markdown'" />
        </div>
      </section>
      <section v-else class="welcome glass-panel"><FileText :size="44" /><h2>Select or upload a document</h2><p>Files and OCR results stay on this Mac.</p></section>
    </main>
    <ConfirmDeleteModal v-if="deleteTarget" :name="deleteTarget.name" :busy="deleting" @cancel="deleteTarget = null" @confirm="confirmDelete" />
  </div>
</template>

<style scoped>
.parser-page{box-sizing:border-box;color:var(--ink);max-width:100%;min-height:100vh;overflow-x:clip;padding-bottom:4rem;width:100%}.parser-hero{align-items:end;border-bottom:0;padding-top:1.25rem}.parser-hero h1 span{background:linear-gradient(105deg,var(--accent),#5e5ce6 58%,#af52de);background-clip:text;color:transparent}.parser-hero>p{color:var(--muted);font-size:.78rem}.glass-panel{background:color-mix(in srgb,var(--surface) 78%,transparent);border:1px solid color-mix(in srgb,var(--line) 85%,white);border-radius:24px;box-shadow:0 24px 60px color-mix(in srgb,#000 13%,transparent),inset 0 1px 0 color-mix(in srgb,white 35%,transparent);backdrop-filter:blur(28px) saturate(145%)}.parser-shell{display:grid;gap:1rem;grid-template-columns:minmax(0,292px) minmax(0,1fr);max-width:100%;min-width:0;width:100%}.parser-shell>*{min-width:0}.library{align-self:start;overflow:visible;padding:.8rem;position:sticky;top:1rem}.upload-zone{align-items:center;background:color-mix(in srgb,var(--accent) 5%,transparent);border:1px dashed color-mix(in srgb,var(--accent) 50%,var(--line));border-radius:18px;cursor:pointer;display:flex;flex-direction:column;gap:.3rem;overflow:hidden;padding:1rem;text-align:center}.upload-zone>svg{color:var(--accent)}.upload-zone strong{font-size:.75rem}.upload-zone span{color:var(--muted);font-size:.61rem}.upload-zone b{background:var(--surface-solid);border:1px solid var(--line);border-radius:999px;font-size:.62rem;margin-top:.2rem;padding:.32rem .65rem}.upload-zone small{color:var(--accent);font-size:.58rem;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.upload-zone input{display:none!important}.upload-button,.parser-page button{align-items:center;background:var(--surface-solid);border:1px solid var(--line);border-radius:11px;color:var(--ink);cursor:pointer;display:inline-flex;font-family:var(--font-display);font-size:.68rem;font-weight:700;gap:.35rem;justify-content:center;min-height:36px;padding:.45rem .7rem}.upload-button{background:linear-gradient(180deg,#2997ff,#087ff5);color:white;margin-top:.6rem;width:100%}.parser-page button:disabled{cursor:not-allowed;opacity:.4}.library-heading{align-items:center;display:flex;justify-content:space-between;margin:1.1rem .25rem .45rem}.library-heading h2{font-size:.72rem;letter-spacing:.08em;margin:0;text-transform:uppercase}.library-heading span{color:var(--muted);font-size:.68rem}.document-list{display:grid;gap:.25rem;max-height:52vh;overflow-x:hidden;overflow-y:auto}.document-item{background:color-mix(in srgb,var(--surface-solid) 42%,transparent);border:1px solid transparent;border-radius:14px;display:grid;grid-template-columns:minmax(0,1fr) 28px;min-width:0;position:relative}.document-item:hover{background:color-mix(in srgb,var(--surface-solid) 72%,transparent)}.document-item.selected{background:color-mix(in srgb,var(--accent) 11%,var(--surface-solid));border-color:color-mix(in srgb,var(--accent) 35%,var(--line))}.document-select{background:transparent!important;border:0!important;border-radius:13px 0 0 13px!important;box-shadow:none!important;display:grid!important;grid-template-columns:32px minmax(0,1fr);justify-content:stretch!important;min-width:0;padding:.5rem!important;text-align:left}.file-icon{align-items:center;background:color-mix(in srgb,var(--accent) 11%,transparent);border:1px solid color-mix(in srgb,var(--accent) 20%,var(--line));border-radius:8px;color:var(--accent);display:flex;height:28px;justify-content:center;width:28px}.document-copy{align-self:center;min-width:0}.document-copy strong,.document-copy small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.document-copy strong{font-size:.7rem}.document-copy small{color:var(--muted);font-size:.57rem;margin-top:.12rem}.operation{align-self:center;background:transparent!important;border:0!important;box-shadow:none!important;height:28px;min-height:28px!important;padding:0!important;width:28px}.operation-menu{background:var(--surface-solid);border:1px solid var(--line);border-radius:11px;box-shadow:0 12px 34px #0003;padding:.3rem;position:absolute;right:.25rem;top:calc(100% - .2rem);z-index:20}.operation-menu button{color:var(--danger);min-height:32px;white-space:nowrap}.document-list>p{color:var(--muted);font-size:.68rem;text-align:center}.workspace{display:grid;grid-template-rows:auto minmax(0,1fr);max-width:100%;min-height:720px;min-width:0;overflow:hidden;width:100%}.tabs{background:color-mix(in srgb,var(--surface-solid) 35%,transparent);border-bottom:1px solid var(--line);display:flex;gap:.25rem;max-width:100%;overflow-x:auto;padding:.65rem}.tabs button{background:transparent;border-color:transparent;box-shadow:none;white-space:nowrap}.tabs button.active{background:var(--surface-solid);border-color:var(--line);box-shadow:0 5px 16px #00000012;color:var(--accent)}.tab-content{max-width:100%;min-height:0;min-width:0;overflow:hidden;width:100%}.tab-content>*{box-sizing:border-box;max-width:100%;min-width:0;width:100%}.welcome{align-items:center;display:flex;flex-direction:column;justify-content:center;min-height:620px;text-align:center}.welcome svg{color:var(--accent)}.welcome h2{margin:.7rem 0 .2rem}.welcome p{color:var(--muted);font-size:.72rem}.notice{align-items:center;display:flex;justify-content:space-between;margin-bottom:.8rem}.error-banner{color:var(--danger)}.error-banner button{height:28px;min-height:28px;padding:0;width:28px}@media(max-width:980px){.parser-shell{grid-template-columns:minmax(0,1fr)}.library{position:static}.document-list{max-height:230px}.workspace{min-height:650px}}@media(max-width:620px){.parser-hero{align-items:flex-start;flex-direction:column}.tabs{overflow-x:auto}.tabs button{white-space:nowrap}}@media(prefers-color-scheme:dark){.glass-panel{background:color-mix(in srgb,#171923 82%,transparent);box-shadow:0 28px 70px #0007,inset 0 1px 0 #ffffff14}}
</style>
