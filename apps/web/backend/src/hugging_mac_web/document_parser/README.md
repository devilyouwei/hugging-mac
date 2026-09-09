# Document Parser Backend Technical Design

## Summary

`document_parser` accepts one PDF or an ordered image set. Uploading only stores
and indexes the document; OCR starts explicitly in `markdown` mode. A separate
`structure` job derives metadata and an outline deterministically from completed
Markdown. Metadata and work state live in a dedicated SQLite
WAL database; binary inputs, page renders, raw responses, and exports use a
SHA-256 content-addressed cache. Only one MLX inference work unit is active at a
time.

The default model is `baidu/unlimited-ocr`, selected as the `4bit` variant on
the `mlx` runtime. Users may instead select the optional
`stepfun-ai/got-ocr2.0` or `zai-org/glm-ocr` `8bit` MLX models. Every choice
consumes the same SDK `document-parsing` capability. Native PDF text is a
validation and title-fallback source, not the authoritative final text.
Markdown jobs may additionally select `paddlepaddle/pp-doclayout-v3`. Its
Core ML `document-layout-analysis` result supplies semantic regions and reading
order to the page OCR prompt.

## Module structure

| File | Responsibility |
|---|---|
| `blueprint.py` | Side-effect-free app and router registration |
| `manifest.py` | Catalog metadata and the required model capability |
| `routes.py` | Multipart HTTP, result downloads, and resumable SSE transport |
| `manager.py` | Durable queue, single-lane model lifecycle, retries, cancellation, and exports |
| `store.py` | SQLite WAL schema, transactions, events, revisions, and recovery |
| `cache.py` | Atomic content-addressed objects and document reference counting |
| `pdf.py` | Encryption checks, text/bookmark metadata, and on-demand rendering |
| `pipeline.py` | OCR validation, layout conversion, and deduplication |
| `structure.py` | Deterministic Markdown metadata and outline extraction |
| `schemas.py` | Public document, job, event, and editable-outline views |

## Interfaces

The API prefix is `/api/v1/apps/document-parser`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/apps/document-parser/documents` | Store a PDF or ordered images without starting OCR |
| GET | `/api/v1/apps/document-parser/documents` | List retained documents and current jobs |
| GET | `/api/v1/apps/document-parser/models` | List compatible OCR models and their local resource status |
| GET | `/api/v1/apps/document-parser/layout-models` | List compatible page-layout models and their local resource status |
| GET | `/api/v1/apps/document-parser/documents/{document_id}` | Read document, outline, usage, and job state |
| POST | `/api/v1/apps/document-parser/documents/{document_id}/jobs` | Queue a `markdown` or `structure` job |
| GET | `/api/v1/apps/document-parser/documents/{document_id}/outputs/{kind}` | Read the latest persisted output for one mode |
| GET | `/api/v1/apps/document-parser/jobs` | List jobs, optionally filtered to active jobs or one document |
| GET | `/api/v1/apps/document-parser/jobs/{job_id}` | Read durable job progress |
| GET | `/api/v1/apps/document-parser/jobs/{job_id}/snapshot` | Atomically read a resumable draft and SSE cursor |
| GET | `/api/v1/apps/document-parser/jobs/{job_id}/events` | Stream persisted progress and warning events |
| POST | `/api/v1/apps/document-parser/jobs/{job_id}/cancel` | Stop scheduling work and discard stale inference |
| POST | `/api/v1/apps/document-parser/jobs/{job_id}/retry` | Create a retry job from a terminal/waiting job |
| PATCH | `/api/v1/apps/document-parser/documents/{document_id}/outline` | Save a revision-locked complete outline |
| GET | `/api/v1/apps/document-parser/documents/{document_id}/result` | Return the authoritative structured JSON tree |
| GET | `/api/v1/apps/document-parser/documents/{document_id}/pages/{page_index}/image` | Render or reuse a cached page preview |
| GET | `/api/v1/apps/document-parser/documents/{document_id}/result.md` | Download assembled Markdown with page provenance |
| DELETE | `/api/v1/apps/document-parser/documents/{document_id}` | Delete document state and unreferenced cache objects |

Uploads are capped by the platform-level `max_document_upload_bytes` and
`max_document_pages` settings. PDF and independent images cannot be mixed.
Encrypted PDFs fail before a document is committed.

## Pipeline and recovery

Markdown mode processes one page at a time, streams normalized layout records,
and persists a resumable draft while keeping the prior published result intact.
When page-layout analysis is enabled, `page_layout_started`, `page_layout`,
`page_layout_failed`, `layout_region_started`, and `layout_region_completed` SSE
events expose the current page, semantic region counts, ordered region boxes,
active crop, runtime, device, and timing information. Regions are cropped and
OCRed individually, then assembled in model-provided reading order. The latest
completed page layout is included in the job snapshot for refresh recovery.
Structure mode parses title, authors, abstract, keywords, and outline in five
durable stages without loading another model. OCR follows an explicit request,
validation, corrected-request loop with at most three attempts. Repetition is
detected while tokens are arriving so a looping generation is stopped before the
token limit. Key lifecycle, rendering, cache, inference, retry, cancellation,
and completion nodes emit structured logs through the shared logger utility.

On process startup, running work units and jobs return to `queued`. Completed
cache-equivalent units are reused. Missing model resources move jobs to
`waiting_for_model`; periodic checks resume them after a Models-page download.
Cancellation is observed during token streaming and between stages. A snapshot
cursor plus persisted SSE events rebuild in-flight output after a browser refresh. After a server restart,
the active job is queued again; validated page cache entries are replayed and
only the interrupted page is generated again.

## Tests and review points

Tests cover upload ordering, explicit job creation, format rejection, previews,
deletion, output persistence, native layout conversion, streaming events,
repetition retry, cache reference cleanup, and process recovery.
Review changes against these constraints:

1. Never execute more than one MLX OCR work unit concurrently.
2. Never use native PDF text as final body content.
3. Never publish output from a superseded job.
4. Every cache key must include input, model/runtime, pipeline, prompt, schema,
   revision, and parsing parameters.
5. Persist progress before emitting it and keep cancellation/recovery idempotent.
