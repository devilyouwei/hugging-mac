"""Durable single-lane MLX document parsing coordinator."""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import UploadFile
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import DocumentLayoutAnalysis, DocumentParsing
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.detection import ImageInput
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRequest
from hugging_mac_sdk.schemas.document_parsing import DocumentImage, DocumentParsingRequest
from PIL import Image, ImageDraw

from hugging_mac_web.config import WebSettings
from hugging_mac_web.shared.utils.log_util import get_logger

from .cache import ContentCache
from .pdf import inspect_image, inspect_pdf, render_image_page, render_pdf_page
from .pipeline import (
    HeadingCandidate,
    build_tree,
    filter_layout_regions,
    has_repetition,
    infer_outline_nodes,
    layout_region_task_prompt,
    merge_fragments,
    native_heading_candidates,
    normalize_layout_region_markdown,
    normalize_outline_hierarchy,
    ocr_heading_candidates,
    outline_markdown,
    section_ranges,
    streaming_ocr_markdown,
    validate_ocr_page,
    validate_outline_ocr,
)
from .schemas import (
    DocumentOutput,
    DocumentView,
    JobSnapshot,
    JobView,
    OutlineNodeInput,
    OutlineNodeView,
    ParserModelView,
    StructuredDocument,
)
from .store import DocumentParserStore, now_iso
from .structure import (
    STRUCTURE_VERSION,
    extract_abstract,
    extract_authors,
    extract_keywords,
    extract_title,
    markdown_digest,
)
from .structure import build_outline as build_markdown_outline

logger = get_logger("document_parser.manager")
DEFAULT_MODEL_ID = "baidu/unlimited-ocr"
PIPELINE_VERSION = "document-parser-v5"
PROMPT_VERSION = "document-ocr-v3"
TITLE_EXTRACTION_VERSION = "native-layout-headings-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class DocumentParserManager:
    def __init__(self, settings: WebSettings, models: ModelSdk) -> None:
        self.settings = settings
        self.models = models
        self.store = DocumentParserStore(settings.data_dir / "document-parser.sqlite3")
        self.cache = ContentCache(settings.cache_dir / "document-parser", self.store)
        self._wake = asyncio.Event()
        self._event_wakes: dict[str, asyncio.Event] = {}
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    async def start(self) -> None:
        self.store.recover()
        self.cache.clear_staging()
        self._worker = asyncio.create_task(self._run(), name="document-parser-worker")
        self._wake.set()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        self.store.close()

    async def create_document(self, uploads: list[UploadFile]) -> DocumentView:
        if not uploads:
            raise ValueError("Choose one PDF or at least one image")
        suffixes = [Path(upload.filename or "").suffix.lower() for upload in uploads]
        is_pdf = len(uploads) == 1 and suffixes[0] == ".pdf"
        if not is_pdf and (
            any(suffix == ".pdf" for suffix in suffixes)
            or any(suffix not in IMAGE_SUFFIXES for suffix in suffixes)
        ):
            raise ValueError("Use either one PDF or an ordered JPG, PNG, or WebP image set")
        source_kind = "pdf" if is_pdf else "images"
        document_id = uuid.uuid4().hex
        stamp = now_iso()
        log = logger.bind(document_id=document_id, source_kind=source_kind)
        log.info(
            "document_upload_started",
            file_count=len(uploads),
            filenames=[upload.filename for upload in uploads],
        )
        self.store.execute(
            "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                document_id,
                uploads[0].filename or "Untitled",
                source_kind,
                "ready",
                stamp,
                stamp,
            ),
        )
        staged_files: list[tuple[Path, str, int, UploadFile]] = []
        try:
            remaining = self.settings.max_document_upload_bytes
            for upload in uploads:
                staged, digest, size = await asyncio.to_thread(
                    self.cache.stage_stream, upload.file, max_bytes=remaining
                )
                remaining -= size
                staged_files.append((staged, digest, size, upload))
            pages: list[tuple[int, float, float, str, str | None, int]] = []
            source_rows: list[tuple[str, str, Path]] = []
            bookmarks: list[tuple[int, str, int]] = []
            for position, (staged, digest, size, upload) in enumerate(staged_files):
                suffix = Path(upload.filename or "").suffix.lower()
                path = self.cache.commit_staged(
                    staged, digest, size, namespace="sources", suffix=suffix
                )
                file_id = uuid.uuid4().hex
                cache_key = f"source:{position}:{digest}"
                self.store.execute(
                    "INSERT INTO source_files(id,document_id,position,filename,content_type,digest,size,cache_key) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        file_id,
                        document_id,
                        position,
                        upload.filename or path.name,
                        upload.content_type,
                        digest,
                        size,
                        cache_key,
                    ),
                )
                self.cache.ref(document_id, cache_key, digest)
                source_rows.append((file_id, digest, path))
                if is_pdf:
                    metadata = await asyncio.to_thread(
                        inspect_pdf, path, max_pages=self.settings.max_document_pages
                    )
                    bookmarks.extend(metadata.bookmarks)
                    pages.extend(
                        (
                            page.index,
                            page.width,
                            page.height,
                            page.native_text,
                            page.label,
                            page.index,
                        )
                        for page in metadata.pages
                    )
                else:
                    width, height = await asyncio.to_thread(inspect_image, path)
                    pages.append((position, width, height, "", str(position + 1), 0))
            if len(pages) > self.settings.max_document_pages:
                raise ValueError(
                    f"The document exceeds the {self.settings.max_document_pages}-page limit"
                )
            for page_index, width, height, native_text, label, source_page in pages:
                source_file_id = source_rows[0][0] if is_pdf else source_rows[page_index][0]
                self.store.execute(
                    "INSERT INTO pages(document_id,page_index,display_page,page_label,width,height,native_text,source_file_id,source_page) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        page_index,
                        page_index + 1,
                        label,
                        width,
                        height,
                        native_text,
                        source_file_id,
                        source_page,
                    ),
                )
            warnings: list[str] = []
            native_payload = json.dumps(
                [
                    {"page": page_index, "label": label, "text": native_text}
                    for page_index, _width, _height, native_text, label, _source_page in pages
                ],
                ensure_ascii=False,
            ).encode()
            native_digest, _ = self.cache.put_bytes(
                native_payload, namespace="native-text", suffix=".json"
            )
            self.cache.ref(document_id, "native-text:v1", native_digest)
            self.store.execute(
                "INSERT INTO document_hints(document_id,bookmarks_json) VALUES(?,?)",
                (document_id, json.dumps(bookmarks, ensure_ascii=False)),
            )
            self.store.execute(
                "UPDATE documents SET page_count=?, warnings_json=?, updated_at=? WHERE id=?",
                (len(pages), json.dumps(warnings, ensure_ascii=False), now_iso(), document_id),
            )
            document = self.get_document(document_id)
            log.info(
                "document_upload_completed",
                page_count=len(pages),
                disk_bytes=document.disk_bytes,
            )
            return document
        except BaseException as error:
            log.exception("document_upload_failed", error_type=type(error).__name__)
            for staged, *_ in staged_files:
                staged.unlink(missing_ok=True)
            self.cache.release_document(document_id)
            self.store.execute("DELETE FROM documents WHERE id=?", (document_id,))
            raise

    def list_documents(self) -> list[DocumentView]:
        rows = self.store.all(
            "SELECT id FROM documents WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        return [self.get_document(row["id"]) for row in rows]

    def get_document(self, document_id: str) -> DocumentView:
        row = self._document_row(document_id)
        if row.get("deleted_at"):
            raise KeyError("Document not found")
        revision = int(row["outline_revision"])
        nodes = self._outline_nodes(document_id, revision) if revision else []
        disk = self.store.one(
            "SELECT COALESCE(SUM(DISTINCT o.size),0) AS size FROM cache_refs r "
            "JOIN cache_objects o ON o.digest=r.digest WHERE r.document_id=?",
            (document_id,),
        )
        latest_jobs: dict[str, JobView | None] = {"markdown": None, "structure": None}
        for kind in latest_jobs:
            latest = self.store.one(
                "SELECT id FROM jobs WHERE document_id=? AND kind=? ORDER BY created_at DESC LIMIT 1",
                (document_id, kind),
            )
            if latest:
                latest_jobs[kind] = self.get_job(latest["id"])
        job = self.get_job(row["current_job_id"]) if row.get("current_job_id") else None
        return DocumentView(
            id=row["id"],
            name=row["name"],
            source_kind=row["source_kind"],
            status=row["status"],
            page_count=row["page_count"],
            outline_revision=revision,
            warnings=json.loads(row["warnings_json"]),
            disk_bytes=int(disk["size"] if disk else 0),
            current_job_id=row.get("current_job_id"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            outline=[OutlineNodeView.model_validate(node) for node in build_tree(nodes)],
            job=job,
            jobs=latest_jobs,
        )

    def get_job(self, job_id: str) -> JobView:
        row = self.store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            raise KeyError("Task not found")
        event_row = self.store.one(
            "SELECT COALESCE(MAX(sequence),0) AS sequence FROM job_events WHERE job_id=?",
            (job_id,),
        )
        return self._job_view(row, int(event_row["sequence"] if event_row else 0))

    async def parser_models(self) -> list[ParserModelView]:
        return await self._available_models("document-parsing")

    async def layout_models(self) -> list[ParserModelView]:
        return await self._available_models("document-layout-analysis")

    async def _available_models(self, capability: str) -> list[ParserModelView]:
        choices: list[ParserModelView] = []
        for definition in self.models.registry.list(capability=capability):
            manifest = definition.manifest
            model_id = str(manifest.model_id)
            variant = manifest.default_variant
            runtime = manifest.default_runtime or definition.supported_runtimes[0]
            ready = False
            try:
                status = await self.models.resources.status(
                    model_id,
                    variant=variant,
                    options={"model_home": self.settings.model_home},
                )
                ready = any(
                    item.runtime == runtime and item.available for item in status.runtimes
                )
            except Exception as error:
                logger.warning(
                    "document_parser_model_status_failed",
                    model_id=model_id,
                    variant=variant,
                    runtime=runtime,
                    error_type=type(error).__name__,
                )
            choices.append(
                ParserModelView(
                    model_id=model_id,
                    name=manifest.display_name,
                    description=manifest.description or "",
                    variant=variant,
                    runtime=runtime,
                    ready=ready,
                    source_url=str(manifest.source_url) if manifest.source_url else None,
                )
            )
        return choices

    @staticmethod
    def _job_view(row: dict[str, Any], event_sequence: int) -> JobView:
        total = int(row["total_units"])
        completed = int(row["completed_units"])
        return JobView(
            id=row["id"],
            document_id=row["document_id"],
            kind=row.get("kind", "markdown"),
            model_id=row["model_id"],
            model_variant=row.get("model_variant") or (
                "4bit" if row["model_id"] == DEFAULT_MODEL_ID else ""
            ),
            model_runtime=row.get("model_runtime") or "mlx",
            layout_model_id=row.get("layout_model_id"),
            layout_model_variant=row.get("layout_model_variant"),
            layout_model_runtime=row.get("layout_model_runtime"),
            state=row["state"],
            stage=row["stage"],
            completed_units=completed,
            total_units=total,
            progress=(completed / total if total else 0.0),
            cancel_requested=bool(row["cancel_requested"]),
            error=row["error"],
            event_sequence=event_sequence,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_jobs(
        self, *, active_only: bool = False, document_id: str | None = None
    ) -> list[JobView]:
        conditions: list[str] = []
        arguments: list[Any] = []
        if active_only:
            conditions.append("state IN ('queued','waiting_for_model','running')")
        if document_id is not None:
            conditions.append("document_id=?")
            arguments.append(document_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.store.all(
            f"SELECT id FROM jobs{where} ORDER BY created_at DESC",
            tuple(arguments),
        )
        return [self.get_job(row["id"]) for row in rows]

    def events(self, job_id: str, after: int) -> list[dict[str, Any]]:
        self.get_job(job_id)
        rows = self.store.all(
            "SELECT sequence,event,data_json,created_at FROM job_events WHERE job_id=? AND sequence>? ORDER BY sequence",
            (job_id, after),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row.pop("data_json"))
            result.append({**row, "data": payload})
        return result

    def snapshot(self, job_id: str) -> JobSnapshot:
        row, draft, maximum_sequence, draft_sequence = self.store.job_snapshot(job_id)
        if not row:
            raise KeyError("Task not found")
        job = self._job_view(row, maximum_sequence)
        return JobSnapshot(
            job=job,
            event_sequence=draft_sequence,
            draft=json.loads(draft["content_json"]) if draft else {},
        )

    async def wait_for_events(self, job_id: str, after: int, timeout: float) -> None:
        wake = self._event_wakes.setdefault(job_id, asyncio.Event())
        wake.clear()
        if self.events(job_id, after):
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wake.wait(), timeout=timeout)

    def cancel(self, job_id: str) -> JobView:
        job = self.get_job(job_id)
        if job.state in {"completed", "completed_with_warnings", "failed", "cancelled"}:
            return job
        self.store.execute(
            "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?", (now_iso(), job_id)
        )
        self._event(job_id, "cancelling", {"stage": job.stage})
        logger.info(
            "document_job_cancel_requested",
            job_id=job_id,
            document_id=job.document_id,
            job_kind=job.kind,
            stage=job.stage,
        )
        return self.get_job(job_id)

    def retry(self, job_id: str) -> JobView:
        old = self.get_job(job_id)
        if old.kind == "titles":
            raise ValueError("Legacy title extraction tasks cannot be retried")
        if old.state not in {"failed", "cancelled", "completed_with_warnings", "waiting_for_model"}:
            raise ValueError("Only failed, cancelled, warning, or waiting tasks can be retried")
        new_id = uuid.uuid4().hex
        revision = self._document_row(old.document_id)["outline_revision"]
        self._insert_job(
            new_id,
            old.document_id,
            revision=int(revision),
            kind=old.kind,
            model_id=old.model_id if old.kind == "markdown" else None,
            layout_model_id=old.layout_model_id if old.kind == "markdown" else None,
        )
        self.store.execute(
            "UPDATE documents SET status='queued', current_job_id=?, updated_at=? WHERE id=?",
            (new_id, now_iso(), old.document_id),
        )
        self._event(new_id, "queued", {"retry_of": job_id})
        self._wake.set()
        return self.get_job(new_id)

    def create_job(
        self,
        document_id: str,
        kind: str = "markdown",
        model_id: str | None = None,
        layout_model_id: str | None = None,
    ) -> JobView:
        document = self._document_row(document_id)
        if kind not in {"markdown", "structure"}:
            raise ValueError("Unsupported parsing mode")
        if kind == "structure" and not self.store.one(
            "SELECT 1 FROM document_outputs WHERE document_id=? AND kind='markdown' AND content<>''",
            (document_id,),
        ):
            raise ValueError("markdown_required")
        if document.get("current_job_id"):
            current = self.get_job(document["current_job_id"])
            if current.state in {"queued", "waiting_for_model", "running"}:
                raise ValueError("This document already has an active task")
        job_id = uuid.uuid4().hex
        self._insert_job(
            job_id,
            document_id,
            revision=int(document["outline_revision"]),
            kind=kind,
            model_id=model_id,
            layout_model_id=layout_model_id if kind == "markdown" else None,
        )
        self.store.execute(
            "UPDATE documents SET status='queued',current_job_id=?,updated_at=? WHERE id=?",
            (job_id, now_iso(), document_id),
        )
        queued = self.get_job(job_id)
        self._event(
            job_id,
            "queued",
            {
                "kind": kind,
                "page_count": document["page_count"],
                "model_id": queued.model_id,
                "layout_model_id": queued.layout_model_id,
            },
        )
        logger.info(
            "document_job_queued",
            job_id=job_id,
            document_id=document_id,
            job_kind=kind,
            model_id=queued.model_id,
            layout_model_id=queued.layout_model_id,
            page_count=document["page_count"],
        )
        self._wake.set()
        return self.get_job(job_id)

    def output(self, document_id: str, kind: str) -> DocumentOutput:
        self._document_row(document_id)
        if kind not in {"markdown", "structure"}:
            raise ValueError("Unsupported output kind")
        row = self.store.one(
            "SELECT * FROM document_outputs WHERE document_id=? AND kind=?",
            (document_id, kind),
        )
        structured: StructuredDocument | None = None
        content = row["content"] if row else ""
        if row and kind == "structure" and content:
            payload = json.loads(content)
            current_markdown = self.store.one(
                "SELECT digest FROM document_outputs WHERE document_id=? AND kind='markdown'",
                (document_id,),
            )
            payload["stale"] = bool(
                current_markdown
                and current_markdown.get("digest")
                and payload.get("source_markdown_digest") != current_markdown["digest"]
            )
            structured = StructuredDocument.model_validate(payload)
        return DocumentOutput(
            document_id=document_id,
            kind=kind,
            content=content if kind == "markdown" else "",
            digest=row.get("digest") if row else None,
            structured=structured,
            warnings=json.loads(row["warnings_json"]) if row else [],
            updated_at=row["updated_at"] if row else None,
        )

    def patch_outline(
        self, document_id: str, base_revision: int, inputs: list[OutlineNodeInput]
    ) -> tuple[DocumentView, JobView | None]:
        document = self._document_row(document_id)
        current = int(document["outline_revision"])
        if current != base_revision:
            raise RuntimeError(f"The outline has changed; current revision is {current}")
        ids = {node.id for node in inputs}
        if len(ids) != len(inputs) or any(
            node.parent_id and node.parent_id not in ids for node in inputs
        ):
            raise ValueError("Outline node IDs are duplicated or a parent is missing")
        if any(node.start_page >= int(document["page_count"]) for node in inputs):
            raise ValueError("An outline node starts outside the document")
        parent_by_id = {node.id: node.parent_id for node in inputs}
        for node in inputs:
            visited = {node.id}
            parent = node.parent_id
            while parent is not None:
                if parent in visited:
                    raise ValueError("The outline contains a parent cycle")
                visited.add(parent)
                parent = parent_by_id.get(parent)
        sibling_positions = [(node.parent_id, node.position) for node in inputs]
        if len(sibling_positions) != len(set(sibling_positions)):
            raise ValueError("Sibling outline positions must be unique")
        old = {node["id"]: node for node in self._outline_nodes(document_id, current)}
        new_revision = current + 1
        claimed = self.store.execute(
            "UPDATE documents SET outline_revision=?,updated_at=? "
            "WHERE id=? AND outline_revision=?",
            (new_revision, now_iso(), document_id, current),
        )
        if claimed.rowcount != 1:
            latest = self._document_row(document_id)["outline_revision"]
            raise RuntimeError(f"The outline has changed; current revision is {latest}")
        new_nodes = [node.model_dump() | {"user_edited": True} for node in inputs]
        self._save_outline(document_id, new_revision, new_nodes, "user")
        page_count = int(document["page_count"])
        old_signatures = _boundary_signatures(list(old.values()), page_count)
        new_signatures = _boundary_signatures(new_nodes, page_count)
        boundary_changed = {
            node.id
            for node in inputs
            if node.id not in old or old_signatures.get(node.id) != new_signatures.get(node.id)
        }
        for node in inputs:
            if node.id not in boundary_changed and node.id in old:
                section = self.store.one(
                    "SELECT * FROM sections WHERE document_id=? AND revision=? AND node_id=?",
                    (document_id, current, node.id),
                )
                if section:
                    self.store.execute(
                        "INSERT INTO sections(document_id,revision,node_id,markdown,fragments_json,page_spans_json,warnings_json,source_work_unit) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            document_id,
                            new_revision,
                            node.id,
                            section["markdown"],
                            section["fragments_json"],
                            section["page_spans_json"],
                            section["warnings_json"],
                            section["source_work_unit"],
                        ),
                    )
        missing_sections = {
            node.id
            for node in inputs
            if not self.store.one(
                "SELECT 1 FROM sections WHERE document_id=? AND revision=? AND node_id=?",
                (document_id, new_revision, node.id),
            )
        }
        affected = boundary_changed | missing_sections
        job = self.create_job(document_id, "markdown") if affected else None
        if not job:
            self._write_exports(document_id, new_revision)
        return self.get_document(document_id), job

    def result(self, document_id: str) -> dict[str, Any]:
        document = self._document_row(document_id)
        revision = int(document["outline_revision"])
        nodes = self._outline_nodes(document_id, revision)
        ranges = section_ranges(nodes, int(document["page_count"]))
        sections = {
            row["node_id"]: row
            for row in self.store.all(
                "SELECT * FROM sections WHERE document_id=? AND revision=?", (document_id, revision)
            )
        }
        enriched = []
        ordered = sorted(nodes, key=lambda node: (node["start_page"], node["position"]))
        for node in ordered:
            section = sections.get(node["id"])
            start, end = ranges[node["id"]]
            enriched.append(
                {
                    **node,
                    "start": {"page": start, "bbox": node.get("bbox")},
                    "end": {"page": end, "bbox": None},
                    "content": {
                        "markdown": section["markdown"] if section else "",
                        "page_fragments": (
                            [
                                {"page": int(page), "markdown": markdown}
                                for page, markdown in json.loads(section["fragments_json"]).items()
                            ]
                            if section
                            else []
                        ),
                        "page_spans": json.loads(section["page_spans_json"]) if section else [],
                        "warnings": json.loads(section["warnings_json"])
                        if section
                        else ["Not parsed yet"],
                    },
                }
            )
        return {
            "document_id": document_id,
            "page_count": document["page_count"],
            "outline_revision": revision,
            "warnings": json.loads(document["warnings_json"]),
            "outline": build_tree(enriched),
        }

    def markdown(self, document_id: str) -> str:
        output = self.store.one(
            "SELECT content FROM document_outputs WHERE document_id=? AND kind='markdown'",
            (document_id,),
        )
        if output:
            return output["content"].rstrip() + "\n"
        result = self.result(document_id)
        parts: list[str] = []

        def visit(nodes: list[dict[str, Any]]) -> None:
            for node in nodes:
                parts.append(f"{'#' * min(6, int(node['level']))} {node['title']}")
                parts.append(
                    f"<!-- pages:{node['start']['page'] + 1}-{node['end']['page'] + 1} -->"
                )
                if node["content"]["markdown"]:
                    parts.append(node["content"]["markdown"])
                visit(node["children"])

        visit(result["outline"])
        return "\n\n".join(parts).strip() + "\n"

    def delete_document(self, document_id: str) -> None:
        document = self._document_row(document_id)
        if document.get("deleted_at"):
            raise KeyError("Document not found")
        active = self.store.all(
            "SELECT id FROM jobs WHERE document_id=? AND state IN ('queued','waiting_for_model','running')",
            (document_id,),
        )
        self.store.execute(
            "UPDATE documents SET deleted_at=?,status='deleting',updated_at=? WHERE id=?",
            (now_iso(), now_iso(), document_id),
        )
        for item in active:
            self.store.execute(
                "UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?",
                (now_iso(), item["id"]),
            )
            self._event(item["id"], "cancelling", {"reason": "document_deleted"})
        if not active or not any(self.get_job(item["id"]).state == "running" for item in active):
            self._purge_document(document_id)

    async def page_preview(self, document_id: str, page_index: int) -> Path:
        document = self._document_row(document_id)
        if page_index < 0 or page_index >= int(document["page_count"]):
            raise IndexError("Page not found")
        return await self._page_image(document_id, page_index, 1400, "preview")

    async def _run(self) -> None:
        while not self._closing:
            row = self.store.one(
                "SELECT id FROM jobs WHERE state IN ('queued','waiting_for_model') "
                "ORDER BY CASE state WHEN 'queued' THEN 0 ELSE 1 END, created_at LIMIT 1"
            )
            if not row:
                self._wake.clear()
                await self._wake.wait()
                continue
            try:
                await self._process_job(row["id"])
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(
                    "document_job_unhandled_error",
                    job_id=row["id"],
                    error_type=type(error).__name__,
                )
                self._finish_failed(row["id"], error)

    async def _process_job(self, job_id: str) -> None:
        row = self.store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert row
        job_kind = row.get("kind", "markdown")
        document_id = row["document_id"]
        started = perf_counter()
        log = logger.bind(job_id=job_id, document_id=document_id, job_kind=job_kind)
        log.info("document_job_started")
        if row["cancel_requested"]:
            self._finish_cancelled(job_id)
            return
        if job_kind == "structure":
            await self._process_structure_job(job_id, document_id, started)
            return
        model_id = row["model_id"]
        model_variant = row.get("model_variant") or (
            "4bit" if model_id == DEFAULT_MODEL_ID else ""
        )
        model_runtime = row.get("model_runtime") or "mlx"
        layout_model_id = row.get("layout_model_id")
        layout_model_variant = row.get("layout_model_variant")
        layout_model_runtime = row.get("layout_model_runtime")
        log = log.bind(
            model_id=model_id,
            model_variant=model_variant,
            model_runtime=model_runtime,
            layout_model_id=layout_model_id,
        )
        log.info("document_model_status_check_started")
        status = await self.models.resources.status(
            model_id,
            variant=model_variant,
            options={"model_home": self.settings.model_home},
        )
        runtime_ready = any(
            runtime.runtime == model_runtime and runtime.available for runtime in status.runtimes
        )
        if not runtime_ready:
            self._set_job(job_id, state="waiting_for_model", stage="waiting_for_model")
            self.store.execute(
                "UPDATE documents SET status='waiting_for_model',updated_at=? WHERE id=?",
                (now_iso(), row["document_id"]),
            )
            self._event(
                job_id,
                "waiting_for_model",
                {
                    "model_id": model_id,
                    "variant": model_variant,
                    "runtime": model_runtime,
                },
            )
            log.warning("document_model_unavailable")
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=10)
            return
        if layout_model_id:
            layout_status = await self.models.resources.status(
                layout_model_id,
                variant=layout_model_variant,
                options={"model_home": self.settings.model_home},
            )
            layout_ready = any(
                runtime.runtime == layout_model_runtime and runtime.available
                for runtime in layout_status.runtimes
            )
            if not layout_ready:
                self._set_job(job_id, state="waiting_for_model", stage="waiting_for_layout_model")
                self.store.execute(
                    "UPDATE documents SET status='waiting_for_model',updated_at=? WHERE id=?",
                    (now_iso(), document_id),
                )
                self._event(
                    job_id,
                    "waiting_for_model",
                    {
                        "model_id": layout_model_id,
                        "variant": layout_model_variant,
                        "runtime": layout_model_runtime,
                        "role": "layout",
                    },
                )
                log.warning("document_layout_model_unavailable")
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=10)
                return
        self._set_job(job_id, state="running", stage="loading_model")
        self.store.execute(
            "UPDATE documents SET status='running',updated_at=? WHERE id=?",
            (now_iso(), row["document_id"]),
        )
        self._event(job_id, "progress", {"stage": "loading_model"})
        model_started = perf_counter()
        log.info("document_model_load_started")
        async with contextlib.AsyncExitStack() as stack:
            handle = await stack.enter_async_context(
                await self.models.load(
                    model_id,
                    variant=model_variant,
                    runtime=model_runtime,
                    options={"model_home": self.settings.model_home},
                )
            )
            log.info(
                "document_model_load_completed",
                elapsed_ms=round((perf_counter() - model_started) * 1000, 3),
            )
            parser = handle.require(DocumentParsing)  # type: ignore[type-abstract]
            layout_analyzer: DocumentLayoutAnalysis | None = None
            if layout_model_id:
                layout_started = perf_counter()
                log.info("document_layout_model_load_started")
                layout_handle = await stack.enter_async_context(
                    await self.models.load(
                        layout_model_id,
                        variant=layout_model_variant,
                        runtime=layout_model_runtime,
                        options={"model_home": self.settings.model_home},
                    )
                )
                layout_analyzer = layout_handle.require(DocumentLayoutAnalysis)  # type: ignore[type-abstract]
                log.info(
                    "document_layout_model_load_completed",
                    elapsed_ms=round((perf_counter() - layout_started) * 1000, 3),
                )
            if not self._draft(job_id):
                self._event(job_id, "output_reset", {"kind": job_kind, "page": -1})
            if job_kind == "titles":
                content, warnings = await self._extract_titles_only(
                    job_id, document_id, parser
                )
            else:
                content, warnings = await self._parse_full_markdown(
                    job_id, document_id, parser, layout_analyzer
                )
            if self._cancelled(job_id):
                self._finish_cancelled(job_id)
                return
            if self._is_stale(job_id, document_id, int(row["outline_revision"])):
                self._set_job(job_id, state="cancelled", stage="superseded")
                self._event(job_id, "superseded", {})
                return
            self._save_output(document_id, job_kind, content, warnings)
            self._event(
                job_id,
                "output_complete",
                {
                    "kind": job_kind,
                    "output_characters": len(content),
                    "warning_count": len(warnings),
                },
            )
            state = "completed_with_warnings" if warnings else "completed"
            self._set_job(job_id, state=state, stage="done")
            self.store.execute(
                "UPDATE documents SET status=?,updated_at=? WHERE id=?",
                (state, now_iso(), document_id),
            )
            self._event(job_id, state, {"stage": "done", "kind": job_kind})
            log.info(
                "document_job_completed",
                state=state,
                warning_count=len(warnings),
                output_characters=len(content),
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )

    async def _parse_full_markdown(
        self,
        job_id: str,
        document_id: str,
        parser: DocumentParsing,
        layout_analyzer: DocumentLayoutAnalysis | None = None,
    ) -> tuple[str, list[str]]:
        page_count = int(self._document_row(document_id)["page_count"])
        draft = self._draft(job_id)
        fragments = {int(page): value for page, value in draft.get("pages", {}).items()}
        warnings: list[str] = list(draft.get("warnings", []))
        self._set_job(
            job_id,
            stage="parsing_markdown",
            completed_units=len(fragments),
            total_units=page_count,
        )
        for page in range(page_count):
            if self._cancelled(job_id):
                break
            path: Path | None = None
            layout_payload: dict[str, Any] | None = None
            if layout_analyzer is not None:
                path = await self._page_image(document_id, page, 2400, "content")
                layout_payload, layout_warning = await self._analyze_page_layout(
                    job_id, document_id, page, page_count, path, layout_analyzer
                )
                if layout_warning and layout_warning not in warnings:
                    warnings.append(layout_warning)
            if page in fragments:
                self._progress(job_id, page + 1, page_count, "parsing_markdown")
                continue
            logger.info(
                "document_page_parse_started",
                job_id=job_id,
                document_id=document_id,
                job_kind="markdown",
                page=page + 1,
                page_count=page_count,
            )
            path = path or await self._page_image(document_id, page, 2400, "content")
            native = self.store.one(
                "SELECT native_text FROM pages WHERE document_id=? AND page_index=?",
                (document_id, page),
            )
            result: Any | None = None
            warning: str | None = None
            if layout_payload and layout_payload.get("regions"):
                page_markdown, region_warnings = await self._parse_layout_regions(
                    job_id, page, path, parser, layout_payload
                )
                warnings.extend(region_warnings)
                if page_markdown.strip():
                    result = ({page: page_markdown}, [])
            if result is None:
                result, warning = await self._request_with_feedback(
                    parser,
                    [path],
                    self._layout_aware_prompt(layout_payload),
                    lambda text, page=page, native=native: validate_ocr_page(
                        text, page=page, native_text=native["native_text"] if native else ""
                    ),
                    job_id=job_id,
                    kind="markdown",
                    start=page,
                    end=page,
                )
            if result:
                fragments.update(result[0])
                warnings.extend(result[1])
            if warning:
                warnings.append(warning)
            persisted_draft = self._draft(job_id)
            persisted_draft.update({"pages": fragments, "warnings": warnings})
            self._save_draft(job_id, "markdown", persisted_draft)
            self._progress(job_id, page + 1, page_count, "parsing_markdown")
            logger.info(
                "document_page_parse_completed",
                job_id=job_id,
                document_id=document_id,
                job_kind="markdown",
                page=page + 1,
                output_characters=len(fragments.get(page, "")),
                warning=warning,
            )
        return self._join_pages(fragments), warnings

    async def _parse_layout_regions(
        self,
        job_id: str,
        page: int,
        page_path: Path,
        parser: DocumentParsing,
        layout: dict[str, Any],
    ) -> tuple[str, list[str]]:
        raw_regions = layout.get("regions", [])
        regions = [
            region
            for region in filter_layout_regions(raw_regions)
            if layout_region_task_prompt(str(region.get("label", "content"))) is not None
        ]
        if not regions:
            return "", []
        logger.info(
            "document_layout_regions_prepared",
            job_id=job_id,
            page=page + 1,
            detected_regions=len(raw_regions),
            parsed_regions=len(regions),
            discarded_regions=len(raw_regions) - len(regions),
        )
        assembled = ""
        warnings: list[str] = []
        self._event(job_id, "output_reset", {"kind": "markdown", "page": page, "attempt": 1})
        with tempfile.TemporaryDirectory(dir=self.cache.root / "staging") as temp:
            for index, region in enumerate(regions):
                if self._cancelled(job_id):
                    break
                box = region.get("box", {})
                crop_path = Path(temp) / f"region-{index:03d}.jpg"
                try:
                    await asyncio.to_thread(
                        self._crop_layout_region,
                        page_path,
                        crop_path,
                        box,
                        region.get("polygon", []),
                    )
                except (OSError, ValueError) as error:
                    warning = f"Page {page + 1} region {index + 1} could not be cropped"
                    warnings.append(warning)
                    logger.warning(
                        "document_layout_region_crop_failed",
                        job_id=job_id,
                        page=page + 1,
                        region=index + 1,
                        error_type=type(error).__name__,
                    )
                    continue
                label = str(region.get("label", "content"))
                task_prompt = layout_region_task_prompt(label)
                assert task_prompt is not None
                self._event(
                    job_id,
                    "layout_region_started",
                    {
                        "page": page,
                        "display_page": page + 1,
                        "region": index + 1,
                        "region_count": len(regions),
                        "label": label,
                        "box": box,
                    },
                )
                result, warning = await self._request_with_feedback(
                    parser,
                    [crop_path],
                    task_prompt,
                    lambda text, page=page: validate_ocr_page(text, page=page, native_text=""),
                    job_id=job_id,
                    kind="layout_region",
                    start=page,
                    end=page,
                    section_id=f"region:{index}",
                )
                if warning:
                    warnings.append(warning)
                if not result:
                    continue
                region_markdown = normalize_layout_region_markdown(
                    label, result[0].get(page, "")
                )
                if not region_markdown:
                    continue
                assembled = merge_fragments(assembled, region_markdown)
                self._event(
                    job_id,
                    "output_commit",
                    {
                        "kind": "markdown",
                        "page": page,
                        "markdown": assembled,
                        "layout_region": index + 1,
                    },
                )
                self._event(
                    job_id,
                    "layout_region_completed",
                    {
                        "page": page,
                        "display_page": page + 1,
                        "region": index + 1,
                        "region_count": len(regions),
                        "label": label,
                        "output_characters": len(region_markdown),
                    },
                )
        return assembled, warnings

    @staticmethod
    def _crop_layout_region(
        source: Path,
        target: Path,
        box: dict[str, Any],
        polygon: list[dict[str, Any]] | None = None,
    ) -> None:
        with Image.open(source) as image:
            left = max(0, min(image.width - 1, int(float(box.get("x1", 0)))))
            top = max(0, min(image.height - 1, int(float(box.get("y1", 0)))))
            right = max(left + 1, min(image.width, int(float(box.get("x2", image.width)))))
            bottom = max(top + 1, min(image.height, int(float(box.get("y2", image.height)))))
            if right <= left or bottom <= top:
                raise ValueError("The layout region is empty")
            crop = image.crop((left, top, right, bottom)).convert("RGB")
            points = [
                (float(point["x"]) - left, float(point["y"]) - top)
                for point in (polygon or [])
                if "x" in point and "y" in point
            ]
            if len(points) >= 3:
                mask = Image.new("L", crop.size, 0)
                ImageDraw.Draw(mask).polygon(points, fill=255)
                crop = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
            crop.save(target, "JPEG", quality=95)

    async def _analyze_page_layout(
        self,
        job_id: str,
        document_id: str,
        page: int,
        page_count: int,
        path: Path,
        analyzer: DocumentLayoutAnalysis,
    ) -> tuple[dict[str, Any] | None, str | None]:
        cached = self.store.one(
            "SELECT result_json FROM page_layouts WHERE job_id=? AND page_index=?",
            (job_id, page),
        )
        if cached:
            payload = json.loads(cached["result_json"])
            payload["cached"] = True
            draft = self._draft(job_id)
            draft["current_layout"] = payload
            self._save_draft(job_id, "markdown", draft)
            self._event(job_id, "page_layout", payload)
            return payload, None

        job = self.get_job(job_id)
        self._event(
            job_id,
            "page_layout_started",
            {
                "page": page,
                "display_page": page + 1,
                "page_count": page_count,
                "model_id": job.layout_model_id,
            },
        )
        started = perf_counter()
        logger.info(
            "document_page_layout_started",
            job_id=job_id,
            document_id=document_id,
            page=page + 1,
            page_count=page_count,
            layout_model_id=job.layout_model_id,
        )
        try:
            response = await analyzer.analyze_layout(
                DocumentLayoutRequest(image=ImageInput(path=path), confidence=0.3)
            )
            region_counts: dict[str, int] = {}
            for region in response.regions:
                region_counts[region.label] = region_counts.get(region.label, 0) + 1
            payload = {
                "page": page,
                "display_page": page + 1,
                "page_count": page_count,
                "model_id": response.model_id,
                "runtime": response.runtime,
                "device": response.device,
                "image_size": response.image_size.model_dump(mode="json"),
                "region_count": len(response.regions),
                "region_counts": region_counts,
                "regions": [region.model_dump(mode="json") for region in response.regions],
                "timings": response.timings.model_dump(mode="json"),
                "cached": False,
            }
            self.store.execute(
                "INSERT OR REPLACE INTO page_layouts(job_id,page_index,model_id,result_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (job_id, page, response.model_id, json.dumps(payload), now_iso()),
            )
            draft = self._draft(job_id)
            draft["current_layout"] = payload
            self._save_draft(job_id, "markdown", draft)
            self._event(job_id, "page_layout", payload)
            logger.info(
                "document_page_layout_completed",
                job_id=job_id,
                document_id=document_id,
                page=page + 1,
                region_count=len(response.regions),
                region_counts=region_counts,
                elapsed_ms=round((perf_counter() - started) * 1000, 3),
            )
            return payload, None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            warning = f"Page {page + 1} layout analysis failed: {error}"
            self._event(
                job_id,
                "page_layout_failed",
                {"page": page, "display_page": page + 1, "message": warning},
            )
            logger.exception(
                "document_page_layout_failed",
                job_id=job_id,
                document_id=document_id,
                page=page + 1,
                error_type=type(error).__name__,
            )
            return None, warning

    @staticmethod
    def _layout_aware_prompt(layout: dict[str, Any] | None) -> str:
        if not layout:
            return "document parsing."
        regions = layout.get("regions", [])
        lines: list[str] = []
        for region in regions:
            box = region.get("box", {})
            lines.append(
                f"{int(region.get('order', len(lines))) + 1}. {region.get('label', 'content')} "
                f"[{box.get('x1', 0):.0f},{box.get('y1', 0):.0f},"
                f"{box.get('x2', 0):.0f},{box.get('y2', 0):.0f}]"
            )
        layout_text = "\n".join(lines)
        return (
            "Parse this document page into complete Markdown. Follow the detected reading "
            "order below. Preserve headings, paragraphs, lists, tables, formulas, figure "
            "captions, and footnotes; do not emit the coordinates or explain the layout.\n\n"
            f"Detected regions in reading order:\n{layout_text}"
        )

    async def _process_structure_job(
        self, job_id: str, document_id: str, started: float
    ) -> None:
        markdown_row = self.store.one(
            "SELECT content,digest FROM document_outputs WHERE document_id=? AND kind='markdown'",
            (document_id,),
        )
        if not markdown_row or not markdown_row["content"]:
            raise ValueError("markdown_required")
        source = markdown_row["content"]
        source_digest = markdown_row.get("digest") or markdown_digest(source)
        stages = (
            "extracting_title",
            "extracting_authors",
            "extracting_abstract",
            "extracting_keywords",
            "building_outline",
        )
        existing = self._draft(job_id)
        payload: dict[str, Any] = existing.get(
            "structured",
            {
                "document_id": document_id,
                "source_markdown_digest": source_digest,
                "stale": False,
                "title": None,
                "authors": [],
                "affiliations": [],
                "abstract": None,
                "keywords": [],
                "outline": [],
                "warnings": [],
                "updated_at": None,
                "parser_version": STRUCTURE_VERSION,
            },
        )
        completed = min(int(self.get_job(job_id).completed_units), len(stages))
        self._set_job(
            job_id,
            state="running",
            stage=stages[completed] if completed < len(stages) else "building_outline",
            total_units=len(stages),
        )
        self.store.execute(
            "UPDATE documents SET status='running',updated_at=? WHERE id=?",
            (now_iso(), document_id),
        )
        for index, stage in enumerate(stages):
            if index < completed:
                continue
            if self._cancelled(job_id):
                self._finish_cancelled(job_id)
                return
            self._set_job(job_id, stage=stage)
            self._event(job_id, "structure_field_started", {"stage": stage, "field": stage})
            logger.info(
                "document_structure_field_started",
                job_id=job_id,
                document_id=document_id,
                stage=stage,
                source_markdown_digest=source_digest,
            )
            if stage == "extracting_title":
                payload["title"] = extract_title(source)
                if not payload["title"]:
                    payload["warnings"].append("A document title could not be identified.")
            elif stage == "extracting_authors":
                authors, affiliations, warnings = extract_authors(source, payload.get("title"))
                payload["authors"] = authors
                payload["affiliations"] = affiliations
                payload["warnings"] = list(dict.fromkeys([*payload["warnings"], *warnings]))
                if not authors:
                    payload["warnings"].append("Author information could not be identified.")
            elif stage == "extracting_abstract":
                payload["abstract"] = extract_abstract(source)
                if not payload["abstract"]:
                    payload["warnings"].append("An abstract could not be identified.")
            elif stage == "extracting_keywords":
                payload["keywords"] = extract_keywords(source)
            else:
                payload["outline"] = build_markdown_outline(source, payload.get("title"))
            payload["warnings"] = list(dict.fromkeys(payload["warnings"]))
            payload["updated_at"] = now_iso()
            self._save_draft(job_id, "structure", {"structured": payload})
            self._event(
                job_id,
                "structure_snapshot",
                {"kind": "structure", "stage": stage, "structured": payload},
            )
            self._progress(job_id, index + 1, len(stages), stage)
            logger.info(
                "document_structure_field_completed",
                job_id=job_id,
                document_id=document_id,
                stage=stage,
                completed_units=index + 1,
                total_units=len(stages),
            )
            await asyncio.sleep(0)
        if self._cancelled(job_id):
            self._finish_cancelled(job_id)
            return
        content = json.dumps(payload, ensure_ascii=False)
        warnings = list(payload["warnings"])
        self._save_output(document_id, "structure", content, warnings)
        self._event(
            job_id,
            "output_complete",
            {"kind": "structure", "warning_count": len(warnings)},
        )
        state = "completed_with_warnings" if warnings else "completed"
        self._set_job(job_id, state=state, stage="done")
        self.store.execute(
            "UPDATE documents SET status=?,updated_at=? WHERE id=?",
            (state, now_iso(), document_id),
        )
        self._event(job_id, state, {"stage": "done", "kind": "structure"})
        logger.info(
            "document_job_completed",
            job_id=job_id,
            document_id=document_id,
            job_kind="structure",
            state=state,
            warning_count=len(warnings),
            elapsed_ms=round((perf_counter() - started) * 1000, 3),
        )

    async def _extract_titles_only(
        self, job_id: str, document_id: str, parser: DocumentParsing
    ) -> tuple[str, list[str]]:
        page_count = int(self._document_row(document_id)["page_count"])
        self._set_job(
            job_id, stage="extracting_titles", completed_units=0, total_units=page_count
        )
        candidates: list[HeadingCandidate] = []
        warnings: list[str] = []
        for page in range(page_count):
            if self._cancelled(job_id):
                break
            logger.info(
                "document_page_parse_started",
                job_id=job_id,
                document_id=document_id,
                job_kind="titles",
                page=page + 1,
                page_count=page_count,
            )
            path = await self._page_image(document_id, page, 1400, "outline")
            result, warning = await self._request_with_feedback(
                parser,
                [path],
                "document parsing.",
                lambda text, page=page: validate_outline_ocr(text, page=page),
                job_id=job_id,
                kind="titles",
                start=page,
                end=page,
            )
            previous_count = len(candidates)
            resolved = result or []
            candidate_source = "native-layout"
            if result is None:
                native = self.store.one(
                    "SELECT native_text FROM pages WHERE document_id=? AND page_index=?",
                    (document_id, page),
                )
                fallback = native_heading_candidates(
                    native["native_text"] if native else "", page=page
                )
                resolved = fallback
                candidate_source = "native-text"
            candidates = normalize_outline_hierarchy(candidates + resolved)
            for candidate in candidates[previous_count:]:
                self._event(
                    job_id,
                    "outline_candidate",
                    {**dataclasses.asdict(candidate), "source": candidate_source},
                )
            if warning:
                warnings.append(warning)
            content = self._titles_markdown(candidates)
            self._save_output(document_id, "titles", content, warnings)
            self._event(
                job_id,
                "output_snapshot",
                {"kind": "titles", "page": page, "content": content},
            )
            self._progress(job_id, page + 1, page_count, "extracting_titles")
            logger.info(
                "document_page_parse_completed",
                job_id=job_id,
                document_id=document_id,
                job_kind="titles",
                page=page + 1,
                title_count=len(candidates) - previous_count,
                cumulative_title_count=len(candidates),
                warning=warning,
            )
            logger.info(
                "document_outline_state_advanced",
                job_id=job_id,
                document_id=document_id,
                page=page + 1,
                previous_title_count=previous_count,
                appended_title_count=len(candidates) - previous_count,
                cumulative_title_count=len(candidates),
                source=candidate_source,
            )
        return self._titles_markdown(candidates), warnings

    def _save_output(
        self, document_id: str, kind: str, content: str, warnings: list[str]
    ) -> None:
        digest = markdown_digest(content)
        self.store.execute(
            "INSERT OR REPLACE INTO document_outputs(document_id,kind,content,digest,warnings_json,updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                document_id,
                kind,
                content,
                digest,
                json.dumps(warnings, ensure_ascii=False),
                now_iso(),
            ),
        )

    def _draft(self, job_id: str) -> dict[str, Any]:
        row = self.store.one("SELECT content_json FROM job_drafts WHERE job_id=?", (job_id,))
        return json.loads(row["content_json"]) if row else {}

    def _save_draft(self, job_id: str, kind: str, value: dict[str, Any]) -> None:
        event = self.store.one(
            "SELECT COALESCE(MAX(sequence),0) AS sequence FROM job_events WHERE job_id=?",
            (job_id,),
        )
        self.store.execute(
            "INSERT OR REPLACE INTO job_drafts(job_id,kind,content_json,event_sequence,updated_at) VALUES(?,?,?,?,?)",
            (
                job_id,
                kind,
                json.dumps(value, ensure_ascii=False),
                int(event["sequence"] if event else 0),
                now_iso(),
            ),
        )

    @staticmethod
    def _join_pages(fragments: dict[int, str]) -> str:
        return "\n\n".join(
            f"<!-- Page {page + 1} -->\n\n{fragments[page]}"
            for page in sorted(fragments)
            if fragments[page].strip()
        )

    @staticmethod
    def _titles_markdown(candidates: list[HeadingCandidate]) -> str:
        return outline_markdown(candidates)

    async def _discover_outline(
        self, job_id: str, document_id: str, parser: DocumentParsing
    ) -> int:
        page_count = self._document_row(document_id)["page_count"]
        windows = _windows(page_count, size=1, overlap=0)
        self._set_job(job_id, stage="discovering_outline", total_units=len(windows))
        candidates: list[HeadingCandidate] = []
        for index, (start, end) in enumerate(windows):
            if self._cancelled(job_id) or self._is_stale(job_id, document_id, 0):
                return 0
            paths = [
                await self._page_image(document_id, page, 1400, "outline")
                for page in range(start, end + 1)
            ]
            hints = self.store.all(
                "SELECT page_index,native_text FROM pages WHERE document_id=? AND page_index BETWEEN ? AND ?",
                (document_id, start, end),
            )
            prompt = self._outline_prompt(start, end, hints, [])
            result, warning = await self._request_with_feedback(
                parser,
                paths,
                prompt,
                lambda text, page=start: validate_outline_ocr(text, page=page),
                job_id=job_id,
                kind="outline",
                start=start,
                end=end,
            )
            resolved = result or []
            if not resolved and hints:
                resolved = native_heading_candidates(hints[0]["native_text"], page=start)
                for candidate in resolved:
                    self._event(
                        job_id,
                        "outline_candidate",
                        {**dataclasses.asdict(candidate), "source": "native-text"},
                    )
            candidates.extend(resolved)
            if warning:
                self._append_document_warning(document_id, warning)
            self._progress(job_id, index + 1, len(windows), "discovering_outline")
        if self._is_stale(job_id, document_id, 0):
            return 0
        nodes = infer_outline_nodes(candidates, page_count=page_count)
        claimed = self.store.execute(
            "UPDATE documents SET outline_revision=1,updated_at=? "
            "WHERE id=? AND outline_revision=0 AND current_job_id=?",
            (now_iso(), document_id, job_id),
        )
        if claimed.rowcount != 1:
            return 0
        self._save_outline(document_id, 1, nodes, "model")
        self.store.execute("UPDATE jobs SET outline_revision=1 WHERE id=?", (job_id,))
        self._event(job_id, "outline_ready", {"revision": 1, "node_count": len(nodes)})
        return 1

    async def _extract_sections(
        self,
        job_id: str,
        document_id: str,
        revision: int,
        parser: DocumentParsing,
        targets: list[str] | None,
    ) -> None:
        nodes = self._outline_nodes(document_id, revision)
        page_count = self._document_row(document_id)["page_count"]
        ranges = section_ranges(nodes, page_count)
        ordered_nodes = sorted(
            nodes,
            key=lambda item: (
                int(item["start_page"]),
                float(item["bbox"][1]) if item.get("bbox") else -1,
            ),
        )
        next_nodes = {
            node["id"]: ordered_nodes[index + 1] if index + 1 < len(ordered_nodes) else None
            for index, node in enumerate(ordered_nodes)
        }
        selected = [node for node in nodes if targets is None or node["id"] in targets]
        units: list[tuple[dict[str, Any], tuple[int, int]]] = []
        for node in selected:
            range_start, range_end = ranges[node["id"]]
            units.extend((node, (page, page)) for page in range(range_start, range_end + 1))
        self._set_job(
            job_id, stage="extracting_sections", completed_units=0, total_units=len(units)
        )
        fragments: dict[str, dict[int, str]] = {node["id"]: {} for node in selected}
        warnings: dict[str, list[str]] = {node["id"]: [] for node in selected}
        for completed, (node, (start, end)) in enumerate(units, start=1):
            if self._cancelled(job_id) or self._is_stale(job_id, document_id, revision):
                return
            paths = [
                await self._section_page_image(document_id, page, node, next_nodes[node["id"]])
                for page in range(start, end + 1)
            ]
            prompt = self._content_prompt(node, next_nodes[node["id"]], start, end, "")
            native_rows = self.store.all(
                "SELECT page_index,native_text FROM pages "
                "WHERE document_id=? AND page_index BETWEEN ? AND ?",
                (document_id, start, end),
            )
            native_text = {int(row["page_index"]): row["native_text"] for row in native_rows}
            result, warning = await self._request_with_feedback(
                parser,
                paths,
                prompt,
                lambda text, page=start, native=native_text: validate_ocr_page(
                    text, page=page, native_text=native.get(page, "")
                ),
                job_id=job_id,
                kind="content",
                start=start,
                end=end,
                section_id=node["id"],
            )
            if result:
                page_fragments, result_warnings = result
                for page, fragment in page_fragments.items():
                    prior = fragments[node["id"]].get(page, "")
                    fragments[node["id"]][page] = fragment if len(fragment) >= len(prior) else prior
                warnings[node["id"]].extend(result_warnings)
            if warning:
                warnings[node["id"]].append(warning)
            self._progress(job_id, completed, len(units), "extracting_sections")
        for node in selected:
            section_text = ""
            for page in sorted(fragments[node["id"]]):
                section_text = merge_fragments(section_text, fragments[node["id"]][page])
            start, end = ranges[node["id"]]
            source_unit = self.store.one(
                "SELECT id FROM work_units WHERE job_id=? AND section_id=? "
                "AND state='succeeded' ORDER BY updated_at DESC LIMIT 1",
                (job_id, node["id"]),
            )
            self.store.execute(
                "INSERT OR REPLACE INTO sections(document_id,revision,node_id,markdown,fragments_json,page_spans_json,warnings_json,source_work_unit) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    revision,
                    node["id"],
                    section_text,
                    json.dumps(fragments[node["id"]], ensure_ascii=False),
                    json.dumps([{"start": start, "end": end}]),
                    json.dumps(warnings[node["id"]], ensure_ascii=False),
                    source_unit["id"] if source_unit else None,
                ),
            )

    async def _request_with_feedback(
        self,
        parser: DocumentParsing,
        paths: list[Path],
        prompt: str,
        validator: Any,
        *,
        job_id: str,
        kind: str,
        start: int,
        end: int,
        section_id: str | None = None,
    ) -> tuple[Any | None, str | None]:
        previous = ""
        errors: list[str] = []
        unit_id = uuid.uuid4().hex
        image_fingerprint = ":".join(
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        )
        job_revision = self.store.one(
            "SELECT outline_revision,model_id,model_variant,model_runtime FROM jobs WHERE id=?",
            (job_id,),
        )
        model_id = job_revision["model_id"] if job_revision else DEFAULT_MODEL_ID
        model_variant = (
            job_revision.get("model_variant") if job_revision else None
        ) or ("4bit" if model_id == DEFAULT_MODEL_ID else "")
        model_runtime = (
            job_revision.get("model_runtime") if job_revision else None
        ) or "mlx"
        cache_key = hashlib.sha256(
            (
                f"{image_fingerprint}|{model_id}|{model_variant}|{model_runtime}|"
                f"{PIPELINE_VERSION}|model-internal-config|max8192|"
                f"{PROMPT_VERSION}|title-extraction:"
                f"{TITLE_EXTRACTION_VERSION if kind == 'titles' else 'n/a'}|"
                f"revision:{job_revision['outline_revision'] if job_revision else 0}|"
                f"{kind}|{start}|{end}|{prompt}"
            ).encode()
        ).hexdigest()
        cached = self.store.one(
            "SELECT result_json FROM work_units "
            "WHERE cache_key=? AND state='succeeded' AND result_json IS NOT NULL "
            "ORDER BY updated_at DESC LIMIT 1",
            (cache_key,),
        )
        if cached:
            stamp = now_iso()
            outline_revision = int(job_revision["outline_revision"]) if job_revision else 0
            self.store.execute(
                "INSERT INTO work_units(id,job_id,kind,page_start,page_end,section_id,prompt_type,cache_key,attempt,state,outline_revision,result_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,0,'succeeded',?,?,?,?)",
                (
                    unit_id,
                    job_id,
                    kind,
                    start,
                    end,
                    section_id,
                    kind,
                    cache_key,
                    outline_revision,
                    cached["result_json"],
                    stamp,
                    stamp,
                ),
            )
            normalized = json.loads(cached["result_json"])
            self._event(job_id, "cache_hit", {"kind": kind, "start": start, "end": end})
            if kind in {"outline", "titles"}:
                cached_candidates = [HeadingCandidate(**item) for item in normalized]
                for candidate in cached_candidates:
                    self._event(
                        job_id,
                        "outline_candidate",
                        {**dataclasses.asdict(candidate), "cached": True},
                    )
                return cached_candidates, None
            fragments = {int(page): text for page, text in normalized[0].items()}
            if kind == "content" and section_id:
                self._event(
                    job_id,
                    "section_content_reset",
                    {"section_id": section_id, "page": start, "attempt": 0, "cached": True},
                )
                for page, markdown in fragments.items():
                    self._event(
                        job_id,
                        "section_content_commit",
                        {
                            "section_id": section_id,
                            "page": page,
                            "markdown": markdown,
                            "cached": True,
                        },
                    )
            elif kind == "markdown":
                self._event(
                    job_id,
                    "output_reset",
                    {"kind": "markdown", "page": start, "attempt": 0, "cached": True},
                )
                for page, markdown in fragments.items():
                    self._event(
                        job_id,
                        "output_commit",
                        {
                            "kind": "markdown",
                            "page": page,
                            "markdown": markdown,
                            "cached": True,
                        },
                    )
            logger.info(
                "document_ocr_cache_hit",
                job_id=job_id,
                job_kind=kind,
                page_start=start + 1,
                page_end=end + 1,
            )
            return (fragments, list(normalized[1])), None
        stamp = now_iso()
        outline_revision = int(job_revision["outline_revision"]) if job_revision else 0
        self.store.execute(
            "INSERT INTO work_units(id,job_id,kind,page_start,page_end,section_id,prompt_type,cache_key,attempt,state,outline_revision,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,0,'running',?,?,?)",
            (
                unit_id,
                job_id,
                kind,
                start,
                end,
                section_id,
                kind,
                cache_key,
                outline_revision,
                stamp,
                stamp,
            ),
        )
        for attempt in range(1, 4):
            if self._cancelled(job_id):
                break
            request_prompt = prompt
            if errors and kind != "titles":
                request_prompt = (
                    f"{prompt}\n\nThe previous response failed validation: {errors[-1]}. "
                    "Return a corrected response that follows the original schema exactly. "
                    "Do not explain the correction. Previous invalid response:\n"
                    f"{previous[-12_000:]}"
                )
            request = DocumentParsingRequest(
                images=tuple(DocumentImage(path=path) for path in paths),
                prompt=request_prompt,
                max_tokens=8192,
            )
            stream_document = getattr(parser, "stream_document", None)
            previous = ""
            streamed_markdown = ""
            pending_delta = ""
            emitted_candidates: set[tuple[str, int, int]] = set()
            logger.info(
                "document_ocr_attempt_started",
                job_id=job_id,
                job_kind=kind,
                page_start=start + 1,
                page_end=end + 1,
                attempt=attempt,
            )
            if kind in {"outline", "titles"}:
                self._event(
                    job_id,
                    "outline_stream_window",
                    {"start": start, "end": end, "attempt": attempt},
                )
            elif kind == "content" and section_id:
                self._event(
                    job_id,
                    "section_content_reset",
                    {"section_id": section_id, "page": start, "attempt": attempt},
                )
            elif kind == "markdown":
                self._event(
                    job_id,
                    "output_reset",
                    {"kind": "markdown", "page": start, "attempt": attempt},
                )
            if callable(stream_document):
                async for event in stream_document(request):
                    previous += event.delta
                    if not event.delta:
                        continue
                    if has_repetition(previous):
                        self._event(
                            job_id,
                            "generation_guard",
                            {
                                "kind": kind,
                                "start": start,
                                "end": end,
                                "attempt": attempt,
                                "reason": "repetition",
                            },
                        )
                        logger.warning(
                            "document_ocr_generation_guard_triggered",
                            job_id=job_id,
                            job_kind=kind,
                            page=start + 1,
                            attempt=attempt,
                            reason="repetition",
                        )
                        break
                    if self._cancelled(job_id):
                        logger.info(
                            "document_ocr_stream_cancelled",
                            job_id=job_id,
                            job_kind=kind,
                            page=start + 1,
                            attempt=attempt,
                        )
                        break
                    if kind in {"outline", "titles"}:
                        for candidate in ocr_heading_candidates(
                            previous, page=start, complete_only=True
                        ):
                            identity = (candidate.title.casefold(), candidate.page, candidate.level)
                            if identity in emitted_candidates:
                                continue
                            emitted_candidates.add(identity)
                            self._event(
                                job_id,
                                "outline_candidate",
                                {**dataclasses.asdict(candidate), "attempt": attempt},
                            )
                    elif (kind == "content" and section_id) or kind == "markdown":
                        markdown = streaming_ocr_markdown(previous)
                        if markdown.startswith(streamed_markdown):
                            pending_delta = markdown[len(streamed_markdown) :]
                        elif markdown != streamed_markdown:
                            if kind == "markdown":
                                self._event(
                                    job_id,
                                    "output_reset",
                                    {"kind": "markdown", "page": start, "attempt": attempt},
                                )
                            else:
                                self._event(
                                    job_id,
                                    "section_content_reset",
                                    {"section_id": section_id, "page": start, "attempt": attempt},
                                )
                            streamed_markdown = ""
                            pending_delta = markdown
                        if pending_delta and (
                            len(pending_delta) >= 96 or pending_delta.endswith("\n")
                        ):
                            if kind == "markdown":
                                self._event(
                                    job_id,
                                    "output_delta",
                                    {
                                        "kind": "markdown",
                                        "page": start,
                                        "delta": pending_delta,
                                        "attempt": attempt,
                                    },
                                )
                            else:
                                self._event(
                                    job_id,
                                    "section_content_delta",
                                    {
                                        "section_id": section_id,
                                        "page": start,
                                        "delta": pending_delta,
                                        "attempt": attempt,
                                    },
                                )
                            streamed_markdown += pending_delta
                            pending_delta = ""
            else:
                response = await parser.parse_document(request)
                previous = response.text
            if (kind == "content" and section_id) or kind == "markdown":
                markdown = streaming_ocr_markdown(previous)
                remaining = markdown[len(streamed_markdown) :] if markdown.startswith(streamed_markdown) else markdown
                if remaining:
                    if kind == "markdown":
                        self._event(
                            job_id,
                            "output_delta",
                            {
                                "kind": "markdown",
                                "page": start,
                                "delta": remaining,
                                "attempt": attempt,
                            },
                        )
                    else:
                        self._event(
                            job_id,
                            "section_content_delta",
                            {
                                "section_id": section_id,
                                "page": start,
                                "delta": remaining,
                                "attempt": attempt,
                            },
                        )
            job = self.get_job(job_id)
            if self._cancelled(job_id) or self._is_stale(job_id, job.document_id, outline_revision):
                self.store.execute(
                    "UPDATE work_units SET state='cancelled',updated_at=? WHERE id=?",
                    (now_iso(), unit_id),
                )
                return None, None
            raw_digest, _ = self.cache.put_bytes(
                previous.encode(), namespace="ocr-raw", suffix=".txt"
            )
            self.cache.ref(self.get_job(job_id).document_id, f"ocr:{unit_id}:{attempt}", raw_digest)
            try:
                result = validator(previous)
                self.store.execute(
                    "UPDATE work_units SET attempt=?,state='succeeded',result_json=?,updated_at=? WHERE id=?",
                    (
                        attempt,
                        json.dumps(_jsonable(result), ensure_ascii=False),
                        now_iso(),
                        unit_id,
                    ),
                )
                normalized_payload = json.dumps(
                    _jsonable(result), ensure_ascii=False, sort_keys=True
                ).encode()
                normalized_digest, _ = self.cache.put_bytes(
                    normalized_payload, namespace="ocr-normalized", suffix=".json"
                )
                self.cache.ref(
                    self.get_job(job_id).document_id,
                    f"ocr-normalized:{unit_id}",
                    normalized_digest,
                )
                if kind == "content" and section_id:
                    fragments = result[0]
                    for page, markdown in fragments.items():
                        self._event(
                            job_id,
                            "section_content_commit",
                            {
                                "section_id": section_id,
                                "page": page,
                                "markdown": markdown,
                                "attempt": attempt,
                            },
                        )
                elif kind == "markdown":
                    fragments = result[0]
                    for page, markdown in fragments.items():
                        self._event(
                            job_id,
                            "output_commit",
                            {
                                "kind": "markdown",
                                "page": page,
                                "markdown": markdown,
                                "attempt": attempt,
                            },
                        )
                logger.info(
                    "document_ocr_attempt_completed",
                    job_id=job_id,
                    job_kind=kind,
                    page_start=start + 1,
                    page_end=end + 1,
                    attempt=attempt,
                    output_characters=len(previous),
                )
                return result, None
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                errors.append(str(error))
                self.store.execute(
                    "UPDATE work_units SET attempt=?,error=?,updated_at=? WHERE id=?",
                    (attempt, str(error)[:1000], now_iso(), unit_id),
                )
                logger.warning(
                    "document_ocr_validation_failed",
                    job_id=job_id,
                    job_kind=kind,
                    page_start=start + 1,
                    page_end=end + 1,
                    attempt=attempt,
                    validation_error=str(error),
                )
        warning = (
            f"Pages {start + 1}–{end + 1} could not be parsed after three attempts"
        )
        self.store.execute(
            "UPDATE work_units SET state='failed',error=?,updated_at=? WHERE id=?",
            ("; ".join(errors)[-2000:], now_iso(), unit_id),
        )
        self._event(job_id, "warning", {"message": warning, "validation_errors": errors})
        return None, warning

    async def _page_image(
        self, document_id: str, page_index: int, edge: int, namespace: str
    ) -> Path:
        page = self.store.one(
            "SELECT p.*,s.digest,s.cache_key FROM pages p JOIN source_files s ON s.id=p.source_file_id "
            "WHERE p.document_id=? AND p.page_index=?",
            (document_id, page_index),
        )
        assert page
        key = f"render:{namespace}:{page['digest']}:{page['source_page']}:{edge}:v1"
        ref = self.store.one(
            "SELECT o.path FROM cache_refs r JOIN cache_objects o ON o.digest=r.digest "
            "WHERE r.document_id=? AND r.cache_key=?",
            (document_id, key),
        )
        if ref and Path(ref["path"]).exists():
            logger.debug(
                "document_page_render_cache_hit",
                document_id=document_id,
                page=page_index + 1,
                namespace=namespace,
                longest_edge=edge,
            )
            return Path(ref["path"])
        source = self.cache.path_for_digest(page["digest"])
        assert source
        with tempfile.TemporaryDirectory(dir=self.cache.root / "staging") as temp:
            render_started = perf_counter()
            logger.info(
                "document_page_render_started",
                document_id=document_id,
                page=page_index + 1,
                namespace=namespace,
                longest_edge=edge,
            )
            rendered = Path(temp) / "page.jpg"
            if self._document_row(document_id)["source_kind"] == "pdf":
                await asyncio.to_thread(
                    render_pdf_page, source, int(page["source_page"]), rendered, longest_edge=edge
                )
            else:
                await asyncio.to_thread(render_image_page, source, rendered, longest_edge=edge)
            digest, path = self.cache.put_bytes(
                rendered.read_bytes(), namespace=f"{namespace}-pages", suffix=".jpg"
            )
        self.cache.ref(document_id, key, digest)
        logger.info(
            "document_page_render_completed",
            document_id=document_id,
            page=page_index + 1,
            namespace=namespace,
            longest_edge=edge,
            elapsed_ms=round((perf_counter() - render_started) * 1000, 3),
        )
        return path

    async def _section_page_image(
        self,
        document_id: str,
        page_index: int,
        node: dict[str, Any],
        next_node: dict[str, Any] | None,
    ) -> Path:
        full = await self._page_image(document_id, page_index, 2400, "content")
        top_bbox = node.get("bbox") if int(node["start_page"]) == page_index else None
        bottom_bbox = (
            next_node.get("bbox")
            if next_node and int(next_node["start_page"]) == page_index
            else None
        )
        if not top_bbox and not bottom_bbox:
            return full
        outline = await self._page_image(document_id, page_index, 1400, "outline")
        crop_key = hashlib.sha256(
            (
                f"section-crop:v1:{full.name}:{outline.name}:{page_index}:{top_bbox}:{bottom_bbox}"
            ).encode()
        ).hexdigest()
        cached = self.store.one(
            "SELECT o.path FROM cache_refs r JOIN cache_objects o ON o.digest=r.digest "
            "WHERE r.document_id=? AND r.cache_key=?",
            (document_id, crop_key),
        )
        if cached and Path(cached["path"]).exists():
            return Path(cached["path"])
        with Image.open(outline) as outline_image, Image.open(full) as content_image:
            scale = content_image.height / max(1, outline_image.height)
            top = int(float(top_bbox[3]) * scale) if top_bbox else 0
            bottom = int(float(bottom_bbox[1]) * scale) if bottom_bbox else content_image.height
            top = max(0, min(content_image.height - 1, top))
            bottom = max(top + 1, min(content_image.height, bottom))
            cropped = content_image.crop((0, top, content_image.width, bottom))
            with tempfile.TemporaryDirectory(dir=self.cache.root / "staging") as temp:
                output = Path(temp) / "section.jpg"
                cropped.save(output, format="JPEG", quality=92)
                digest, path = self.cache.put_bytes(
                    output.read_bytes(), namespace="content-pages", suffix=".jpg"
                )
        self.cache.ref(document_id, crop_key, digest)
        return path

    def _save_outline(
        self, document_id: str, revision: int, nodes: list[dict[str, Any]], source: str
    ) -> None:
        self.store.execute(
            "INSERT INTO outline_revisions(document_id,revision,source,created_at) VALUES(?,?,?,?)",
            (document_id, revision, source, now_iso()),
        )
        for node in nodes:
            self.store.execute(
                "INSERT INTO outline_nodes(document_id,revision,id,parent_id,position,title,level,start_page,bbox_json,confidence,user_edited) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    revision,
                    node["id"],
                    node.get("parent_id"),
                    node["position"],
                    node["title"],
                    node["level"],
                    node["start_page"],
                    json.dumps(node.get("bbox")),
                    node.get("confidence", 1),
                    int(node.get("user_edited", False)),
                ),
            )

    def _outline_nodes(self, document_id: str, revision: int) -> list[dict[str, Any]]:
        rows = self.store.all(
            "SELECT id,parent_id,position,title,level,start_page,bbox_json,confidence,user_edited "
            "FROM outline_nodes WHERE document_id=? AND revision=? ORDER BY start_page,position",
            (document_id, revision),
        )
        return [
            {
                **row,
                "bbox": json.loads(row.pop("bbox_json")),
                "user_edited": bool(row["user_edited"]),
            }
            for row in rows
        ]

    def _write_exports(self, document_id: str, revision: int) -> None:
        payload = json.dumps(self.result(document_id), ensure_ascii=False, indent=2).encode()
        markdown = self.markdown(document_id).encode()
        json_digest, _ = self.cache.put_bytes(payload, namespace="exports", suffix=".json")
        md_digest, _ = self.cache.put_bytes(markdown, namespace="exports", suffix=".md")
        self.cache.ref(document_id, f"export:{revision}:json", json_digest)
        self.cache.ref(document_id, f"export:{revision}:markdown", md_digest)
        self.store.execute(
            "INSERT OR REPLACE INTO exports(document_id,revision,json_digest,markdown_digest,created_at) VALUES(?,?,?,?,?)",
            (document_id, revision, json_digest, md_digest, now_iso()),
        )

    def _insert_job(
        self,
        job_id: str,
        document_id: str,
        *,
        revision: int,
        kind: str = "markdown",
        target_nodes: list[str] | None = None,
        model_id: str | None = None,
        layout_model_id: str | None = None,
    ) -> None:
        if kind == "structure":
            selected_model_id = "builtin/markdown-structure"
            model_variant = "builtin"
            model_runtime = "builtin"
        else:
            selected_model_id, model_variant, model_runtime = self._resolve_parser_model(
                model_id or DEFAULT_MODEL_ID
            )
        resolved_layout_id: str | None = None
        layout_variant: str | None = None
        layout_runtime: str | None = None
        if kind == "markdown" and layout_model_id:
            resolved_layout_id, layout_variant, layout_runtime = self._resolve_layout_model(
                layout_model_id
            )
        stamp = now_iso()
        self.store.execute(
            "INSERT INTO jobs(id,document_id,kind,state,stage,model_id,model_variant,model_runtime,layout_model_id,layout_model_variant,layout_model_runtime,pipeline_version,outline_revision,target_nodes_json,created_at,updated_at) "
            "VALUES(?,?,?,'queued','preparing',?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                document_id,
                kind,
                selected_model_id,
                model_variant,
                model_runtime,
                resolved_layout_id,
                layout_variant,
                layout_runtime,
                STRUCTURE_VERSION if kind == "structure" else PIPELINE_VERSION,
                revision,
                json.dumps(target_nodes) if target_nodes is not None else None,
                stamp,
                stamp,
            ),
        )

    def _resolve_parser_model(self, model_id: str) -> tuple[str, str, str]:
        try:
            definition = self.models.registry.get(model_id)
        except ResourceNotFoundError as error:
            raise ValueError("The selected OCR model is not registered") from error
        manifest = definition.manifest
        if "document-parsing" not in manifest.capabilities:
            raise ValueError("The selected model does not support document parsing")
        variant = manifest.default_variant
        runtime = manifest.default_runtime or (
            definition.supported_runtimes[0] if definition.supported_runtimes else ""
        )
        if not runtime or runtime not in definition.supported_runtimes:
            raise ValueError("The selected OCR model has no supported runtime")
        return model_id, variant, runtime

    def _resolve_layout_model(self, model_id: str) -> tuple[str, str, str]:
        try:
            definition = self.models.registry.get(model_id)
        except ResourceNotFoundError as error:
            raise ValueError("The selected layout model is not registered") from error
        manifest = definition.manifest
        if "document-layout-analysis" not in manifest.capabilities:
            raise ValueError("The selected model does not support document layout analysis")
        variant = manifest.default_variant
        runtime = manifest.default_runtime or (
            definition.supported_runtimes[0] if definition.supported_runtimes else ""
        )
        if not runtime or runtime not in definition.supported_runtimes:
            raise ValueError("The selected layout model has no supported runtime")
        return model_id, variant, runtime

    def _document_row(self, document_id: str) -> dict[str, Any]:
        row = self.store.one("SELECT * FROM documents WHERE id=?", (document_id,))
        if not row:
            raise KeyError("Document not found")
        return row

    def _set_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return
        updates["updated_at"] = now_iso()
        fields = ",".join(f"{key}=?" for key in updates)
        self.store.execute(f"UPDATE jobs SET {fields} WHERE id=?", (*updates.values(), job_id))

    def _progress(self, job_id: str, completed: int, total: int, stage: str) -> None:
        self._set_job(job_id, completed_units=completed, total_units=total, stage=stage)
        self._event(job_id, "progress", {"stage": stage, "completed": completed, "total": total})

    def _event(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        self.store.add_event(job_id, event, data)
        self._event_wakes.setdefault(job_id, asyncio.Event()).set()

    def _cancelled(self, job_id: str) -> bool:
        row = self.store.one("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,))
        return bool(row and row["cancel_requested"])

    def _is_stale(self, job_id: str, document_id: str, revision: int) -> bool:
        document = self._document_row(document_id)
        return (
            int(document["outline_revision"]) != revision
            or document.get("current_job_id") != job_id
        )

    def _finish_cancelled(self, job_id: str) -> None:
        job = self.get_job(job_id)
        self._set_job(job_id, state="cancelled", stage="cancelled")
        self.store.execute(
            "UPDATE documents SET status='cancelled',updated_at=? WHERE id=?",
            (now_iso(), job.document_id),
        )
        self._event(job_id, "cancelled", {})
        logger.info(
            "document_job_cancelled",
            job_id=job_id,
            document_id=job.document_id,
            job_kind=job.kind,
        )
        document = self.store.one("SELECT deleted_at FROM documents WHERE id=?", (job.document_id,))
        if document and document.get("deleted_at"):
            self._purge_document(job.document_id)

    def _purge_document(self, document_id: str) -> None:
        self.cache.release_document(document_id)
        self.store.execute("DELETE FROM documents WHERE id=?", (document_id,))
        logger.info("document_deleted", document_id=document_id)

    def _finish_failed(self, job_id: str, error: Exception) -> None:
        with contextlib.suppress(KeyError):
            job = self.get_job(job_id)
            self._set_job(job_id, state="failed", stage="failed", error=str(error)[:2000])
            self.store.execute(
                "UPDATE documents SET status='failed',updated_at=? WHERE id=?",
                (now_iso(), job.document_id),
            )
            self._event(job_id, "failed", {"message": str(error)[:1000]})
            logger.error(
                "document_job_failed",
                job_id=job_id,
                document_id=job.document_id,
                job_kind=job.kind,
                error_type=type(error).__name__,
                error=str(error)[:1000],
            )

    def _append_document_warning(self, document_id: str, warning: str) -> None:
        row = self._document_row(document_id)
        warnings = json.loads(row["warnings_json"])
        warnings.append(warning)
        self.store.execute(
            "UPDATE documents SET warnings_json=?,updated_at=? WHERE id=?",
            (json.dumps(warnings, ensure_ascii=False), now_iso(), document_id),
        )

    @staticmethod
    def _outline_prompt(
        start: int,
        end: int,
        hints: list[dict[str, Any]],
        bookmarks: list[list[Any]],
    ) -> str:
        return "document parsing."

    @staticmethod
    def _content_prompt(
        node: dict[str, Any],
        next_node: dict[str, Any] | None,
        start: int,
        end: int,
        previous: str,
    ) -> str:
        return "document parsing."


def _windows(page_count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    return _windows_range(0, max(0, page_count - 1), size, overlap) if page_count else []


def _windows_range(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Window size must be greater than overlap")
    result: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        window_end = min(end, cursor + size - 1)
        result.append((cursor, window_end))
        if window_end == end:
            break
        cursor = window_end - overlap + 1
    return result


def _adaptive_windows_range(
    start: int,
    end: int,
    text_sizes: dict[int, int],
    *,
    max_pages: int = 4,
    text_budget: int = 12_000,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        window_end = cursor
        used = 0
        while window_end <= end and window_end - cursor < max_pages:
            next_size = text_sizes.get(window_end, 0)
            if window_end > cursor and used + next_size > text_budget:
                break
            used += next_size
            window_end += 1
        window_end = max(cursor, window_end - 1)
        result.append((cursor, window_end))
        if window_end == end:
            break
        cursor = window_end if window_end > cursor else cursor + 1
    return result


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    return value


def _boundary_signatures(
    nodes: list[dict[str, Any]], page_count: int
) -> dict[str, tuple[Any, ...]]:
    ordered = sorted(
        nodes,
        key=lambda item: (
            int(item["start_page"]),
            float(item["bbox"][1]) if item.get("bbox") else -1,
        ),
    )
    ranges = section_ranges(nodes, page_count)
    signatures: dict[str, tuple[Any, ...]] = {}
    for index, node in enumerate(ordered):
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        signatures[str(node["id"])] = (
            ranges[str(node["id"])],
            tuple(node["bbox"]) if node.get("bbox") else None,
            str(following["id"]) if following else None,
            int(following["start_page"]) if following else None,
            tuple(following["bbox"]) if following and following.get("bbox") else None,
        )
    return signatures
