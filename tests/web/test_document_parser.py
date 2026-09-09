from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient
from hugging_mac_sdk.schemas.detection import BoundingBox, DetectionTimings, ImageSize
from hugging_mac_sdk.schemas.document_layout import (
    DocumentLayoutRegion,
    DocumentLayoutResponse,
)
from hugging_mac_web.config import WebSettings
from hugging_mac_web.context import create_context
from hugging_mac_web.document_parser.cache import ContentCache
from hugging_mac_web.document_parser.pipeline import (
    HeadingCandidate,
    filter_layout_regions,
    infer_outline_nodes,
    layout_region_task_prompt,
    merge_fragments,
    native_heading_candidates,
    normalize_layout_region_markdown,
    ocr_heading_candidates,
    ocr_to_markdown,
    outline_markdown,
    section_ranges,
    streaming_heading_candidates,
    streaming_ocr_markdown,
    validate_headings,
    validate_ocr_page,
    validate_outline_ocr,
    validate_page_fragments,
)
from hugging_mac_web.document_parser.schemas import OutlineNodeInput
from hugging_mac_web.document_parser.store import DocumentParserStore, now_iso
from hugging_mac_web.document_parser.structure import build_outline as build_markdown_outline
from hugging_mac_web.document_parser.structure import (
    extract_abstract,
    extract_authors,
    extract_keywords,
    extract_title,
)
from hugging_mac_web.lifecycle import close_context
from hugging_mac_web.main import create_app
from PIL import Image


def settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_home=tmp_path / "models",
        database_path=tmp_path / "platform.json",
    )


def png_bytes(color: str = "white") -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (320, 480), color).save(payload, format="PNG")
    return payload.getvalue()


def pdf_bytes(*, encrypted: bool = False) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1 Introduction")
    options: dict[str, object] = {}
    if encrypted:
        options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner",
            "user_pw": "secret",
        }
    payload = document.tobytes(**options)
    document.close()
    return payload


def test_document_parser_upload_order_preview_and_delete(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[
                ("files", ("002.png", png_bytes("white"), "image/png")),
                ("files", ("001.png", png_bytes("gray"), "image/png")),
            ],
        )
        assert created.status_code == 202
        document = created.json()["data"]["document"]
        assert document["source_kind"] == "images"
        assert document["page_count"] == 2

        assert created.json()["data"]["job"] is None
        assert document["status"] == "ready"
        active_jobs = client.get(
            "/api/v1/apps/document-parser/jobs", params={"active_only": "true"}
        )
        assert active_jobs.status_code == 200
        assert active_jobs.json()["data"] == []

        queued = client.post(
            f"/api/v1/apps/document-parser/documents/{document['id']}/jobs",
            json={"kind": "markdown"},
        )
        assert queued.status_code == 202
        assert queued.json()["data"]["kind"] == "markdown"
        assert queued.json()["data"]["event_sequence"] >= 1

        detail = client.get(f"/api/v1/apps/document-parser/documents/{document['id']}")
        assert detail.status_code == 200
        preview = client.get(
            f"/api/v1/apps/document-parser/documents/{document['id']}/pages/0/image"
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"

        deleted = client.delete(f"/api/v1/apps/document-parser/documents/{document['id']}")
        assert deleted.status_code == 204
        assert (
            client.get(f"/api/v1/apps/document-parser/documents/{document['id']}").status_code
            == 404
        )


def test_document_parser_lists_and_persists_ocr_model_selection(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        available = client.get("/api/v1/apps/document-parser/models")
        assert available.status_code == 200
        models = {item["model_id"]: item for item in available.json()["data"]}
        assert models["baidu/unlimited-ocr"]["variant"] == "4bit"
        assert models["stepfun-ai/got-ocr2.0"]["variant"] == "8bit"
        assert models["stepfun-ai/got-ocr2.0"]["runtime"] == "mlx"
        assert models["zai-org/glm-ocr"]["variant"] == "8bit"
        assert models["zai-org/glm-ocr"]["runtime"] == "mlx"
        layout_models = client.get("/api/v1/apps/document-parser/layout-models")
        assert layout_models.status_code == 200
        assert [item["model_id"] for item in layout_models.json()["data"]] == [
            "paddlepaddle/pp-doclayout-v3"
        ]

        created = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[("files", ("page.png", png_bytes(), "image/png"))],
        ).json()["data"]["document"]
        queued = client.post(
            f"/api/v1/apps/document-parser/documents/{created['id']}/jobs",
            json={
                "kind": "markdown",
                "model_id": "stepfun-ai/got-ocr2.0",
                "layout_model_id": "paddlepaddle/pp-doclayout-v3",
            },
        )
        assert queued.status_code == 202
        job = queued.json()["data"]
        assert job["model_id"] == "stepfun-ai/got-ocr2.0"
        assert job["model_variant"] == "8bit"
        assert job["model_runtime"] == "mlx"
        assert job["layout_model_id"] == "paddlepaddle/pp-doclayout-v3"
        assert job["layout_model_variant"] == "base"
        assert job["layout_model_runtime"] == "coreml"

        detail = client.get(
            f"/api/v1/apps/document-parser/documents/{created['id']}"
        ).json()["data"]
        assert detail["jobs"]["markdown"]["model_id"] == "stepfun-ai/got-ocr2.0"
        assert (
            detail["jobs"]["markdown"]["layout_model_id"]
            == "paddlepaddle/pp-doclayout-v3"
        )


@pytest.mark.asyncio
async def test_page_layout_is_persisted_and_emitted_for_sse_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = create_context(settings(tmp_path))
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,page_count,created_at,updated_at) "
        "VALUES('doc','doc','images','running',1,?,?)",
        (stamp, stamp),
    )
    manager._insert_job(
        "job",
        "doc",
        revision=0,
        layout_model_id="paddlepaddle/pp-doclayout-v3",
    )
    manager.store.execute("UPDATE documents SET current_job_id='job' WHERE id='doc'")
    image = tmp_path / "page.png"
    image.write_bytes(png_bytes())

    class FakeLayoutAnalyzer:
        async def analyze_layout(self, request: Any) -> DocumentLayoutResponse:
            assert request.image.path == image
            return DocumentLayoutResponse(
                model_id="paddlepaddle/pp-doclayout-v3",
                instance_id="layout",
                runtime="coreml",
                device="all",
                image_size=ImageSize(width=320, height=480),
                regions=(
                    DocumentLayoutRegion(
                        box=BoundingBox(x1=10, y1=20, x2=300, y2=80),
                        confidence=0.98,
                        class_id=6,
                        label="doc_title",
                        order=0,
                    ),
                    DocumentLayoutRegion(
                        box=BoundingBox(x1=10, y1=100, x2=300, y2=220),
                        confidence=0.95,
                        class_id=21,
                        label="table",
                        order=1,
                    ),
                ),
                timings=DetectionTimings(inference_ms=3),
            )

    payload, warning = await manager._analyze_page_layout(
        "job", "doc", 0, 1, image, FakeLayoutAnalyzer()  # type: ignore[arg-type]
    )
    assert warning is None
    assert payload is not None
    assert payload["region_counts"] == {"doc_title": 1, "table": 1}
    assert payload["regions"][0]["box"] == {"x1": 10.0, "y1": 20.0, "x2": 300.0, "y2": 80.0}
    assert manager.snapshot("job").draft["current_layout"]["display_page"] == 1
    assert [event["event"] for event in manager.events("job", 0)][-2:] == [
        "page_layout_started",
        "page_layout",
    ]
    prompt = manager._layout_aware_prompt(payload)
    assert "1. doc_title [10,20,300,80]" in prompt

    prompts: list[str] = []

    async def fake_region_request(
        parser: Any,
        paths: list[Path],
        prompt: str,
        validator: Any,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        prompts.append(prompt)
        text = "# Paper title" if len(prompts) == 1 else "| A | B |\n|---|---|\n| 1 | 2 |"
        return ({0: text}, []), None

    monkeypatch.setattr(manager, "_request_with_feedback", fake_region_request)
    markdown, warnings = await manager._parse_layout_regions(
        "job", 0, image, object(), payload  # type: ignore[arg-type]
    )
    assert warnings == []
    assert markdown == "# Paper title\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    assert prompts == ["Text Recognition:", "Table Recognition:"]
    assert any(event["event"] == "layout_region_started" for event in manager.events("job", 0))
    assert any(event["event"] == "layout_region_completed" for event in manager.events("job", 0))
    await close_context(context)


def test_layout_regions_are_filtered_routed_and_rendered_as_markdown() -> None:
    regions = [
        {
            "label": "table",
            "order": 2,
            "box": {"x1": 0, "y1": 0, "x2": 200, "y2": 200},
        },
        {
            "label": "formula",
            "order": 1,
            "box": {"x1": 20, "y1": 20, "x2": 80, "y2": 60},
        },
        {
            "label": "footer",
            "order": 3,
            "box": {"x1": 0, "y1": 220, "x2": 200, "y2": 240},
        },
    ]

    assert [region["label"] for region in filter_layout_regions(regions)] == ["table"]
    assert layout_region_task_prompt("table") == "Table Recognition:"
    assert layout_region_task_prompt("display_formula") == "Formula Recognition:"
    assert layout_region_task_prompt("paragraph_title") == "Text Recognition:"
    assert layout_region_task_prompt("footer") is None
    assert normalize_layout_region_markdown("doc_title", "BERT") == "# BERT"
    assert normalize_layout_region_markdown("paragraph_title", "3.2 Attention") == (
        "### 3.2 Attention"
    )
    assert normalize_layout_region_markdown("formula", r"\[x^2\]") == "$$\nx^2\n$$"
    assert normalize_layout_region_markdown(
        "text", "\\section*{Introduction}\nBody"
    ) == "## Introduction\nBody"


def test_document_parser_rejects_unknown_ocr_model(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[("files", ("page.png", png_bytes(), "image/png"))],
        ).json()["data"]["document"]
        response = client.post(
            f"/api/v1/apps/document-parser/documents/{created['id']}/jobs",
            json={"kind": "markdown", "model_id": "missing/ocr"},
        )
    assert response.status_code == 409
    assert "not registered" in response.json()["error"]["message"]


def test_document_parser_rejects_mixed_inputs(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[
                ("files", ("document.pdf", b"%PDF-1.7", "application/pdf")),
                ("files", ("page.png", png_bytes(), "image/png")),
            ],
        )
    assert response.status_code == 422


def test_document_parser_accepts_pdf_and_rejects_encrypted_pdf(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[("files", ("plain.pdf", pdf_bytes(), "application/pdf"))],
        )
        rejected = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[
                (
                    "files",
                    ("encrypted.pdf", pdf_bytes(encrypted=True), "application/pdf"),
                )
            ],
        )
    assert accepted.status_code == 202
    assert accepted.json()["data"]["document"]["page_count"] == 1
    assert rejected.status_code == 422


def test_pipeline_validates_outline_and_fragments() -> None:
    headings = validate_headings(
        json.dumps(
            {
                "headings": [
                    {
                        "title": "1 Introduction",
                        "page": 0,
                        "level": 1,
                        "bbox": [10, 20, 200, 60],
                        "type": "heading",
                        "confidence": 0.9,
                    }
                ]
            }
        ),
        page_start=0,
        page_end=3,
    )
    nodes = infer_outline_nodes(headings, page_count=4)
    assert nodes[0]["title"] == "1 Introduction"
    assert section_ranges(nodes, 4)[nodes[0]["id"]] == (0, 3)

    fragments, warnings = validate_page_fragments(
        '{"pages":[{"page":0,"markdown":"alpha"}],"complete":true}',
        expected_pages=[0],
    )
    assert fragments == {0: "alpha"}
    assert warnings == []
    assert merge_fragments("one shared tail", "shared tail two") == "one shared tail two"


def test_streaming_outline_emits_only_complete_heading_objects() -> None:
    partial = '{"headings":[{"title":"Chapter 1","page":0,"level":1'
    assert streaming_heading_candidates(partial, page_start=0, page_end=2) == []

    complete = partial + ',"bbox":null,"type":"heading","confidence":0.9}'
    candidates = streaming_heading_candidates(complete, page_start=0, page_end=2)
    assert [(item.title, item.page, item.level) for item in candidates] == [
        ("Chapter 1", 0, 1)
    ]


def test_code_assembles_native_headings_into_markdown_outline() -> None:
    result = [
        HeadingCandidate("1 Introduction", 0, 1, None, "title", 0.9),
        HeadingCandidate("2 Method", 1, 1, None, "title", 0.9),
        HeadingCandidate("2.1 Data", 1, 2, None, "heading", 0.9),
        HeadingCandidate("2.1 Data", 1, 2, None, "heading", 0.8),
    ]
    assert outline_markdown(result) == (
        "- 1 Introduction · Page 1\n"
        "- 2 Method · Page 2\n"
        "  - 2.1 Data · Page 2"
    )


@pytest.mark.asyncio
async def test_title_scan_uses_native_layout_then_accumulates_in_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = create_context(settings(tmp_path))
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents"
        "(id,name,source_kind,status,page_count,created_at,updated_at) "
        "VALUES('doc','doc','images','running',2,?,?)",
        (stamp, stamp),
    )
    manager._insert_job("job", "doc", revision=0, kind="titles")
    prompts: list[str] = []

    async def fake_page_image(*args: Any, **kwargs: Any) -> Path:
        return tmp_path / "page.jpg"

    async def fake_request(
        parser: Any,
        paths: list[Path],
        prompt: str,
        validator: Any,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        prompts.append(prompt)
        output = (
            "<|det|>text [10,80,500,200]<|/det|>Body text is fully OCRed.\n"
            "<|det|>title [10,20,300,60]<|/det|>1 Introduction\n"
        )
        if len(prompts) == 2:
            output = (
                "<|det|>text [10,80,500,200]<|/det|>More complete page text.\n"
                "<|det|>title [10,20,300,60]<|/det|>2 Method\n"
                "<|det|>heading [10,220,260,250]<|/det|>2.1 Data\n"
            )
        return validator(output), None

    monkeypatch.setattr(manager, "_page_image", fake_page_image)
    monkeypatch.setattr(manager, "_request_with_feedback", fake_request)
    content, warnings = await manager._extract_titles_only("job", "doc", object())  # type: ignore[arg-type]

    assert warnings == []
    assert prompts == ["document parsing.", "document parsing."]
    assert content == (
        "- 1 Introduction · Page 1\n"
        "- 2 Method · Page 2\n"
        "  - 2.1 Data · Page 2"
    )
    snapshots = [
        event for event in manager.events("job", 0) if event["event"] == "output_snapshot"
    ]
    assert len(snapshots) == 2
    await close_context(context)


def test_unlimited_ocr_native_layout_stream_becomes_outline_and_markdown() -> None:
    partial = "<|det|>title [35,38,685,88]<|/det|>Unlimited OCR"
    assert ocr_heading_candidates(partial, page=2, complete_only=True) == []

    output = partial + "\n<|det|>text [33,114,761,150]<|/det|>Body text\n"
    headings = ocr_heading_candidates(output, page=2, complete_only=True)
    assert [(item.title, item.page, item.bbox) for item in headings] == [
        ("Unlimited OCR", 2, (35.0, 38.0, 685.0, 88.0))
    ]
    counting_loop = " ".join(f"{index}." for index in range(1, 21))
    assert ocr_to_markdown(f"{counting_loop} {output}") == "## Unlimited OCR\nBody text"
    assert streaming_ocr_markdown(partial) == "## Unlimited OCR"
    assert streaming_ocr_markdown(output) == "## Unlimited OCR\nBody text"
    native = "MM / DD / YYYY\n01:50:17 UTC\nDEFENDANT'S AUTHORIZATION\n"
    assert [item.title for item in native_heading_candidates(native, page=0)] == [
        "DEFENDANT'S AUTHORIZATION"
    ]
    paper_text = (
        "5 Training\n2014 English-French dataset consisting of 36M sentences and split "
        "tokens into a 32000 word-piece\n"
    )
    assert [item.title for item in native_heading_candidates(paper_text, page=6)] == [
        "5 Training"
    ]


def test_outline_creates_preamble_and_hierarchy() -> None:
    nodes = infer_outline_nodes(
        [
            HeadingCandidate("1 Chapter", 2, 1, None, "heading", 0.9),
            HeadingCandidate("1.1 Detail", 3, 2, None, "heading", 0.8),
        ],
        page_count=5,
    )
    assert nodes[0]["title"] == "Preamble"
    assert nodes[2]["parent_id"] == nodes[1]["id"]


def test_cache_reference_counting_and_recovery(tmp_path: Path) -> None:
    store = DocumentParserStore(tmp_path / "parser.sqlite3")
    cache = ContentCache(tmp_path / "objects", store)
    stamp = now_iso()
    for document_id in ("a", "b"):
        store.execute(
            "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
            "VALUES(?,?,'images','queued',?,?)",
            (document_id, document_id, stamp, stamp),
        )
    digest, path = cache.put_bytes(b"shared", namespace="test")
    cache.ref("a", "source", digest)
    cache.ref("b", "source", digest)
    cache.release_document("a")
    assert path.exists()
    cache.release_document("b")
    assert not path.exists()
    store.close()


@pytest.mark.asyncio
async def test_native_ocr_retries_repetition_and_streams_heading(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    context = create_context(configured)
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
        "VALUES('doc','doc','images','running',?,?)",
        (stamp, stamp),
    )
    manager._insert_job("job", "doc", revision=0)
    manager.store.execute("UPDATE documents SET current_job_id='job' WHERE id='doc'")
    image = tmp_path / "page.jpg"
    image.write_bytes(png_bytes())

    class FakeParser:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def stream_document(self, request: Any):
            prompt = str(request.prompt)
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                text = "loop " * 80
            else:
                text = "<|det|>title [10,20,200,60]<|/det|>Introduction\n"
            for index in range(0, len(text), 11):
                yield SimpleNamespace(delta=text[index : index + 11])

    parser = FakeParser()
    result, warning = await manager._request_with_feedback(
        parser,  # type: ignore[arg-type]
        [image],
        "document parsing.",
        lambda text: validate_outline_ocr(text, page=0),
        job_id="job",
        kind="outline",
        start=0,
        end=0,
    )
    assert result[0].title == "Introduction"
    assert warning is None
    assert len(parser.prompts) == 2
    assert parser.prompts[0] == "document parsing."
    assert "previous response failed validation" in parser.prompts[1]
    assert "loop loop" in parser.prompts[1]
    assert any(event["event"] == "outline_candidate" for event in manager.events("job", 0))
    await close_context(context)


@pytest.mark.asyncio
async def test_native_ocr_streams_markdown_and_commits_validated_page(
    tmp_path: Path,
) -> None:
    context = create_context(settings(tmp_path))
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
        "VALUES('doc','doc','images','running',?,?)",
        (stamp, stamp),
    )
    manager._insert_job("job", "doc", revision=0)
    manager.store.execute("UPDATE documents SET current_job_id='job' WHERE id='doc'")
    image = tmp_path / "page.jpg"
    image.write_bytes(png_bytes())

    class FakeParser:
        async def stream_document(self, request: Any):
            text = (
                "<|det|>heading [10,20,200,60]<|/det|>Introduction\n"
                "<|det|>text [10,80,500,200]<|/det|>Streaming body text arrives now.\n"
            )
            for index in range(0, len(text), 9):
                yield SimpleNamespace(delta=text[index : index + 9])

    result, warning = await manager._request_with_feedback(
        FakeParser(),  # type: ignore[arg-type]
        [image],
        "document parsing.",
        lambda text: validate_ocr_page(text, page=0),
        job_id="job",
        kind="markdown",
        start=0,
        end=0,
    )

    assert warning is None
    assert result[0][0] == "## Introduction\nStreaming body text arrives now."
    events = manager.events("job", 0)
    assert any(event["event"] == "output_reset" for event in events)
    deltas = [
        event["data"]["delta"]
        for event in events
        if event["event"] == "output_delta"
    ]
    assert "".join(deltas) == result[0][0]
    commits = [event for event in events if event["event"] == "output_commit"]
    assert commits[-1]["data"]["markdown"] == result[0][0]
    await close_context(context)


@pytest.mark.asyncio
async def test_job_event_wait_wakes_without_database_polling(tmp_path: Path) -> None:
    context = create_context(settings(tmp_path))
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
        "VALUES('doc','doc','images','queued',?,?)",
        (stamp, stamp),
    )
    manager._insert_job("job", "doc", revision=0)

    waiter = asyncio.create_task(manager.wait_for_events("job", after=0, timeout=1))
    await asyncio.sleep(0)
    manager._event("job", "progress", {"stage": "loading_model"})
    await asyncio.wait_for(waiter, timeout=0.2)

    job = manager.get_job("job")
    assert job.event_sequence == 1
    assert manager.events("job", after=0)[0]["event"] == "progress"
    await close_context(context)


@pytest.mark.asyncio
async def test_outline_revision_reuses_text_and_invalidates_boundary_neighbors(
    tmp_path: Path,
) -> None:
    context = create_context(settings(tmp_path))
    manager = context.document_parser
    stamp = now_iso()
    manager.store.execute(
        "INSERT INTO documents"
        "(id,name,source_kind,status,page_count,outline_revision,created_at,updated_at) "
        "VALUES('doc','doc','images','completed',3,1,?,?)",
        (stamp, stamp),
    )
    original = [
        {
            "id": "a",
            "parent_id": None,
            "position": 0,
            "title": "A",
            "level": 1,
            "start_page": 0,
            "bbox": [0, 10, 100, 30],
            "confidence": 1,
            "user_edited": False,
        },
        {
            "id": "b",
            "parent_id": None,
            "position": 1,
            "title": "B",
            "level": 1,
            "start_page": 1,
            "bbox": [0, 20, 100, 40],
            "confidence": 1,
            "user_edited": False,
        },
    ]
    manager._save_outline("doc", 1, original, "model")
    for node_id in ("a", "b"):
        manager.store.execute(
            "INSERT INTO sections"
            "(document_id,revision,node_id,markdown,fragments_json,page_spans_json,warnings_json) "
            "VALUES('doc',1,?,'body','{}','[]','[]')",
            (node_id,),
        )

    renamed = [
        OutlineNodeInput.model_validate(node | {"title": f"{node['title']} renamed"})
        for node in original
    ]
    document, job = manager.patch_outline("doc", 1, renamed)
    assert document.outline_revision == 2
    assert job is None
    assert (
        len(manager.store.all("SELECT 1 FROM sections WHERE document_id='doc' AND revision=2")) == 2
    )

    moved_bbox = [
        node.model_copy(update={"bbox": (0, 40, 100, 60)}) if node.id == "b" else node
        for node in renamed
    ]
    _document, job = manager.patch_outline("doc", 2, moved_bbox)
    assert job is not None
    assert job.kind == "markdown"
    await close_context(context)


def test_document_output_is_persisted_by_kind(tmp_path: Path) -> None:
    context = create_context(settings(tmp_path))
    stamp = now_iso()
    context.document_parser.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
        "VALUES('doc','doc','images','ready',?,?)",
        (stamp, stamp),
    )
    context.document_parser._save_output("doc", "markdown", "# Hello", ["warning"])

    output = context.document_parser.output("doc", "markdown")
    assert output.content == "# Hello"
    assert output.warnings == ["warning"]
    assert context.document_parser.output("doc", "structure").structured is None
    context.document_parser.store.close()


def test_markdown_structure_extracts_paper_metadata_and_outline() -> None:
    markdown = """<!-- Page 1 -->

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova
Google AI Language
{jacobdevlin,mingweichang,kentonl,kristout}@google.com
## Abstract
We introduce a new language representation model.
Keywords: language models; pre-training
## 1 Introduction
Introduction body.

<!-- Page 2 -->

## 2 Related Work
## 2.1 Language Models
## References
"""
    title = extract_title(markdown)
    authors, affiliations, warnings = extract_authors(markdown, title)
    assert title == (
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    )
    assert [author["name"] for author in authors] == [
        "Jacob Devlin",
        "Ming-Wei Chang",
        "Kenton Lee",
        "Kristina Toutanova",
    ]
    assert authors[0]["emails"] == ["jacobdevlin@google.com"]
    assert affiliations == ["Google AI Language"]
    assert warnings == []
    assert extract_abstract(markdown) == "We introduce a new language representation model."
    assert extract_keywords(markdown) == ["language models", "pre-training"]
    outline = build_markdown_outline(markdown, title)
    assert [(item["title"], item["page"]) for item in outline] == [
        ("1 Introduction", 1),
        ("2 Related Work", 2),
        ("References", 2),
    ]
    assert outline[1]["children"][0]["title"] == "2.1 Language Models"


def test_structure_job_requires_completed_markdown_and_snapshot_is_durable(
    tmp_path: Path,
) -> None:
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/apps/document-parser/documents",
            files=[("files", ("page.png", png_bytes(), "image/png"))],
        ).json()["data"]["document"]
        missing = client.post(
            f"/api/v1/apps/document-parser/documents/{created['id']}/jobs",
            json={"kind": "structure"},
        )
        assert missing.status_code == 409
        manager = app.state.context.document_parser
        manager._save_output(created["id"], "markdown", "## Title\n## Abstract\nText", [])
        queued = client.post(
            f"/api/v1/apps/document-parser/documents/{created['id']}/jobs",
            json={"kind": "structure"},
        )
        assert queued.status_code == 202
        job = queued.json()["data"]
        snapshot = client.get(
            f"/api/v1/apps/document-parser/jobs/{job['id']}/snapshot"
        )
        assert snapshot.status_code == 200
        assert snapshot.json()["data"]["job"]["kind"] == "structure"


def test_structure_output_becomes_stale_when_markdown_changes(tmp_path: Path) -> None:
    context = create_context(settings(tmp_path))
    stamp = now_iso()
    context.document_parser.store.execute(
        "INSERT INTO documents(id,name,source_kind,status,created_at,updated_at) "
        "VALUES('doc','doc','images','ready',?,?)",
        (stamp, stamp),
    )
    context.document_parser._save_output("doc", "markdown", "## Version one", [])
    digest = context.document_parser.output("doc", "markdown").digest
    structure = {
        "document_id": "doc",
        "source_markdown_digest": digest,
        "stale": False,
        "title": "Version one",
        "authors": [],
        "affiliations": [],
        "abstract": None,
        "keywords": [],
        "outline": [],
        "warnings": [],
        "updated_at": stamp,
    }
    context.document_parser._save_output(
        "doc", "structure", json.dumps(structure), []
    )
    assert context.document_parser.output("doc", "structure").structured.stale is False
    context.document_parser._save_output("doc", "markdown", "## Version two", [])
    assert context.document_parser.output("doc", "structure").structured.stale is True
    context.document_parser.store.close()
