"""HTTP and SSE endpoints for Document Parser."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now

from .schemas import (
    CreateJobRequest,
    DocumentCreated,
    DocumentOutput,
    DocumentView,
    JobSnapshot,
    JobView,
    OutlinePatch,
    ParserModelView,
)

PREFIX = "/api/v1/apps/document-parser"


def response(data: object) -> ApiResponse[object]:
    return ApiResponse(data=data, meta=ResponseMeta(generated_at=utc_now()))


def create_router() -> APIRouter:
    router = APIRouter(prefix=PREFIX, tags=["document-parser"])

    @router.post(
        "/documents",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ApiResponse[DocumentCreated],
    )
    async def create_document(
        context: ContextDependency, files: Annotated[list[UploadFile], File()]
    ) -> ApiResponse[DocumentCreated]:
        try:
            document = await context.document_parser.create_document(files)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            for upload in files:
                await upload.close()
        return response(DocumentCreated(document=document))  # type: ignore[return-value]

    @router.get("/documents", response_model=ApiResponse[list[DocumentView]])
    async def list_documents(context: ContextDependency) -> ApiResponse[list[DocumentView]]:
        return response(context.document_parser.list_documents())  # type: ignore[return-value]

    @router.get("/models", response_model=ApiResponse[list[ParserModelView]])
    async def parser_models(
        context: ContextDependency,
    ) -> ApiResponse[list[ParserModelView]]:
        return response(await context.document_parser.parser_models())  # type: ignore[return-value]

    @router.get("/layout-models", response_model=ApiResponse[list[ParserModelView]])
    async def layout_models(
        context: ContextDependency,
    ) -> ApiResponse[list[ParserModelView]]:
        return response(await context.document_parser.layout_models())  # type: ignore[return-value]

    @router.get("/documents/{document_id}", response_model=ApiResponse[DocumentView])
    async def get_document(
        document_id: str, context: ContextDependency
    ) -> ApiResponse[DocumentView]:
        try:
            return response(context.document_parser.get_document(document_id))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post(
        "/documents/{document_id}/jobs",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ApiResponse[JobView],
    )
    async def create_job(
        document_id: str, context: ContextDependency, request: CreateJobRequest
    ) -> ApiResponse[JobView]:
        try:
            return response(
                context.document_parser.create_job(
                    document_id,
                    request.kind,
                    model_id=request.model_id,
                    layout_model_id=request.layout_model_id,
                )
            )  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get(
        "/documents/{document_id}/outputs/{kind}",
        response_model=ApiResponse[DocumentOutput],
    )
    async def document_output(
        document_id: str, kind: str, context: ContextDependency
    ) -> ApiResponse[DocumentOutput]:
        try:
            return response(context.document_parser.output(document_id, kind))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/jobs/{job_id}", response_model=ApiResponse[JobView])
    async def get_job(job_id: str, context: ContextDependency) -> ApiResponse[JobView]:
        try:
            return response(context.document_parser.get_job(job_id))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/jobs/{job_id}/snapshot", response_model=ApiResponse[JobSnapshot])
    async def job_snapshot(
        job_id: str, context: ContextDependency
    ) -> ApiResponse[JobSnapshot]:
        try:
            return response(context.document_parser.snapshot(job_id))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.get("/jobs", response_model=ApiResponse[list[JobView]])
    async def list_jobs(
        context: ContextDependency,
        active_only: bool = False,
        document_id: str | None = None,
    ) -> ApiResponse[list[JobView]]:
        return response(
            context.document_parser.list_jobs(
                active_only=active_only, document_id=document_id
            )
        )  # type: ignore[return-value]

    @router.get("/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        context: ContextDependency,
        after: int = 0,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            context.document_parser.get_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def stream() -> AsyncIterator[str]:
            try:
                header_cursor = int(last_event_id or 0)
            except ValueError:
                header_cursor = 0
            cursor = max(after, header_cursor)
            while True:
                events = context.document_parser.events(job_id, cursor)
                for event in events:
                    cursor = int(event["sequence"])
                    yield (
                        f"id: {cursor}\nevent: {event['event']}\n"
                        f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                    )
                job = context.document_parser.get_job(job_id)
                if job.state in {"completed", "completed_with_warnings", "failed", "cancelled"}:
                    return
                if not events:
                    yield ": heartbeat\n\n"
                await context.document_parser.wait_for_events(
                    job_id, cursor, context.settings.sse_heartbeat_seconds
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/jobs/{job_id}/cancel", response_model=ApiResponse[JobView])
    async def cancel(job_id: str, context: ContextDependency) -> ApiResponse[JobView]:
        try:
            return response(context.document_parser.cancel(job_id))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/jobs/{job_id}/retry", response_model=ApiResponse[JobView])
    async def retry(job_id: str, context: ContextDependency) -> ApiResponse[JobView]:
        try:
            return response(context.document_parser.retry(job_id))  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.patch(
        "/documents/{document_id}/outline",
        response_model=ApiResponse[dict[str, DocumentView | JobView | None]],
    )
    async def patch_outline(
        document_id: str, request: OutlinePatch, context: ContextDependency
    ) -> ApiResponse[dict[str, DocumentView | JobView | None]]:
        try:
            document, job = context.document_parser.patch_outline(
                document_id, request.base_revision, request.nodes
            )
            return response({"document": document, "job": job})  # type: ignore[return-value]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/documents/{document_id}/result")
    async def result(document_id: str, context: ContextDependency) -> JSONResponse:
        try:
            payload = context.document_parser.result(document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="{document_id}.json"'},
        )

    @router.get("/documents/{document_id}/pages/{page_index}/image")
    async def page_image(
        document_id: str, page_index: int, context: ContextDependency
    ) -> FileResponse:
        try:
            path = await context.document_parser.page_preview(document_id, page_index)
        except (KeyError, IndexError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/documents/{document_id}/result.md", response_class=PlainTextResponse)
    async def markdown(document_id: str, context: ContextDependency) -> PlainTextResponse:
        try:
            content = context.document_parser.markdown(document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return PlainTextResponse(
            content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{document_id}.md"'},
        )

    @router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(document_id: str, context: ContextDependency) -> None:
        try:
            context.document_parser.delete_document(document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router
