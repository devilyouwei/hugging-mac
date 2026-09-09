"""Public and internal schemas for durable document parsing."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JobState = Literal[
    "queued",
    "waiting_for_model",
    "running",
    "completed",
    "completed_with_warnings",
    "failed",
    "cancelled",
]
JobKind = Literal["markdown", "structure", "titles"]
CreatableJobKind = Literal["markdown", "structure"]


class OutlineNodeInput(BaseModel):
    id: str
    parent_id: str | None = None
    position: int = Field(ge=0)
    title: str = Field(min_length=1)
    level: int = Field(ge=1, le=12)
    start_page: int = Field(ge=0)
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None and (value[2] < value[0] or value[3] < value[1]):
            raise ValueError("bbox coordinates are reversed")
        return value


class OutlinePatch(BaseModel):
    base_revision: int = Field(ge=0)
    nodes: list[OutlineNodeInput]


class OutlineNodeView(OutlineNodeInput):
    user_edited: bool = False
    children: list[OutlineNodeView] = Field(default_factory=list)


class JobView(BaseModel):
    id: str
    document_id: str
    kind: JobKind = "markdown"
    model_id: str
    model_variant: str
    model_runtime: str
    layout_model_id: str | None = None
    layout_model_variant: str | None = None
    layout_model_runtime: str | None = None
    state: JobState
    stage: str
    completed_units: int
    total_units: int
    progress: float
    cancel_requested: bool
    error: str | None = None
    event_sequence: int = Field(default=0, ge=0)
    created_at: str
    updated_at: str


class DocumentView(BaseModel):
    id: str
    name: str
    source_kind: Literal["pdf", "images"]
    status: str
    page_count: int
    outline_revision: int
    warnings: list[str]
    disk_bytes: int
    current_job_id: str | None
    created_at: str
    updated_at: str
    outline: list[OutlineNodeView] = Field(default_factory=list)
    job: JobView | None = None
    jobs: dict[str, JobView | None] = Field(default_factory=dict)


class DocumentCreated(BaseModel):
    document: DocumentView
    job: JobView | None = None


class JobEvent(BaseModel):
    sequence: int
    event: str
    data: dict[str, Any]
    created_at: str


class CreateJobRequest(BaseModel):
    kind: CreatableJobKind
    model_id: str | None = None
    layout_model_id: str | None = None


class ParserModelView(BaseModel):
    model_id: str
    name: str
    description: str
    variant: str
    runtime: str
    ready: bool
    source_url: str | None = None


class DocumentOutput(BaseModel):
    document_id: str
    kind: JobKind
    content: str
    digest: str | None = None
    structured: StructuredDocument | None = None
    warnings: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class AuthorInfo(BaseModel):
    name: str
    affiliations: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)


class OutlineItem(BaseModel):
    id: str
    title: str
    level: int = Field(ge=1)
    page: int = Field(ge=1)
    children: list[OutlineItem] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    document_id: str
    source_markdown_digest: str
    stale: bool = False
    title: str | None = None
    authors: list[AuthorInfo] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    outline: list[OutlineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class JobSnapshot(BaseModel):
    job: JobView
    event_sequence: int = Field(ge=0)
    draft: dict[str, Any] = Field(default_factory=dict)
