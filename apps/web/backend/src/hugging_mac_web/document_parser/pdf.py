"""Streaming PDF inspection and on-demand page rendering."""
# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image


@dataclass(frozen=True, slots=True)
class PageMetadata:
    index: int
    label: str | None
    width: float
    height: float
    native_text: str


@dataclass(frozen=True, slots=True)
class PdfMetadata:
    pages: tuple[PageMetadata, ...]
    bookmarks: tuple[tuple[int, str, int], ...]


def inspect_pdf(path: Path, *, max_pages: int) -> PdfMetadata:
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted or password-protected PDFs are not supported")
        if document.page_count > max_pages:
            raise ValueError(f"The PDF exceeds the {max_pages}-page limit")
        labels = _page_labels(document)
        pages = []
        for index in range(document.page_count):
            page = document.load_page(index)
            pages.append(
                PageMetadata(
                    index=index,
                    label=labels[index] if index < len(labels) else None,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    native_text=page.get_text("text") or "",
                )
            )
        bookmarks = tuple(
            (max(1, int(level)), str(title), max(0, int(page_number) - 1))
            for level, title, page_number, *_ in document.get_toc(simple=True)
            if int(page_number) > 0
        )
        return PdfMetadata(tuple(pages), bookmarks)


def _page_labels(document: pymupdf.Document) -> list[str]:
    labels: list[str] = []
    for index in range(document.page_count):
        try:
            labels.append(document.load_page(index).get_label())
        except (AttributeError, RuntimeError):
            labels.append(str(index + 1))
    return labels


def render_pdf_page(
    source: Path,
    page_index: int,
    output: Path,
    *,
    longest_edge: int,
    crop: tuple[float, float, float, float] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(source) as document:
        page = document.load_page(page_index)
        clip = pymupdf.Rect(crop) if crop else page.rect
        scale = min(longest_edge / max(clip.width, clip.height), 4.0)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
        pixmap.save(output)
    return output


def inspect_image(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        image.verify()
        return float(image.width), float(image.height)


def render_image_page(source: Path, output: Path, *, longest_edge: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        converted = image.convert("RGB")
        converted.thumbnail((longest_edge, longest_edge))
        converted.save(output, format="JPEG", quality=92)
    return output
