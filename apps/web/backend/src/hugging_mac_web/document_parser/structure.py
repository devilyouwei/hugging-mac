"""Deterministic structured metadata extraction from completed OCR Markdown."""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

STRUCTURE_VERSION = "markdown-structure-v1"

_PAGE = re.compile(r"^<!--\s*Page\s+(\d+)\s*-->\s*$", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:\.?)[\s:–—-]+(?P<title>.+)$")
_APPENDIX = re.compile(r"^(?:appendix\s+)?([A-Z])(?:\.\d+)*(?:[\s:–—-]+|$)", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Z0-9._%+\-{} ,]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
_ORCID = re.compile(r"(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-[\dX]{4}", re.IGNORECASE)
_META_HEADINGS = {"abstract", "keywords", "key words", "author", "authors"}
_AFFILIATION_WORDS = (
    "university", "institute", "institution", "department", "laboratory", "lab ",
    "school of", "college", "academy", "research", "google", "microsoft", "meta ",
    "openai", "inc.", "corp.",
)


@dataclass(frozen=True)
class Heading:
    title: str
    markdown_level: int
    page: int
    line_index: int


def markdown_digest(markdown: str) -> str:
    return hashlib.sha256(markdown.encode()).hexdigest()


def parse_headings(markdown: str) -> list[Heading]:
    page = 1
    result: list[Heading] = []
    for index, raw in enumerate(markdown.splitlines()):
        if match := _PAGE.match(raw.strip()):
            page = int(match.group(1))
            continue
        if match := _HEADING.match(raw.strip()):
            title = re.sub(r"\s+", " ", match.group(2)).strip()
            if title:
                result.append(Heading(title, len(match.group(1)), page, index))
    return result


def extract_title(markdown: str) -> str | None:
    headings = parse_headings(markdown)
    for heading in headings:
        normalized = heading.title.casefold().rstrip(":")
        if normalized in _META_HEADINGS or _NUMBERED.match(heading.title):
            continue
        return heading.title
    return None


def extract_abstract(markdown: str) -> str | None:
    lines = markdown.splitlines()
    start: int | None = None
    start_level = 6
    content: list[str] = []
    for index, raw in enumerate(lines):
        match = _HEADING.match(raw.strip())
        if start is None:
            if match and match.group(2).strip().casefold().rstrip(":") == "abstract":
                start = index + 1
                start_level = len(match.group(1))
            continue
        if match and len(match.group(1)) <= start_level:
            break
        if re.match(r"^(?:key\s*words|keywords)\s*[:—-]", raw.strip(), re.IGNORECASE):
            break
        if not _PAGE.match(raw.strip()):
            content.append(raw)
    value = "\n".join(content).strip()
    return value or None


def extract_keywords(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        heading = _HEADING.match(stripped)
        if heading and heading.group(2).strip().casefold().rstrip(":") in {"keywords", "key words"}:
            value = ""
            for line in lines[index + 1 :]:
                if _HEADING.match(line.strip()):
                    break
                if line.strip() and not _PAGE.match(line.strip()):
                    value = line.strip()
                    break
            return _split_keywords(value)
        if match := re.match(r"^(?:key\s*words|keywords)\s*[:—-]\s*(.+)$", stripped, re.IGNORECASE):
            return _split_keywords(match.group(1))
    return []


def extract_authors(markdown: str, title: str | None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    lines = markdown.splitlines()
    title_line = next((i for i, line in enumerate(lines) if title and title in line), -1)
    end = next(
        (i for i, line in enumerate(lines) if i > title_line and _HEADING.match(line.strip()) and _HEADING.match(line.strip()).group(2).strip().casefold().rstrip(":") == "abstract"),
        min(len(lines), title_line + 16),
    )
    preamble = [line.strip() for line in lines[title_line + 1 : end] if line.strip() and not _PAGE.match(line.strip())]
    expanded_emails: list[str] = []
    for line in preamble:
        for candidate in _EMAIL.findall(line):
            expanded_emails.extend(_expand_email(candidate))
    affiliations = _unique(
        re.sub(r"^[\d*†‡,\s]+", "", line).strip()
        for line in preamble
        if any(word in f" {line.casefold()} " for word in _AFFILIATION_WORDS) and "@" not in line
    )
    identifiers = _unique(match.group(0) for line in preamble for match in _ORCID.finditer(line))
    author_line = next(
        (
            line for line in preamble
            if "@" not in line
            and line not in affiliations
            and not _ORCID.search(line)
            and len(re.findall(r"[A-Z][\w'’-]+", line)) >= 2
        ),
        "",
    )
    names = _split_author_names(author_line, len(expanded_emails))
    authors: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        emails = [expanded_emails[index]] if index < len(expanded_emails) else []
        authors.append(
            {
                "name": name,
                "affiliations": affiliations if len(affiliations) == 1 else [],
                "emails": emails,
                "identifiers": [identifiers[index]] if index < len(identifiers) else [],
            }
        )
    warnings: list[str] = []
    if affiliations and authors and len(affiliations) > 1:
        warnings.append("Multiple affiliations were found, but their author mapping was ambiguous.")
    if expanded_emails and authors and len(expanded_emails) != len(authors):
        warnings.append("Some email addresses could not be mapped to an author reliably.")
    return authors, affiliations, warnings


def build_outline(markdown: str, document_title: str | None) -> list[dict[str, Any]]:
    candidates: list[tuple[Heading, int]] = []
    headings = parse_headings(markdown)
    base = min((item.markdown_level for item in headings), default=1)
    for heading in headings:
        normalized = heading.title.casefold().rstrip(":")
        if normalized in _META_HEADINGS or heading.title == document_title:
            continue
        if numbered := _NUMBERED.match(heading.title):
            level = numbered.group("number").count(".") + 1
        elif _APPENDIX.match(heading.title) or normalized in {"references", "bibliography", "acknowledgements", "acknowledgments"}:
            level = 1
        else:
            level = max(1, heading.markdown_level - base + 1)
        candidates.append((heading, level))

    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for position, (heading, level) in enumerate(candidates):
        level = min(level, len(stack) + 1)
        node = {
            "id": hashlib.sha1(f"{position}:{heading.page}:{heading.title}".encode()).hexdigest()[:16],
            "title": heading.title,
            "level": level,
            "page": heading.page,
            "children": [],
        }
        while len(stack) >= level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _expand_email(value: str) -> list[str]:
    compact = re.sub(r"\s+", "", value).strip(".,;()[]")
    if "@" not in compact:
        return []
    local, domain = compact.rsplit("@", 1)
    if local.startswith("{") and "}" in local:
        local = local.strip("{}")
        return [f"{item}@{domain}" for item in local.split(",") if item]
    return [compact]


def _split_author_names(value: str, email_count: int) -> list[str]:
    clean = re.sub(r"[\d*†‡]+", " ", value)
    separated = [item.strip() for item in re.split(r"\s*(?:,|;|\band\b|·)\s*", clean) if item.strip()]
    if len(separated) > 1:
        return separated
    words = clean.split()
    if email_count > 1 and len(words) >= email_count * 2:
        size, extra = divmod(len(words), email_count)
        result: list[str] = []
        cursor = 0
        for index in range(email_count):
            take = size + (1 if index < extra else 0)
            result.append(" ".join(words[cursor : cursor + take]))
            cursor += take
        return result
    return [clean.strip()] if clean.strip() else []


def _split_keywords(value: str) -> list[str]:
    return _unique(item.strip(" .") for item in re.split(r"[,;·•|]", value) if item.strip(" ."))


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if value and normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return result
