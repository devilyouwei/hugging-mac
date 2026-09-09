"""Deterministic validation, merging and assembly for OCR responses."""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True, slots=True)
class HeadingCandidate:
    title: str
    page: int
    level: int
    bbox: tuple[float, float, float, float] | None
    kind: str
    confidence: float


_DETECTION = re.compile(
    r"<\|det\|>\s*([^\[]+?)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*"
    r"([\d.]+)\s*,\s*([\d.]+)\s*\]\s*<\|/det\|>\s*(.*?)(?="
    r"<\|det\|>|<PAGE>|\Z)",
    re.DOTALL,
)

_LAYOUT_ABANDON_LABELS = frozenset(
    {
        "aside_text",
        "footer",
        "footer_image",
        "footnote",
        "header",
        "header_image",
        "number",
        "reference",
    }
)
_LAYOUT_IMAGE_LABELS = frozenset({"chart", "image"})
_LAYOUT_FORMULA_LABELS = frozenset({"display_formula", "formula", "inline_formula"})


def parse_json_response(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start_candidates = [
        position for position in (value.find("{"), value.find("[")) if position >= 0
    ]
    if start_candidates:
        value = value[min(start_candidates) :]
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Output is not valid JSON: {error.msg}") from error


def validate_headings(text: str, *, page_start: int, page_end: int) -> list[HeadingCandidate]:
    data = parse_json_response(text)
    items = data.get("headings") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("Output must contain a headings array")
    headings: list[HeadingCandidate] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            raise ValueError(f"headings[{index}] is missing title text")
        page = int(item.get("page", -1))
        if page < page_start or page > page_end:
            raise ValueError(f"headings[{index}] page {page} is outside the current window")
        bbox_value = item.get("bbox")
        bbox = None
        if bbox_value is not None:
            if not isinstance(bbox_value, list) or len(bbox_value) != 4:
                raise ValueError(f"headings[{index}] bbox must contain four coordinates")
            bbox = (
                float(bbox_value[0]),
                float(bbox_value[1]),
                float(bbox_value[2]),
                float(bbox_value[3]),
            )
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError(f"headings[{index}] bbox coordinates are invalid")
        headings.append(
            HeadingCandidate(
                title=str(item["title"]).strip(),
                page=page,
                level=max(1, min(12, int(item.get("level", 1)))),
                bbox=bbox,
                kind=str(item.get("type", "heading")),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
            )
        )
    return headings


def streaming_heading_candidates(
    text: str, *, page_start: int, page_end: int
) -> list[HeadingCandidate]:
    """Extract complete heading objects while the surrounding JSON is still arriving."""

    candidates: list[HeadingCandidate] = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            item = json.loads(match.group(0))
            if not isinstance(item, dict) or "title" not in item:
                continue
            candidates.extend(
                validate_headings(
                    json.dumps({"headings": [item]}),
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return candidates


def ocr_heading_candidates(
    text: str, *, page: int, complete_only: bool = False
) -> list[HeadingCandidate]:
    """Read semantic title detections from Unlimited-OCR's native output."""

    candidates: list[HeadingCandidate] = []
    for match in _DETECTION.finditer(text):
        if complete_only and match.end() == len(text) and not text.endswith(("\n", "<PAGE>")):
            continue
        kind = match.group(1).strip().casefold()
        if kind not in {"title", "heading"}:
            continue
        title = re.sub(r"\s+", " ", match.group(6)).strip()
        if not title or len(title) > 300:
            continue
        bbox = tuple(float(match.group(index)) for index in range(2, 6))
        candidates.append(
            HeadingCandidate(
                title=title,
                page=page,
                level=_heading_level(title),
                bbox=bbox,  # type: ignore[arg-type]
                kind=kind,
                confidence=0.95,
            )
        )
    return merge_heading_candidates(candidates)


def native_heading_candidates(text: str, *, page: int) -> list[HeadingCandidate]:
    """Conservative fallback for PDFs whose visual OCR emits no title label."""

    candidates: list[HeadingCandidate] = []
    for line in text.splitlines()[:40]:
        title = re.sub(r"\s+", " ", line).strip(" -\t")
        letters = [character for character in title if character.isalpha()]
        timestamp_or_placeholder = re.fullmatch(
            r"(?i)(?:(?:MM|DD|YYYY|UTC)|[\d\s:/.,-])+", title
        )
        if not (3 <= len(title) <= 120 and len(letters) >= 5) or timestamp_or_placeholder:
            continue
        uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
        numeric = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", title)
        numbered = bool(
            (numeric and ("." in numeric.group(1) or int(numeric.group(1)) <= 50))
            or re.match(r"^[IVXLC]+[.)]\s+", title)
        )
        if uppercase_ratio < 0.82 and not numbered:
            continue
        candidates.append(
            HeadingCandidate(
                title=title,
                page=page,
                level=_heading_level(title),
                bbox=None,
                kind="native-heading",
                confidence=0.55,
            )
        )
    return merge_heading_candidates(candidates)


def validate_outline_ocr(text: str, *, page: int) -> list[HeadingCandidate]:
    if has_repetition(text):
        raise ValueError("A repetitive OCR generation loop was detected")
    if not text.strip():
        raise ValueError("OCR returned no content")
    return ocr_heading_candidates(text, page=page)


def normalize_outline_hierarchy(
    candidates: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    """Preserve list order while removing duplicates and invalid level jumps."""
    normalized: list[HeadingCandidate] = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        identity = (_normal(candidate.title), candidate.page)
        if identity in seen:
            continue
        seen.add(identity)
        requested_level = max(1, min(12, candidate.level))
        level = 1 if not normalized else min(requested_level, normalized[-1].level + 1)
        normalized.append(replace(candidate, level=level))
    return normalized


def outline_markdown(candidates: list[HeadingCandidate]) -> str:
    """Serialize the accumulated outline as a human- and model-friendly list."""

    return "\n".join(
        f"{'  ' * (candidate.level - 1)}- {candidate.title} · Page {candidate.page + 1}"
        for candidate in normalize_outline_hierarchy(candidates)
    )


def validate_ocr_page(
    text: str, *, page: int, native_text: str = ""
) -> tuple[dict[int, str], list[str]]:
    if has_repetition(text):
        raise ValueError("A repetitive OCR generation loop was detected")
    markdown = ocr_to_markdown(text)
    if not markdown:
        raise ValueError("OCR returned no document content")
    native_size = len(re.sub(r"\s+", "", native_text))
    output_size = len(re.sub(r"\s+", "", markdown))
    warnings: list[str] = []
    if native_size >= 80 and output_size < native_size * 0.15:
        warnings.append(f"Page {page + 1} has low character coverage compared with its text layer")
    return {page: markdown}, warnings


def ocr_to_markdown(text: str) -> str:
    value = text.replace("<PAGE>", "\n\n<!-- page-break -->\n\n")
    value = re.sub(r"^(?:\s*\d+\.\s+){20,}", "", value)

    value = _DETECTION.sub(_detection_markdown, value)
    value = re.sub(r"<\|/?(?:det|ref)\|>", "", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def filter_layout_regions(
    regions: list[dict[str, Any]], *, min_overlap_ratio: float = 0.8
) -> list[dict[str, Any]]:
    """Drop non-content and nested layout boxes before region OCR.

    PP-DocLayout can emit a table/text container plus formula boxes inside it.
    Parsing both produces duplicated and interleaved Markdown. The outer region
    is authoritative when at least this fraction of a smaller region is inside
    it, matching GLM-OCR's formatter default.
    """

    candidates = [
        region
        for region in regions
        if str(region.get("label", "")).casefold() not in _LAYOUT_ABANDON_LABELS
    ]
    kept: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_box = candidate.get("box", {})
        candidate_area = _box_area(candidate_box)
        if candidate_area <= 0:
            continue
        nested = False
        for other_index, other in enumerate(candidates):
            if other_index == index:
                continue
            other_box = other.get("box", {})
            other_area = _box_area(other_box)
            if other_area <= candidate_area:
                continue
            if _box_intersection(candidate_box, other_box) / candidate_area >= min_overlap_ratio:
                nested = True
                break
        if not nested:
            kept.append(candidate)
    return sorted(kept, key=lambda item: int(item.get("order", 0)))


def layout_region_task_prompt(label: str) -> str | None:
    """Map PP-DocLayout labels to GLM-OCR's trained task vocabulary."""

    normalized = label.casefold()
    if normalized in _LAYOUT_IMAGE_LABELS or normalized in _LAYOUT_ABANDON_LABELS:
        return None
    if normalized == "table":
        return "Table Recognition:"
    if normalized in _LAYOUT_FORMULA_LABELS:
        return "Formula Recognition:"
    return "Text Recognition:"


def normalize_layout_region_markdown(label: str, text: str) -> str:
    """Turn one typed layout region into deterministic Markdown syntax."""

    normalized_label = label.casefold()
    value = re.sub(r"<\|[^>]+\|>", "", text).strip()
    value = re.sub(r"^```(?:markdown|md|html|latex|tex)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value).strip()
    value = _normalize_latex_structure(value)
    if not value:
        return ""
    if normalized_label == "doc_title":
        return f"# {_plain_heading(value)}"
    if normalized_label == "paragraph_title":
        heading = _plain_heading(value)
        number = re.match(r"^(?:chapter\s+)?(\d+(?:\.\d+)*)[.)]?\s+", heading, re.I)
        depth = number.group(1).count(".") if number else 0
        return f"{'#' * min(6, depth + 2)} {heading}"
    if normalized_label == "figure_title":
        caption = re.sub(r"\s+", " ", value).strip()
        return f"*{caption}*"
    if normalized_label in _LAYOUT_FORMULA_LABELS:
        formula = value
        if formula.startswith("\\[") and formula.endswith("\\]"):
            formula = formula[2:-2].strip()
        formula = formula.strip("$").strip()
        return f"$$\n{formula}\n$$"
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _normalize_latex_structure(value: str) -> str:
    replacements = (
        (r"\\title\*?\{([^{}]+)\}", r"# \1"),
        (r"\\section\*?\{([^{}]+)\}", r"## \1"),
        (r"\\subsection\*?\{([^{}]+)\}", r"### \1"),
        (r"\\subsubsection\*?\{([^{}]+)\}", r"#### \1"),
        (r"\\paragraph\*?\{([^{}]+)\}", r"##### \1"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"\\begin\{abstract\}", "## Abstract\n\n", value)
    value = re.sub(r"\\end\{abstract\}", "", value)
    return value.strip()


def _plain_heading(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    heading = lines[0] if lines else value
    return re.sub(r"^#{1,6}\s+", "", heading).strip()


def _box_area(box: dict[str, Any]) -> float:
    return max(0.0, float(box.get("x2", 0)) - float(box.get("x1", 0))) * max(
        0.0, float(box.get("y2", 0)) - float(box.get("y1", 0))
    )


def _box_intersection(left: dict[str, Any], right: dict[str, Any]) -> float:
    width = max(
        0.0,
        min(float(left.get("x2", 0)), float(right.get("x2", 0)))
        - max(float(left.get("x1", 0)), float(right.get("x1", 0))),
    )
    height = max(
        0.0,
        min(float(left.get("y2", 0)), float(right.get("y2", 0)))
        - max(float(left.get("y1", 0)), float(right.get("y1", 0))),
    )
    return width * height


def streaming_ocr_markdown(text: str) -> str:
    """Render native layout records as soon as their header and some content arrive."""

    rendered = [_detection_markdown(match).strip() for match in _DETECTION.finditer(text)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(filter(None, rendered))).strip()


def _detection_markdown(match: re.Match[str]) -> str:
    kind = match.group(1).strip().casefold()
    content = match.group(6).strip()
    if kind in {"title", "heading"}:
        return f"## {content}\n"
    return f"{content}\n"


def _heading_level(title: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", title)
    return min(12, match.group(1).count(".") + 1) if match else 1


def merge_heading_candidates(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    merged: list[HeadingCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.page, _bbox_y(item.bbox))):
        duplicate_index = next(
            (
                index
                for index, prior in enumerate(merged)
                if abs(prior.page - candidate.page) <= 0
                and SequenceMatcher(None, _normal(prior.title), _normal(candidate.title)).ratio()
                >= 0.88
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif candidate.confidence > merged[duplicate_index].confidence:
            merged[duplicate_index] = candidate
    return merged


def merge_heading_segments(
    candidates: list[HeadingCandidate], *, group_size: int = 40
) -> list[HeadingCandidate]:
    """Bound merge memory and progressively combine long heading lists."""
    if group_size < 1:
        raise ValueError("group_size must be positive")
    ordered = sorted(candidates, key=lambda item: (item.page, _bbox_y(item.bbox)))
    groups = [
        merge_heading_candidates(ordered[index : index + group_size])
        for index in range(0, len(ordered), group_size)
    ]
    while len(groups) > 1:
        groups = [
            merge_heading_candidates(
                groups[index] + (groups[index + 1] if index + 1 < len(groups) else [])
            )
            for index in range(0, len(groups), 2)
        ]
    return groups[0] if groups else []


def conflicting_heading_pages(
    windows: list[list[HeadingCandidate]],
) -> set[int]:
    """Find overlapping-window candidates that disagree at the same visual position."""
    conflicts: set[int] = set()
    for index, current in enumerate(windows):
        for later in windows[index + 1 :]:
            for left_side, right_side in ((current, later), (later, current)):
                for left in left_side:
                    nearby = [
                        right
                        for right in right_side
                        if left.page == right.page
                        and abs(_bbox_y(left.bbox) - _bbox_y(right.bbox)) <= 80
                    ]
                    if (
                        nearby
                        and max(
                            SequenceMatcher(None, _normal(left.title), _normal(right.title)).ratio()
                            for right in nearby
                        )
                        < 0.75
                    ):
                        conflicts.add(left.page)
    return conflicts


def infer_outline_nodes(
    candidates: list[HeadingCandidate], *, page_count: int
) -> list[dict[str, Any]]:
    if not candidates:
        return [_node("Document", 1, 0, None, 0, None, 0.25)]
    ordered = merge_heading_segments(candidates)
    result: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    root_positions: Counter[str | None] = Counter()
    if ordered[0].page > 0 or _bbox_y(ordered[0].bbox) > 150:
        preamble = _node("Preamble", 1, 0, None, 0, None, 0.6)
        result.append(preamble)
        root_positions[None] = 1
    for candidate in ordered:
        level = infer_level(candidate.title, candidate.level)
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()
        parent_id = str(stack[-1]["id"]) if stack else None
        node = _node(
            candidate.title,
            level,
            candidate.page,
            candidate.bbox,
            root_positions[parent_id],
            parent_id,
            candidate.confidence,
        )
        root_positions[parent_id] += 1
        result.append(node)
        stack.append(node)
    return result


def infer_level(title: str, hint: int) -> int:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\s.)、]", title)
    if match:
        return min(12, match.group(1).count(".") + 1)
    if re.match(r"^\s*第[一二三四五六七八九十百千\d]+[章节篇部]", title):
        return 1
    return max(1, min(12, hint))


def section_ranges(nodes: list[dict[str, Any]], page_count: int) -> dict[str, tuple[int, int]]:
    ordered = sorted(nodes, key=lambda node: (int(node["start_page"]), _bbox_y(node.get("bbox"))))
    ranges: dict[str, tuple[int, int]] = {}
    for index, node in enumerate(ordered):
        start = int(node["start_page"])
        end = page_count - 1
        if index + 1 < len(ordered):
            end = int(ordered[index + 1]["start_page"])
        ranges[str(node["id"])] = (start, max(start, end))
    return ranges


def validate_page_fragments(
    text: str,
    *,
    expected_pages: list[int],
    native_text_by_page: dict[int, str] | None = None,
    prior_fragments: dict[int, str] | None = None,
) -> tuple[dict[int, str], list[str]]:
    data = parse_json_response(text)
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ValueError("Output must contain a pages array")
    fragments: dict[int, str] = {}
    for item in data["pages"]:
        if not isinstance(item, dict) or "page" not in item or "markdown" not in item:
            raise ValueError("Each page fragment must contain page and markdown")
        page = int(item["page"])
        if page not in expected_pages:
            raise ValueError(f"The model returned unrequested page {page}")
        fragments[page] = str(item["markdown"]).strip()
    missing = [page for page in expected_pages if page not in fragments]
    if missing:
        raise ValueError(f"Missing pages: {missing}")
    if data.get("complete") is False:
        raise ValueError("The model output may be truncated")
    if any(has_repetition(fragment) for fragment in fragments.values()):
        raise ValueError("Repetitive generated content was detected")
    for page, native_text in (native_text_by_page or {}).items():
        native_size = len(re.sub(r"\s+", "", native_text))
        output_size = len(re.sub(r"\s+", "", fragments.get(page, "")))
        if native_size >= 80 and output_size < native_size * 0.15:
            raise ValueError(f"Page {page} has abnormal character coverage")
    for page, prior in (prior_fragments or {}).items():
        current = fragments.get(page, "")
        if len(prior) >= 50 and len(current) >= 50:
            similarity = SequenceMatcher(None, prior, current).ratio()
            if similarity < 0.25:
                raise ValueError(f"Page {page} is inconsistent across overlapping windows")
    return fragments, []


def has_repetition(text: str, *, n: int = 8) -> bool:
    words = re.findall(r"\S+", text)
    if len(words) < n * 4:
        return False
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(grams)
    return bool(counts and counts.most_common(1)[0][1] >= 4)


def merge_fragments(current: str, incoming: str) -> str:
    if not current:
        return incoming.strip()
    if not incoming:
        return current.strip()
    left, right = current.rstrip(), incoming.lstrip()
    max_overlap = min(len(left), len(right), 2000)
    for size in range(max_overlap, 4, -1):
        if left[-size:] == right[:size]:
            return f"{left}{right[size:]}".strip()
    return f"{left}\n\n{right}".strip()


def build_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = {str(node["id"]): {**node, "children": []} for node in nodes}
    roots: list[dict[str, Any]] = []
    for node in sorted(mapped.values(), key=lambda item: (item["position"], item["start_page"])):
        parent = mapped.get(str(node["parent_id"])) if node.get("parent_id") else None
        (parent["children"] if parent else roots).append(node)
    return roots


def _node(
    title: str,
    level: int,
    page: int,
    bbox: tuple[float, ...] | None,
    position: int,
    parent_id: str | None,
    confidence: float,
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "parent_id": parent_id,
        "position": position,
        "title": title,
        "level": level,
        "start_page": page,
        "bbox": list(bbox) if bbox else None,
        "confidence": confidence,
        "user_edited": False,
    }


def _normal(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).casefold()


def _bbox_y(bbox: Any) -> float:
    return float(bbox[1]) if bbox and len(bbox) == 4 else -1.0
