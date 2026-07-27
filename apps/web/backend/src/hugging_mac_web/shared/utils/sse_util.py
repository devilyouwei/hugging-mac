"""Server-Sent Events formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SseEvent:
    data: Any
    event: str | None = None
    event_id: str | None = None
    retry_ms: int | None = None

    def encode(self) -> str:
        lines: list[str] = []
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.event_id is not None:
            lines.append(f"id: {self.event_id}")
        if self.retry_ms is not None:
            lines.append(f"retry: {self.retry_ms}")
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        lines.extend(f"data: {line}" for line in payload.splitlines() or ("",))
        return "\n".join(lines) + "\n\n"
