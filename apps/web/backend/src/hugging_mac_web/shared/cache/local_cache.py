"""Content-addressed local cache for media and derived model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from hugging_mac_web.shared.storage import TinyDocumentStore

_SAFE_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SAFE_SUFFIX = re.compile(r"^\.[a-zA-Z0-9]{1,12}$")


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_id: str
    namespace: str
    digest: str
    path: Path
    size_bytes: int
    created_at: datetime
    metadata: dict[str, Any]


class LocalCache:
    def __init__(self, root: Path, documents: TinyDocumentStore) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._documents = documents

    def put_bytes(
        self,
        namespace: str,
        data: bytes,
        *,
        suffix: str = ".bin",
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        _validate_namespace(namespace)
        _validate_suffix(suffix)
        digest = hashlib.sha256(data).hexdigest()
        path = self._root / namespace / digest[:2] / f"{digest}{suffix.lower()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            staging = path.parent / f".{path.name}.{uuid4().hex}.partial"
            try:
                with staging.open("xb") as stream:
                    stream.write(data)
                os.replace(staging, path)
            finally:
                if staging.exists():
                    staging.unlink()

        cache_id = f"{namespace}:{digest}"
        created_at = datetime.now(UTC)
        entry = CacheEntry(
            cache_id=cache_id,
            namespace=namespace,
            digest=digest,
            path=path,
            size_bytes=len(data),
            created_at=created_at,
            metadata=metadata or {},
        )
        self._documents.upsert(
            "cache_entries",
            {
                "id": cache_id,
                **entry.model_dump(mode="json"),
                "path": str(path),
            },
        )
        return entry

    def put_json(
        self,
        namespace: str,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self.put_bytes(namespace, data, suffix=".json", metadata=metadata)

    def get(self, cache_id: str) -> CacheEntry | None:
        document = self._documents.get("cache_entries", cache_id)
        if document is None:
            return None
        return CacheEntry.model_validate(document)


def _validate_namespace(namespace: str) -> None:
    if _SAFE_NAMESPACE.fullmatch(namespace) is None:
        raise ValueError(f"Unsafe cache namespace: {namespace}")


def _validate_suffix(suffix: str) -> None:
    if _SAFE_SUFFIX.fullmatch(suffix) is None:
        raise ValueError(f"Unsafe cache suffix: {suffix}")
