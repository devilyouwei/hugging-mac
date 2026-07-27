"""TinyDB-backed local document store.

This store is intended for one local platform process. It does not start or
connect to an external service.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from tinydb import Query, TinyDB


class TinyDocumentStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database = TinyDB(path, sort_keys=True, indent=2)
        self._lock = RLock()

    def insert(self, collection: str, document: dict[str, Any]) -> dict[str, Any]:
        stored = dict(document)
        stored.setdefault("id", uuid4().hex)
        with self._lock:
            self._database.table(collection).insert(stored)
        return stored

    def get(self, collection: str, document_id: str) -> dict[str, Any] | None:
        with self._lock:
            result = self._database.table(collection).get(Query().id == document_id)
        if result is None:
            return None
        document = cast(dict[str, Any], result)
        return dict(document)

    def upsert(self, collection: str, document: dict[str, Any]) -> dict[str, Any]:
        stored = dict(document)
        stored.setdefault("id", uuid4().hex)
        with self._lock:
            self._database.table(collection).upsert(
                stored,
                Query().id == stored["id"],
            )
        return stored

    def list(self, collection: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(document) for document in self._database.table(collection).all())

    def remove(self, collection: str, document_id: str) -> bool:
        with self._lock:
            removed = self._database.table(collection).remove(Query().id == document_id)
        return bool(removed)

    def close(self) -> None:
        with self._lock:
            self._database.close()
