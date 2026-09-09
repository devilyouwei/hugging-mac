"""Content-addressed, atomically written document cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from .store import DocumentParserStore, now_iso


class ContentCache:
    def __init__(self, root: Path, store: DocumentParserStore) -> None:
        self.root = root
        self.store = store
        root.mkdir(parents=True, exist_ok=True)
        (root / "staging").mkdir(exist_ok=True)

    def stage_stream(self, source: BinaryIO, *, max_bytes: int) -> tuple[Path, str, int]:
        digest = hashlib.sha256()
        size = 0
        fd, raw_path = tempfile.mkstemp(prefix="upload-", dir=self.root / "staging")
        path = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as target:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"The upload exceeds the {max_bytes}-byte limit")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            return path, digest.hexdigest(), size
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def commit_staged(
        self, staged: Path, digest: str, size: int, *, namespace: str, suffix: str = ""
    ) -> Path:
        directory = self.root / namespace / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest}{suffix}"
        if target.exists():
            staged.unlink(missing_ok=True)
        else:
            os.replace(staged, target)
        self.store.execute(
            "INSERT OR IGNORE INTO cache_objects"
            "(digest,namespace,path,size,created_at) VALUES(?,?,?,?,?)",
            (digest, namespace, str(target), size, now_iso()),
        )
        return target

    def put_bytes(self, payload: bytes, *, namespace: str, suffix: str = "") -> tuple[str, Path]:
        digest = hashlib.sha256(payload).hexdigest()
        fd, raw_path = tempfile.mkstemp(prefix="object-", dir=self.root / "staging")
        staged = Path(raw_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return digest, self.commit_staged(
            staged, digest, len(payload), namespace=namespace, suffix=suffix
        )

    def ref(self, document_id: str, cache_key: str, digest: str) -> None:
        self.store.execute(
            "INSERT OR REPLACE INTO cache_refs(document_id,cache_key,digest) VALUES(?,?,?)",
            (document_id, cache_key, digest),
        )

    def path_for_digest(self, digest: str) -> Path | None:
        row = self.store.one("SELECT path FROM cache_objects WHERE digest=?", (digest,))
        return Path(row["path"]) if row else None

    def release_document(self, document_id: str) -> None:
        candidates = self.store.all(
            "SELECT digest FROM cache_refs WHERE document_id=?", (document_id,)
        )
        self.store.execute("DELETE FROM cache_refs WHERE document_id=?", (document_id,))
        for candidate in candidates:
            digest = candidate["digest"]
            in_use = self.store.one("SELECT 1 FROM cache_refs WHERE digest=? LIMIT 1", (digest,))
            if in_use:
                continue
            row = self.store.one("SELECT path FROM cache_objects WHERE digest=?", (digest,))
            if row:
                Path(row["path"]).unlink(missing_ok=True)
            self.store.execute("DELETE FROM cache_objects WHERE digest=?", (digest,))

    def clear_staging(self) -> None:
        for path in (self.root / "staging").iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
