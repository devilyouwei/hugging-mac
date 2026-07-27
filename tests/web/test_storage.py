from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_web.shared.cache import LocalCache
from hugging_mac_web.shared.storage import TinyDocumentStore


def test_document_store_and_content_addressed_cache(tmp_path: Path) -> None:
    store = TinyDocumentStore(tmp_path / "platform.json")
    cache = LocalCache(tmp_path / "cache", store)
    first = cache.put_json("embeddings", {"values": [1, 2, 3]})
    second = cache.put_json("embeddings", {"values": [1, 2, 3]})

    assert first.path == second.path
    assert first.path.is_file()
    assert cache.get(first.cache_id) == second
    assert len(store.list("cache_entries")) == 1

    assert store.insert("jobs", {"status": "queued"})["id"]
    job = store.list("jobs")[0]
    assert store.remove("jobs", str(job["id"]))
    store.close()


def test_cache_rejects_unsafe_names(tmp_path: Path) -> None:
    store = TinyDocumentStore(tmp_path / "platform.json")
    cache = LocalCache(tmp_path / "cache", store)

    with pytest.raises(ValueError):
        cache.put_bytes("../escape", b"payload")
    with pytest.raises(ValueError):
        cache.put_bytes("uploads", b"payload", suffix="../../bad")

    store.close()
