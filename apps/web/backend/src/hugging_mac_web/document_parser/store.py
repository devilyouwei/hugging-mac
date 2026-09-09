"""SQLite/WAL persistence for long-running document jobs."""
# ruff: noqa: E501

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DocumentParserStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, source_kind TEXT NOT NULL,
          status TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
          outline_revision INTEGER NOT NULL DEFAULT 0, warnings_json TEXT NOT NULL DEFAULT '[]',
          current_job_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_files (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, filename TEXT NOT NULL, content_type TEXT,
          digest TEXT NOT NULL, size INTEGER NOT NULL, cache_key TEXT NOT NULL,
          UNIQUE(document_id, position)
        );
        CREATE TABLE IF NOT EXISTS document_hints (
          document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
          bookmarks_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS pages (
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          page_index INTEGER NOT NULL, display_page INTEGER NOT NULL, page_label TEXT,
          width REAL NOT NULL, height REAL NOT NULL, native_text TEXT NOT NULL DEFAULT '',
          source_file_id TEXT, source_page INTEGER, PRIMARY KEY(document_id, page_index)
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          state TEXT NOT NULL, stage TEXT NOT NULL, completed_units INTEGER NOT NULL DEFAULT 0,
          total_units INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
          model_id TEXT NOT NULL, model_variant TEXT NOT NULL DEFAULT '',
          model_runtime TEXT NOT NULL DEFAULT 'mlx', pipeline_version TEXT NOT NULL,
          layout_model_id TEXT, layout_model_variant TEXT, layout_model_runtime TEXT,
          outline_revision INTEGER NOT NULL DEFAULT 0,
          target_nodes_json TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, created_at);
        CREATE TABLE IF NOT EXISTS work_units (
          id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
          section_id TEXT, prompt_type TEXT NOT NULL, cache_key TEXT,
          attempt INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL,
          outline_revision INTEGER NOT NULL, result_json TEXT, error TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS outline_revisions (
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          revision INTEGER NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL,
          PRIMARY KEY(document_id, revision)
        );
        CREATE TABLE IF NOT EXISTS outline_nodes (
          document_id TEXT NOT NULL, revision INTEGER NOT NULL, id TEXT NOT NULL,
          parent_id TEXT, position INTEGER NOT NULL, title TEXT NOT NULL, level INTEGER NOT NULL,
          start_page INTEGER NOT NULL, bbox_json TEXT, confidence REAL NOT NULL,
          user_edited INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(document_id, revision, id),
          FOREIGN KEY(document_id, revision) REFERENCES outline_revisions(document_id, revision) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS sections (
          document_id TEXT NOT NULL, revision INTEGER NOT NULL, node_id TEXT NOT NULL,
          markdown TEXT NOT NULL, fragments_json TEXT NOT NULL DEFAULT '{}',
          page_spans_json TEXT NOT NULL, warnings_json TEXT NOT NULL,
          source_work_unit TEXT, PRIMARY KEY(document_id, revision, node_id)
        );
        CREATE TABLE IF NOT EXISTS job_events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
          event TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS job_events_idx ON job_events(job_id, sequence);
        CREATE TABLE IF NOT EXISTS cache_objects (
          digest TEXT PRIMARY KEY, namespace TEXT NOT NULL, path TEXT NOT NULL,
          size INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cache_refs (
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          cache_key TEXT NOT NULL, digest TEXT NOT NULL REFERENCES cache_objects(digest),
          PRIMARY KEY(document_id, cache_key)
        );
        CREATE TABLE IF NOT EXISTS exports (
          document_id TEXT NOT NULL, revision INTEGER NOT NULL, json_digest TEXT NOT NULL,
          markdown_digest TEXT NOT NULL, created_at TEXT NOT NULL,
          PRIMARY KEY(document_id, revision)
        );
        CREATE TABLE IF NOT EXISTS document_outputs (
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, content TEXT NOT NULL DEFAULT '',
          digest TEXT, warnings_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL,
          PRIMARY KEY(document_id, kind)
        );
        CREATE TABLE IF NOT EXISTS job_drafts (
          job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, content_json TEXT NOT NULL DEFAULT '{}',
          event_sequence INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS page_layouts (
          job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
          page_index INTEGER NOT NULL, model_id TEXT NOT NULL,
          result_json TEXT NOT NULL, created_at TEXT NOT NULL,
          PRIMARY KEY(job_id, page_index)
        );
        """)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(sections)").fetchall()}
        if "fragments_json" not in columns:
            self._db.execute(
                "ALTER TABLE sections ADD COLUMN fragments_json TEXT NOT NULL DEFAULT '{}'"
            )
        job_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "kind" not in job_columns:
            self._db.execute(
                "ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'markdown'"
            )
        if "model_variant" not in job_columns:
            self._db.execute(
                "ALTER TABLE jobs ADD COLUMN model_variant TEXT NOT NULL DEFAULT ''"
            )
            self._db.execute(
                "UPDATE jobs SET model_variant='4bit' "
                "WHERE model_id='baidu/unlimited-ocr' AND model_variant=''"
            )
        if "model_runtime" not in job_columns:
            self._db.execute(
                "ALTER TABLE jobs ADD COLUMN model_runtime TEXT NOT NULL DEFAULT 'mlx'"
            )
        if "layout_model_id" not in job_columns:
            self._db.execute("ALTER TABLE jobs ADD COLUMN layout_model_id TEXT")
        if "layout_model_variant" not in job_columns:
            self._db.execute("ALTER TABLE jobs ADD COLUMN layout_model_variant TEXT")
        if "layout_model_runtime" not in job_columns:
            self._db.execute("ALTER TABLE jobs ADD COLUMN layout_model_runtime TEXT")
        output_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(document_outputs)").fetchall()
        }
        if "digest" not in output_columns:
            self._db.execute("ALTER TABLE document_outputs ADD COLUMN digest TEXT")
        document_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "deleted_at" not in document_columns:
            self._db.execute("ALTER TABLE documents ADD COLUMN deleted_at TEXT")
        draft_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(job_drafts)").fetchall()
        }
        if "event_sequence" not in draft_columns:
            self._db.execute(
                "ALTER TABLE job_drafts ADD COLUMN event_sequence INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def execute(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def transaction(self) -> sqlite3.Connection:
        return self._db

    def one(self, sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def add_event(self, job_id: str, event: str, data: dict[str, Any]) -> int:
        cursor = self.execute(
            "INSERT INTO job_events(job_id,event,data_json,created_at) VALUES(?,?,?,?)",
            (job_id, event, json.dumps(data, ensure_ascii=False), now_iso()),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def job_snapshot(
        self, job_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, int]:
        """Read job, draft, and event cursor under one lock for race-free SSE resume."""
        with self._lock:
            job = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            draft = self._db.execute(
                "SELECT content_json,event_sequence FROM job_drafts WHERE job_id=?", (job_id,)
            ).fetchone()
            event = self._db.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM job_events WHERE job_id=?", (job_id,)
            ).fetchone()
        maximum = int(event[0]) if event else 0
        return (
            dict(job) if job else None,
            dict(draft) if draft else None,
            maximum,
            int(draft["event_sequence"]) if draft else 0,
        )

    def recover(self) -> None:
        stamp = now_iso()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "UPDATE work_units SET state='queued', updated_at=? WHERE state='running'",
                    (stamp,),
                )
                self._db.execute(
                    "UPDATE jobs SET state='queued', stage='resuming', updated_at=? WHERE state='running'",
                    (stamp,),
                )
                self._db.execute(
                    "UPDATE jobs SET state='cancelled',stage='legacy_retired',updated_at=? "
                    "WHERE kind='titles' AND state IN ('queued','waiting_for_model','running')",
                    (stamp,),
                )
                self._db.execute(
                    "UPDATE documents SET status='ready',updated_at=? WHERE current_job_id IN "
                    "(SELECT id FROM jobs WHERE kind='titles' AND stage='legacy_retired')",
                    (stamp,),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
