"""Resumable checkpoint store for the bulk adjudication pass (Phase 3).

A standalone SQLite store (NOT metadata.db, NOT the production sidecar) holding one row
per (album_id, prompt_version). `is_done` is true only for a `complete` row whose
`input_hash` still matches — so failed rows retry and a contract/payload change re-runs.
Each result is committed immediately, so a killed/limit-hit run resumes losing nothing.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS adjudications (
  album_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  release_key TEXT,
  input_hash TEXT,
  model TEXT,
  status TEXT,
  response_json TEXT,
  dropped_file_tags_json TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  total_tokens INTEGER,
  error TEXT,
  updated_at TEXT,
  PRIMARY KEY (album_id, prompt_version)
)
"""


class AdjudicationStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def is_done(self, album_id: str, prompt_version: str, input_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT status, input_hash FROM adjudications WHERE album_id=? AND prompt_version=?",
            (album_id, prompt_version),
        ).fetchone()
        return bool(row and row[0] == "complete" and row[1] == input_hash)

    def save(
        self,
        *,
        album_id: str,
        prompt_version: str,
        status: str,
        release_key: str | None = None,
        input_hash: str | None = None,
        model: str | None = None,
        response: dict[str, Any] | None = None,
        dropped_file_tags: list[str] | None = None,
        tokens: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        tokens = tokens or {}
        self._conn.execute(
            """
            INSERT INTO adjudications (album_id, prompt_version, release_key, input_hash, model,
                status, response_json, dropped_file_tags_json, input_tokens, output_tokens,
                total_tokens, error, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(album_id, prompt_version) DO UPDATE SET
                release_key=excluded.release_key, input_hash=excluded.input_hash, model=excluded.model,
                status=excluded.status, response_json=excluded.response_json,
                dropped_file_tags_json=excluded.dropped_file_tags_json,
                input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                total_tokens=excluded.total_tokens, error=excluded.error, updated_at=excluded.updated_at
            """,
            (
                album_id, prompt_version, release_key, input_hash, model, status,
                json.dumps(response, ensure_ascii=False) if response is not None else None,
                json.dumps(dropped_file_tags) if dropped_file_tags is not None else None,
                tokens.get("input_tokens"), tokens.get("output_tokens"), tokens.get("total_tokens"),
                error, time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        return {
            status: n
            for status, n in self._conn.execute(
                "SELECT status, COUNT(*) FROM adjudications GROUP BY status"
            )
        }

    def total_tokens(self) -> int:
        return int(self._conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM adjudications"
        ).fetchone()[0])

    def iter_complete(self):
        for album_id, release_key, resp, dropped in self._conn.execute(
            "SELECT album_id, release_key, response_json, dropped_file_tags_json "
            "FROM adjudications WHERE status='complete'"
        ):
            yield {
                "album_id": album_id,
                "release_key": release_key,
                "response": json.loads(resp) if resp else None,
                "dropped_file_tags": json.loads(dropped) if dropped else [],
            }

    def close(self) -> None:
        self._conn.close()
