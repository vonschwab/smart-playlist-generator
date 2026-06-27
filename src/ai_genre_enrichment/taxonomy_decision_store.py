"""Durable staging store for taxonomy-term adjudication decisions.

Mirrors escalation_queue.py one layer up: instead of album->genre assignments,
this stages vocabulary-level verdicts (add/alias/reject) for terms not yet in
the taxonomy graph. Accept records a row here instantly (cheap, revertable,
survives restart); a separate tracked Apply job writes them into
data/layered_genre_taxonomy.yaml. Lives in the sidecar (data/ai_genre_enrichment.db).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS taxonomy_term_decisions (
  term            TEXT PRIMARY KEY,
  raw_term        TEXT,
  verdict         TEXT NOT NULL,
  proposal_json   TEXT,
  claude_json     TEXT,
  human_edited    INTEGER NOT NULL DEFAULT 0,
  status          TEXT NOT NULL DEFAULT 'pending',
  created_at      TEXT,
  applied_at      TEXT,
  batch_version   TEXT
)
"""


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "term": row["term"],
        "raw_term": row["raw_term"],
        "verdict": row["verdict"],
        "proposal": json.loads(row["proposal_json"] or "null"),
        "claude": json.loads(row["claude_json"] or "null"),
        "human_edited": int(row["human_edited"] or 0),
        "status": row["status"],
        "created_at": row["created_at"],
        "applied_at": row["applied_at"],
        "batch_version": row["batch_version"],
    }


def list_decisions(db_path, *, status: str = "pending") -> list[dict]:
    """Read-only list of decisions by status. Opens mode=ro, does NO DDL — safe on
    the worker reader thread (2026-06-12 timeout-incident rule). Returns [] if the
    table doesn't exist yet."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM taxonomy_term_decisions WHERE status=? ORDER BY created_at",
            (status,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except sqlite3.OperationalError:
        # table not created yet (no decisions recorded)
        return []
    finally:
        conn.close()


class TaxonomyDecisionStore:
    """Read/write staging store. Instantiated on the worker thread for quick writes
    (record/revert/mark_applied); the read path on the reader thread uses the
    module-level ``list_decisions`` (mode=ro, no DDL)."""

    def __init__(self, db_path: "str | Path") -> None:
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path)
        self._c.row_factory = sqlite3.Row
        self._c.execute(_SCHEMA)
        self._c.commit()

    def record_decision(self, *, term: str, raw_term: str, verdict: str,
                        proposal_json: str, claude_json: str,
                        human_edited: int) -> None:
        if verdict not in ("add", "alias", "reject"):
            raise ValueError(f"unknown verdict {verdict!r}")
        self._c.execute(
            """
            INSERT INTO taxonomy_term_decisions
                (term, raw_term, verdict, proposal_json, claude_json, human_edited,
                 status, created_at, applied_at, batch_version)
            VALUES (?,?,?,?,?,?, 'pending', ?, NULL, NULL)
            ON CONFLICT(term) DO UPDATE SET
                raw_term=excluded.raw_term, verdict=excluded.verdict,
                proposal_json=excluded.proposal_json, claude_json=excluded.claude_json,
                human_edited=excluded.human_edited, status='pending',
                applied_at=NULL, batch_version=NULL
            """,
            (term, raw_term, verdict, proposal_json, claude_json, int(human_edited),
             time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._c.commit()

    def revert(self, term: str) -> None:
        self._c.execute("DELETE FROM taxonomy_term_decisions WHERE term=?", (term,))
        self._c.commit()

    def _list(self, status: str) -> list[dict]:
        rows = self._c.execute(
            "SELECT * FROM taxonomy_term_decisions WHERE status=? ORDER BY created_at",
            (status,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_pending(self) -> list[dict]:
        return self._list("pending")

    def list_applied(self) -> list[dict]:
        return self._list("applied")

    def get(self, term: str) -> "dict | None":
        row = self._c.execute(
            "SELECT * FROM taxonomy_term_decisions WHERE term=?", (term,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def decided_terms(self) -> set[str]:
        return {r["term"] for r in self._c.execute(
            "SELECT term FROM taxonomy_term_decisions")}

    def mark_applied(self, terms: list[str], batch_version: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._c.executemany(
            "UPDATE taxonomy_term_decisions SET status='applied', applied_at=?, "
            "batch_version=? WHERE term=?",
            [(now, batch_version, t) for t in terms],
        )
        self._c.commit()

    def close(self) -> None:
        self._c.close()
