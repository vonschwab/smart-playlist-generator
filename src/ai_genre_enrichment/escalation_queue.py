"""Durable queue of held-back escalated album adjudications.

The SP1 <-> SP2 contract: apply() enqueues escalations; the CLI (SP1) and the web GUI
(SP2) both read via list_pending()/get() and decide via record_decision(). Lives in the
sidecar (data/ai_genre_enrichment.db).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sqlite3
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS adjudication_escalations (
  album_id TEXT PRIMARY KEY,
  release_key TEXT,
  artist TEXT,
  album TEXT,
  prior_observed_leaf_json TEXT,
  proposed_genres_json TEXT,
  escalate_reason TEXT,
  dropped_file_tags_json TEXT,
  prompt_version TEXT,
  model TEXT,
  input_hash TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  decision_genres_json TEXT,
  created_at TEXT,
  decided_at TEXT
)
"""


class EscalationQueue:
    def __init__(self, db_path: "str | Path") -> None:
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path)
        self._c.row_factory = sqlite3.Row
        self._c.execute(_SCHEMA)
        self._c.commit()

    def enqueue(
        self, *, album_id: str, release_key: str, artist: str, album: str,
        prior_observed_leaf: list[str], proposed_genres: list[dict],
        escalate_reason: str, dropped_file_tags: list[str], prompt_version: str,
        model: str, input_hash: str,
    ) -> None:
        existing = self._c.execute(
            "SELECT status, input_hash FROM adjudication_escalations WHERE album_id=?",
            (album_id,),
        ).fetchone()
        # A decided row with an unchanged proposal is left alone; a changed proposal re-opens it.
        if existing is not None and existing["status"] != "pending" \
                and existing["input_hash"] == input_hash:
            return
        self._c.execute(
            """
            INSERT INTO adjudication_escalations (album_id, release_key, artist, album,
                prior_observed_leaf_json, proposed_genres_json, escalate_reason,
                dropped_file_tags_json, prompt_version, model, input_hash, status,
                decision_genres_json, created_at, decided_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?, 'pending', NULL, ?, NULL)
            ON CONFLICT(album_id) DO UPDATE SET
                release_key=excluded.release_key, artist=excluded.artist, album=excluded.album,
                prior_observed_leaf_json=excluded.prior_observed_leaf_json,
                proposed_genres_json=excluded.proposed_genres_json,
                escalate_reason=excluded.escalate_reason,
                dropped_file_tags_json=excluded.dropped_file_tags_json,
                prompt_version=excluded.prompt_version, model=excluded.model,
                input_hash=excluded.input_hash, status='pending',
                decision_genres_json=NULL, decided_at=NULL
            """,
            (album_id, release_key, artist, album, json.dumps(prior_observed_leaf),
             json.dumps(proposed_genres), escalate_reason, json.dumps(dropped_file_tags),
             prompt_version, model, input_hash, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._c.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "album_id": row["album_id"], "release_key": row["release_key"],
            "artist": row["artist"], "album": row["album"],
            "prior_observed_leaf": json.loads(row["prior_observed_leaf_json"] or "[]"),
            "proposed_genres": json.loads(row["proposed_genres_json"] or "[]"),
            "escalate_reason": row["escalate_reason"],
            "dropped_file_tags": json.loads(row["dropped_file_tags_json"] or "[]"),
            "prompt_version": row["prompt_version"], "model": row["model"],
            "input_hash": row["input_hash"], "status": row["status"],
            "decision_genres": json.loads(row["decision_genres_json"] or "null"),
            "created_at": row["created_at"], "decided_at": row["decided_at"],
        }

    def list_pending(self) -> list[dict]:
        rows = self._c.execute(
            "SELECT * FROM adjudication_escalations WHERE status='pending' ORDER BY created_at"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, album_id: str) -> "dict | None":
        row = self._c.execute(
            "SELECT * FROM adjudication_escalations WHERE album_id=?", (album_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def decided_ids(self) -> set[str]:
        return {r["album_id"] for r in self._c.execute(
            "SELECT album_id FROM adjudication_escalations WHERE status != 'pending'"
        )}

    def _mark(self, album_id: str, *, status: str, decision_genres: "list[str] | None") -> None:
        self._c.execute(
            "UPDATE adjudication_escalations SET status=?, decision_genres_json=?, decided_at=? "
            "WHERE album_id=?",
            (status, json.dumps(decision_genres) if decision_genres is not None else None,
             time.strftime("%Y-%m-%dT%H:%M:%S"), album_id),
        )
        self._c.commit()

    def record_decision(
        self, album_id: str, decision: str, *, genres: "list[str] | None" = None,
        sidecar_store: Any, taxonomy: Any, model: str = "review",
    ) -> None:
        from .adjudication_materializer import materialize_adjudication

        row = self.get(album_id)
        if row is None:
            raise KeyError(f"no escalation queued for album_id={album_id!r}")
        if decision not in ("accept", "edit", "reject"):
            raise ValueError(f"unknown decision {decision!r}")
        if decision == "reject":
            self._mark(album_id, status="rejected", decision_genres=None)
            return
        if decision == "edit":
            terms = list(genres or [])
        else:  # accept -> use the proposed terms as-is
            terms = [g["term"] for g in row["proposed_genres"]]
        response = {"genres": [{"term": t, "confidence": 0.8, "layer": "core"} for t in terms],
                    "facets": [], "escalate": False}
        materialize_adjudication(
            sidecar_store, album_id=album_id, artist=row["artist"], album=row["album"],
            response=response, taxonomy=taxonomy,
            prompt_version=row["prompt_version"], model=model,
        )
        self._mark(album_id, status=("edited" if decision == "edit" else "accepted"),
                   decision_genres=terms)

    def close(self) -> None:
        self._c.close()
