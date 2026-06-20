#!/usr/bin/env python
"""Human review of the escalated adjudications (album-grain). Reads pending escalations
from the sidecar EscalationQueue. accept/edit -> materialize via queue.record_decision();
reject -> leave the album's existing authority untouched.

Usage:
  python scripts/research/review_escalated.py            # interactive review
  python scripts/research/review_escalated.py --apply    # materialize decided accept/edit
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_album_adjudicator import resolve_db  # noqa: E402

from src.ai_genre_enrichment.escalation_queue import EscalationQueue  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402


def parse_decision(line: str) -> tuple[str, list[str]]:
    """Parse a user decision line.

    "accept"                    -> ("accept", [])
    "reject"                    -> ("reject", [])
    "skip"                      -> ("skip", [])
    "quit"                      -> ("quit", [])
    "edit a, b ,c"              -> ("edit", ["a", "b", "c"])
    """
    line = line.strip()
    if line.startswith("edit"):
        rest = line[len("edit"):].strip()
        genres = [g.strip() for g in rest.split(",") if g.strip()]
        return ("edit", genres)
    word = line.split()[0] if line.split() else ""
    return (word, [])


class ReviewDecisionStore:
    """Tiny SQLite-backed store for escalation review decisions. Resumable."""

    def __init__(self, db_path: "str | Path") -> None:
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path)
        self._c.execute(
            "CREATE TABLE IF NOT EXISTS escalation_decisions "
            "(album_id TEXT PRIMARY KEY, decision TEXT, genres_json TEXT, updated_at TEXT)"
        )
        self._c.commit()

    def save(self, album_id: str, decision: str, genres: list[str]) -> None:
        import time
        self._c.execute(
            "INSERT INTO escalation_decisions (album_id, decision, genres_json, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET "
            "decision=excluded.decision, genres_json=excluded.genres_json, "
            "updated_at=excluded.updated_at",
            (album_id, decision, json.dumps(genres), time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self._c.commit()

    def get(self, album_id: str) -> "dict | None":
        row = self._c.execute(
            "SELECT decision, genres_json FROM escalation_decisions WHERE album_id=?",
            (album_id,),
        ).fetchone()
        return {"decision": row[0], "genres": json.loads(row[1])} if row else None

    def decided_ids(self) -> set[str]:
        return {r[0] for r in self._c.execute("SELECT album_id FROM escalation_decisions")}

    def close(self) -> None:
        self._c.close()


def _sidecar_path() -> str:
    return str(resolve_db("ai_genre_enrichment.db"))


def apply_decisions(*, sidecar_path: str, decisions: dict) -> int:
    """Materialize accept/edit decisions via the queue; reject is a no-op. Returns count applied."""
    store = SidecarStore(sidecar_path)
    store.initialize()
    queue = EscalationQueue(sidecar_path)
    taxonomy = load_default_layered_taxonomy()
    n = 0
    for album_id, (decision, genres) in decisions.items():
        if decision not in ("accept", "edit", "reject"):
            continue
        queue.record_decision(album_id, decision, genres=genres or None,
                              sidecar_store=store, taxonomy=taxonomy)
        if decision in ("accept", "edit"):
            n += 1
    queue.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Human review of escalated album adjudications.")
    ap.add_argument("--apply", action="store_true",
                    help="materialize the decisions captured interactively (no prompts)")
    args = ap.parse_args()

    sidecar = _sidecar_path()
    queue = EscalationQueue(sidecar)
    pending = queue.list_pending()
    print(f"pending escalations: {len(pending)}")
    decisions: dict = {}
    for i, row in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {row['artist']} — {row['album']}")
        print(f"   prior    = {row['prior_observed_leaf']}")
        print(f"   proposed = {[g['term'] for g in row['proposed_genres']]}")
        print(f"   reason   = {row['escalate_reason']}")
        if row["dropped_file_tags"]:
            print(f"   DROPPED FILE TAGS = {row['dropped_file_tags']}")
        line = input("   [accept / reject / edit a,b,c / skip / quit] > ")
        decision, genres = parse_decision(line)
        if decision == "quit":
            break
        if decision == "skip":
            continue
        if decision in ("accept", "reject", "edit"):
            decisions[row["album_id"]] = (decision, genres)
    queue.close()
    n = apply_decisions(sidecar_path=sidecar, decisions=decisions)
    print(f"applied {n} accept/edit decisions (reject = no-op)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
