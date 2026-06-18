#!/usr/bin/env python
"""Human review of the escalated adjudications (album-grain). Resumable: decisions
persist to escalation_decisions in the shadow DB. accept/edit -> materialize via the
SAME path as the auto lane; reject -> leave the album's existing authority untouched.

Usage:
  python scripts/research/review_escalated.py            # interactive review
  python scripts/research/review_escalated.py --apply    # materialize decided accept/edit
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adjudicator_bulk import effective_prompt_version  # noqa: E402
from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_materializer import materialize_adjudication  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import canonicalize_proposed  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402
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


def _escalated(shadow_db: str, tho_pv: str) -> dict:
    """Return best response per album_id for albums that have escalate=True."""
    conn = sqlite3.connect(f"file:{shadow_db}?mode=ro", uri=True)
    raw = [
        (a, pv, json.loads(rj))
        for a, pv, rj in conn.execute(
            "SELECT album_id, prompt_version, response_json FROM adjudications WHERE status='complete'"
        )
    ]
    conn.close()
    best: dict = {}
    for a, pv, resp in raw:
        if a not in best or pv == tho_pv:
            best[a] = resp
    return {a: r for a, r in best.items() if r.get("escalate")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Human review of escalated album adjudications."
    )
    ap.add_argument(
        "--shadow-db",
        default=str(_ROOT / "data" / "adjudication_pass1.db"),
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="materialize decided accept/edit decisions (no interactive prompts)",
    )
    args = ap.parse_args()

    tho_pv = effective_prompt_version(thorough=True)
    std_pv = effective_prompt_version(thorough=False)
    escalated = _escalated(args.shadow_db, tho_pv)
    decisions = ReviewDecisionStore(args.shadow_db)
    adapter = load_graph_adapter()
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {
        r[0]: r[1]
        for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")
    }

    if args.apply:
        taxonomy = load_default_layered_taxonomy()
        store = SidecarStore(str(resolve_db("ai_genre_enrichment.db")))
        n = 0
        for album_id in escalated:
            d = decisions.get(album_id)
            if not d or d["decision"] not in ("accept", "edit"):
                continue
            ev = build_evidence(meta, album_id, id2name)
            resp = escalated[album_id]
            if d["decision"] == "edit":
                resp = {
                    **resp,
                    "genres": [
                        {"term": g, "confidence": 0.8, "layer": "core"}
                        for g in d["genres"]
                    ],
                }
            materialize_adjudication(
                store,
                album_id=album_id,
                artist=ev["artist"],
                album=ev["album"],
                response=resp,
                taxonomy=taxonomy,
                prompt_version=std_pv,
                model="review",
            )
            n += 1
        meta.close()
        decisions.close()
        print(f"applied {n} accept/edit decisions")
        return 0

    # Interactive review loop
    done = decisions.decided_ids()
    todo = [a for a in escalated if a not in done]
    print(f"escalated={len(escalated)} decided={len(done)} remaining={len(todo)}")
    for i, album_id in enumerate(todo, 1):
        ev = build_evidence(meta, album_id, id2name)
        resp = escalated[album_id]
        canon = canonicalize_proposed(
            [g["term"] for g in resp["genres"]], adapter.canonicalize_tag
        )["canonical"]
        print(f"\n[{i}/{len(todo)}] {ev['artist']} — {ev['album']}")
        print(f"   prior    = {ev['current_observed_leaf']}")
        print(f"   proposed = {canon}")
        print(f"   reason   = {resp.get('escalate_reason', '')}")
        if resp.get("dropped_file_tags"):
            print(f"   DROPPED FILE TAGS = {resp['dropped_file_tags']}")
        line = input("   [accept / reject / edit a,b,c / skip / quit] > ")
        decision, genres = parse_decision(line)
        if decision == "quit":
            break
        if decision == "skip":
            continue
        if decision in ("accept", "reject", "edit"):
            decisions.save(album_id, decision, genres)

    meta.close()
    decisions.close()
    print("review session saved. Re-run to resume; then --apply to materialize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
