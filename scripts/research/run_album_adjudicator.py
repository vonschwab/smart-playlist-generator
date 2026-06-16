#!/usr/bin/env python
"""Phase-1 shadow runner: album-adjudicator-v1 over the gold corpus.

Reads `docs/genre_adjudication/corpus.yaml`, builds a read-only evidence payload per
release from metadata.db (resolved to the MAIN checkout — a worktree's data/ lacks the
gitignored .db), calls Claude (Haiku by default) with the album-adjudicator-v1 contract,
canonicalizes the proposed terms against the taxonomy, and writes a non-authoritative
shadow JSON under docs/genre_adjudication/shadow/. Writes NOTHING to any database.

Usage:
  python scripts/research/run_album_adjudicator.py --limit 3            # smoke test
  python scripts/research/run_album_adjudicator.py                      # full 50
  python scripts/research/run_album_adjudicator.py --dry-run            # no API, estimates only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from src.ai_genre_enrichment.album_adjudicator import (  # noqa: E402
    ADJUDICATOR_INSTRUCTIONS,
    adjudicator_response_format,
    build_adjudicator_payload,
    build_adjudicator_prompt,
    canonicalize_proposed,
    validate_adjudicator_response,
)
from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

CORPUS = _ROOT / "docs" / "genre_adjudication" / "corpus.yaml"
SHADOW_DIR = _ROOT / "docs" / "genre_adjudication" / "shadow"


def resolve_db(name: str) -> Path:
    """Prefer a data/<name> that actually exists (main checkout in a worktree)."""
    candidates = [_ROOT / "data" / name]
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=str(_ROOT), text=True
        ).strip()
        candidates.insert(0, Path(common).resolve().parent / "data" / name)
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"could not locate data/{name}")


def open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build_evidence(conn: sqlite3.Connection, album_id: str, id2name: dict[str, str]) -> dict:
    row = conn.execute(
        "SELECT artist, title, release_year, musicbrainz_release_id FROM albums WHERE album_id=?",
        (album_id,),
    ).fetchone()
    artist, title, year, mbid = row if row else (None, None, None, None)
    tracks = [r[0] for r in conn.execute(
        "SELECT title FROM tracks WHERE album_id=? AND title IS NOT NULL LIMIT 8", (album_id,)
    )]
    by_source: dict[str, list[str]] = {}
    for genre, source in conn.execute(
        "SELECT DISTINCT genre, source FROM album_genres WHERE album_id=? AND genre NOT IN ('__EMPTY__','__empty__')",
        (album_id,),
    ):
        by_source.setdefault((source or "album").lower(), []).append(genre)
    trk = [g for (g,) in conn.execute(
        "SELECT DISTINCT tg.genre FROM track_genres tg JOIN tracks t ON t.track_id=tg.track_id "
        "WHERE t.album_id=? AND tg.genre NOT IN ('__EMPTY__','__empty__')",
        (album_id,),
    )]
    if trk:
        by_source["file_track"] = sorted(set(trk))
    observed = [
        id2name[g] for (g,) in conn.execute(
            "SELECT genre_id FROM release_effective_genres WHERE album_id=? AND assignment_layer='observed_leaf'",
            (album_id,),
        ) if g in id2name
    ]
    identifiers = {"mbid": mbid} if mbid else {}
    return {
        "artist": artist, "album": title, "album_id": album_id, "year": year,
        "identifiers": identifiers, "track_titles": tracks,
        "existing_genres_by_source": by_source, "current_observed_leaf": observed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bucket", default=None, help="filter to one bucket")
    ap.add_argument("--concurrency", type=int, default=6, help="parallel Claude calls")
    args = ap.parse_args()

    corpus = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    entries = corpus["entries"]
    if args.bucket:
        entries = [e for e in entries if e["bucket"] == args.bucket]
    if args.limit:
        entries = entries[: args.limit]

    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}
    # Phase A: build all payloads sequentially off the single DB connection.
    preps = []
    for entry in entries:
        evidence = build_evidence(meta, entry["album_id"], id2name)
        payload = build_adjudicator_payload(evidence)
        preps.append({
            "entry": entry, "evidence": evidence, "payload": payload,
            "prompt": build_adjudicator_prompt(payload),
        })
    meta.close()

    # Phase B: fan out the INDEPENDENT Claude calls. Per-call client (cheap) avoids
    # the shared last_token_usage race; concurrency overlaps the per-call SDK startup.
    def _adjudicate(prep):
        client = ClaudeCodeEnrichmentClient(model=args.model, dry_run=args.dry_run)
        return prep["entry"]["album_id"], client.request_structured(
            payload=prep["payload"], prompt=prep["prompt"],
            response_format=adjudicator_response_format(),
            validator=validate_adjudicator_response,
            instructions=ADJUDICATOR_INSTRUCTIONS,
            estimated_output_tokens=700,
        )

    t0 = time.time()
    res_by_id: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(_adjudicate, p) for p in preps]
        for n, fut in enumerate(as_completed(futures), 1):
            aid, res = fut.result()
            res_by_id[aid] = res
            print(f"  ({n}/{len(preps)}) [{round(time.time() - t0)}s]")

    # Phase C: canonicalize + assemble in corpus order (sequential; adapter cache).
    adapter = load_graph_adapter()
    results = []
    for prep in preps:
        entry, evidence, payload = prep["entry"], prep["evidence"], prep["payload"]
        res = res_by_id[entry["album_id"]]
        rec = {
            "release_key": entry.get("release_key"), "album_id": entry["album_id"],
            "artist": evidence["artist"], "title": evidence["album"], "bucket": entry["bucket"],
            "failure_mode": entry.get("failure_mode"),
            "gold_genres": entry["gold_genres"], "gold_facets": entry.get("gold_facets", []),
            "must_preserve": entry.get("must_preserve", []),
            "taxonomy_gaps_gold": entry.get("taxonomy_gaps", []),
            "current_observed_leaf": payload["current_observed_leaf"],
        }
        if res.status == "complete":
            r = res.response_json
            canon = canonicalize_proposed([g["term"] for g in r["genres"]], adapter.canonicalize_tag)
            rec["proposed"] = {
                "genres": r["genres"], "canonical": canon["canonical"], "gaps": canon["gaps"],
                "facets": r["facets"], "escalate": r["escalate"],
                "escalate_reason": r["escalate_reason"], "overall_confidence": r["overall_confidence"],
                "warnings": r["warnings"], "tokens": res.token_usage,
            }
            print(f"  {evidence['artist']} — {evidence['album']}: {canon['canonical']}"
                  + (f"  GAPS={canon['gaps']}" if canon['gaps'] else "")
                  + ("  [ESCALATE]" if r["escalate"] else ""))
        else:
            rec["proposed"] = {"status": res.status, "error": res.error_message}
            print(f"  {evidence['artist']} — {evidence['album']}: {res.status} {res.error_message or ''}")
        results.append(rec)

    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = SHADOW_DIR / f"run_{args.model}_{stamp}{'_dryrun' if args.dry_run else ''}.json"
    out.write_text(json.dumps({
        "model": args.model, "dry_run": args.dry_run, "prompt_version": "album-adjudicator-v1",
        "count": len(results), "elapsed_s": round(time.time() - t0, 1), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(results)} releases, {round(time.time()-t0,1)}s)")
    meta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
