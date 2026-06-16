#!/usr/bin/env python
"""Phase-3 RESUMABLE bulk adjudication pass — shadow, checkpointed, usage-safe.

Writes each release's result to a standalone checkpoint DB (NOT metadata.db, NOT the
production sidecar) the moment it completes, so a killed / usage-limit-hit run loses
NOTHING — rerun and it resumes, skipping every (album_id, prompt_version, input_hash)
already complete. `--max-calls` caps a run to stay under a usage window; a streak of
failures (the rate-limit wall) stops the run cleanly.

Usage:
  python scripts/research/run_adjudicator_bulk.py --source corpus --limit 5       # smoke
  python scripts/research/run_adjudicator_bulk.py --source library                # full pass
  python scripts/research/run_adjudicator_bulk.py --source library --max-calls 500  # one window
  # rerun the same command to resume.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse the eval runner's helpers

import yaml  # noqa: E402

from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import (  # noqa: E402
    ADJUDICATOR_INSTRUCTIONS,
    ADJUDICATOR_PROMPT_VERSION,
    adjudicator_response_format,
    build_adjudicator_payload,
    build_adjudicator_prompt,
    enforce_file_tag_floor,
    validate_adjudicator_response,
)
from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient  # noqa: E402
from src.ai_genre_enrichment.model_prior import stable_input_hash  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

DEFAULT_DB = _ROOT / "data" / "genre_adjudication_shadow.db"
FAIL_STREAK_STOP = 8  # consecutive failures => likely a usage/rate-limit wall


def effective_prompt_version() -> str:
    """Bump automatically whenever the contract instructions change."""
    h = hashlib.sha256(ADJUDICATOR_INSTRUCTIONS.encode("utf-8")).hexdigest()[:8]
    return f"{ADJUDICATOR_PROMPT_VERSION}+{h}"


def target_albums(source: str, meta, corpus_path: str):
    if source == "corpus":
        doc = yaml.safe_load(Path(corpus_path).read_text(encoding="utf-8"))
        return [(e["album_id"], e.get("release_key")) for e in doc["entries"]]
    return [(r[0], None) for r in meta.execute("SELECT album_id FROM albums ORDER BY album_id")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["corpus", "library"], default="corpus")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=0, help="0=unlimited; cap a run to one usage window")
    ap.add_argument("--checkpoint-db", default=str(DEFAULT_DB))
    ap.add_argument("--corpus", default=str(_ROOT / "docs" / "genre_adjudication" / "corpus.yaml"))
    args = ap.parse_args()

    pv = effective_prompt_version()
    store = AdjudicationStore(args.checkpoint_db)
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}
    targets = target_albums(args.source, meta, args.corpus)
    if args.limit:
        targets = targets[: args.limit]

    # Phase A: build payloads + skip already-done (single DB thread).
    todo, skipped = [], 0
    for album_id, release_key in targets:
        evidence = build_evidence(meta, album_id, id2name)
        payload = build_adjudicator_payload(evidence)
        ih = stable_input_hash(payload)
        if store.is_done(album_id, pv, ih):
            skipped += 1
            continue
        todo.append({
            "album_id": album_id, "release_key": release_key, "payload": payload,
            "prompt": build_adjudicator_prompt(payload), "input_hash": ih,
            "file_tags": payload["user_file_tags"],
        })
    meta.close()
    run_now = todo[: args.max_calls] if args.max_calls else todo
    print(f"checkpoint={args.checkpoint_db}\nprompt_version={pv}")
    print(f"targets={len(targets)} already_done={skipped} remaining={len(todo)} -> this run={len(run_now)}")
    if not run_now:
        print(f"nothing to do. stats={store.stats()}")
        store.close()
        return 0

    adapter = load_graph_adapter()

    def _is_broad(name: str) -> bool:
        node = adapter.node(name)
        return bool(node is not None and node.is_broad)

    def worker(item):
        client = ClaudeCodeEnrichmentClient(model=args.model)
        return item, client.request_structured(
            payload=item["payload"], prompt=item["prompt"],
            response_format=adjudicator_response_format(),
            validator=validate_adjudicator_response,
            instructions=ADJUDICATOR_INSTRUCTIONS, estimated_output_tokens=700,
        )

    t0, done, fail_streak, stopped = time.time(), 0, 0, False
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, it) for it in run_now]
        for fut in as_completed(futures):
            item, res = fut.result()
            if res.status == "complete":
                r = enforce_file_tag_floor(
                    res.response_json, file_tags=item["file_tags"],
                    canonicalize_fn=adapter.canonicalize_tag, is_broad_fn=_is_broad,
                )
                store.save(
                    album_id=item["album_id"], prompt_version=pv, release_key=item["release_key"],
                    input_hash=item["input_hash"], model=args.model, status="complete",
                    response=r, dropped_file_tags=r.get("dropped_file_tags", []), tokens=res.token_usage,
                )
                fail_streak = 0
            else:
                store.save(
                    album_id=item["album_id"], prompt_version=pv, release_key=item["release_key"],
                    input_hash=item["input_hash"], model=args.model, status="failed", error=res.error_message,
                )
                fail_streak += 1
            done += 1
            if done % 20 == 0 or done == len(run_now):
                print(f"  {done}/{len(run_now)} [{round(time.time() - t0)}s] "
                      f"tokens={store.total_tokens()} fail_streak={fail_streak}")
            if fail_streak >= FAIL_STREAK_STOP:
                print(f"  !! {FAIL_STREAK_STOP} consecutive failures — likely a usage/rate-limit wall. "
                      f"Stopping cleanly; rerun to resume.")
                stopped = True
                for f in futures:
                    f.cancel()
                break

    print(f"\n{'STOPPED EARLY' if stopped else 'run complete'}. "
          f"stats={store.stats()} total_tokens={store.total_tokens()}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
