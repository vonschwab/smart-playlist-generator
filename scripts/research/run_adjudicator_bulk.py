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

Second-pass (thorough) for shallow results (≤2 genres, not escalated):
  python scripts/research/run_adjudicator_bulk.py --source shallow --model sonnet
  # --thorough is implied by --source shallow; stored under a different prompt_version key.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse the eval runner's helpers

import yaml  # noqa: E402

from run_album_adjudicator import build_evidence, open_ro, resolve_db  # noqa: E402

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore  # noqa: E402
from src.ai_genre_enrichment.album_adjudicator import (  # noqa: E402
    ADJUDICATOR_INSTRUCTIONS,
    ADJUDICATOR_INSTRUCTIONS_THOROUGH,
    ADJUDICATOR_PROMPT_VERSION,
    ADJUDICATOR_PROMPT_VERSION_THOROUGH,
    adjudicator_response_format,
    build_adjudicator_payload,
    build_adjudicator_prompt,
    canonicalize_proposed,
    enforce_file_tag_floor,
    validate_adjudicator_response,
)
from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient  # noqa: E402
from src.ai_genre_enrichment.model_prior import stable_input_hash  # noqa: E402
from src.genre.graph_adapter import load_graph_adapter  # noqa: E402

DEFAULT_DB = _ROOT / "data" / "adjudication_pass1.db"
FAIL_STREAK_STOP = 8  # consecutive failures => likely a usage/rate-limit wall


def effective_prompt_version(thorough: bool = False) -> str:
    """Bump automatically whenever the contract instructions change."""
    instructions = ADJUDICATOR_INSTRUCTIONS_THOROUGH if thorough else ADJUDICATOR_INSTRUCTIONS
    base = ADJUDICATOR_PROMPT_VERSION_THOROUGH if thorough else ADJUDICATOR_PROMPT_VERSION
    h = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]
    return f"{base}+{h}"


class _StopRun(Exception):
    """Raised from the per-item callback to stop a session run cleanly (resumable)."""


def exclude_done_album_ids(targets, done_ids):
    """Drop targets whose album_id already has a complete adjudication row.

    Unlike the runner's per-item `is_done` check (which keys on input_hash and so
    re-runs an album once its evidence drifts), this excludes by album_id alone.
    Needed for the remaining-library pass: publishing the first pass rewrote
    `current_observed_leaf` for the already-done albums, changing their input_hash —
    without this they would all be re-adjudicated.
    """
    done = set(done_ids)
    return [(aid, key) for (aid, key) in targets if aid not in done]


def target_albums(source: str, meta, corpus_path: str, overtag_min: int, store=None):
    if source == "corpus":
        doc = yaml.safe_load(Path(corpus_path).read_text(encoding="utf-8"))
        return [(e["album_id"], e.get("release_key")) for e in doc["entries"]]
    if source == "targeted":
        # high-value: sparse (0 observed_leaf) OR over-tagged (>= overtag_min).
        counts = {r[0]: r[1] for r in meta.execute(
            "SELECT album_id, COUNT(*) FROM release_effective_genres "
            "WHERE assignment_layer='observed_leaf' GROUP BY album_id")}
        return [(aid, None) for (aid,) in meta.execute("SELECT album_id FROM albums ORDER BY album_id")
                if counts.get(aid, 0) == 0 or counts.get(aid, 0) >= overtag_min]
    if source == "shallow":
        # Second-pass targets: albums where the standard pass returned ≤2 genres, not escalated.
        if store is None:
            raise ValueError("--source shallow requires a checkpoint DB (--checkpoint-db)")
        standard_pv = effective_prompt_version(thorough=False)
        album_ids = store.shallow_album_ids(standard_pv)
        return [(aid, None) for aid in album_ids]
    return [(r[0], None) for r in meta.execute("SELECT album_id FROM albums ORDER BY album_id")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["corpus", "library", "targeted", "shallow"], default="targeted")
    ap.add_argument("--thorough", action="store_true",
                    help="use the completeness-nudged second-pass instructions (auto-set for --source shallow)")
    ap.add_argument("--overtag-min", type=int, default=7,
                    help="targeted: min observed_leaf count for 'over-tagged' (sparse=0 always included)")
    ap.add_argument("--exclude-done", action="store_true",
                    help="skip albums with any complete adjudication row (by album_id, ignoring "
                         "input_hash). Use for the remaining-library pass after the first pass "
                         "was published — otherwise evidence drift re-adjudicates done albums.")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=0, help="0=unlimited; cap a run to one usage window")
    ap.add_argument("--reset-every", type=int, default=25, help="fresh SDK session every N items (bounds context)")
    ap.add_argument("--checkpoint-db", default=str(DEFAULT_DB))
    ap.add_argument("--corpus", default=str(_ROOT / "docs" / "genre_adjudication" / "corpus.yaml"))
    args = ap.parse_args()

    thorough = args.thorough or args.source == "shallow"
    instructions = ADJUDICATOR_INSTRUCTIONS_THOROUGH if thorough else ADJUDICATOR_INSTRUCTIONS
    pv = effective_prompt_version(thorough=thorough)
    store = AdjudicationStore(args.checkpoint_db)
    meta = open_ro(resolve_db("metadata.db"))
    id2name = {r[0]: r[1] for r in meta.execute("SELECT genre_id, name FROM genre_graph_canonical_genres")}
    targets = target_albums(args.source, meta, args.corpus, args.overtag_min, store=store)
    if args.exclude_done:
        done_ids = {row["album_id"] for row in store.iter_complete()}
        before = len(targets)
        targets = exclude_done_album_ids(targets, done_ids)
        print(f"exclude_done: {before} -> {len(targets)} ({before - len(targets)} already complete)")
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

    # Single persistent session (cached prefix, one process => no config race), per-item
    # checkpoint via on_result, fresh session every --reset-every items to bound context.
    client = ClaudeCodeEnrichmentClient(model=args.model)
    prep_by_id = {it["album_id"]: it for it in run_now}
    state = {"done": 0, "fail_streak": 0, "stopped": False}
    t0 = time.time()
    total = len(run_now)

    def on_result(album_id, parsed, err, usage):
        it = prep_by_id[album_id]
        artist = (it["payload"].get("artist") or "?")[:24]
        album = (it["payload"].get("album") or "?")[:30]
        if parsed is not None:
            r = enforce_file_tag_floor(
                parsed, file_tags=it["file_tags"],
                canonicalize_fn=adapter.canonicalize_tag, is_broad_fn=_is_broad,
            )
            canon = canonicalize_proposed([g["term"] for g in r["genres"]], adapter.canonicalize_tag)["canonical"]
            store.save(
                album_id=album_id, prompt_version=pv, release_key=it["release_key"],
                input_hash=it["input_hash"], model=args.model, status="complete",
                response=r, dropped_file_tags=r.get("dropped_file_tags", []), tokens=usage or {},
            )
            state["fail_streak"] = 0
            tail = f"-> {canon}" + ("  [ESC]" if r.get("escalate") else "")
        else:
            store.save(
                album_id=album_id, prompt_version=pv, release_key=it["release_key"],
                input_hash=it["input_hash"], model=args.model, status="failed", error=err,
            )
            state["fail_streak"] += 1
            tail = f"!! FAILED ({(err or '')[:40]})"
        state["done"] += 1
        print(f"  [{state['done']}/{total}] {artist} — {album}  {tail}", flush=True)
        if state["done"] % 25 == 0:
            print(f"  ---- {state['done']}/{total}  {round(time.time()-t0)}s  "
                  f"tokens={store.total_tokens()}  status={store.stats()} ----", flush=True)
        if state["fail_streak"] >= FAIL_STREAK_STOP:
            state["stopped"] = True
            raise _StopRun()

    items = [(it["album_id"], it["prompt"]) for it in run_now]
    try:
        client.call_structured_session(
            items, response_format=adjudicator_response_format(),
            validator=validate_adjudicator_response, instructions=instructions,
            on_result=on_result, reset_every=args.reset_every,
        )
    except _StopRun:
        print(f"  !! {FAIL_STREAK_STOP} consecutive failures — stopping cleanly; rerun to resume.", flush=True)

    print(f"\n{'STOPPED EARLY' if state['stopped'] else 'run complete'}. "
          f"stats={store.stats()} total_tokens={store.total_tokens()}", flush=True)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
