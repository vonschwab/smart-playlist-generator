"""Incremental, resumable single-model adjudication runner (library form).

Promoted from scripts/research/run_adjudicator_bulk.py's main loop. Builds the to-do
set (skipping albums already complete in the checkpoint), runs one structured call per
album through an injected client, and checkpoints each result the moment it lands.
"""
from __future__ import annotations

from dataclasses import dataclass

from .album_adjudicator import (
    adjudicator_response_format,
    build_adjudicator_payload,
    build_adjudicator_prompt,
    enforce_file_tag_floor,
    validate_adjudicator_response,
)
from .album_evidence import build_evidence
from .model_prior import stable_input_hash

FAIL_STREAK_STOP = 8


@dataclass
class AdjudicationRunSummary:
    adjudicated: int
    failed: int
    paused: bool
    pause_reason: "str | None"


class _StopRun(Exception):
    pass


def build_todo(store, conn, id2name, album_ids, *, prompt_version) -> list[dict]:
    todo: list[dict] = []
    for album_id in album_ids:
        evidence = build_evidence(conn, album_id, id2name)
        payload = build_adjudicator_payload(evidence)
        ih = stable_input_hash(payload)
        if store.is_done(album_id, prompt_version, ih):
            continue
        todo.append({
            "album_id": album_id, "release_key": None, "payload": payload,
            "prompt": build_adjudicator_prompt(payload), "input_hash": ih,
            "file_tags": payload["user_file_tags"],
        })
    return todo


def run_adjudication(store, todo, *, model, instructions, prompt_version, adapter,
                     client, reset_every: int = 25) -> AdjudicationRunSummary:
    prep = {it["album_id"]: it for it in todo}
    state = {"done": 0, "failed": 0, "fail_streak": 0, "paused": False, "reason": None}

    def _is_broad(name: str) -> bool:
        node = adapter.node(name)
        return bool(node is not None and getattr(node, "is_broad", False))

    def on_result(album_id, parsed, err, usage):
        it = prep[album_id]
        if parsed is not None:
            r = enforce_file_tag_floor(
                parsed, file_tags=it["file_tags"],
                canonicalize_fn=adapter.canonicalize_tag, is_broad_fn=_is_broad,
            )
            store.save(
                album_id=album_id, prompt_version=prompt_version, release_key=it["release_key"],
                input_hash=it["input_hash"], model=model, status="complete",
                response=r, dropped_file_tags=r.get("dropped_file_tags", []), tokens=usage or {},
            )
            state["fail_streak"] = 0
            state["done"] += 1
        else:
            store.save(
                album_id=album_id, prompt_version=prompt_version, release_key=it["release_key"],
                input_hash=it["input_hash"], model=model, status="failed", error=err,
            )
            state["failed"] += 1
            state["fail_streak"] += 1
        if state["fail_streak"] >= FAIL_STREAK_STOP:
            state["paused"] = True
            state["reason"] = f"{FAIL_STREAK_STOP} consecutive failures (likely usage wall)"
            raise _StopRun()

    items = [(it["album_id"], it["prompt"]) for it in todo]
    if items:
        try:
            client.call_structured_session(
                items, response_format=adjudicator_response_format(),
                validator=validate_adjudicator_response, instructions=instructions,
                on_result=on_result, reset_every=reset_every,
            )
        except _StopRun:
            pass
    return AdjudicationRunSummary(
        adjudicated=state["done"], failed=state["failed"],
        paused=state["paused"], pause_reason=state["reason"],
    )
