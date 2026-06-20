# Analyze Adjudicate Stage (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the album-grain Sonnet adjudicator a first-class pair of Analyze Library stages (`adjudicate` → `apply`) that replace the tag-grain `enrich` stage, holding escalations in a durable queue.

**Architecture:** Promote the proven shadow-flow research code (`build_evidence`, the bulk-runner loop, the apply logic) into importable `src/ai_genre_enrichment/` modules. Add two thin stages to `scripts/analyze_library.py`: `adjudicate` runs one Sonnet call per new/changed album and checkpoints raw responses into the sidecar; `apply` deterministically materializes non-escalated results and enqueues escalated ones. `publish` is unchanged. A new `EscalationQueue` store is the contract the future GUI (SP2) builds on.

**Tech Stack:** Python 3.11+, SQLite (sidecar `data/ai_genre_enrichment.db`), the existing `album_adjudicator` contract, `ClaudeCodeEnrichmentClient` (Agent SDK / Claude Max), pytest.

## Global Constraints

- **Single Sonnet pass per album in the pipeline.** Model default for the `adjudicate` stage is `sonnet`; prompt_version is the standard (non-thorough) `effective_prompt_version(thorough=False)`.
- **Hold + queue escalations.** Escalated albums are never materialized by `apply`; they go to `adjudication_escalations` (status `pending`). Prior authority is preserved.
- **File-tag-floor drops force-escalate** (already in the contract via `enforce_file_tag_floor`) — they must never auto-publish.
- **Unknown terms are skipped, never invented** (the materializer already does this).
- **A configured knob that can't act is a startup error, not a silent no-op** (CLAUDE.md). Misconfig raises; transient LLM failures return a `paused` result.
- **No live `metadata.db` / sidecar writes in tests.** Use `tmp_path` temp DBs and an injected fake adjudicator client.
- **Checkpoint lives in the sidecar** (`data/ai_genre_enrichment.db`), table `adjudications` (created by `AdjudicationStore`), alongside the layered-graph tables.
- **`publish()` remains the sole writer of `release_effective_genres`.** These stages write only the sidecar.

## Reference signatures (already exist — consume, don't redefine)

- `src/ai_genre_enrichment/adjudication_materializer.py`:
  `materialize_adjudication(store, *, album_id, artist, album, response, taxonomy, prompt_version, model) -> AdjudicationMaterializeSummary`
  and `compute_adjudication_rows(response, taxonomy, *, prompt_version, model) -> (genre_rows, facet_rows, skipped)`.
- `src/ai_genre_enrichment/adjudication_store.py`: `AdjudicationStore(db_path)` with
  `is_done(album_id, pv, input_hash) -> bool`,
  `save(*, album_id, prompt_version, status, release_key=None, input_hash=None, model=None, response=None, dropped_file_tags=None, tokens=None, error=None)`,
  `complete_album_ids(pv) -> set[str]`, `iter_complete()`, `stats()`.
- `src/ai_genre_enrichment/album_adjudicator.py`: `ADJUDICATOR_INSTRUCTIONS`, `ADJUDICATOR_PROMPT_VERSION`,
  `build_adjudicator_payload(evidence) -> dict`, `build_adjudicator_prompt(payload) -> str`,
  `adjudicator_response_format()`, `validate_adjudicator_response`,
  `enforce_file_tag_floor(parsed, *, file_tags, canonicalize_fn, is_broad_fn) -> dict`,
  `canonicalize_proposed(terms, canonicalize_fn) -> {"canonical": [...]}`.
- `src/ai_genre_enrichment/model_prior.py`: `stable_input_hash(payload) -> str`.
- `src/ai_genre_enrichment/claude_client.py`: `ClaudeCodeEnrichmentClient(model=...)` with
  `call_structured_session(items, *, response_format, validator, instructions, on_result, reset_every)`,
  where `items = [(album_id, prompt), ...]` and `on_result(album_id, parsed, err, usage)`.
- `src/ai_genre_enrichment/storage.py`: `SidecarStore(path)` with
  `replace_layered_assignments_for_release(*, release_id, artist, album, genre_assignments, facet_assignments)`.
- `src/ai_genre_enrichment/layered_taxonomy.py`: `load_default_layered_taxonomy()`.
- `src/genre/graph_adapter.py`: `load_graph_adapter()` → adapter with `.canonicalize_tag(tag)` and `.node(name).is_broad`.
- `scripts/research/run_adjudicator_bulk.py`: `effective_prompt_version(thorough=False) -> str`.
- `scripts/analyze_library.py`: `STAGE_FUNCS` dict, `STAGE_ORDER_DEFAULT` list, `ENRICHMENT_DB_PATH`,
  `compute_stage_fingerprint(ctx, stage)`, `estimate_stage_units(ctx, stage)`, `_hash_obj(obj)`,
  stages take `ctx: Dict` and return a result dict; a `{"paused": True, "pause_reason": ...}` result
  pauses the pipeline.

---

## Task 1: Promote `build_evidence` to `src/ai_genre_enrichment/album_evidence.py`

**Files:**
- Create: `src/ai_genre_enrichment/album_evidence.py`
- Modify: `scripts/research/run_album_adjudicator.py` (re-export for back-compat)
- Test: `tests/unit/test_album_evidence.py`

**Interfaces:**
- Produces: `build_evidence(conn: sqlite3.Connection, album_id: str, id2name: dict[str, str]) -> dict`
  returning keys `artist, album, album_id, year, identifiers, track_titles, file_tags,
  existing_genres_by_source, current_observed_leaf`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_album_evidence.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.album_evidence import build_evidence


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO tracks VALUES ('t1','a1','Alison');
        INSERT INTO album_genres VALUES ('a1','shoegaze','discogs_release');
        INSERT INTO track_genres VALUES ('t1','dream pop');
        """
    )
    conn.commit()
    return conn


def test_build_evidence_collects_sources_titles_and_observed(tmp_path):
    conn = _db(tmp_path)
    ev = build_evidence(conn, "a1", id2name={})
    assert ev["artist"] == "Slowdive"
    assert ev["album"] == "Souvlaki"
    assert ev["year"] == 1993
    assert ev["track_titles"] == ["Alison"]
    assert ev["file_tags"] == ["dream pop"]
    assert ev["existing_genres_by_source"]["discogs_release"] == ["shoegaze"]
    assert ev["existing_genres_by_source"]["file_track"] == ["dream pop"]
    assert ev["current_observed_leaf"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_album_evidence.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.album_evidence'`.

- [ ] **Step 3: Create the module by moving the function verbatim**

Create `src/ai_genre_enrichment/album_evidence.py` with the exact body currently in
`scripts/research/run_album_adjudicator.py::build_evidence` (do not alter behavior):

```python
"""Per-album evidence assembly for the album adjudicator.

Promoted from scripts/research/run_album_adjudicator.py so the analyze pipeline stages
can import it. Reads only metadata.db tables (albums, tracks, album_genres, track_genres,
release_effective_genres).
"""
from __future__ import annotations

import sqlite3


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
    file_tags = sorted(set(trk))
    observed = [
        id2name[g] for (g,) in conn.execute(
            "SELECT genre_id FROM release_effective_genres WHERE album_id=? AND assignment_layer='observed_leaf'",
            (album_id,),
        ) if g in id2name
    ]
    identifiers = {"mbid": mbid} if mbid else {}
    return {
        "artist": artist, "album": title, "album_id": album_id, "year": year,
        "identifiers": identifiers, "track_titles": tracks, "file_tags": file_tags,
        "existing_genres_by_source": by_source, "current_observed_leaf": observed,
    }
```

Note: the test DB has no `release_effective_genres` table; the function must tolerate that. The
existing function would raise `OperationalError`. Wrap the observed-leaf query so a missing table
yields `[]`:

```python
    try:
        observed = [
            id2name[g] for (g,) in conn.execute(
                "SELECT genre_id FROM release_effective_genres WHERE album_id=? AND assignment_layer='observed_leaf'",
                (album_id,),
            ) if g in id2name
        ]
    except sqlite3.OperationalError:
        observed = []
```

- [ ] **Step 4: Re-export from the old location for back-compat**

In `scripts/research/run_album_adjudicator.py`, replace the `build_evidence` definition with an import
so existing scripts (`run_adjudicator_bulk.py`, `apply_adjudication.py`, `review_escalated.py`) keep working:

```python
from src.ai_genre_enrichment.album_evidence import build_evidence  # noqa: F401
```

(Leave `open_ro` and `resolve_db` where they are.)

- [ ] **Step 5: Run tests + the existing scripts' importers**

Run: `python -m pytest tests/unit/test_album_evidence.py -q`
Expected: PASS (3 assertions).
Run: `python -c "import sys; sys.path.insert(0,'scripts/research'); from run_album_adjudicator import build_evidence; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/album_evidence.py scripts/research/run_album_adjudicator.py tests/unit/test_album_evidence.py
git commit -m "refactor(genre): promote build_evidence to src/ai_genre_enrichment/album_evidence"
```

---

## Task 2: Compound-facet split in the materializer

**Files:**
- Modify: `src/ai_genre_enrichment/adjudication_materializer.py` (the facet loop in `compute_adjudication_rows`)
- Test: `tests/unit/test_adjudication_materializer.py` (add a case)

**Interfaces:**
- No signature change. Behavior: a facet `term` containing commas (e.g.
  `"grief-stricken, meditative, confessional"`) is split into atomic terms; each atomic term is
  classified and routed to the facet table if it resolves; unresolvable atoms are dropped silently
  (facets are secondary). Genres are never affected.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_adjudication_materializer.py
from src.ai_genre_enrichment.adjudication_materializer import compute_adjudication_rows
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


def test_compound_facet_string_is_split_into_atomic_terms():
    taxonomy = load_default_layered_taxonomy()
    # Use facet atoms known to exist in the taxonomy facet vocabulary.
    response = {
        "genres": [{"term": "shoegaze", "confidence": 0.9}],
        "facets": [{"term": "instrumental, lo-fi"}],
        "overall_confidence": 0.8,
    }
    _, facet_rows, _ = compute_adjudication_rows(
        response, taxonomy, prompt_version="pv", model="sonnet")
    facet_ids = {r["facet_id"] for r in facet_rows}
    # both atoms resolve to facets and are present; the compound string is NOT a single row
    assert "instrumental" in {taxonomy.facet_by_id(fid).name for fid in facet_ids}
    assert "lo-fi" in {taxonomy.facet_by_id(fid).name for fid in facet_ids}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_adjudication_materializer.py::test_compound_facet_string_is_split_into_atomic_terms -q`
Expected: FAIL — the compound string `"instrumental, lo-fi"` classifies as `review`/unknown, so no
facet rows are produced.

- [ ] **Step 3: Implement the split in the facet loop**

In `compute_adjudication_rows`, replace the facet loop:

```python
    for f in response.get("facets", []):
        term = f.get("term", "")
        cls = classify_layered_term(taxonomy, term)
        if cls.term_kind in ("facet", "alias") and cls.canonical_id and taxonomy.facet_by_id(cls.canonical_id):
            _facet(cls.canonical_id, response.get("overall_confidence", 0.8), term)
```

with a version that splits on commas first:

```python
    for f in response.get("facets", []):
        raw = f.get("term", "")
        atoms = [a.strip() for a in raw.split(",") if a.strip()] or [raw]
        for term in atoms:
            cls = classify_layered_term(taxonomy, term)
            if cls.term_kind in ("facet", "alias") and cls.canonical_id and taxonomy.facet_by_id(cls.canonical_id):
                _facet(cls.canonical_id, response.get("overall_confidence", 0.8), term)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_adjudication_materializer.py -q`
Expected: PASS (existing cases + the new one).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/adjudication_materializer.py tests/unit/test_adjudication_materializer.py
git commit -m "feat(genre): split compound facet strings into atomic terms in materializer"
```

---

## Task 3: `EscalationQueue` store

**Files:**
- Create: `src/ai_genre_enrichment/escalation_queue.py`
- Test: `tests/unit/test_escalation_queue.py`

**Interfaces:**
- Consumes: `materialize_adjudication` (Task ref: existing), `load_default_layered_taxonomy`,
  `SidecarStore`.
- Produces:
  - `EscalationQueue(db_path)` with table `adjudication_escalations`.
  - `enqueue(*, album_id, release_key, artist, album, prior_observed_leaf, proposed_genres,
    escalate_reason, dropped_file_tags, prompt_version, model, input_hash) -> None`
    (upsert; re-open a decided row only when `input_hash` differs).
  - `list_pending() -> list[dict]`, `get(album_id) -> dict | None`, `decided_ids() -> set[str]`.
  - `record_decision(album_id, decision, *, genres=None, sidecar_store, taxonomy, model="review") -> None`
    where `decision in {"accept","edit","reject"}`; accept/edit materialize, reject does not.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_escalation_queue.py
from __future__ import annotations

from src.ai_genre_enrichment.escalation_queue import EscalationQueue


def _enq(q, album_id="a1", proposed=("shoegaze",), input_hash="h1"):
    q.enqueue(
        album_id=album_id, release_key="slowdive::souvlaki", artist="Slowdive",
        album="Souvlaki", prior_observed_leaf=["indie rock"],
        proposed_genres=[{"term": g, "confidence": 0.9} for g in proposed],
        escalate_reason="sparse", dropped_file_tags=[], prompt_version="pv",
        model="sonnet", input_hash=input_hash,
    )


def test_enqueue_and_list_pending(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q)
    pending = q.list_pending()
    assert len(pending) == 1
    row = pending[0]
    assert row["album_id"] == "a1"
    assert row["prior_observed_leaf"] == ["indie rock"]
    assert row["proposed_genres"][0]["term"] == "shoegaze"
    assert row["status"] == "pending"


def test_decided_row_not_reopened_when_proposal_unchanged(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, input_hash="h1")
    q._mark(  # test helper: force a decided state without materializing
        "a1", status="rejected", decision_genres=None)
    _enq(q, input_hash="h1")  # same proposal
    assert q.get("a1")["status"] == "rejected"
    assert q.list_pending() == []


def test_changed_proposal_reopens_decided_row(tmp_path):
    q = EscalationQueue(tmp_path / "side.db")
    _enq(q, input_hash="h1")
    q._mark("a1", status="rejected", decision_genres=None)
    _enq(q, proposed=("dream pop",), input_hash="h2")  # changed proposal
    assert q.get("a1")["status"] == "pending"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_escalation_queue.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.escalation_queue'`.

- [ ] **Step 3: Implement the store (schema + enqueue + reads + `_mark`)**

```python
# src/ai_genre_enrichment/escalation_queue.py
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

    def close(self) -> None:
        self._c.close()
```

- [ ] **Step 4: Run the read/enqueue tests**

Run: `python -m pytest tests/unit/test_escalation_queue.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Write the failing test for `record_decision`**

```python
# add to tests/unit/test_escalation_queue.py
import sqlite3 as _sqlite3

from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


def test_record_decision_accept_materializes_and_marks(tmp_path):
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    q = EscalationQueue(side)
    _enq(q, proposed=("shoegaze",))
    taxonomy = load_default_layered_taxonomy()
    q.record_decision("a1", "accept", sidecar_store=store, taxonomy=taxonomy)
    assert q.get("a1")["status"] == "accepted"
    # an observed_leaf row was written for the release
    conn = _sqlite3.connect(side)
    n = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    conn.close()
    assert n >= 1


def test_record_decision_reject_does_not_materialize(tmp_path):
    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    q = EscalationQueue(side)
    _enq(q)
    taxonomy = load_default_layered_taxonomy()
    q.record_decision("a1", "reject", sidecar_store=store, taxonomy=taxonomy)
    assert q.get("a1")["status"] == "rejected"
    conn = _sqlite3.connect(side)
    n = conn.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments").fetchone()[0]
    conn.close()
    assert n == 0
```

- [ ] **Step 6: Run to verify the new tests fail**

Run: `python -m pytest tests/unit/test_escalation_queue.py -k record_decision -q`
Expected: FAIL with `AttributeError: 'EscalationQueue' object has no attribute 'record_decision'`.

- [ ] **Step 7: Implement `record_decision`**

Add to `EscalationQueue` (imports at top of the method to avoid a module import cycle):

```python
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
```

- [ ] **Step 8: Run the full queue test file**

Run: `python -m pytest tests/unit/test_escalation_queue.py -q`
Expected: PASS (5 tests).

- [ ] **Step 9: Commit**

```bash
git add src/ai_genre_enrichment/escalation_queue.py tests/unit/test_escalation_queue.py
git commit -m "feat(genre): EscalationQueue store (enqueue/list/get/record_decision)"
```

---

## Task 4: `adjudication_runner` — incremental single-model session runner

**Files:**
- Create: `src/ai_genre_enrichment/adjudication_runner.py`
- Test: `tests/unit/test_adjudication_runner.py`

**Interfaces:**
- Consumes: `build_evidence`, `build_adjudicator_payload`, `build_adjudicator_prompt`,
  `stable_input_hash`, `enforce_file_tag_floor`, `validate_adjudicator_response`,
  `adjudicator_response_format`, `AdjudicationStore`, an adapter (`canonicalize_tag`, `node().is_broad`),
  and a client exposing `call_structured_session(items, *, response_format, validator, instructions,
  on_result, reset_every)`.
- Produces:
  - `build_todo(store, conn, id2name, album_ids, *, prompt_version) -> list[dict]` — one work item
    per not-yet-done album: keys `album_id, release_key(None), payload, prompt, input_hash, file_tags`.
  - `@dataclass AdjudicationRunSummary(adjudicated:int, failed:int, paused:bool, pause_reason:str|None)`.
  - `run_adjudication(store, todo, *, model, instructions, prompt_version, adapter, client,
    reset_every=25) -> AdjudicationRunSummary`.

- [ ] **Step 1: Write the failing test (with a fake client)**

```python
# tests/unit/test_adjudication_runner.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.adjudication_runner import build_todo, run_adjudication
from src.ai_genre_enrichment.adjudication_store import AdjudicationStore


class FakeClient:
    """Mimics ClaudeCodeEnrichmentClient.call_structured_session for tests."""
    def __init__(self, responses):
        self._responses = responses  # album_id -> parsed dict (or Exception)

    def call_structured_session(self, items, *, response_format, validator, instructions,
                                on_result, reset_every):
        for album_id, _prompt in items:
            r = self._responses.get(album_id)
            if isinstance(r, Exception):
                on_result(album_id, None, str(r), {})
            else:
                on_result(album_id, r, None, {"total_tokens": 10})


class FakeAdapter:
    def canonicalize_tag(self, tag):
        return tag
    def node(self, name):
        return None


def _meta(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO albums VALUES ('a2','Nowhere','Ride',1990,NULL);
        """
    )
    conn.commit()
    return conn


def _resp(genres, escalate=False):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "overall_confidence": 0.9}


def test_run_adjudication_checkpoints_each_album(tmp_path):
    conn = _meta(tmp_path)
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    assert len(todo) == 2
    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["shoegaze", "dream pop"])})
    summary = run_adjudication(store, todo, model="sonnet", instructions="x",
                               prompt_version="pv", adapter=FakeAdapter(), client=client)
    assert summary.adjudicated == 2
    assert summary.paused is False
    assert store.complete_album_ids("pv") == {"a1", "a2"}


def test_run_adjudication_skips_already_done(tmp_path):
    conn = _meta(tmp_path)
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["shoegaze"])})
    run_adjudication(store, todo, model="sonnet", instructions="x",
                     prompt_version="pv", adapter=FakeAdapter(), client=client)
    todo2 = build_todo(store, conn, {}, ["a1", "a2"], prompt_version="pv")
    assert todo2 == []  # both already complete -> nothing to do
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_adjudication_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.adjudication_runner'`.

- [ ] **Step 3: Implement the runner**

```python
# src/ai_genre_enrichment/adjudication_runner.py
"""Incremental, resumable single-model adjudication runner (library form).

Promoted from scripts/research/run_adjudicator_bulk.py's main loop. Builds the to-do
set (skipping albums already complete in the checkpoint), runs one structured call per
album through an injected client, and checkpoints each result the moment it lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .album_adjudicator import (
    adjudicator_response_format,
    build_adjudicator_payload,
    build_adjudicator_prompt,
    canonicalize_proposed,
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
```

Note: `canonicalize_proposed` is imported for parity with the bulk runner's provenance logging but
is not required by the loop; if a linter flags it as unused, remove the import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_adjudication_runner.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the pause test**

```python
# add to tests/unit/test_adjudication_runner.py
def test_run_adjudication_pauses_after_fail_streak(tmp_path):
    conn = _meta(tmp_path)
    # 8 albums that all fail -> pause
    ids = [f"a{i}" for i in range(8)]
    conn.executemany("INSERT INTO albums VALUES (?,?,?,?,?)",
                     [(i, f"T{i}", f"Art{i}", 2000, None) for i in ids])
    conn.commit()
    store = AdjudicationStore(tmp_path / "side.db")
    todo = build_todo(store, conn, {}, ids, prompt_version="pv")
    client = FakeClient({i: RuntimeError("rate limit") for i in ids})
    summary = run_adjudication(store, todo, model="sonnet", instructions="x",
                               prompt_version="pv", adapter=FakeAdapter(), client=client)
    assert summary.paused is True
    assert summary.failed >= 8
```

- [ ] **Step 6: Run + commit**

Run: `python -m pytest tests/unit/test_adjudication_runner.py -q`
Expected: PASS (3 tests).

```bash
git add src/ai_genre_enrichment/adjudication_runner.py tests/unit/test_adjudication_runner.py
git commit -m "feat(genre): incremental single-model adjudication runner (library form)"
```

---

## Task 5: `adjudication_apply` — best-result selection + materialize/enqueue

**Files:**
- Create: `src/ai_genre_enrichment/adjudication_apply.py`
- Test: `tests/unit/test_adjudication_apply.py`

**Interfaces:**
- Consumes: `AdjudicationStore.iter_complete` is not pv-aware; this module reads the checkpoint via a
  passed-in list of `(album_id, prompt_version, response)` rows. Also consumes `build_evidence`,
  `materialize_adjudication`, `EscalationQueue.enqueue`, `canonicalize_proposed`, a taxonomy, an adapter.
- Produces:
  - `best_results(rows, *, thorough_pv) -> dict[str, dict]` (thorough wins).
  - `@dataclass ApplySummary(materialized:int, escalated:int)`.
  - `apply_adjudications(*, rows, thorough_pv, std_pv, meta_conn, id2name, taxonomy, adapter,
    sidecar_store, queue, model="sonnet") -> ApplySummary`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_adjudication_apply.py
from __future__ import annotations

import sqlite3

from src.ai_genre_enrichment.adjudication_apply import apply_adjudications, best_results
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore


class FakeAdapter:
    def canonicalize_tag(self, tag):
        return tag
    def node(self, name):
        return None


def _meta(tmp_path):
    conn = sqlite3.connect(tmp_path / "m.db")
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO albums VALUES ('a2','X','Y',2000,NULL);
        """
    )
    conn.commit()
    return conn


def _resp(genres, escalate=False, reason=""):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "escalate_reason": reason,
            "dropped_file_tags": [], "overall_confidence": 0.9}


def test_best_results_prefers_thorough():
    rows = [("a1", "std", _resp(["x"])), ("a1", "tho", _resp(["x", "y"]))]
    best = best_results(rows, thorough_pv="tho")
    assert len(best["a1"]["genres"]) == 2


def test_apply_materializes_nonescalated_and_enqueues_escalated(tmp_path):
    conn = _meta(tmp_path)
    side = tmp_path / "side.db"
    store = SidecarStore(str(side)); store.initialize()
    queue = EscalationQueue(side)
    rows = [
        ("a1", "std", _resp(["shoegaze"])),                       # non-escalated -> materialize
        ("a2", "std", _resp(["dream pop"], escalate=True, reason="sparse")),  # -> queue
    ]
    summary = apply_adjudications(
        rows=rows, thorough_pv="tho", std_pv="std", meta_conn=conn, id2name={},
        taxonomy=load_default_layered_taxonomy(), adapter=FakeAdapter(),
        sidecar_store=store, queue=queue,
    )
    assert summary.materialized == 1
    assert summary.escalated == 1
    # a1 materialized, a2 NOT materialized
    c = sqlite3.connect(side)
    a1 = c.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
                   "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    a2 = c.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
                   "WHERE release_id='y::x'").fetchone()[0]
    c.close()
    assert a1 >= 1 and a2 == 0
    pending = queue.list_pending()
    assert [p["album_id"] for p in pending] == ["a2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_adjudication_apply.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.adjudication_apply'`.

- [ ] **Step 3: Implement the apply module**

```python
# src/ai_genre_enrichment/adjudication_apply.py
"""Deterministic apply: checkpoint best-results -> sidecar (non-escalated) + queue (escalated).

No LLM calls. Idempotent (materialize is replace-by-release-key). Safe to re-run after a
taxonomy-growth pass to pick up new canonical mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adjudication_materializer import materialize_adjudication
from .album_adjudicator import canonicalize_proposed
from .album_evidence import build_evidence
from .normalization import normalize_release_artist, normalize_release_name


def best_results(rows, *, thorough_pv) -> dict:
    best: dict = {}
    for album_id, pv, resp in rows:
        if album_id not in best or pv == thorough_pv:
            best[album_id] = resp
    return best


@dataclass
class ApplySummary:
    materialized: int
    escalated: int


def apply_adjudications(*, rows, thorough_pv, std_pv, meta_conn, id2name, taxonomy, adapter,
                        sidecar_store, queue, model: str = "sonnet") -> ApplySummary:
    best = best_results(rows, thorough_pv=thorough_pv)
    materialized = 0
    escalated = 0
    for album_id, resp in best.items():
        ev = build_evidence(meta_conn, album_id, id2name)
        if resp.get("escalate"):
            canon = canonicalize_proposed(
                [g["term"] for g in resp.get("genres", [])], adapter.canonicalize_tag)["canonical"]
            release_key = f"{normalize_release_artist(ev['artist'])}::{normalize_release_name(ev['album'])}"
            queue.enqueue(
                album_id=album_id, release_key=release_key, artist=ev["artist"], album=ev["album"],
                prior_observed_leaf=ev["current_observed_leaf"],
                proposed_genres=[{"term": t, "confidence": 0.8} for t in canon],
                escalate_reason=resp.get("escalate_reason", ""),
                dropped_file_tags=resp.get("dropped_file_tags", []),
                prompt_version=std_pv, model=model, input_hash=resp.get("input_hash", ""),
            )
            escalated += 1
            continue
        materialize_adjudication(
            sidecar_store, album_id=album_id, artist=ev["artist"], album=ev["album"],
            response=resp, taxonomy=taxonomy, prompt_version=std_pv, model=model,
        )
        materialized += 1
    return ApplySummary(materialized=materialized, escalated=escalated)
```

Note on `input_hash`: the checkpoint rows carry it, but `iter_complete()` does not currently surface
it. The stage (Task 6) reads rows with `SELECT album_id, prompt_version, response_json, input_hash`
and injects `input_hash` into each `resp` dict before calling `apply_adjudications`, so the queue's
re-open rule has a stable key. The test above omits it (empty string) — acceptable for the unit test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_adjudication_apply.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/adjudication_apply.py tests/unit/test_adjudication_apply.py
git commit -m "feat(genre): adjudication apply (best-result -> materialize/enqueue)"
```

---

## Task 6: Wire `adjudicate` + `apply` stages into `analyze_library.py`

**Files:**
- Modify: `scripts/analyze_library.py` (imports; `STAGE_ORDER_DEFAULT`; `STAGE_FUNCS`; new
  `stage_adjudicate`, `stage_apply`; `compute_stage_fingerprint`; `estimate_stage_units`; an
  `--adjudicate-model` default and the injected-client hook)
- Test: `tests/unit/test_analyze_adjudicate_stages.py`

**Interfaces:**
- Consumes: `build_todo`, `run_adjudication`, `apply_adjudications`, `AdjudicationStore`,
  `EscalationQueue`, `load_default_layered_taxonomy`, `load_graph_adapter`, `effective_prompt_version`,
  `ADJUDICATOR_INSTRUCTIONS`.
- Produces: stages `adjudicate` and `apply` registered in `STAGE_FUNCS`; `STAGE_ORDER_DEFAULT` with
  `enrich` replaced by `adjudicate, apply`.

- [ ] **Step 1: Write the failing integration test (injected fake client)**

```python
# tests/unit/test_analyze_adjudicate_stages.py
from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import scripts.analyze_library as al
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.storage import SidecarStore


class FakeClient:
    def __init__(self, responses):
        self._responses = responses
    def call_structured_session(self, items, *, response_format, validator, instructions,
                                on_result, reset_every):
        for album_id, _ in items:
            on_result(album_id, self._responses[album_id], None, {"total_tokens": 5})


def _metadata_db(tmp_path: Path) -> str:
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        INSERT INTO albums VALUES ('a2','X','Y',2000,NULL);
        """
    )
    conn.commit(); conn.close()
    return str(db)


def _resp(genres, escalate=False):
    return {"genres": [{"term": g, "confidence": 0.9} for g in genres],
            "facets": [], "escalate": escalate, "escalate_reason": "sparse",
            "dropped_file_tags": [], "overall_confidence": 0.9}


def test_adjudicate_then_apply_materializes_and_queues(tmp_path, monkeypatch):
    db = _metadata_db(tmp_path)
    side = tmp_path / "ai_genre_enrichment.db"
    SidecarStore(str(side)).initialize()
    monkeypatch.setattr(al, "ENRICHMENT_DB_PATH", side)

    client = FakeClient({"a1": _resp(["shoegaze"]), "a2": _resp(["dream pop"], escalate=True)})
    args = Namespace(adjudicate_model="sonnet", adjudicate_client=client, limit=None)
    ctx = {"args": args, "db_path": db}

    out_adj = al.stage_adjudicate(ctx)
    assert out_adj["adjudicated"] == 2 and not out_adj.get("paused")

    out_apply = al.stage_apply(ctx)
    assert out_apply["materialized"] == 1
    assert out_apply["escalated"] == 1

    q = EscalationQueue(side)
    assert [p["album_id"] for p in q.list_pending()] == ["a2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_analyze_adjudicate_stages.py -q`
Expected: FAIL with `AttributeError: module 'scripts.analyze_library' has no attribute 'stage_adjudicate'`.

- [ ] **Step 3: Add imports near the other enrichment imports (top of `analyze_library.py`)**

```python
import hashlib

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore
from src.ai_genre_enrichment.adjudication_runner import build_todo, run_adjudication
from src.ai_genre_enrichment.adjudication_apply import apply_adjudications
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.album_adjudicator import (
    ADJUDICATOR_INSTRUCTIONS,
    ADJUDICATOR_INSTRUCTIONS_THOROUGH,
    ADJUDICATOR_PROMPT_VERSION,
    ADJUDICATOR_PROMPT_VERSION_THOROUGH,
)
from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.genre.graph_adapter import load_graph_adapter
```

Do NOT import `effective_prompt_version` from `run_adjudicator_bulk` — that module lives in
`scripts/research/` and is not on `analyze_library.py`'s import path. Define the helper inline
(verbatim logic from `run_adjudicator_bulk.py:57-62`) so the stage is self-contained:

```python
def effective_prompt_version(thorough: bool = False) -> str:
    instructions = ADJUDICATOR_INSTRUCTIONS_THOROUGH if thorough else ADJUDICATOR_INSTRUCTIONS
    base = ADJUDICATOR_PROMPT_VERSION_THOROUGH if thorough else ADJUDICATOR_PROMPT_VERSION
    h = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]
    return f"{base}+{h}"
```

- [ ] **Step 4: Implement the two stages (place beside `stage_enrich`)**

```python
def stage_adjudicate(ctx: Dict) -> Dict:
    """Album-grain Sonnet adjudication. One call per new/changed album -> sidecar checkpoint.

    Incremental: skips albums already complete (is_done by input_hash). Returns a `paused`
    result on the rate wall (resumable). Writes only the sidecar `adjudications` table.
    """
    args = ctx["args"]
    conn = sqlite3.connect(ctx["db_path"])
    id2name = {r[0]: r[1] for r in conn.execute(
        "SELECT genre_id, name FROM genre_graph_canonical_genres")}
    album_ids = [r[0] for r in conn.execute("SELECT album_id FROM albums ORDER BY album_id")]
    limit = getattr(args, "limit", None)
    if limit and limit > 0:
        album_ids = album_ids[:limit]
    store = AdjudicationStore(str(ENRICHMENT_DB_PATH))
    pv = effective_prompt_version(thorough=False)
    todo = build_todo(store, conn, id2name, album_ids, prompt_version=pv)
    conn.close()
    if not todo:
        logger.info("Skipping adjudicate stage (no new/changed albums)")
        return {"skipped": True, "reason": "nothing_pending", "adjudicated": 0}
    model = getattr(args, "adjudicate_model", None) or "sonnet"
    client = getattr(args, "adjudicate_client", None) or ClaudeCodeEnrichmentClient(model=model)
    adapter = load_graph_adapter()
    summary = run_adjudication(
        store, todo, model=model, instructions=ADJUDICATOR_INSTRUCTIONS,
        prompt_version=pv, adapter=adapter, client=client)
    store.close()
    if summary.paused:
        return {"paused": True, "pause_reason": summary.pause_reason,
                "adjudicated": summary.adjudicated, "failed": summary.failed}
    return {"adjudicated": summary.adjudicated, "failed": summary.failed,
            "total": summary.adjudicated}


def stage_apply(ctx: Dict) -> Dict:
    """Deterministic apply of checkpointed adjudications: materialize non-escalated, queue escalated."""
    args = ctx["args"]
    std_pv = effective_prompt_version(thorough=False)
    tho_pv = effective_prompt_version(thorough=True)
    rows = []
    side = sqlite3.connect(str(ENRICHMENT_DB_PATH))
    for album_id, pv, rj, ih in side.execute(
        "SELECT album_id, prompt_version, response_json, input_hash "
        "FROM adjudications WHERE status='complete'"
    ):
        resp = json.loads(rj) if rj else None
        if resp is None:
            continue
        resp["input_hash"] = ih
        rows.append((album_id, pv, resp))
    side.close()
    if not rows:
        logger.info("Skipping apply stage (no complete adjudications)")
        return {"skipped": True, "reason": "no_adjudications", "materialized": 0, "escalated": 0}
    conn = sqlite3.connect(ctx["db_path"])
    id2name = {r[0]: r[1] for r in conn.execute(
        "SELECT genre_id, name FROM genre_graph_canonical_genres")}
    taxonomy = load_default_layered_taxonomy()
    adapter = load_graph_adapter()
    store = SidecarStore(str(ENRICHMENT_DB_PATH)); store.initialize()
    queue = EscalationQueue(ENRICHMENT_DB_PATH)
    summary = apply_adjudications(
        rows=rows, thorough_pv=tho_pv, std_pv=std_pv, meta_conn=conn, id2name=id2name,
        taxonomy=taxonomy, adapter=adapter, sidecar_store=store, queue=queue,
        model=getattr(args, "adjudicate_model", None) or "sonnet")
    conn.close(); queue.close()
    return {"materialized": summary.materialized, "escalated": summary.escalated,
            "total": summary.materialized}
```

(Confirm `import json` and `import sqlite3` are already present at the top of `analyze_library.py`;
they are used by other stages, so no new import is needed.)

- [ ] **Step 5: Swap the stage order and registry**

Change `STAGE_ORDER_DEFAULT` (line 48) from:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "mert", "enrich", "publish", "genre-sim", "artifacts", "genre-embedding", "verify"]
```

to:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "mert", "adjudicate", "apply", "publish", "genre-sim", "artifacts", "genre-embedding", "verify"]
```

Add both stages to `STAGE_FUNCS` (keep `enrich` registered so `--stages enrich` still works):

```python
    "enrich": stage_enrich,
    "adjudicate": stage_adjudicate,
    "apply": stage_apply,
    "publish": stage_publish,
```

- [ ] **Step 6: Add fingerprint + unit-estimate entries**

In `compute_stage_fingerprint`, add before the final `return _hash_obj({"stage": stage})`:

```python
    if stage == "adjudicate":
        conn = sqlite3.connect(ctx["db_path"])
        total_albums = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
        conn.close()
        done = _sidecar_count(
            "SELECT COUNT(DISTINCT album_id) FROM adjudications WHERE status='complete'")
        return _hash_obj({"stage": stage, "total_albums": total_albums, "done": done})
    if stage == "apply":
        complete = _sidecar_count(
            "SELECT COUNT(*) FROM adjudications WHERE status='complete'")
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        tax_version = load_default_layered_taxonomy().version
        return _hash_obj({"stage": stage, "complete": complete, "taxonomy": tax_version})
```

In `estimate_stage_units`, add (mirroring the `enrich` branch around line 474):

```python
        if stage == "adjudicate":
            conn = sqlite3.connect(ctx["db_path"])
            total = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
            conn.close()
            done = _sidecar_count(
                "SELECT COUNT(DISTINCT album_id) FROM adjudications WHERE status='complete'")
            return max(0, total - done), "albums to adjudicate"
        if stage == "apply":
            complete = _sidecar_count("SELECT COUNT(*) FROM adjudications WHERE status='complete'")
            return complete, "adjudications to apply"
```

(`_sidecar_count` already exists and counts against `ENRICHMENT_DB_PATH`, returning 0 if the table
is absent.)

- [ ] **Step 7: Run the integration test**

Run: `python -m pytest tests/unit/test_analyze_adjudicate_stages.py -q`
Expected: PASS (1 test).

- [ ] **Step 8: Run the existing analyze-stages test to confirm no regression**

Run: `python -m pytest tests/unit/test_analyze_graph_stages.py -q`
Expected: PASS (enrich/lastfm/publish still work; `enrich` remains registered).

- [ ] **Step 9: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_adjudicate_stages.py
git commit -m "feat(analyze): adjudicate + apply stages replace enrich in default order"
```

---

## Task 7: Publish regression — escalated album is never cleared

**Files:**
- Test: `tests/unit/test_genre_publish.py` (add a case)

**Interfaces:**
- Consumes: `src.genre.genre_publish.publish` (existing). No production change expected; this task
  proves the hold is structural. If the test fails, add a guard in `apply` (not publish) and note it.

- [ ] **Step 1: Write the test**

```python
# add to tests/unit/test_genre_publish.py
def test_escalated_album_retains_prior_assignments(tmp_path):
    """An album with prior graph assignments that is later escalated (and therefore NOT
    re-materialized by apply) keeps its prior observed_leaf after publish."""
    import sqlite3
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.escalation_queue import EscalationQueue
    from src.ai_genre_enrichment.adjudication_apply import apply_adjudications
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    class FakeAdapter:
        def canonicalize_tag(self, t): return t
        def node(self, n): return None

    side = tmp_path / "side.db"
    store = SidecarStore(str(side)); store.initialize()
    meta = sqlite3.connect(tmp_path / "m.db")
    meta.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        """
    )
    meta.commit()
    taxonomy = load_default_layered_taxonomy()
    queue = EscalationQueue(side)

    # First apply: non-escalated -> materializes shoegaze.
    rows1 = [("a1", "std", {"genres": [{"term": "shoegaze", "confidence": 0.9}],
                            "facets": [], "escalate": False})]
    apply_adjudications(rows=rows1, thorough_pv="tho", std_pv="std", meta_conn=meta,
                        id2name={}, taxonomy=taxonomy, adapter=FakeAdapter(),
                        sidecar_store=store, queue=queue)
    before = sqlite3.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    assert before >= 1

    # Second apply: same album now ESCALATED -> must NOT clear prior assignments.
    rows2 = [("a1", "std", {"genres": [{"term": "dream pop", "confidence": 0.9}],
                            "facets": [], "escalate": True, "escalate_reason": "x",
                            "dropped_file_tags": []})]
    apply_adjudications(rows=rows2, thorough_pv="tho", std_pv="std", meta_conn=meta,
                        id2name={}, taxonomy=taxonomy, adapter=FakeAdapter(),
                        sidecar_store=store, queue=queue)
    after = sqlite3.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    meta.close()
    assert after == before  # prior assignments preserved; nothing cleared
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/unit/test_genre_publish.py::test_escalated_album_retains_prior_assignments -q`
Expected: PASS — because `apply_adjudications` never touches the assignments of an escalated album.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_genre_publish.py
git commit -m "test(genre): escalated album retains prior assignments (hold is structural)"
```

---

## Task 8: Adapt `review_escalated.py` to the `EscalationQueue`

**Files:**
- Modify: `scripts/research/review_escalated.py`
- Test: `tests/unit/test_review_escalated.py` (keep `parse_decision` tests; add a queue-backed test)

**Interfaces:**
- Consumes: `EscalationQueue` (`list_pending`, `get`, `record_decision`), `load_default_layered_taxonomy`,
  `SidecarStore`.
- Produces: the CLI reads pending escalations from the sidecar queue (not the old shadow
  `_escalated()` scan) and, with `--apply`, calls `queue.record_decision(...)` for decided albums.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_review_escalated.py
def test_review_apply_materializes_via_queue(tmp_path, monkeypatch):
    import sqlite3
    import scripts.research.review_escalated as re_mod
    from src.ai_genre_enrichment.escalation_queue import EscalationQueue
    from src.ai_genre_enrichment.storage import SidecarStore

    side = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(str(side)); store.initialize()
    q = EscalationQueue(side)
    q.enqueue(album_id="a1", release_key="slowdive::souvlaki", artist="Slowdive",
              album="Souvlaki", prior_observed_leaf=["indie rock"],
              proposed_genres=[{"term": "shoegaze", "confidence": 0.9}],
              escalate_reason="sparse", dropped_file_tags=[], prompt_version="pv",
              model="sonnet", input_hash="h1")
    # mark a decision directly, then run --apply path
    q.record_decision  # ensure attribute exists
    # Point the CLI's sidecar resolver at our temp DB.
    monkeypatch.setattr(re_mod, "_sidecar_path", lambda: str(side))

    # Decide accept, then apply.
    re_mod.apply_decisions(sidecar_path=str(side), decisions={"a1": ("accept", [])})
    n = sqlite3.connect(side).execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki'").fetchone()[0]
    assert n >= 1
    assert q.get("a1")["status"] == "accepted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_review_escalated.py -k apply_materializes_via_queue -q`
Expected: FAIL with `AttributeError: module 'scripts.research.review_escalated' has no attribute 'apply_decisions'`.

- [ ] **Step 3: Rewrite the CLI body to use the queue**

Replace the shadow-DB `_escalated()` scan and the `--apply` block in `review_escalated.py` with
queue-backed logic. Keep `parse_decision` unchanged. New helpers + main:

```python
from src.ai_genre_enrichment.escalation_queue import EscalationQueue
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.storage import SidecarStore
from run_album_adjudicator import resolve_db  # resolves data/ai_genre_enrichment.db


def _sidecar_path() -> str:
    return str(resolve_db("ai_genre_enrichment.db"))


def apply_decisions(*, sidecar_path: str, decisions: dict) -> int:
    """Materialize accept/edit decisions via the queue; reject is a no-op. Returns count applied."""
    store = SidecarStore(sidecar_path); store.initialize()
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
```

Remove the now-unused imports (`build_evidence`, `materialize_adjudication`, `canonicalize_proposed`,
`load_graph_adapter`, `effective_prompt_version`, `ReviewDecisionStore` usage in `main`, `time`,
`json` if unused). Keep `ReviewDecisionStore` and `parse_decision` defined (other tests import them).

- [ ] **Step 4: Run the review tests**

Run: `python -m pytest tests/unit/test_review_escalated.py -q`
Expected: PASS (existing `parse_decision` tests + the new queue test).

- [ ] **Step 5: Commit**

```bash
git add scripts/research/review_escalated.py tests/unit/test_review_escalated.py
git commit -m "feat(genre): review_escalated reads the sidecar EscalationQueue"
```

---

## Task 9: One-time importer — `adjudication_pass1.db` → sidecar `adjudications`

**Files:**
- Create: `scripts/research/import_backfill_adjudications.py`
- Test: `tests/unit/test_import_backfill_adjudications.py`

**Interfaces:**
- Produces: `import_adjudications(src_db: str, sidecar_db: str) -> int` — copies all rows from the
  source `adjudications` table into the sidecar's `adjudications` table (upsert by
  `(album_id, prompt_version)`), returning the number imported.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_import_backfill_adjudications.py
from __future__ import annotations

import sqlite3

from scripts.research.import_backfill_adjudications import import_adjudications
from src.ai_genre_enrichment.adjudication_store import AdjudicationStore


def test_import_copies_rows_into_sidecar(tmp_path):
    src = tmp_path / "adjudication_pass1.db"
    s = AdjudicationStore(src)
    s.save(album_id="a1", prompt_version="pv", input_hash="h", status="complete",
           response={"genres": []}, tokens={"total_tokens": 1})
    s.save(album_id="a2", prompt_version="pv", input_hash="h", status="complete",
           response={"genres": []})
    s.close()

    side = tmp_path / "ai_genre_enrichment.db"
    AdjudicationStore(side).close()  # create the table in the sidecar
    n = import_adjudications(str(src), str(side))
    assert n == 2
    dst = AdjudicationStore(side)
    assert dst.complete_album_ids("pv") == {"a1", "a2"}
    dst.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_import_backfill_adjudications.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.research.import_backfill_adjudications'`.

- [ ] **Step 3: Implement the importer**

```python
#!/usr/bin/env python
"""One-time importer: copy backfill adjudications into the sidecar.

The Pass-1/Pass-2 backfill writes to data/adjudication_pass1.db. The analyze pipeline
reads the checkpoint from the sidecar (data/ai_genre_enrichment.db). Run this once after
the backfill so the pipeline inherits all the work instead of re-calling the LLM.

Usage:
  python scripts/research/import_backfill_adjudications.py \
    --src data/adjudication_pass1.db --sidecar data/ai_genre_enrichment.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore  # noqa: E402

_COLS = (
    "album_id, prompt_version, release_key, input_hash, model, status, response_json, "
    "dropped_file_tags_json, input_tokens, output_tokens, total_tokens, error, updated_at"
)


def import_adjudications(src_db: str, sidecar_db: str) -> int:
    AdjudicationStore(sidecar_db).close()  # ensure the table exists in the sidecar
    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    rows = src.execute(f"SELECT {_COLS} FROM adjudications").fetchall()
    src.close()
    dst = sqlite3.connect(sidecar_db)
    placeholders = ",".join(["?"] * len(_COLS.split(",")))
    dst.executemany(
        f"INSERT INTO adjudications ({_COLS}) VALUES ({placeholders}) "
        "ON CONFLICT(album_id, prompt_version) DO UPDATE SET "
        "release_key=excluded.release_key, input_hash=excluded.input_hash, model=excluded.model, "
        "status=excluded.status, response_json=excluded.response_json, "
        "dropped_file_tags_json=excluded.dropped_file_tags_json, input_tokens=excluded.input_tokens, "
        "output_tokens=excluded.output_tokens, total_tokens=excluded.total_tokens, "
        "error=excluded.error, updated_at=excluded.updated_at",
        rows,
    )
    dst.commit()
    n = len(rows)
    dst.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--sidecar", default=str(_ROOT / "data" / "ai_genre_enrichment.db"))
    args = ap.parse_args()
    n = import_adjudications(args.src, args.sidecar)
    print(f"imported {n} adjudication rows -> {args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/unit/test_import_backfill_adjudications.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/research/import_backfill_adjudications.py tests/unit/test_import_backfill_adjudications.py
git commit -m "feat(genre): one-time importer for backfill adjudications -> sidecar"
```

---

## Task 10: Full-suite gate + docs note

**Files:**
- Modify: `docs/genre_adjudication/` (add a short `ANALYZE_ADJUDICATE_STAGE.md` operator note)
- No code change.

- [ ] **Step 1: Run the full fast suite**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS — all prior tests plus the new `test_album_evidence`, `test_escalation_queue`,
`test_adjudication_runner`, `test_adjudication_apply`, `test_analyze_adjudicate_stages`,
`test_import_backfill_adjudications`, and the added materializer/publish/review cases. Quote the
real pass/fail counts.

- [ ] **Step 2: Lint the new/changed files**

Run: `python -m ruff check src/ai_genre_enrichment/ scripts/analyze_library.py scripts/research/review_escalated.py scripts/research/import_backfill_adjudications.py`
Expected: `All checks passed!` (remove any unused imports flagged).

- [ ] **Step 3: Write the operator note**

Create `docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md` documenting: the new stage order
(`… mert → adjudicate → apply → publish …`), that `enrich` is opt-in legacy (`--stages enrich`),
the single-Sonnet default and `--adjudicate-model` override, the hold+queue policy, how to clear the
queue (`python scripts/research/review_escalated.py`), and the one-time backfill import command.

- [ ] **Step 4: Commit**

```bash
git add docs/genre_adjudication/ANALYZE_ADJUDICATE_STAGE.md
git commit -m "docs(genre): operator note for the analyze adjudicate/apply stages"
```

---

## Notes for the executor

- **Do not** run the live `adjudicate` stage against the real `metadata.db`/sidecar as part of this
  plan — all tests use temp DBs + an injected fake client. Live runs are an operator step after merge.
- **`--adjudicate-model` arg:** add it to `parse_args` in `analyze_library.py` (default `None`) so the
  stage's `getattr(args, "adjudicate_model", None) or "sonnet"` resolves; the existing `--model` arg
  remains the `enrich` override. Fold this one-line argparse add into Task 6, Step 4.
- **Background backfill:** a Pass-2 Sonnet backfill may still be writing `data/adjudication_pass1.db`
  while this plan is implemented. The tests never touch that file. Run Task 9's importer only after
  the backfill completes.
