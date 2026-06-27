# Taxonomy Term Adjudication Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new "Taxonomy" tab in the right-sidebar `AdvancedPanel` that surfaces library genre *terms* not yet in the taxonomy graph, lets the user summon Claude for an add/alias/reject verdict, ratify/edit it, and — in a deliberate second step — write accepted decisions into `data/layered_genre_taxonomy.yaml`.

**Architecture:** Mirror the existing album-level Genre Review path one layer up (vocabulary level instead of album→genre level). Two-phase write: Accept records a decision to a new staging table (`taxonomy_term_decisions` in `data/ai_genre_enrichment.db`) instantly; a separate tracked "Apply N decisions" job validates the whole batch, backs up + writes the YAML, and reloads the graph. Backend is four new focused modules in `src/ai_genre_enrichment/`; wiring mirrors the escalation-queue routes/handlers exactly.

**Tech Stack:** Python 3.11 (backend), pytest (TDD), FastAPI + NDJSON worker bridge (web), React + TypeScript + Vite + Tailwind (frontend). SQLite (sidecar staging). YAML (taxonomy graph). `pyyaml`, existing `graph_growth`/`layered_taxonomy`/`album_adjudicator`/`escalation_queue` modules.

## Global Constraints

- **This feature NEVER touches `metadata.db`.** Write target is `data/layered_genre_taxonomy.yaml` (git-tracked, recoverable) + the sidecar staging table only. (Handoff §6 Safety invariants.)
- **validate-before-write, backup-before-write.** One Apply = one timestamped YAML backup = one reviewable diff = one commit (Dylan commits it himself). (Handoff §6.)
- **Read genres from the authority only.** Never wire this panel to an internal/stale genre layer. Taxonomy *structure* is read via `layered_taxonomy.load_*`; we never read `metadata.db` genre tables here. (`genre-data-authority` skill.)
- **Untracked worker handlers run on the READER thread → read-only + WAL, no DDL on the read path.** A blocking/DDL write there wedges every untracked command incl. cancel (2026-06-12 incident). Reads use `mode=ro`. (`web-gui` skill.)
- **Worker result lines are single NDJSON lines read with a 16 MB cap but the page must stay small** — always paginate (`limit`/`offset`); never return the whole queue in one line. (`web-gui` skill, 64 KB trap.)
- **After any `web/src` edit, rebuild `web/dist`; after any worker/`src` edit, restart `serve_web.py`.** The served GUI runs `dist`, and the running worker holds the code it was spawned with. (`web-gui` Core Rules 1–2.)
- **Pytest: run bounded, never piped.** `python -m pytest -q -m "not slow"` with the tool timeout. (project CLAUDE.md.)
- **Carry the `taxonomy-growth` pre-existing-failure deselects** when running the genre-enrichment suite (they fail on unmodified taxonomy; not our regressions):
  `--deselect tests/unit/test_ai_genre_hybrid_cli.py::test_hybrid_enrich_one_apply_persists_accepted_signature --deselect tests/unit/test_ai_genre_hybrid_cli.py::test_hybrid_enrich_one_apply_can_include_provisional_lastfm_terms --deselect tests/unit/test_ai_genre_hybrid_evidence.py::test_lastfm_only_is_rejected_noise --deselect tests/unit/test_ai_genre_hybrid_evidence.py::test_specific_lastfm_only_terms_are_provisional_when_release_evidence_exists`
- **Activate, don't scaffold.** The feature must be wired through every layer and demonstrably exercised end-to-end before it's "done" (project failure-mode #1). (CLAUDE.md design principle 22.)
- **Environment:** work only in this worktree (`feat/taxonomy-term-adjudication`). `data/` here is a real dir holding the git-tracked YAML files; `metadata.db` / `ai_genre_enrichment.db` / `config.yaml` are ABSENT. Phases 1–3 + 4-tests use temp copies only. Phase 6 data access is resolved in Task 6.1 — **never symlink a SQLite DB into the worktree** (WAL-aliasing corruption rule).

---

## Verified reuse map (re-verified against the live tree 2026-06-26)

| Need | Reuse (verified symbol @ line) |
|---|---|
| Unmapped-term discovery + impact annotations | `graph_growth.gather_growth_candidates(store, taxonomy, *, min_album_freq=3, max_examples=3, max_cooccurring=8) -> list[GrowthCandidate]` (`graph_growth.py:30`); each `GrowthCandidate` has `term, album_frequency, cooccurring_tags, examples, variants` (`:21`) |
| Spacing-variant dedup | `graph_growth.collapse_variants(list[GrowthCandidate]) -> list[GrowthCandidate]` (`:90`) — merges by `_variant_key`, records merged spellings in `.variants` |
| Adjudicator gap split | `album_adjudicator.canonicalize_proposed(terms, canonicalize_fn) -> {"canonical": [...], "gaps": [...]}` (`album_adjudicator.py:245`); `canonicalize_fn` = `src/genre/graph_adapter.py:119 GraphAdapter.canonicalize_tag` |
| Record schema + ingest | `graph_growth.GrowthProposal` (`:189`, fields: `name, kind, status, specificity_score, parent_edges=[], similar_to=[], alias_variants=[], term_kind_confirm="genre", rationale="", facet_type=None, canonical_target=None`); `validate_proposal(taxonomy, proposal) -> list[str]` (`:389`, `[]`==OK; proposable kinds = `{genre, subgenre, umbrella, facet, alias}`; `reject`/`microgenre` are NOT proposable → "Unsupported kind"); `append_approved_to_taxonomy(taxonomy_path, approved: list[GrowthProposal], *, new_version) -> AppendResult` (`:550`, NO `dry_run`) |
| Name normalization | `layered_taxonomy.normalize_taxonomy_name(value) -> str` (`:15`) |
| Taxonomy load (fresh, NO lru_cache) | `layered_taxonomy.load_default_layered_taxonomy()` (`:243`) / `load_layered_taxonomy(path)` (`:247`); `DEFAULT_TAXONOMY_PATH` (`:11`); `LayeredTaxonomy.genre_by_name(name)` (`:123`), `.facet_by_name(name)` (`:133`), `.genres` tuple, `.rejected_term_by_name(name)` (`:158`) |
| Reject record shape (loader) | `_structured_taxonomy_from_data` reads reject records by `kind=="reject"` (or `status=="rejected"`) requiring `reject_reason` ∈ enum (`layered_taxonomy.py:382`). Enum: `label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise` |
| Claude call | `provider.create_enrichment_client(*, model=None, dry_run=False, web_mode=OFF, config_path=None) -> client` (`provider.py:64`); `client.call_structured(prompt, response_format, *, instructions) -> dict` (`claude_client.py:137`). Usage template: `graph_growth.propose_placement` (`graph_growth.py:218`) |
| Adjudicator contract template | `album_adjudicator.ADJUDICATOR_INSTRUCTIONS` (`:31`), `ADJUDICATOR_RESPONSE_SCHEMA` (`:67`), `adjudicator_response_format()` (`:106`), `build_adjudicator_prompt(payload)` (`:115`), `validate_adjudicator_response(data)` (`:130`), `build_adjudicator_payload(evidence)` (`:175`) |
| Staging-store template | `escalation_queue.py` — module-level read-only `list_page(db_path, *, status, search, limit, offset) -> dict` (`:55`); class `EscalationQueue` (`:87`, `__init__` opens R/W conn + DDL), `record_decision` (`:162`), `revert` (`:190`), `list_pending` (`:136`), `_mark` (`:153`), `close` (`:208`) |
| Sidecar store | `storage.py:156 SidecarStore(db_path="data/ai_genre_enrichment.db")`; `.connect()` (`:162`, R/W WAL), `.connect_readonly(*, busy_timeout_ms=2000)` (`:177`, mode=ro, raises if file absent), `.all_collected_tags() -> list[Row]` (`:1044`, rows have `normalized_tag, release_key, normalized_artist, normalized_album`) |
| Worker untracked read handler | `handle_get_escalation_queue` (`worker.py:2684`) — explicit `request_id` + `job_id=None` discipline |
| Worker untracked decision handler | `handle_apply_escalation_decision` (`worker.py:2726`) — instantiates store inline, quick write |
| Worker tracked Apply job (backup template) | `handle_publish_decided` (`worker.py:2550`) — `emit_progress` → timestamped backup → work → `emit_result`/`emit_done(summary=...)` |
| Handler registration | `TRACKED_COMMAND_HANDLERS` (`worker.py:2798`), `UNTRACKED_COMMAND_HANDLERS` (`worker.py:2842`); `SIDECAR_DB_PATH = "data/ai_genre_enrichment.db"` (`worker.py:78`) |
| API review routes (mirror) | `src/playlist_web/app.py` review routes ~`:276` (re-verify at execution) |
| FE panel/tab/api/types (mirror) | `web/src/components/GenreReviewPanel.tsx`, `AdvancedPanel.tsx`, `web/src/lib/api.ts` (`reviewQueue`/`reviewDecision`/`reviewPublish`), `web/src/lib/types.ts` (`EscalationOut`/`EscalationDecisionRequest`) |

## File structure (new + modified)

**New backend (`src/ai_genre_enrichment/`):**
- `taxonomy_decision_store.py` — staging store: `taxonomy_term_decisions` DDL, module-level read-only `list_decisions`, class `TaxonomyDecisionStore` (record/revert/list_pending/list_applied/mark_applied/get/decided_terms).
- `taxonomy_review_queue.py` — `TaxonomyCandidate` dataclass, `build_candidate_index(store, taxonomy)`, module-level read-only `list_page(...)` that joins the computed candidate index with staged decisions.
- `taxonomy_term_adjudicator.py` — Claude contract: instructions, schema, payload builder, `validate_response()` → `GrowthProposal | RejectVerdict`, `adjudicate_term()`.
- `taxonomy_apply.py` — `apply_decisions(...)` (backup → validate → order → write add/alias + reject → version bump → stats), `_reject_record`, `_order_for_forward_refs`, `ApplyResult`.

**New tests (`tests/unit/`):** `test_taxonomy_decision_store.py`, `test_taxonomy_review_queue.py`, `test_taxonomy_term_adjudicator.py`, `test_taxonomy_apply.py`. **New integration:** `tests/integration/test_taxonomy_web_api.py`.

**Modified backend:** `src/playlist_web/app.py` (+5 routes), `src/playlist_web/schemas.py` (+request/JobOut fields), `src/playlist_gui/worker.py` (+5 handlers, +2 registrations), `tests/fixtures/fake_worker.py` (+command branches).

**Modified frontend:** `web/src/components/TaxonomyReviewPanel.tsx` (new), `web/src/components/AdvancedPanel.tsx` (+tab), `web/src/lib/api.ts` (+methods), `web/src/lib/types.ts` (+types).

---

# PHASE 1 — Queue builder + staging store

Pure backend, no Claude, no GUI. Fully testable in isolation against temp DB/YAML copies.

### Task 1.1: Staging store — `taxonomy_term_decisions`

**Files:**
- Create: `src/ai_genre_enrichment/taxonomy_decision_store.py`
- Test: `tests/unit/test_taxonomy_decision_store.py`

**Interfaces:**
- Produces:
  - `TaxonomyDecisionStore(db_path)` with: `record_decision(*, term, raw_term, verdict, proposal_json, claude_json, human_edited) -> None`, `revert(term) -> None`, `list_pending() -> list[dict]`, `list_applied() -> list[dict]`, `get(term) -> dict | None`, `decided_terms() -> set[str]`, `mark_applied(terms, batch_version) -> None`, `close() -> None`.
  - module-level `list_decisions(db_path, *, status="pending") -> list[dict]` (read-only, mode=ro, no DDL).
  - Each row dict: `{term, raw_term, verdict, proposal, claude, human_edited, status, created_at, applied_at, batch_version}` (`proposal`/`claude` are JSON-decoded).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_taxonomy_decision_store.py
import json
from src.ai_genre_enrichment.taxonomy_decision_store import (
    TaxonomyDecisionStore, list_decisions,
)


def _store(tmp_path):
    return TaxonomyDecisionStore(tmp_path / "ai_genre_enrichment.db")


def test_record_and_list_pending(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(
            term="vapor wave", raw_term="Vapor Wave", verdict="add",
            proposal_json=json.dumps({"name": "vaporwave", "kind": "genre"}),
            claude_json=json.dumps({"verdict": "add"}), human_edited=0,
        )
        pending = store.list_pending()
        assert [r["term"] for r in pending] == ["vapor wave"]
        assert pending[0]["verdict"] == "add"
        assert pending[0]["proposal"] == {"name": "vaporwave", "kind": "genre"}
        assert pending[0]["status"] == "pending"
        assert store.decided_terms() == {"vapor wave"}
    finally:
        store.close()


def test_revert_removes_from_pending(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="reject",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.revert("t")
        assert store.list_pending() == []
        assert store.get("t") is None
    finally:
        store.close()


def test_record_decision_upserts(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="add",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.record_decision(term="t", raw_term="t", verdict="reject",
                              proposal_json="{}", claude_json="{}", human_edited=1)
        rows = store.list_pending()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "reject"
        assert rows[0]["human_edited"] == 1
    finally:
        store.close()


def test_mark_applied_moves_to_applied(tmp_path):
    store = _store(tmp_path)
    try:
        store.record_decision(term="t", raw_term="t", verdict="add",
                              proposal_json="{}", claude_json="{}", human_edited=0)
        store.mark_applied(["t"], batch_version="0.9.0-gui-20260626-grown")
        assert store.list_pending() == []
        applied = store.list_applied()
        assert applied[0]["status"] == "applied"
        assert applied[0]["batch_version"] == "0.9.0-gui-20260626-grown"
        assert applied[0]["applied_at"] is not None
    finally:
        store.close()


def test_list_decisions_readonly_missing_table_is_empty(tmp_path):
    # mode=ro read on a fresh DB with no table must NOT raise.
    (tmp_path / "ai_genre_enrichment.db").write_bytes(b"")  # empty file
    assert list_decisions(tmp_path / "ai_genre_enrichment.db", status="pending") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_taxonomy_decision_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai_genre_enrichment.taxonomy_decision_store'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_genre_enrichment/taxonomy_decision_store.py
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
    the worker reader thread. Returns [] if the table doesn't exist yet."""
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
        return []
    finally:
        conn.close()


class TaxonomyDecisionStore:
    def __init__(self, db_path: "str | Path") -> None:
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path)
        self._c.row_factory = sqlite3.Row
        self._c.execute(_SCHEMA)
        self._c.commit()

    def record_decision(self, *, term: str, raw_term: str, verdict: str,
                        proposal_json: str, claude_json: str, human_edited: int) -> None:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_taxonomy_decision_store.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/taxonomy_decision_store.py tests/unit/test_taxonomy_decision_store.py
git commit -m "feat(taxonomy-review): staging store for term adjudication decisions"
```

---

### Task 1.2: Queue builder — merged/deduped candidate index + `list_page`

**Files:**
- Create: `src/ai_genre_enrichment/taxonomy_review_queue.py`
- Test: `tests/unit/test_taxonomy_review_queue.py`

**Interfaces:**
- Consumes: `graph_growth.gather_growth_candidates`, `collapse_variants`; `TaxonomyDecisionStore`/`list_decisions` (Task 1.1); `layered_taxonomy.load_layered_taxonomy`, `normalize_taxonomy_name`; `storage.SidecarStore`.
- Produces:
  - `TaxonomyCandidate` dataclass: `term, raw_term, album_frequency, cooccurring_tags, examples, variants, source` (`source` ∈ `"growth"`/`"adjudicator"`/`"both"`).
  - `build_candidate_index(store, taxonomy, *, min_album_freq=3) -> dict[str, TaxonomyCandidate]` keyed by `normalize_taxonomy_name(term)`.
  - `list_page(sidecar_db_path, taxonomy_path, *, status="untriaged", search=None, limit=50, offset=0) -> dict` → `{"terms": [row...], "untriaged_terms": N, "decided_terms": N}`. Each row: candidate fields + `decision` (None or the staged-decision dict). Read-only: opens the sidecar via `SidecarStore.connect_readonly()` and decisions via `list_decisions`.

**Design notes (resolves Handoff §10 impact-count + §2-Decision-1 union):**
- Impact count = `album_frequency` from `gather_growth_candidates` = distinct releases where the term appears as an observed/legacy collected tag (the recommended §10 semantics: distinct albums, observed/legacy).
- Source union: `gather_growth_candidates` (taxonomy-aware unmapped detection over collected tags) is the spine. `collapse_variants` dedups spacing variants. The adjudicator-`taxonomy_gaps` source canonicalizes the SAME collected tags through the graph and yields the same gap set, so for v1 the union is gather-dominated; each item is stamped `source="growth"`. The `source` field + the union seam are left as the documented extension point if a stored-adjudication-proposed-terms accessor is later added. (Recorded as the §10 resolution; not a silent cap.)
- "untriaged" = candidate terms with NO staged decision; "decided" = terms that DO have a staged decision (pending or applied). The page subtracts/joins the decision set.

- [ ] **Step 1: Write the failing test** (uses a fake store + a tiny temp taxonomy, no real DB)

```python
# tests/unit/test_taxonomy_review_queue.py
import sqlite3
from pathlib import Path

import yaml

from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore
from src.ai_genre_enrichment import taxonomy_review_queue as trq


def _tiny_taxonomy(tmp_path: Path) -> Path:
    """A minimal records-based taxonomy with one family so loads validate."""
    data = {
        "taxonomy_version": "0.0.1-test",
        "enums": {
            "reject_reason": ["source_noise", "malformed", "label"],
            "facet_type": ["mood", "instrumentation"],
        },
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
        ],
    }
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


class _FakeStore:
    """Stands in for SidecarStore.all_collected_tags()."""
    def __init__(self, rows):
        self._rows = rows

    def all_collected_tags(self):
        return self._rows


def _row(tag, rk, artist="A", album="B"):
    return {"normalized_tag": tag, "release_key": rk,
            "normalized_artist": artist, "normalized_album": album}


def test_build_candidate_index_finds_unmapped_terms(tmp_path):
    tax = load_layered_taxonomy(_tiny_taxonomy(tmp_path))
    # "vaporwave" appears on 3 distinct releases -> a candidate; "rock" is mapped.
    rows = [
        _row("vaporwave", "r1"), _row("vaporwave", "r2"), _row("vaporwave", "r3"),
        _row("rock", "r1"),
    ]
    index = trq.build_candidate_index(_FakeStore(rows), tax, min_album_freq=3)
    assert "vaporwave" in index
    assert index["vaporwave"].album_frequency == 3
    assert index["vaporwave"].source == "growth"
    assert "rock" not in index  # mapped -> excluded


def test_build_candidate_index_collapses_spacing_variants(tmp_path):
    tax = load_layered_taxonomy(_tiny_taxonomy(tmp_path))
    rows = [
        _row("vapor wave", "r1"), _row("vapor wave", "r2"), _row("vapor wave", "r3"),
        _row("vaporwave", "r4"), _row("vaporwave", "r5"),
    ]
    index = trq.build_candidate_index(_FakeStore(rows), tax, min_album_freq=3)
    # collapsed into one representative with summed frequency
    assert len(index) == 1
    rep = next(iter(index.values()))
    assert rep.album_frequency == 5
    assert "vaporwave" in rep.variants or "vapor wave" in rep.variants


def test_list_page_joins_decisions_and_filters_status(tmp_path, monkeypatch):
    tax_path = _tiny_taxonomy(tmp_path)
    db = tmp_path / "ai_genre_enrichment.db"
    rows = [_row("vaporwave", "r1"), _row("vaporwave", "r2"), _row("vaporwave", "r3"),
            _row("slowcore", "r1"), _row("slowcore", "r2"), _row("slowcore", "r3")]

    # Patch SidecarStore so list_page reads our fake collected tags.
    monkeypatch.setattr(trq, "_open_store_readonly", lambda p: _FakeStore(rows))

    store = TaxonomyDecisionStore(db)
    store.record_decision(term="slowcore", raw_term="slowcore", verdict="add",
                          proposal_json="{}", claude_json="{}", human_edited=0)
    store.close()

    untriaged = trq.list_page(db, tax_path, status="untriaged")
    decided = trq.list_page(db, tax_path, status="decided")
    assert [t["term"] for t in untriaged["terms"]] == ["vaporwave"]
    assert untriaged["untriaged_terms"] == 1
    assert untriaged["decided_terms"] == 1
    assert [t["term"] for t in decided["terms"]] == ["slowcore"]
    assert decided["terms"][0]["decision"]["verdict"] == "add"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_taxonomy_review_queue.py -q`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_genre_enrichment/taxonomy_review_queue.py
"""Merged/deduped candidate queue for taxonomy-term adjudication.

Surfaces genre terms present in the library but absent from the taxonomy graph,
annotated with reach (distinct albums), co-occurring tags, and example releases.
The queue is a DERIVED view: candidates are computed from the sidecar's collected
tags via graph_growth.gather_growth_candidates; staged decisions (Task 1.1) are
joined in to split untriaged vs decided. Read path is mode=ro (reader-thread safe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .graph_growth import collapse_variants, gather_growth_candidates
from .layered_taxonomy import (
    LayeredTaxonomy, load_layered_taxonomy, normalize_taxonomy_name,
)
from .storage import SidecarStore
from .taxonomy_decision_store import list_decisions


@dataclass
class TaxonomyCandidate:
    term: str
    raw_term: str
    album_frequency: int
    cooccurring_tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    source: str = "growth"


def build_candidate_index(
    store, taxonomy: LayeredTaxonomy, *, min_album_freq: int = 3,
) -> dict[str, TaxonomyCandidate]:
    """Normalized-name -> candidate. Spine = gather_growth_candidates (taxonomy-aware
    unmapped detection over collected tags) + collapse_variants."""
    candidates = collapse_variants(
        gather_growth_candidates(store, taxonomy, min_album_freq=min_album_freq)
    )
    index: dict[str, TaxonomyCandidate] = {}
    for c in candidates:
        key = normalize_taxonomy_name(c.term)
        index[key] = TaxonomyCandidate(
            term=key,
            raw_term=c.term,
            album_frequency=c.album_frequency,
            cooccurring_tags=list(c.cooccurring_tags),
            examples=list(c.examples),
            variants=list(c.variants),
            source="growth",
        )
    return index


def _open_store_readonly(sidecar_db_path) -> SidecarStore:
    """Indirection seam so tests can inject a fake collected-tags store.

    SidecarStore.all_collected_tags() opens its own connection; we keep the store
    object read-only by construction here (no initialize()/DDL on this path)."""
    return SidecarStore(sidecar_db_path)


def list_page(
    sidecar_db_path, taxonomy_path, *,
    status: str = "untriaged", search: "str | None" = None,
    limit: int = 50, offset: int = 0,
) -> dict:
    """Read-only page of candidate terms joined with staged decisions.

    status: 'untriaged' (no staged decision) or 'decided' (has one).
    Never writes; safe on the worker reader thread.
    """
    taxonomy = load_layered_taxonomy(taxonomy_path)
    store = _open_store_readonly(sidecar_db_path)
    index = build_candidate_index(store, taxonomy)

    decisions = {d["term"]: d for d in list_decisions(sidecar_db_path, status="pending")}
    decisions.update({d["term"]: d for d in list_decisions(sidecar_db_path, status="applied")})

    rows: list[dict] = []
    for key, cand in index.items():
        decision = decisions.get(key)
        is_decided = decision is not None
        if status == "untriaged" and is_decided:
            continue
        if status == "decided" and not is_decided:
            continue
        if search and search.lower() not in key.lower():
            continue
        rows.append({
            "term": cand.term, "raw_term": cand.raw_term,
            "album_frequency": cand.album_frequency,
            "cooccurring_tags": cand.cooccurring_tags,
            "examples": cand.examples, "variants": cand.variants,
            "source": cand.source, "decision": decision,
        })

    rows.sort(key=lambda r: (-r["album_frequency"], r["term"]))
    untriaged_total = sum(1 for k in index if k not in decisions)
    decided_total = sum(1 for k in index if k in decisions)
    page = rows[offset:offset + limit]
    return {"terms": page, "untriaged_terms": untriaged_total,
            "decided_terms": decided_total}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_taxonomy_review_queue.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/taxonomy_review_queue.py tests/unit/test_taxonomy_review_queue.py
git commit -m "feat(taxonomy-review): merged/deduped candidate queue with impact annotations"
```

---

# PHASE 2 — Claude adjudicator contract

Model on `album_adjudicator.py`. One on-demand call per term → an add/alias/reject verdict that `validate_response()` maps onto the existing proposal helpers.

### Task 2.1: Adjudicator instructions, schema, payload, validate_response, adjudicate_term

**Files:**
- Create: `src/ai_genre_enrichment/taxonomy_term_adjudicator.py`
- Test: `tests/unit/test_taxonomy_term_adjudicator.py`

**Interfaces:**
- Consumes: `graph_growth.GrowthProposal`, `graph_growth._build_taxonomy_context` (bounded taxonomy slice), `layered_taxonomy.normalize_taxonomy_name`, `provider.create_enrichment_client`, `client.call_structured`.
- Produces:
  - `TAXONOMY_ADJUDICATOR_INSTRUCTIONS: str` (placement guardrails baked in, verbatim from `taxonomy-growth`).
  - `RejectVerdict` dataclass: `term, reject_reason, rationale`.
  - `taxonomy_adjudicator_response_format() -> dict`.
  - `build_payload(candidate, taxonomy) -> dict`.
  - `build_prompt(payload) -> str`.
  - `validate_response(data, *, term, taxonomy) -> GrowthProposal | RejectVerdict` (raises `ValueError` on contract violations).
  - `adjudicate_term(candidate, taxonomy, *, client) -> GrowthProposal | RejectVerdict`.

**Reject-reason enum** (must match `layered_genre_taxonomy.yaml` `enums.reject_reason`): `label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise`.

- [ ] **Step 1: Write the failing test** (validate_response mapping is the load-bearing unit; no live Claude)

```python
# tests/unit/test_taxonomy_term_adjudicator.py
import pytest
import yaml

from src.ai_genre_enrichment.graph_growth import GrowthProposal, validate_proposal
from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment import taxonomy_term_adjudicator as tta


def _taxonomy(tmp_path):
    data = {
        "taxonomy_version": "0.0.1-test",
        "enums": {"reject_reason": ["source_noise", "malformed", "label"],
                  "facet_type": ["mood"]},
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
            {"name": "shoegaze", "kind": "genre", "status": "active",
             "specificity_score": 0.6,
             "parent_edges": [{"target": "rock", "edge_type": "family_context",
                               "weight": 0.5, "confidence": 0.8}]},
        ],
    }
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_layered_taxonomy(p)


def test_validate_add_maps_to_growthproposal(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {
        "verdict": "add", "name": "dreampop", "kind": "genre", "status": "active",
        "specificity_score": 0.62,
        "parent_edges": [{"target": "rock", "edge_type": "family_context",
                          "weight": 0.5, "confidence": 0.8}],
        "similar_to": ["shoegaze"], "alias_variants": ["dream pop"],
        "reject_reason": "", "canonical_target": "", "rationale": "ok",
    }
    out = tta.validate_response(data, term="dreampop", taxonomy=tax)
    assert isinstance(out, GrowthProposal)
    assert out.kind == "genre"
    assert out.term_kind_confirm == "genre"
    # The mapped proposal must pass the real growth validator against this taxonomy.
    assert validate_proposal(tax, out) == []


def test_validate_alias_maps_to_alias_proposal(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "alias", "name": "shoe gaze", "kind": "alias",
            "canonical_target": "shoegaze", "rationale": "spelling",
            "parent_edges": [], "similar_to": [], "alias_variants": [],
            "specificity_score": 0.0, "status": "alias_only", "reject_reason": ""}
    out = tta.validate_response(data, term="shoe gaze", taxonomy=tax)
    assert isinstance(out, GrowthProposal)
    assert out.kind == "alias"
    assert out.canonical_target == "shoegaze"
    assert validate_proposal(tax, out) == []


def test_validate_reject_maps_to_rejectverdict(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "reject", "name": "my favorite albums", "kind": "reject",
            "reject_reason": "user_list", "rationale": "a personal list, not a genre",
            "parent_edges": [], "similar_to": [], "alias_variants": [],
            "specificity_score": 0.0, "status": "rejected", "canonical_target": ""}
    out = tta.validate_response(data, term="my favorite albums", taxonomy=tax)
    assert isinstance(out, tta.RejectVerdict)
    assert out.reject_reason == "user_list"


def test_validate_reject_rejects_bad_reason(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "reject", "name": "x", "reject_reason": "not_an_enum",
            "rationale": "", "parent_edges": [], "similar_to": [],
            "alias_variants": [], "specificity_score": 0.0}
    with pytest.raises(ValueError, match="reject_reason"):
        tta.validate_response(data, term="x", taxonomy=tax)


def test_validate_alias_unknown_target_raises(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "alias", "name": "x", "canonical_target": "no_such_genre",
            "rationale": "", "parent_edges": [], "similar_to": [],
            "alias_variants": [], "specificity_score": 0.0}
    with pytest.raises(ValueError):
        tta.validate_response(data, term="x", taxonomy=tax)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_taxonomy_term_adjudicator.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_genre_enrichment/taxonomy_term_adjudicator.py
"""Claude contract for adjudicating a single taxonomy-vocabulary term.

One on-demand call per term. Claude returns an add/alias/reject verdict;
validate_response() maps it onto graph_growth's proposal helpers (the same
records the Apply engine writes). Modeled on album_adjudicator.py, but the
concern is *what genres exist in the vocabulary*, not *which genres an album gets*.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .graph_growth import (
    GrowthProposal, _alias_target_error, _build_taxonomy_context,
    _parent_target_error,
)
from .layered_taxonomy import LayeredTaxonomy, normalize_taxonomy_name

REJECT_REASONS = {
    "label", "artist_name", "release_title", "place", "format", "era",
    "user_list", "malformed", "joke_tag", "negative_tag", "retail_bucket",
    "source_noise", "unknown_noise",
}

ADJUDICATOR_PROMPT_VERSION = "taxonomy-term-adjudicator-v1"

TAXONOMY_ADJUDICATOR_INSTRUCTIONS = """You place ONE candidate genre term into an existing hierarchical music-genre taxonomy, or reject it.

Return ONE JSON object only — no prose, no markdown.

Decide a `verdict`:
- "add": the term is a real GENRE or subgenre that belongs in the taxonomy. Give `kind` (umbrella/genre/subgenre), `status`, `specificity_score`, `parent_edges` (1-2, each target an EXISTING taxonomy name exactly as given), optional `similar_to` (existing names), optional `alias_variants`, and a `rationale`.
- "alias": the term is a spelling/naming variant of an EXISTING canonical genre. Give `canonical_target` (an existing canonical name) and `rationale`. Do NOT collapse a genuinely distinct genre into an alias (uk garage is NOT garage rock).
- "reject": the term is not a genre (a label, artist name, release title, place, format, era, user list, malformed/joke/negative tag, retail bucket, or source noise). Give a `reject_reason` from the allowed list and a `rationale`.

Placement guardrails (hard rules):
- Umbrellas are broad context, LOW specificity (~0.24-0.42), with spread parentage — no single child branch gets a strong parent weight.
- Instrument-led terms (piano jazz, jazz guitar) are FACETS, not genre leaves — reject them here as source_noise/format unless there's a real scene/style tradition beyond the instrument.
- Specificity ladder: umbrella 0.24-0.42 · genre 0.48-0.66 · subgenre 0.62-0.82.
- A leaf (genre/subgenre) needs at least one parent edge to an existing taxonomy name.
- Broad or noisy-but-real terms get status "review", not "active".
- parent_edges / similar_to targets MUST be names present in the provided `existing_taxonomy_names`. Do not invent edges to names not given.

Allowed reject_reason values: label, artist_name, release_title, place, format, era, user_list, malformed, joke_tag, negative_tag, retail_bucket, source_noise, unknown_noise.
"""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "name", "kind", "status", "specificity_score",
                 "parent_edges", "similar_to", "alias_variants",
                 "canonical_target", "reject_reason", "rationale"],
    "properties": {
        "verdict": {"type": "string", "enum": ["add", "alias", "reject"]},
        "name": {"type": "string"},
        "kind": {"type": "string",
                 "enum": ["umbrella", "genre", "subgenre", "alias", "reject"]},
        "status": {"type": "string",
                   "enum": ["active", "review", "alias_only", "rejected"]},
        "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "parent_edges": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["target", "edge_type", "weight", "confidence"],
                "properties": {
                    "target": {"type": "string"},
                    "edge_type": {"type": "string",
                                  "enum": ["is_a", "family_context"]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "similar_to": {"type": "array", "items": {"type": "string"}},
        "alias_variants": {"type": "array", "items": {"type": "string"}},
        "canonical_target": {"type": "string"},
        "reject_reason": {"type": "string"},
        "rationale": {"type": "string"},
    },
}


def taxonomy_adjudicator_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "name": "taxonomy_term_adjudicator_v1",
            "strict": True, "schema": deepcopy(_SCHEMA)}


@dataclass
class RejectVerdict:
    term: str
    reject_reason: str
    rationale: str = ""


def build_payload(candidate, taxonomy: LayeredTaxonomy) -> dict[str, Any]:
    """A *relevant slice* of the taxonomy (candidate parents/aliases by token +
    co-occurrence, plus family/umbrella anchors) so Claude places against the
    real graph, not the whole file. `candidate` is a TaxonomyCandidate-like
    object (.term/.raw_term/.album_frequency/.cooccurring_tags/.examples/.variants)."""
    context_names = _build_taxonomy_context(taxonomy, candidate)
    return {
        "candidate_term": candidate.raw_term,
        "normalized_term": candidate.term,
        "album_frequency": candidate.album_frequency,
        "cooccurring_tags": list(candidate.cooccurring_tags),
        "spelling_variants": list(candidate.variants),
        "examples": list(candidate.examples),
        "existing_taxonomy_names": context_names,
        "prompt_version": ADJUDICATOR_PROMPT_VERSION,
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return ("Adjudicate this candidate genre term. Use the payload only.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def validate_response(data: dict[str, Any], *, term: str,
                      taxonomy: LayeredTaxonomy) -> "GrowthProposal | RejectVerdict":
    if not isinstance(data, dict):
        raise ValueError("adjudicator response must be a JSON object")
    verdict = str(data.get("verdict") or "").strip()
    if verdict not in ("add", "alias", "reject"):
        raise ValueError(f"unknown verdict: {verdict!r}")

    if verdict == "reject":
        reason = str(data.get("reject_reason") or "").strip()
        if reason not in REJECT_REASONS:
            raise ValueError(f"unsupported reject_reason: {reason!r}")
        return RejectVerdict(term=term, reject_reason=reason,
                             rationale=str(data.get("rationale") or ""))

    if verdict == "alias":
        target = str(data.get("canonical_target") or "").strip()
        err = _alias_target_error(taxonomy, target)
        if err is not None:
            raise ValueError(err)
        return GrowthProposal(
            name=str(data.get("name") or term), kind="alias", status="alias_only",
            specificity_score=0.0, parent_edges=[], similar_to=[], alias_variants=[],
            term_kind_confirm="genre", canonical_target=target,
            rationale=str(data.get("rationale") or ""))

    # verdict == "add"
    kind = str(data.get("kind") or "genre")
    if kind not in ("umbrella", "genre", "subgenre"):
        raise ValueError(f"add verdict has non-leaf/umbrella kind: {kind!r}")
    parent_edges = list(data.get("parent_edges") or [])
    similar_to = list(data.get("similar_to") or [])
    for e in parent_edges:
        err = _parent_target_error(taxonomy, str(e.get("target") or ""))
        if err is not None:
            raise ValueError(err)
    for t in similar_to:
        err = _parent_target_error(taxonomy, str(t))
        if err is not None:
            raise ValueError(err)
    if kind in ("genre", "subgenre") and not parent_edges:
        raise ValueError("an add leaf needs at least one parent edge")
    return GrowthProposal(
        name=str(data.get("name") or term), kind=kind,
        status=str(data.get("status") or "active"),
        specificity_score=float(data.get("specificity_score") or 0.5),
        parent_edges=parent_edges, similar_to=similar_to,
        alias_variants=list(data.get("alias_variants") or []),
        term_kind_confirm="genre", rationale=str(data.get("rationale") or ""))


def adjudicate_term(candidate, taxonomy: LayeredTaxonomy, *,
                    client) -> "GrowthProposal | RejectVerdict":
    """One structured Claude call. `client` exposes call_structured (see
    provider.create_enrichment_client)."""
    payload = build_payload(candidate, taxonomy)
    data = client.call_structured(
        build_prompt(payload), taxonomy_adjudicator_response_format(),
        instructions=TAXONOMY_ADJUDICATOR_INSTRUCTIONS)
    return validate_response(data, term=candidate.term, taxonomy=taxonomy)
```

> **Note on private imports:** `_build_taxonomy_context`, `_parent_target_error`, `_alias_target_error` are reused from `graph_growth` so the adjudicator validates against the SAME loader-resolution rules the Apply step uses (single source of truth). If a reviewer prefers public surface, promote these to public names in `graph_growth` in a separate refactor commit — do not duplicate the logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_taxonomy_term_adjudicator.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/taxonomy_term_adjudicator.py tests/unit/test_taxonomy_term_adjudicator.py
git commit -m "feat(taxonomy-review): Claude term adjudicator contract (add/alias/reject)"
```

---

# PHASE 3 — Apply engine (riskiest; before the GUI)

Validate the whole batch, order same-batch forward refs, timestamped backup, write add/alias via the public ingest + reject records directly, bump version, return stats. Runs inside the tracked worker job (Phase 4).

### Task 3.1: `apply_decisions` — batch validate + forward-ref order + backup + write + stats

**Files:**
- Create: `src/ai_genre_enrichment/taxonomy_apply.py`
- Test: `tests/unit/test_taxonomy_apply.py`

**Interfaces:**
- Consumes: `graph_growth.GrowthProposal`, `validate_proposal`, `append_approved_to_taxonomy`, `normalize_taxonomy_name`, `layered_taxonomy.load_layered_taxonomy`.
- Produces:
  - `@dataclass ApplyResult`: `added: int, aliased: int, rejected: int, deferred_edges: list[dict], backup_path: str, new_version: str, validation_failures: list[tuple[str, list[str]]]`.
  - `@dataclass Decision`: `term, verdict, proposal (GrowthProposal|None), reject_reason (str|None), rationale (str)`.
  - `apply_decisions(taxonomy_path, decisions: list[Decision], *, new_version: str, backup_dir=None) -> ApplyResult`.
  - `preflight(taxonomy, decisions) -> list[tuple[str, list[str]]]` (validation failures; `[]` == OK).

**Algorithm (Handoff §6, exactly):**
1. Re-read YAML fresh from disk (`load_layered_taxonomy`) — never trust a snapshot from record time.
2. Build add/alias `GrowthProposal`s from decisions; reject → reject records.
3. `preflight`: `validate_proposal` per add/alias proposal (skip reject — validator doesn't support it). If ANY fail → abort the write, return failures.
4. Forward-ref ordering: if a proposal's parent/`similar_to` target is another pending NEW term in this batch, topologically order so parents land first. If a true forward/cyclic edge remains, trim it and record a `deferred_edge`. Common case (targets are existing canonical genres) is a no-op.
5. Timestamped backup of the YAML.
6. Write add/alias via `append_approved_to_taxonomy(path, ordered_proposals, new_version=...)`; then append reject records and re-stamp the version (one job, two writes, both post-backup).
7. (Reload is the worker's job — Phase 4 — but `apply_decisions` returns `new_version` so the caller can verify.)
8. Stats: added/aliased/rejected counts, deferred edges, backup path.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_taxonomy_apply.py
import shutil
from pathlib import Path

import yaml

from src.ai_genre_enrichment.graph_growth import GrowthProposal
from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment import taxonomy_apply as ta


def _seed_taxonomy(tmp_path: Path) -> Path:
    data = {
        "taxonomy_version": "0.8.0-test",
        "enums": {"reject_reason": ["user_list", "source_noise"],
                  "facet_type": ["mood"]},
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
            {"name": "shoegaze", "kind": "genre", "status": "active",
             "specificity_score": 0.6,
             "parent_edges": [{"target": "rock", "edge_type": "family_context",
                               "weight": 0.5, "confidence": 0.8}]},
        ],
    }
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def _add(name, parent="rock"):
    return ta.Decision(
        term=name, verdict="add",
        proposal=GrowthProposal(
            name=name, kind="genre", status="active", specificity_score=0.6,
            parent_edges=[{"target": parent, "edge_type": "family_context",
                           "weight": 0.5, "confidence": 0.8}],
            term_kind_confirm="genre"),
        reject_reason=None, rationale="")


def test_apply_add_writes_record_and_bumps_version(tmp_path):
    path = _seed_taxonomy(tmp_path)
    result = ta.apply_decisions(
        path, [_add("dreampop")],
        new_version="0.9.0-gui-20260626-grown", backup_dir=tmp_path / "bak")
    assert result.added == 1
    assert result.validation_failures == []
    tax = load_layered_taxonomy(path)
    assert tax.genre_by_name("dreampop") is not None
    assert tax.version == "0.9.0-gui-20260626-grown"
    assert Path(result.backup_path).exists()


def test_apply_alias_writes_alias(tmp_path):
    path = _seed_taxonomy(tmp_path)
    d = ta.Decision(term="shoe gaze", verdict="alias",
                    proposal=GrowthProposal(
                        name="shoe gaze", kind="alias", status="alias_only",
                        specificity_score=0.0, canonical_target="shoegaze",
                        term_kind_confirm="genre"),
                    reject_reason=None, rationale="")
    result = ta.apply_decisions(path, [d], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.aliased == 1
    tax = load_layered_taxonomy(path)
    # alias resolves to the canonical genre
    assert tax.genre_by_name("shoe gaze").name == "shoegaze"


def test_apply_reject_writes_reject_record(tmp_path):
    path = _seed_taxonomy(tmp_path)
    d = ta.Decision(term="my list", verdict="reject", proposal=None,
                    reject_reason="user_list", rationale="not a genre")
    result = ta.apply_decisions(path, [d], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.rejected == 1
    tax = load_layered_taxonomy(path)  # loads -> reject_reason enum-validated
    assert tax.rejected_term_by_name("my list") is not None


def test_apply_aborts_on_validation_failure_without_writing(tmp_path):
    path = _seed_taxonomy(tmp_path)
    before = path.read_text(encoding="utf-8")
    bad = _add("bad", parent="no_such_parent")  # parent doesn't exist -> fail
    result = ta.apply_decisions(path, [bad], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.validation_failures  # non-empty
    assert result.added == 0
    assert path.read_text(encoding="utf-8") == before  # untouched


def test_apply_orders_same_batch_forward_reference(tmp_path):
    # "child" parents on "parentnew", which is ALSO new in this batch.
    path = _seed_taxonomy(tmp_path)
    parentnew = _add("parentnew")  # parents on rock (exists)
    child = _add("childnew", parent="parentnew")  # parents on a same-batch new term
    result = ta.apply_decisions(
        path, [child, parentnew],  # deliberately out of order
        new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.validation_failures == []
    assert result.added == 2
    tax = load_layered_taxonomy(path)
    assert tax.genre_by_name("childnew") is not None
    assert tax.genre_by_name("parentnew") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_taxonomy_apply.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_genre_enrichment/taxonomy_apply.py
"""Apply staged taxonomy-term decisions into data/layered_genre_taxonomy.yaml.

The load-bearing, riskiest part: validate the WHOLE batch first, order same-batch
forward references (parents before children), timestamped backup, then write
add/alias via the public ingest and reject records directly, bumping the version.
Runs inside the tracked Apply worker job. NEVER touches metadata.db.
"""
from __future__ import annotations

import datetime
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .graph_growth import GrowthProposal, append_approved_to_taxonomy, validate_proposal
from .layered_taxonomy import (
    DEFAULT_TAXONOMY_PATH, load_layered_taxonomy, normalize_taxonomy_name,
)


@dataclass
class Decision:
    term: str
    verdict: str                       # 'add' | 'alias' | 'reject'
    proposal: "GrowthProposal | None"  # for add/alias
    reject_reason: "str | None" = None
    rationale: str = ""


@dataclass
class ApplyResult:
    added: int = 0
    aliased: int = 0
    rejected: int = 0
    deferred_edges: list[dict] = field(default_factory=list)
    backup_path: str = ""
    new_version: str = ""
    validation_failures: list[tuple[str, list[str]]] = field(default_factory=list)


def _reject_record(name: str, reject_reason: str, notes: str) -> dict:
    return {
        "name": name, "kind": "reject", "role": "reject", "status": "rejected",
        "facet_type": None, "specificity_score": None, "canonical_target": None,
        "parent_edges": [], "secondary_roles": [], "reject_reason": reject_reason,
        "alias_policy": None, "source_policy": "growth",
        "possible_context_target": None,
        "notes": notes or "Rejected via taxonomy term adjudication.",
    }


def preflight(taxonomy, decisions: list[Decision]) -> list[tuple[str, list[str]]]:
    """validate_proposal per add/alias proposal. reject is skipped (the validator
    doesn't support it; the loader validates reject_reason at write/reload time)."""
    failures: list[tuple[str, list[str]]] = []
    for d in decisions:
        if d.verdict == "reject":
            continue
        if d.proposal is None:
            failures.append((d.term, ["missing proposal for add/alias decision"]))
            continue
        errors = validate_proposal(taxonomy, d.proposal)
        if errors:
            failures.append((d.term, errors))
    return failures


def _order_for_forward_refs(
    add_alias: list[Decision], taxonomy,
) -> tuple[list[Decision], list[dict]]:
    """Order so a proposal whose parent/similar_to target is ANOTHER new term in
    this batch lands after that term. Trim a residual cyclic edge -> deferred."""
    new_names = {normalize_taxonomy_name(d.proposal.name) for d in add_alias
                 if d.proposal is not None}

    def targets(d: Decision) -> set[str]:
        if d.proposal is None:
            return set()
        names = {str(e.get("target") or "") for e in d.proposal.parent_edges}
        names |= {str(t) for t in d.proposal.similar_to}
        return {normalize_taxonomy_name(n) for n in names}

    # Kahn's algorithm over same-batch dependency edges only.
    by_name = {normalize_taxonomy_name(d.proposal.name): d for d in add_alias
               if d.proposal is not None}
    deps: dict[str, set[str]] = {
        n: (targets(d) & new_names) - {n} for n, d in by_name.items()}
    ordered: list[Decision] = []
    resolved: set[str] = set()
    deferred: list[dict] = []
    # Aliases (proposal is None for none here; alias has a proposal) carry no
    # same-batch parent deps, so they order freely.
    progress = True
    while len(resolved) < len(by_name) and progress:
        progress = False
        for n, d in by_name.items():
            if n in resolved:
                continue
            if deps[n] <= resolved:
                ordered.append(d)
                resolved.add(n)
                progress = True
    # Anything left is in a cycle: trim its same-batch edges and emit it deferred.
    for n, d in by_name.items():
        if n in resolved:
            continue
        for t in sorted(deps[n] - resolved):
            deferred.append({"source": by_name[n].proposal.name, "target": t,
                             "reason": "same-batch cycle; edge trimmed"})
        # trim same-batch parent/similar edges so the record still lands
        p = d.proposal
        p.parent_edges = [e for e in p.parent_edges
                          if normalize_taxonomy_name(str(e.get("target") or "")) not in (deps[n] - resolved)]
        p.similar_to = [t for t in p.similar_to
                        if normalize_taxonomy_name(str(t)) not in (deps[n] - resolved)]
        ordered.append(d)
        resolved.add(n)
    return ordered, deferred


def apply_decisions(taxonomy_path, decisions: list[Decision], *,
                    new_version: str, backup_dir=None) -> ApplyResult:
    path = Path(taxonomy_path or DEFAULT_TAXONOMY_PATH)
    taxonomy = load_layered_taxonomy(path)  # fresh read (step 1)

    add_alias = [d for d in decisions if d.verdict in ("add", "alias")]
    rejects = [d for d in decisions if d.verdict == "reject"]

    failures = preflight(taxonomy, add_alias)  # step 3
    if failures:
        return ApplyResult(validation_failures=failures, new_version=taxonomy.version)

    ordered, deferred = _order_for_forward_refs(add_alias, taxonomy)  # step 4

    # step 5 — timestamped backup
    backup_dir = Path(backup_dir) if backup_dir else path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.bak.{ts}"
    shutil.copy2(path, backup_path)

    # step 6 — write add/alias via public ingest, then reject records.
    proposals = [d.proposal for d in ordered if d.proposal is not None]
    if proposals:
        append_approved_to_taxonomy(path, proposals, new_version=new_version)
    aliased = sum(1 for d in ordered if d.proposal and d.proposal.kind == "alias")
    added = len(proposals) - aliased

    if rejects:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records = data.setdefault("records", [])
        for d in rejects:
            records.append(_reject_record(d.term, d.reject_reason or "unknown_noise",
                                          d.rationale))
        data["taxonomy_version"] = new_version
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    elif not proposals:
        # nothing written above but we still must bump version + persist backup intent
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["taxonomy_version"] = new_version
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")

    return ApplyResult(added=added, aliased=aliased, rejected=len(rejects),
                       deferred_edges=deferred, backup_path=str(backup_path),
                       new_version=new_version)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_taxonomy_apply.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full new-module suite + the enrichment suite with deselects**

Run:
```bash
python -m pytest tests/unit/test_taxonomy_decision_store.py tests/unit/test_taxonomy_review_queue.py tests/unit/test_taxonomy_term_adjudicator.py tests/unit/test_taxonomy_apply.py -q
python -m pytest tests/unit/test_graph_growth.py tests/unit/test_layered_taxonomy.py -q \
  --deselect tests/unit/test_ai_genre_hybrid_cli.py::test_hybrid_enrich_one_apply_persists_accepted_signature
```
Expected: all green (the new four pass; the reused-module suites unaffected by our additions).

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/taxonomy_apply.py tests/unit/test_taxonomy_apply.py
git commit -m "feat(taxonomy-review): Apply engine — validate, order forward-refs, backup, write, bump"
```

> **Phases 1–3 produce a fully verified backend core. The GUI session wires against this.**

---

# PHASE 4 — API routes + worker handlers

Wire the five endpoints through every layer (web-gui checklist). Untracked reads/decision/adjudicate first, then the tracked Apply job. Re-read `app.py` review routes (~`:276`) and `schemas.py` against the live tree before editing.

### Task 4.1: Worker handlers + registration

**Files:**
- Modify: `src/playlist_gui/worker.py` (add 5 handlers; register in the two dicts at `:2798`/`:2842`)
- Modify: `tests/fixtures/fake_worker.py` (add command branches)
- Test: `tests/unit/test_worker_taxonomy_handlers.py` (drive handlers with a temp sidecar DB + temp YAML, asserting emitted events)

**Commands (mirror the escalation handlers exactly, incl. explicit `request_id` + `job_id=None` on untracked):**

| Command | Tracked? | Handler | Mirrors |
|---|---|---|---|
| `get_taxonomy_queue` | untracked (read-only) | `handle_get_taxonomy_queue` | `handle_get_escalation_queue` (`:2684`) |
| `get_taxonomy_completed` | untracked (read-only) | `handle_get_taxonomy_completed` | `handle_get_escalation_completed` (`:2705`) |
| `adjudicate_taxonomy_term` | untracked (Claude call; NO persist) | `handle_adjudicate_taxonomy_term` | new — builds candidate, calls `adjudicate_term`, returns verdict |
| `record_taxonomy_decision` | untracked (quick write) | `handle_record_taxonomy_decision` | `handle_apply_escalation_decision` (`:2726`) |
| `apply_taxonomy_decisions` | **tracked** | `handle_apply_taxonomy_decisions` | `handle_publish_decided` (`:2550`) |

Handler skeletons (finalize against the live escalation handlers at execution):

```python
# get_taxonomy_queue — untracked, read-only (reader thread)
def handle_get_taxonomy_queue(cmd_data):
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.taxonomy_review_queue import list_page
        from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
        page = list_page(SIDECAR_DB_PATH, DEFAULT_TAXONOMY_PATH, status="untriaged",
                         search=(cmd_data.get("search") or "").strip() or None,
                         limit=int(cmd_data.get("limit") or 50),
                         offset=int(cmd_data.get("offset") or 0))
        emit_event({"type": "result", "result_type": "taxonomy_queue",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_taxonomy_queue", "ok": True,
                    "detail": f"{page['untriaged_terms']} untriaged",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_taxonomy_queue", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

```python
# adjudicate_taxonomy_term — untracked; calls Claude, returns verdict, does NOT persist.
def handle_adjudicate_taxonomy_term(cmd_data):
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.taxonomy_review_queue import build_candidate_index, _open_store_readonly
        from src.ai_genre_enrichment.taxonomy_term_adjudicator import adjudicate_term, RejectVerdict
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy, normalize_taxonomy_name
        from src.ai_genre_enrichment.provider import create_enrichment_client
        from dataclasses import asdict

        term = normalize_taxonomy_name(str(cmd_data.get("term") or ""))
        taxonomy = load_default_layered_taxonomy()
        index = build_candidate_index(_open_store_readonly(SIDECAR_DB_PATH), taxonomy)
        candidate = index.get(term)
        if candidate is None:
            raise ValueError(f"term not in candidate queue: {term!r}")
        client = create_enrichment_client()  # web_mode off by default
        verdict = adjudicate_term(candidate, taxonomy, client=client)
        if isinstance(verdict, RejectVerdict):
            out = {"verdict": "reject", "term": term,
                   "reject_reason": verdict.reject_reason, "rationale": verdict.rationale}
        else:
            out = {"verdict": verdict.kind == "alias" and "alias" or "add",
                   "term": term, "proposal": asdict(verdict)}
        emit_event({"type": "result", "result_type": "taxonomy_adjudication",
                    "request_id": rid, "job_id": None, **out})
        emit_event({"type": "done", "cmd": "adjudicate_taxonomy_term", "ok": True,
                    "detail": term, "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "adjudicate_taxonomy_term", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

```python
# record_taxonomy_decision — untracked, quick write to staging
def handle_record_taxonomy_decision(cmd_data):
    rid = cmd_data.get("request_id")
    try:
        import json as _json
        from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore
        term = str(cmd_data.get("term") or "")
        verdict = str(cmd_data.get("verdict") or "")
        store = TaxonomyDecisionStore(SIDECAR_DB_PATH)
        try:
            if verdict == "revert":
                store.revert(term); status = "reverted"
            else:
                store.record_decision(
                    term=term, raw_term=str(cmd_data.get("raw_term") or term),
                    verdict=verdict,
                    proposal_json=_json.dumps(cmd_data.get("proposal") or None),
                    claude_json=_json.dumps(cmd_data.get("claude") or None),
                    human_edited=int(bool(cmd_data.get("human_edited"))))
                status = verdict
        finally:
            store.close()
        emit_event({"type": "result", "result_type": "taxonomy_decision",
                    "term": term, "status": status, "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "record_taxonomy_decision", "ok": True,
                    "detail": f"{term}: {status}", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "record_taxonomy_decision", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})
```

```python
# apply_taxonomy_decisions — TRACKED job; backup template = handle_publish_decided
def handle_apply_taxonomy_decisions(cmd_data):
    try:
        from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore
        from src.ai_genre_enrichment.taxonomy_apply import apply_decisions, Decision
        from src.ai_genre_enrichment.graph_growth import GrowthProposal
        from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
        import datetime

        emit_progress("apply_taxonomy_decisions", 0, 3, "reading pending decisions")
        store = TaxonomyDecisionStore(SIDECAR_DB_PATH)
        try:
            pending = store.list_pending()
            decisions = []
            for r in pending:
                proposal = None
                if r["verdict"] in ("add", "alias") and r["proposal"]:
                    proposal = GrowthProposal(**r["proposal"])
                decisions.append(Decision(
                    term=r["term"], verdict=r["verdict"], proposal=proposal,
                    reject_reason=(r["proposal"] or {}).get("reject_reason")
                    if isinstance(r["proposal"], dict) else None,
                    rationale=""))
            check_cancelled()
            ts = datetime.datetime.now().strftime("%Y%m%d")
            version = f"0.X.0-gui-{ts}-grown"  # bump minor at execution per live YAML
            emit_progress("apply_taxonomy_decisions", 1, 3, "validating + writing")
            result = apply_decisions(DEFAULT_TAXONOMY_PATH, decisions,
                                     new_version=version)
            if result.validation_failures:
                emit_result("apply_taxonomy_decisions",
                            {"ok": False, "validation_failures": result.validation_failures})
                emit_done("apply_taxonomy_decisions", False, "validation failed")
                return
            store.mark_applied([d.term for d in decisions], version)
        finally:
            store.close()
        emit_progress("apply_taxonomy_decisions", 2, 3, "reloading graph")
        # bust any long-lived taxonomy cache here (see §6 cache note / Task 4.3)
        emit_result("apply_taxonomy_decisions", {
            "ok": True, "added": result.added, "aliased": result.aliased,
            "rejected": result.rejected, "deferred_edges": result.deferred_edges,
            "backup": result.backup_path, "new_version": result.new_version})
        emit_done("apply_taxonomy_decisions", True,
                  f"Applied {result.added + result.aliased + result.rejected} decisions",
                  summary=f"added={result.added} aliased={result.aliased} rejected={result.rejected}")
    except CancellationError:
        emit_done("apply_taxonomy_decisions", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("apply_taxonomy_decisions", False, str(e))
```

Registration:
```python
# in TRACKED_COMMAND_HANDLERS (worker.py:2798)
"apply_taxonomy_decisions": handle_apply_taxonomy_decisions,
# in UNTRACKED_COMMAND_HANDLERS (worker.py:2842)
"get_taxonomy_queue": handle_get_taxonomy_queue,
"get_taxonomy_completed": handle_get_taxonomy_completed,
"adjudicate_taxonomy_term": handle_adjudicate_taxonomy_term,
"record_taxonomy_decision": handle_record_taxonomy_decision,
```

**Steps:** (1) write `tests/unit/test_worker_taxonomy_handlers.py` driving each untracked handler against a temp `SIDECAR_DB_PATH` (monkeypatch the module constant) + temp YAML, asserting the emitted `result`/`done` events (capture via monkeypatching `emit_event`); (2) run → fail; (3) implement handlers + registration; (4) run → pass; (5) commit.

> **Cache note (Handoff §6 / open question §10-3):** before claiming Apply "reloads the graph," grep the worker + genre runtime for `lru_cache` / module-level taxonomy singletons:
> `Grep "lru_cache|load_default_layered_taxonomy|_TAXONOMY|graph_adapter" src/playlist_gui src/genre`.
> `load_default_layered_taxonomy` reads fresh (no cache at its def), but a long-lived consumer holding a `LayeredTaxonomy` must be busted after write. Resolve this during implementation; if a singleton exists, add an explicit bust in `handle_apply_taxonomy_decisions` after `apply_decisions`.

### Task 4.2: FastAPI routes + schemas

**Files:**
- Modify: `src/playlist_web/app.py` (add 5 routes mirroring the review routes)
- Modify: `src/playlist_web/schemas.py` (request bodies + `JobOut`/result fields)
- Test: extend `tests/integration/test_taxonomy_web_api.py` (Task 4.4) — covered there.

Routes (mirror review routes; untracked use `bridge.command`, tracked uses `registry.create` + `bridge.submit` + 409 guard):
- `GET  /api/taxonomy/queue?status=&search=&limit=&offset=` → `get_taxonomy_queue`
- `GET  /api/taxonomy/completed?...` → `get_taxonomy_completed`
- `POST /api/taxonomy/adjudicate` `{term}` → `adjudicate_taxonomy_term`
- `POST /api/taxonomy/decision` `{term, raw_term, verdict, proposal?, claude?, human_edited?}` → `record_taxonomy_decision`
- `POST /api/taxonomy/apply` → tracked `apply_taxonomy_decisions` → `{job_id}`

Schemas (`schemas.py`): `TaxonomyDecisionRequest`, `TaxonomyAdjudicateRequest`; ensure the non-playlist results (`taxonomy_queue`/`taxonomy_completed`/`taxonomy_adjudication`/`taxonomy_decision`) land via the generic `tool_result` path in `JobRegistry.apply_event` (web-gui checklist item 3) — the untracked ones are request-id replies (not job results), so verify the `bridge.command` return shape carries the `result` payload.

### Task 4.3: Result-capture + cache-bust verification

Confirm (web-gui traps): untracked replies route by `request_id`; the tracked Apply result lands in `job.tool_result` (generic `result` branch); resolve the taxonomy-cache question from Task 4.1's grep.

### Task 4.4: Integration test (real worker, reader-safe)

**Files:** Create `tests/integration/test_taxonomy_web_api.py`.
- `TestClient(create_app(worker_cmd=FAKE))` with `tests/fixtures/fake_worker.py` branches for the 5 commands; assert queue read, decision record, apply job lifecycle (`/api/jobs/{id}` → success).
- **Windows note:** real-worker endpoint verification under `TestClient` can time out spuriously; use the FAKE worker for the integration test, and do the REAL-worker check via the live server in Phase 6 (web-gui skill).

**Commit** after 4.1–4.4 green.

---

# PHASE 5 — Frontend panel + tab

Mirror `GenreReviewPanel.tsx`. Re-read these at execution: `GenreReviewPanel.tsx`, `AdvancedPanel.tsx` (tab buttons ~`:26`, switch ~`:33`), `api.ts` (`reviewQueue`/`reviewDecision`/`reviewPublish` ~`:125`), `types.ts` (`EscalationOut`/`EscalationDecisionRequest` ~`:181`), and `useWorkerEvents`.

### Task 5.1: Types + API client
- `web/src/lib/types.ts`: `TaxonomyQueueItem` (`term, raw_term, album_frequency, cooccurring_tags, examples, variants, source, decision`), `TaxonomyVerdict` (`verdict, proposal?, reject_reason?, rationale`), `TaxonomyDecisionRequest`, `TaxonomyApplyResult`.
- `web/src/lib/api.ts`: `taxonomyQueue(params)`, `taxonomyCompleted(params)`, `taxonomyAdjudicate(term)`, `taxonomyDecision(body)`, `taxonomyApply()` (mirror the review methods + the tracked-job POST shape).

### Task 5.2: `TaxonomyReviewPanel.tsx`
- Queue fetch (untriaged/completed toggle), search, keyboard shortcuts, per-row: term · affected albums · co-occurring tags · example releases · **"Ask Claude"** → render verdict (parent edges + specificity, or alias target, or reject reason) → **Accept / Edit / Override / Reject** → `taxonomyDecision`.
- **"Apply N decisions"** button → `taxonomyApply()` tracked job via `useWorkerEvents`; show result stats incl. "**M albums will re-classify on next publish**" (sum of `album_frequency` over applied add/alias terms) + any deferred edges + backup path.

### Task 5.3: Tab registration
- `AdvancedPanel.tsx`: add a "Taxonomy" tab button + render `<TaxonomyReviewPanel/>` in the switch.

### Task 5.4: Build
- `npm --prefix web run build`; `grep -rl taxonomy web/dist/assets/*.js` to confirm the build is current.
- Optional Playwright spec mirroring an existing `web/tests/*.spec.ts` against the fake worker.

**Commit** after the panel renders and the dist is rebuilt.

---

# PHASE 6 — End-to-end pass (per web-gui §8)

### Task 6.1: Resolve data access (BLOCKER for a real run — do first)
- `config.yaml` is absent here (gitignored). Copy it from the main checkout's `config.yaml` into this worktree (CLAUDE.md worktree note: copy it yourself). Do NOT `git add` it.
- The sidecar `ai_genre_enrichment.db` and `metadata.db` are absent. For the e2e, **copy** (not symlink — WAL-aliasing corruption rule) a sidecar DB into `data/` OR point `SIDECAR_DB_PATH` at a temp copy. The queue read needs `all_collected_tags`; a real sidecar copy gives a realistic queue. If no sidecar is available in this worktree, build a tiny fixture sidecar with a few `ai_genre_source_tags` rows so the queue is non-empty.
- The Apply write target must be a **temp copy** of `layered_genre_taxonomy.yaml` for the e2e (Handoff §8: "confirm a temp-copy YAML gained the record"), so the real tracked file is untouched until Dylan reviews the diff deliberately. Point the Apply at the temp copy via a config/env seam, or run the e2e against a copied worktree path. **Surface this to the user and confirm the data strategy before running anything that writes.**

### Task 6.2: Exercise the real path
- `python tools/serve_web.py --port <spare> --no-browser`; open the Taxonomy tab.
- queue → "Ask Claude" on one term → Accept → "Apply N decisions" → confirm the temp-copy YAML gained the record + version bump; confirm a timestamped backup exists.
- Look for regressions adjacent to the change (escalation/review panels still load; cancel still works).
- Do NOT claim done until this is run and observed (CLAUDE.md verify-before-claiming).

### Task 6.3: Final suite + handoff
- `python -m pytest -q -m "not slow"` with the Global-Constraints deselects; quote real pass/fail counts.
- Summarize for Dylan: what landed, the one-Apply-one-commit reminder, the §7 scope boundaries (does NOT clear the `no canonical name` warnings / re-tag albums by itself), and the §10 resolutions (impact-count = distinct albums; union is gather-spined; cache bust outcome).

---

## Self-review (writing-plans)

- **Spec coverage:** §1 feature (tab) → Phase 5; §2 Decision 1 (merged/deduped queue + impact) → Task 1.2; Decision 2 (two-phase write) → Tasks 1.1 + 3.1; Decision 3 (graph-only + "M albums" message) → Tasks 3.1/5.2/6.2; Decision 4 (per-term Ask Claude, tab in AdvancedPanel) → Tasks 2.1/5.2/5.3. §3 modules → Phases 1–3 modules. §4 staging table + verdict→proposal mapping → Tasks 1.1/2.1. §5 wiring → Phase 4 + 5. §6 Apply algorithm → Task 3.1 (+ cache note Task 4.1). §7 scope boundaries → Task 6.3 summary. §8 test plan → tests in every task + Task 6.3. §9 build order → phase order. §10 open questions → resolved inline (impact-count Task 1.2; Ask-Claude-on-all deferred; cache bust Task 4.1).
- **Placeholder scan:** Phases 1–3 carry full code. Phase 4 carries real handler skeletons + registration; the one literal placeholder is the version string `0.X.0` (intentionally bumped against the live YAML at execution — flagged, not a TODO). Phases 5–6 are mirror tasks against named live files with exact responsibilities.
- **Type consistency:** `GrowthProposal` fields match `graph_growth.py:189`; `apply_decisions`/`Decision`/`ApplyResult` names are consistent across Task 3.1 and Task 4.1; `list_page` return keys (`terms`/`untriaged_terms`/`decided_terms`) are consistent between Task 1.2 and the worker handler; `validate_response` returns `GrowthProposal | RejectVerdict` consistently in Tasks 2.1 and 4.1.
