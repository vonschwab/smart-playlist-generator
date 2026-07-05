# Always-Publish Policy Flip (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace queue-blocking fusion routing with confidence-weighted always-publish, then additively backfill the existing library so ~21k policy-driven review-queue rows become published audit-trail entries — Milestone 1 of `docs/superpowers/specs/2026-07-04-zero-touch-genre-pipeline-design.md`.

**Architecture:** The fusion policy (`fuse_hybrid_evidence`) stops producing an unpublished `needs_review` bucket: Last.fm-only terms publish at a hard confidence cap (0.40) and below-corroboration-bar terms publish at their evidence confidence — both flow into the existing `provisional` bucket, which already materializes to assignments AND records in the review queue (the audit-trail precedent). A new additive-only backfill re-runs fusion per release and merges *only new* assignment rows (never removes, never lowers confidence — the 2026-06-12 anti-wholesale-re-derivation lesson). Confidence attenuation is already real downstream: publish carries confidence into `release_effective_genres`, and the artifact weights genre entries by `confidence × layer_weight` (`scripts/build_beat3tower_artifacts.py:464`), so a 0.40-capped genre has 40% steering weight.

**Tech Stack:** Python 3.11, SQLite (sidecar `data/ai_genre_enrichment.db`, authority in `data/metadata.db`), pytest, existing enrichment modules under `src/ai_genre_enrichment/`.

## Global Constraints

- `data/metadata.db` is irreplaceable: only the `publish` stage writes it (automatic timestamped backup); the real publish run requires Dylan's explicit go after the eval gate (Task 7 STOP).
- Additive-only invariant for the backfill: never remove an assignment row, never lower an existing row's confidence, never touch `rejected_by_user` or user overrides.
- Code tasks (1–6, 10) run in an isolated worktree on a branch. Operational tasks (7–9) run from the **main checkout** — never run data-writing stages against a symlinked SQLite DB (WAL-aliasing corruption rule).
- Pytest: `python -m pytest -q -m "not slow"` directly with a timeout — never piped through `head`/`tail`.
- Music library audio files are permanently read-only.
- No new config knob for the policy: the flip is a pure-function change (rollback = revert commit). Tuning surface = keyword args + module constants in `hybrid_evidence.py`. Rationale: the policy is consumed via 5+ entry points (analyze stage, CLI, worker scan, diagnostics); a half-plumbed config knob is this repo's documented #1 trap, and a pure constant cannot go inert.
- Executor must invoke the `evaluation-methodology` skill before building Task 7's eval package, and the `playlist-testing` skill before Task 9's generation check.

---

### Task 1: Fusion — Last.fm-only terms publish at capped confidence

**Files:**
- Modify: `src/ai_genre_enrichment/hybrid_evidence.py:36` (add constant), `:221-234` (lastfm-only branch)
- Test: `tests/unit/test_ai_genre_hybrid_evidence.py`

**Interfaces:**
- Consumes: existing `EvidenceTerm`, `FusedGenreDecision`, `fuse_hybrid_evidence`.
- Produces: `LASTFM_ONLY_CONFIDENCE_CAP: float = 0.40` module constant; lastfm-only decisions now appear in `report.provisional_genres` with `basis="lastfm_only"` and `confidence <= 0.40`. Task 2 and Task 5 rely on the `basis="lastfm_only"` marker.

- [ ] **Step 1: Write the failing test**

Replace `test_lastfm_only_no_release_evidence_needs_review` (currently at `tests/unit/test_ai_genre_hybrid_evidence.py:30-44`) with:

```python
def test_lastfm_only_publishes_provisionally_at_capped_confidence():
    # Zero-touch policy (2026-07-04): lastfm-only mapped terms publish at a
    # hard-capped confidence instead of blocking in the review queue. The cap
    # keeps artifact weight low (X_genre weight = confidence x layer weight);
    # the 'baroque on Debussy' incident came from this lane publishing at
    # >=0.90 full weight. Junk-tag rejection ("seen live") happens upstream at
    # classification time, and non-taxonomy terms are still dropped by the
    # materializer — this branch only lowers the stakes for real terms.
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[EvidenceTerm(term="shoegaze", source_type="lastfm_tags", confidence=0.95)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    [decision] = [d for d in report.provisional_genres if d.term == "shoegaze"]
    assert decision.basis == "lastfm_only"
    assert decision.confidence <= 0.40
    assert "capped confidence" in decision.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_evidence.py::test_lastfm_only_publishes_provisionally_at_capped_confidence`
Expected: FAIL — the decision is in `report.needs_review`, so the provisional list comprehension is empty (`ValueError: not enough values to unpack`).

- [ ] **Step 3: Implement**

In `src/ai_genre_enrichment/hybrid_evidence.py`, add below `SOURCE_WEIGHTS` (line 34):

```python
# Zero-touch policy (2026-07-04, M1 of the zero-touch genre pipeline spec):
# lastfm-only terms publish at this hard cap instead of queue-blocking. The
# artifact weights genres by confidence, so the cap IS the blast-radius
# control — 0.40 weight nudges steering, never defines it.
LASTFM_ONLY_CONFIDENCE_CAP = 0.40
```

Replace the lastfm-only branch (lines 221-234):

```python
        if all(source in LASTFM_SOURCE_TYPES for source in sources):
            # Published at capped confidence, never review-blocked. History:
            # pre-2026-06-12 this lane published at >=0.90 FULL weight and
            # produced 'baroque' on Debussy at scale; 2026-06-12 made it
            # review-only; 2026-07-04 (zero-touch M1) publishes it capped.
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=min(score, LASTFM_ONLY_CONFIDENCE_CAP),
                basis="lastfm_only",
                sources=sources,
                reason="Last.fm-only mapped signal published at capped confidence pending corroboration.",
            ))
            continue
```

- [ ] **Step 4: Run the module's tests**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_layered_hybrid_policy.py`
Expected: the new test PASSES. Other tests in these files may now fail where they assert the old lastfm-only routing — inspect each failure: if it asserts the old policy (term in `needs_review`), update the assertion to the new policy (term in `provisional_genres`, confidence ≤ 0.40, basis `lastfm_only`). Do NOT weaken tests that assert corroborated-lastfm or noise-rejection behavior — those policies are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/hybrid_evidence.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_layered_hybrid_policy.py
git commit -m "feat(enrich): lastfm-only terms publish at capped confidence (zero-touch M1)"
```

---

### Task 2: Fusion — below-bar terms publish; delete the `needs_review` bucket

**Files:**
- Modify: `src/ai_genre_enrichment/hybrid_evidence.py:9` (DecisionKind), `:69-79` (HybridGenreReport), `:200-353` (fuse internals)
- Modify: `src/ai_genre_enrichment/layered_assignment.py:145,151,251-252`
- Modify: `scripts/ai_genre_enrich.py:2071`
- Test: `tests/unit/test_ai_genre_hybrid_evidence.py`, plus any test asserting `needs_review`

**Interfaces:**
- Consumes: Task 1's lastfm-only branch.
- Produces: `HybridGenreReport` with exactly three lists: `accepted_genres`, `provisional_genres`, `rejected_noise` (field `needs_review` and Literal value `"needs_review"` deleted). `DecisionKind = Literal["accepted", "provisional", "rejected_noise"]`. Every mapped term now lands in one of the three. Tasks 5–6 rely on this shape.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_ai_genre_hybrid_evidence.py`:

```python
def test_below_bar_evidence_publishes_at_evidence_confidence():
    # Zero-touch policy (2026-07-04): "mapped but not strong enough" terms
    # publish at their honest fused confidence instead of queue-blocking.
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[EvidenceTerm(term="dub techno", source_type="discogs", confidence=0.55)],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    [decision] = report.provisional_genres
    assert decision.term == "dub techno"
    assert 0.0 < decision.confidence < 0.72
    assert "below the corroboration bar" in decision.reason
    assert not hasattr(report, "needs_review")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_evidence.py::test_below_bar_evidence_publishes_at_evidence_confidence`
Expected: FAIL — decision lands in `needs_review`; `hasattr(report, "needs_review")` is True.

- [ ] **Step 3: Implement in `hybrid_evidence.py`**

1. Line 9: `DecisionKind = Literal["accepted", "provisional", "rejected_noise"]`
2. `HybridGenreReport` (lines 69-79): delete the `needs_review: list[FusedGenreDecision]` field.
3. In `fuse_hybrid_evidence`: delete `review: list[FusedGenreDecision] = []` (line 203); replace the final fallthrough (lines 339-345) with:

```python
        provisional.append(FusedGenreDecision(
            term=term,
            confidence=score,
            basis=_basis(sources),
            sources=sources,
            reason="Evidence mapped but below the corroboration bar; published at evidence confidence.",
        ))
```

4. Update the return (lines 347-353): drop `needs_review=review`.

- [ ] **Step 4: Update the consumers**

`src/ai_genre_enrichment/layered_assignment.py`:
- Line 145: `review_term_count = len(report.needs_review)` → `review_term_count = 0` (taxonomy-classification review terms still increment it inside `compute_layered_assignment_rows`'s loop).
- Lines 146-154 (`context_terms`): remove `+ list(report.needs_review)` from the concatenation.
- Lines 251-252 (`build_layered_release_diagnostics`): delete the two lines `for decision in report.needs_review: _merge_decision_row(review_terms, decision, "hybrid_fusion", ...)`. The provisional loop directly above it (lines 249-250) now carries the moved classes into the queue with basis `hybrid_provisional` — published + audit-visible.

`scripts/ai_genre_enrich.py`:
- Line 2071: delete the dict entry `"needs_review": [decision.term for decision in fused_report.needs_review],`.

Then enumerate every remaining reference and update per the same rule (old-policy assertion → new-policy assertion; report-shape reference → three-bucket shape):

Run: `grep -rn "needs_review" src/ scripts/ tests/ --include="*.py"`
Expected survivors (different concepts — do NOT touch): `src/ai_genre_enrichment/routing.py:30` (route lane enum), `src/ai_genre_enrichment/storage.py` release-check statuses (`VALID_STATUSES`, CHECK constraints, run-log column), `scripts/ai_genre_enrich.py:1151-1332` (release-checks lane counters).

- [ ] **Step 5: Run the affected suites**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_layered_hybrid_policy.py tests/unit/test_layered_genre_assignments.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_review_queue_logic.py`
Expected: PASS after assertion updates. Then the fast suite: `python -m pytest -q -m "not slow"` (use a 600000ms timeout). Expected: green; quote the real counts.

- [ ] **Step 6: Commit**

```bash
git add -u src tests scripts
git commit -m "feat(enrich): delete the needs_review bucket - every mapped term publishes (zero-touch M1)"
```

---

### Task 3: Review-queue page stats split published-audit vs coverage counts

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py:2933-2949` (`_review_queue_page`)
- Test: `tests/unit/test_review_queue_storage.py`

**Interfaces:**
- Consumes: queue rows whose `basis` is refreshed by rescan (`sync_review_queue_for_release` already UPDATEs `basis` on pending rows — storage.py:2886).
- Produces: `_review_queue_page` return dict gains `pending_published_terms: int` (basis = `hybrid_provisional`, i.e. published-but-audit-tracked) and `pending_coverage_terms: int` (basis = `layered_taxonomy`, i.e. unpublished unknown/review taxonomy terms — M2's target). The GUI/worker pass the dict through unchanged (additive keys are backward-compatible).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_review_queue_storage.py` (follow the file's existing store-fixture pattern for constructing a `SidecarStore` on a tmp path; reuse its helper if one exists):

```python
def test_review_queue_page_splits_pending_by_basis(tmp_path):
    store = _make_store(tmp_path)  # reuse the module's existing store fixture/helper
    store.sync_review_queue_for_release(
        release_key="a::x", normalized_artist="a", normalized_album="x",
        terms=[
            {"term": "shoegaze", "confidence": 0.4, "basis": "hybrid_provisional",
             "sources": ["lastfm_tags"], "reason": "published capped"},
            {"term": "zeuhl", "confidence": 0.6, "basis": "layered_taxonomy",
             "sources": ["discogs"], "reason": "Unknown layered taxonomy term."},
        ],
    )
    page = store.get_review_queue_page()
    assert page["pending_terms"] == 2
    assert page["pending_published_terms"] == 1
    assert page["pending_coverage_terms"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_review_queue_storage.py::test_review_queue_page_splits_pending_by_basis`
Expected: FAIL with `KeyError: 'pending_published_terms'`.

- [ ] **Step 3: Implement**

In `_review_queue_page` (after the `counts` query at storage.py:2942-2945), add:

```python
            by_basis = {
                row["basis"]: row["n"]
                for row in conn.execute(
                    "SELECT basis, COUNT(*) AS n FROM ai_genre_review_queue "
                    "WHERE status = 'pending' GROUP BY basis"
                )
            }
```

and include in the returned dict (alongside the existing `pending_releases` / `pending_terms` keys):

```python
                "pending_published_terms": by_basis.get("hybrid_provisional", 0),
                "pending_coverage_terms": by_basis.get("layered_taxonomy", 0),
```

Also add both keys (value 0) to the two empty-page early returns at storage.py:2922 and :2927 so the readonly path has a consistent shape.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/unit/test_review_queue_storage.py tests/unit/test_worker_review_queue.py tests/integration/test_web_review_api.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_review_queue_storage.py
git commit -m "feat(review): split pending queue counts into published-audit vs coverage"
```

---

### Task 4: Store reader for raw assignment rows (round-trip safe)

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py` (add method next to `layered_release_summary`, ~line 2523)
- Test: `tests/unit/test_review_queue_storage.py` (or the storage test module the fixture lives in)

**Interfaces:**
- Consumes: `genre_graph_release_genre_assignments` / `genre_graph_release_facet_assignments` rows written by `replace_layered_assignments_for_release` (storage.py:2451).
- Produces: `SidecarStore.layered_assignment_rows_for_release(release_id: str) -> dict[str, list[dict]]` returning `{"genre_rows": [...], "facet_rows": [...]}` where each genre row has exactly the keys `replace_layered_assignments_for_release` consumes (`genre_id, assignment_layer, confidence, source_reliability, evidence_count, rejected_by_user, provenance`) and each facet row has (`facet_id, confidence, source, provenance`) — `provenance` decoded from JSON. Task 5 round-trips these through `replace_layered_assignments_for_release`.

- [ ] **Step 1: Write the failing test**

```python
def test_layered_assignment_rows_round_trip(tmp_path):
    store = _make_store(tmp_path)
    genre_rows = [{
        "genre_id": "g_slowcore", "assignment_layer": "observed_leaf",
        "confidence": 0.9, "source_reliability": 0.8, "evidence_count": 2,
        "rejected_by_user": False,
        "provenance": {"term": "slowcore", "sources": ["local_metadata"]},
    }]
    facet_rows = [{
        "facet_id": "f_instrumental", "confidence": 0.7, "source": "local_metadata",
        "provenance": {"term": "instrumental"},
    }]
    store.replace_layered_assignments_for_release(
        release_id="a::x", artist="a", album="x",
        genre_assignments=genre_rows, facet_assignments=facet_rows,
    )
    rows = store.layered_assignment_rows_for_release("a::x")
    assert rows["genre_rows"] == genre_rows
    assert rows["facet_rows"] == facet_rows
    # Round-trip: writing the read rows back must be a no-op shape-wise.
    store.replace_layered_assignments_for_release(
        release_id="a::x", artist="a", album="x",
        genre_assignments=rows["genre_rows"], facet_assignments=rows["facet_rows"],
    )
    assert store.layered_assignment_rows_for_release("a::x") == rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_review_queue_storage.py::test_layered_assignment_rows_round_trip`
Expected: FAIL with `AttributeError: ... 'layered_assignment_rows_for_release'`.

- [ ] **Step 3: Implement**

```python
    def layered_assignment_rows_for_release(self, release_id: str) -> dict[str, list[dict[str, Any]]]:
        """Raw assignment rows in exactly the shape replace_layered_assignments_for_release consumes.

        The additive publish-backfill (zero-touch M1) does read-merge-replace;
        this reader is its read half, so it must round-trip losslessly.
        """
        with self.connect() as conn:
            genre_rows = [
                {
                    "genre_id": row["genre_id"],
                    "assignment_layer": row["assignment_layer"],
                    "confidence": row["confidence"],
                    "source_reliability": row["source_reliability"],
                    "evidence_count": row["evidence_count"],
                    "rejected_by_user": bool(row["rejected_by_user"]),
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
                for row in conn.execute(
                    "SELECT genre_id, assignment_layer, confidence, source_reliability, "
                    "evidence_count, rejected_by_user, provenance_json "
                    "FROM genre_graph_release_genre_assignments WHERE release_id = ?",
                    (release_id,),
                )
            ]
            facet_rows = [
                {
                    "facet_id": row["facet_id"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
                for row in conn.execute(
                    "SELECT facet_id, confidence, source, provenance_json "
                    "FROM genre_graph_release_facet_assignments WHERE release_id = ?",
                    (release_id,),
                )
            ]
        return {"genre_rows": genre_rows, "facet_rows": facet_rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_review_queue_storage.py::test_layered_assignment_rows_round_trip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_review_queue_storage.py
git commit -m "feat(enrich): raw assignment-row reader for the additive backfill"
```

---

### Task 5: Additive backfill planner + applier

**Files:**
- Create: `src/ai_genre_enrichment/publish_backfill.py`
- Test: `tests/unit/test_publish_backfill.py`

**Interfaces:**
- Consumes: `fuse_release_evidence(store, release)` (hybrid_evidence.py:405), `compute_layered_assignment_rows(report, taxonomy)` (layered_assignment.py:133 — its docstring says it was extracted for exactly this dry-run use), `store.layered_assignment_rows_for_release` (Task 4), `store.replace_layered_assignments_for_release` (storage.py:2451).
- Produces:
  - `plan_release_backfill(store, *, taxonomy, release) -> ReleaseBackfillPlan` — pure planning; `ReleaseBackfillPlan` has `release_key: str`, `additions: list[dict]` (full genre-assignment rows for (genre_id, layer) keys not already present), `added_observed_terms: list[dict]` (report rows: `term, genre_id, confidence, basis, sources, reason` for observed_leaf additions only — the eval sample reads these).
  - `apply_release_backfill(store, *, release, plan) -> int` — read-merge-replace; returns rows added. Task 6's CLI drives both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_publish_backfill.py`. Build the fused report through the real fusion (not hand-built reports) with a minimal fake store exposing the four collector methods `hybrid_source_terms_for_release`, `accepted_enriched_genres_for_release`, `latest_check_suggestions_for_release`, `latest_model_prior_terms_for_release` (return `[]` except the first) — mirror the fake-store pattern used in `tests/unit/test_layered_genre_assignments.py`. Use the real `SidecarStore` on tmp_path for the apply-side test (same `_make_store` fixture as Task 4). Taxonomy: reuse the small YAML fixture pattern from `tests/unit/test_layered_genre_assignments.py` with one canonical leaf genre (e.g. `slowcore` under a family) so classification succeeds.

```python
def test_plan_is_additive_only_and_targets_new_terms(...):
    # Store state: release already has observed_leaf 'slowcore' @0.9.
    # Evidence: 'slowcore' (lastfm) + 'dub techno' (lastfm, canonical in fixture taxonomy).
    # Plan must: add 'dub techno' observed_leaf @<=0.40 (+ its inferred parents),
    # NOT touch 'slowcore' (already present at higher confidence),
    # and additions must contain no (genre_id, layer) key already stored.

def test_apply_merges_without_removing_or_lowering(...):
    # After apply: original rows byte-identical (confidence unchanged),
    # new rows present, facet rows untouched, second apply is a no-op (idempotent).

def test_plan_skips_release_with_no_new_terms(...):
    # Evidence identical to stored state -> plan.additions == [].
```

Write all three as real tests with the fixtures above (complete arrange/act/assert — the two store patterns and the taxonomy fixture give every needed building block).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_publish_backfill.py`
Expected: FAIL with `ModuleNotFoundError: src.ai_genre_enrichment.publish_backfill`.

- [ ] **Step 3: Implement `src/ai_genre_enrichment/publish_backfill.py`**

```python
"""Additive backfill for the 2026-07-04 always-publish policy flip (zero-touch M1).

Re-runs the current fusion policy per release and merges ONLY new assignment
rows. Never removes a row, never lowers a confidence, never touches
rejected_by_user or user overrides — the 2026-06-12 lesson (wholesale
re-derivation un-decides good past calls; see assignment_migration.py) applied
to a policy that only got MORE permissive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hybrid_evidence import fuse_release_evidence
from .layered_assignment import compute_layered_assignment_rows


@dataclass(frozen=True)
class ReleaseBackfillPlan:
    release_key: str
    additions: list[dict[str, Any]] = field(default_factory=list)
    added_observed_terms: list[dict[str, Any]] = field(default_factory=list)


def plan_release_backfill(store: Any, *, taxonomy: Any, release: Any) -> ReleaseBackfillPlan:
    report = fuse_release_evidence(store, release)
    computed = compute_layered_assignment_rows(report, taxonomy)
    existing = store.layered_assignment_rows_for_release(release.release_key)
    existing_keys = {
        (row["genre_id"], row["assignment_layer"]) for row in existing["genre_rows"]
    }
    additions = [
        row
        for row in computed["genre_rows"]
        if (row["genre_id"], row["assignment_layer"]) not in existing_keys
    ]
    added_observed_terms = [
        {
            "term": row["provenance"].get("term", ""),
            "genre_id": row["genre_id"],
            "confidence": row["confidence"],
            "basis": row["provenance"].get("basis", ""),
            "sources": row["provenance"].get("sources", []),
            "reason": row["provenance"].get("reason", ""),
        }
        for row in additions
        if row["assignment_layer"] == "observed_leaf"
    ]
    return ReleaseBackfillPlan(
        release_key=release.release_key,
        additions=additions,
        added_observed_terms=added_observed_terms,
    )


def apply_release_backfill(store: Any, *, release: Any, plan: ReleaseBackfillPlan) -> int:
    if not plan.additions:
        return 0
    existing = store.layered_assignment_rows_for_release(release.release_key)
    store.replace_layered_assignments_for_release(
        release_id=release.release_key,
        artist=release.normalized_artist,
        album=release.normalized_album,
        genre_assignments=existing["genre_rows"] + plan.additions,
        facet_assignments=existing["facet_rows"],
    )
    return len(plan.additions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_publish_backfill.py`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/publish_backfill.py tests/unit/test_publish_backfill.py
git commit -m "feat(enrich): additive publish-backfill planner/applier (zero-touch M1)"
```

---

### Task 6: CLI command `backfill-publish` (dry-run default, JSON report)

**Files:**
- Modify: `scripts/ai_genre_enrich.py` (new `cmd_backfill_publish` + subparser registration — follow the file's existing `cmd_*` + subparser pattern; discovery of releases via the same helper the enrich stage uses to build `release` objects with `existing_genres_by_source`, see `scripts/analyze_library.py:1721-1746` and `src/ai_genre_enrichment/discovery.py`)
- Test: `tests/unit/test_ai_genre_hybrid_cli.py` (follow its existing CLI-invocation pattern)

**Interfaces:**
- Consumes: Task 5's `plan_release_backfill` / `apply_release_backfill`; `scan_review_queue` (review_queue.py:48); `load_default_layered_taxonomy`.
- Produces: `python scripts/ai_genre_enrich.py backfill-publish [--apply] [--report-path PATH]`.
  - Dry-run (default): plans every release, writes JSON report `{generated_at, apply: false, releases_affected, additions_total, additions_by_basis, additions_by_confidence_band, releases: [{release_key, added_observed_terms}]}` to `--report-path` (default `docs/run_audits/backfill_always_publish_<UTC-timestamp>.json`), prints the summary counts, mutates nothing.
  - `--apply`: applies every plan, then runs `scan_review_queue` over the store (refreshes queue basis → `hybrid_provisional`), writes the same report with `apply: true` and per-release applied counts.

- [ ] **Step 1: Write the failing CLI test** — dry-run writes a report and does not mutate assignments; `--apply` adds rows and rescans (assert a queue row's basis flips). Use the existing CLI test file's store/taxonomy fixtures and runner pattern; assert on the JSON report structure and on `layered_assignment_rows_for_release` before/after.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_cli.py -k backfill`
Expected: FAIL (unknown command `backfill-publish`).

- [ ] **Step 3: Implement** `cmd_backfill_publish` following the file's established command structure: iterate releases (same discovery as the enrich stage), `plan_release_backfill` each, aggregate stats (`additions_by_basis` from `added_observed_terms[].basis`; confidence bands `<0.45 / 0.45-0.7 / >=0.7`), write the JSON report, and under `--apply` call `apply_release_backfill` per release followed by one `scan_review_queue(store, taxonomy=taxonomy)` pass. Progress-log with the file's `ProgressLogger` pattern.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/unit/test_ai_genre_hybrid_cli.py` then the fast suite `python -m pytest -q -m "not slow"` (600000ms timeout).
Expected: green; quote real counts.

- [ ] **Step 5: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_ai_genre_hybrid_cli.py
git commit -m "feat(enrich): backfill-publish CLI - dry-run report + additive apply + queue rescan"
```

---

### Task 7: OPERATIONAL — dry-run on the real library + eval gate ⛔ STOP for Dylan

Run from the **main checkout** (merge the branch first, or run the branch's code from its worktree pointed at absolute main-checkout data paths — never a symlinked DB).

- [ ] **Step 1:** `python scripts/ai_genre_enrich.py backfill-publish` (dry-run). Capture the report path.
- [ ] **Step 2:** Invoke the `evaluation-methodology` skill, then build the eval package from the report:
  - Headline stats: releases affected, additions by basis (`lastfm_only` vs below-bar), confidence-band histogram.
  - Stratified blind sample: 40 releases with additions — 20 `lastfm_only` (of which ≥5 on short/generic artist names — the Green-House risk class), 20 below-bar. Render as a table: artist / album / current published genres / proposed additions (term + confidence), order randomized, no basis labels visible.
- [ ] **Step 3: STOP.** Present the package to Dylan. Gate: he judges the sampled additions ≥85% "correct or harmless at their stated confidence". Pass → Task 8. Fail → tune `LASTFM_ONLY_CONFIDENCE_CAP` / add a targeted exclusion, re-run Task 7. Record the decision + numbers in the report file's directory.

### Task 8: OPERATIONAL — apply, publish, rescan ⛔ requires Dylan's explicit publish go

- [ ] **Step 1:** `python scripts/ai_genre_enrich.py backfill-publish --apply` (sidecar-only writes + queue rescan).
- [ ] **Step 2:** Publish dry-run: run the publish stage with `dry_run=True` (genre_publish.py:590) via its existing CLI/stage entry; capture stats (rows added by layer, confidence distribution). Sanity: additions ≈ the report's totals; **zero removals**.
- [ ] **Step 3: Ask Dylan to confirm the real publish** (writes `data/metadata.db`; timestamped backup is automatic on the publish path). Then run it and record `PublishStats`.
- [ ] **Step 4:** Verify queue composition: `pending_coverage_terms` ≈ the old "unknown term" population (~12.4k, M2's target); `pending_published_terms` covers the moved classes; total pending no longer contains the two retired policy reasons.

### Task 9: OPERATIONAL — artifact rebuild + geometry & generation sanity

- [ ] **Step 1 (before rebuild):** Create `scripts/research/genre_density_probe.py` — loads the live npz, samples 20k random track pairs (seeded RNG), prints p50/p90/p99 cosine of `X_genre_raw` and `X_genre_smoothed` plus mean nonzero-dims/track. (~25 lines, numpy only; reuse the artifact-path resolution from a neighboring research script.) Run it; save baseline JSON next to the Task 7 report.
- [ ] **Step 2:** Rebuild artifacts via the standard stage (`scripts/build_beat3tower_artifacts.py` entry used by the analyze `artifacts` stage). Confirm `X_sonic_variant` still `muq` after rebuild (hard rule from the MERT-migration incident).
- [ ] **Step 3:** Re-run the probe. Gate: p50 shift ≤ +0.03 and p90 stays well clear of the documented failure zone (2026-06-16 bad state: p50 ≈ 0.42 / p90 ≈ 0.645). Record both runs.
- [ ] **Step 4:** Invoke the `playlist-testing` skill; run one real multi-pier artist-mode generation through the `gui_fidelity` harness with `genre_mode` engaged; read the per-playlist DEBUG log (`logs/playlists/`): genre gate tallies + per-segment pool sizes comparable to a pre-flip reference run; no new starvation.
- [ ] **Step 5:** Commit the probe script + run-audit artifacts:

```bash
git add scripts/research/genre_density_probe.py docs/run_audits/
git commit -m "chore(m1): backfill eval + publish + artifact sanity records"
```

### Task 10: Docs + memory

- [ ] **Step 1:** `docs/AI_GENRE_ENRICHMENT.md`: rewrite the fusion-policy section — three buckets, lastfm cap, always-publish semantics, queue = audit trail (`hybrid_provisional` = published/curatable, `layered_taxonomy` = coverage gap awaiting M2). Cross-link the M1 spec.
- [ ] **Step 2:** `.claude/skills/genre-data-authority/SKILL.md`: update the review-queue row in the layer map (audit trail, non-blocking) per its maintenance protocol.
- [ ] **Step 3:** Update memory `project_zero_touch_genre_pipeline.md` (M1 status + measured queue/eval numbers) via the executing session.
- [ ] **Step 4:** Commit docs; full fast suite once more; then follow `superpowers:finishing-a-development-branch`.

---

## Self-review notes (already applied)

- Spec coverage: every M1 spec-table row maps to a task — lastfm cap (T1), below-bar (T2), unknown-term drop+log (unchanged materializer behavior, verified by existing `test_layered_genre_assignments.py:136` review-term test), broad-term blocklist (untouched — graph-derived), file-tags-win (structurally guaranteed: never-drop branch untouched + backfill is additive-only), audit-trail queues (T2 diagnostics + T3 stats), eval gates (T7-T9).
- The backfill deliberately reuses `compute_layered_assignment_rows` (extracted for this exact purpose per its docstring) and the read-merge-replace shape of `apply_surgical_delta` rather than new machinery.
- Type consistency: `ReleaseBackfillPlan.additions` rows are exactly `replace_layered_assignments_for_release` genre rows; Task 4's reader guarantees the round-trip.
