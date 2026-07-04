# Edge-repair identity-key memoization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `edge_repair` wall-clock (~26% of generation) with bit-identical output by memoizing the per-track-index identity lookups it currently recomputes hundreds of thousands of times.

**Architecture:** Add a per-repair-pass memo object (`_IdentityMemo`) that caches `identity_keys_for_index(bundle, idx)` and `_cap_artist_keys_for_idx(bundle, idx, cfg)` — both pure functions of the track index within one `repair_playlist_edges` call — and thread it through the three helpers that drive `_candidate_refusal_reasons`. Same memoization pattern already accepted as T1-a; bit-identical by construction (string/set ops, no floats). All changes live in one file.

**Tech Stack:** Python 3.13, pytest (markers: `integration`/`slow`), cProfile via `scripts/research/time_golden_replay.py`.

**Spec:** `docs/superpowers/specs/2026-07-04-edge-repair-identity-memoization-design.md`
**Parent effort:** `docs/superpowers/plans/2026-07-03-lossless-generation-speedup.md` (Task 4 = T1-a, the accepted precedent).

## Global Constraints

- **Bit-identical output is the hard gate.** After every change, the ordered `track_ids` from the golden fixture must be **exactly equal** to the frozen golden and `min_transition`/`mean_transition` must match to full float precision (ΔT == 0). Any diff → revert the change before committing.
- **No lossy levers.** Do not change any acceptance decision, floor, margin, or ordering in `edge_repair`. Only remove redundant recomputation.
- **Golden gate needs absolute-path overrides** (the checkout's `data/` is not always usable):
  ```bash
  export PLAYLIST_GOLDEN_ARTIFACT="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"
  export PLAYLIST_GOLDEN_DB="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"
  ```
- **Measure with the profiler, never a single wall-clock run** (wall-clock is load-noisy here; cProfile cumulative is load-independent).
- **Shared master checkout** (a second session may be live): commit with `git commit --only <path>` and inspect the full staged set with `git diff --cached --name-only` **before** committing. Never a bare `git commit`, never `git add -A`/`-u`.
- **Confirm line anchors at edit time.** All `file:line` references are from static analysis and may drift; read the region before editing.

---

## Universal verification block (referenced by every task)

```bash
# 1. Bit-diff gate — MUST pass (porches replays identical, ΔT == 0)
export PLAYLIST_GOLDEN_ARTIFACT="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"
export PLAYLIST_GOLDEN_DB="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"
python -m pytest tests/integration/test_lossless_speedup_golden.py -v

# 2. Fast suite stays green
python -m pytest -q -m "not slow"

# 3. Timing / profile (record before/after in the commit message)
python scripts/research/time_golden_replay.py --fixture porches --profile
```

If step 1 shows any `track_ids` diff or ΔT != 0, the change is **not lossless** — revert it before committing.

---

## Task 1: Add the `_IdentityMemo` primitive + micro-test

The cache primitive, proven correct in isolation before it is wired into the hot path. It is inert until Task 2 (which activates it in the same session — do not stop between Task 1 and Task 2).

**Files:**
- Modify: `src/playlist/repair/edge_repair.py` (import line ~4; new class after `_cap_artist_keys_for_idx`, currently ending ~line 142)
- Create: `tests/test_edge_repair_identity_cache.py`

**Interfaces:**
- Produces: `class _IdentityMemo` with `keys_for_index(bundle, idx) -> TrackIdentityKeys` and `cap_keys(bundle, idx, artist_identity_cfg) -> set[str]`, memoized by `int(idx)`. Consumed by Task 2.
- Consumes: existing `identity_keys_for_index` (imported) and module-level `_cap_artist_keys_for_idx`.

- [ ] **Step 1: Write the failing micro-test**

Create `tests/test_edge_repair_identity_cache.py`:

```python
"""Unit test for the edge_repair per-pass identity memo: cached lookups must
equal the uncached functions and must actually memoize (return the same object)."""
import types

from src.playlist.repair.edge_repair import _IdentityMemo, _cap_artist_keys_for_idx
from src.playlist.identity_keys import identity_keys_for_index


def _fake_bundle():
    # identity_keys_for_index / _cap_artist_keys_for_idx read only these four
    # attributes (each under try/except), so a duck-typed namespace suffices.
    return types.SimpleNamespace(
        track_ids=["t0", "t1", "t2"],
        track_artists=["Miles Davis Quintet", "", "Bill Evans Trio"],
        artist_keys=["miles", "solo", "bill"],
        track_titles=["So What", "Untitled", "Waltz"],
    )


def test_memo_keys_for_index_equal_uncached_and_memoized():
    b = _fake_bundle()
    memo = _IdentityMemo()
    for i in range(3):
        assert memo.keys_for_index(b, i) == identity_keys_for_index(b, i)
    first = memo.keys_for_index(b, 0)
    assert memo.keys_for_index(b, 0) is first  # cached: identical object on re-lookup


def test_memo_cap_keys_equal_uncached_and_memoized():
    b = _fake_bundle()
    memo = _IdentityMemo()
    for i in range(3):
        assert memo.cap_keys(b, i, None) == _cap_artist_keys_for_idx(b, i, None)
    first = memo.cap_keys(b, 0, None)
    assert memo.cap_keys(b, 0, None) is first  # cached: identical object (incl. empty set)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_edge_repair_identity_cache.py -v`
Expected: FAIL — `ImportError: cannot import name '_IdentityMemo'`.

- [ ] **Step 3: Add the `field` import**

In `src/playlist/repair/edge_repair.py`, change the dataclass import (currently `from dataclasses import dataclass`) to:

```python
from dataclasses import dataclass, field
```

Also extend the identity import (currently `from src.playlist.identity_keys import identity_keys_for_index`) to bring in the return type:

```python
from src.playlist.identity_keys import TrackIdentityKeys, identity_keys_for_index
```

- [ ] **Step 4: Implement `_IdentityMemo`**

Insert immediately **after** `_cap_artist_keys_for_idx` (after its `return` around line 142, before `_candidate_refusal_reasons`):

```python
@dataclass
class _IdentityMemo:
    """Per-repair-pass memo for the pure-per-index identity lookups.

    Within one ``repair_playlist_edges`` call the ``bundle`` is immutable and
    ``artist_identity_cfg`` is fixed, so ``identity_keys_for_index(bundle, idx)``
    and ``_cap_artist_keys_for_idx(bundle, idx, cfg)`` are pure functions of the
    integer track index. This memoizes them by index; lifetime is one repair pass
    (a fresh instance per call), mirroring beam.py's per-segment genre_cache /
    trans_cache. Keyed by index only (never by playlist position), so an accepted
    swap never invalidates an entry — the index->keys mapping is immutable.
    """

    _ident: dict[int, TrackIdentityKeys] = field(default_factory=dict)
    _cap: dict[int, set[str]] = field(default_factory=dict)

    def keys_for_index(self, bundle: ArtifactBundle, idx: int) -> TrackIdentityKeys:
        i = int(idx)
        cached = self._ident.get(i)
        if cached is None:  # identity_keys_for_index never returns None → safe sentinel
            cached = identity_keys_for_index(bundle, i)
            self._ident[i] = cached
        return cached

    def cap_keys(
        self,
        bundle: ArtifactBundle,
        idx: int,
        artist_identity_cfg: Optional[ArtistIdentityConfig],
    ) -> set[str]:
        i = int(idx)
        if i not in self._cap:  # membership, not truthiness: an empty set is a valid value
            self._cap[i] = _cap_artist_keys_for_idx(bundle, i, artist_identity_cfg)
        return self._cap[i]
```

- [ ] **Step 5: Run the micro-test to confirm it passes**

Run: `python -m pytest tests/test_edge_repair_identity_cache.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git diff --cached --name-only   # confirm nothing else is staged by another session
git add src/playlist/repair/edge_repair.py tests/test_edge_repair_identity_cache.py
git commit --only src/playlist/repair/edge_repair.py tests/test_edge_repair_identity_cache.py -m "perf(edge-repair): add per-pass identity memo primitive + micro-test

Inert until wired (next commit). Pure-per-index memo mirroring T1-a.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat --oneline HEAD | head   # verify ONLY the two intended files landed
```

---

## Task 2: Wire the memo through `edge_repair` (the core win)

Thread one `_IdentityMemo` instance through the three helpers so every per-index identity lookup in the candidate loop hits the cache. This is the activated, bit-identical, measurable win.

**Files:**
- Modify: `src/playlist/repair/edge_repair.py`
  - `_non_seed_artist_counts_after_replacement` (~line 98)
  - `_candidate_refusal_reasons` (~line 145; internal calls at ~174, ~183, ~196, ~209, ~216)
  - `repair_playlist_edges` (candidate loop call site ~329)

**Interfaces:**
- Consumes: `_IdentityMemo` (Task 1).
- Produces: no signature change visible outside the module; `repair_playlist_edges` public signature is unchanged.

- [ ] **Step 1: Confirm the anchors**

Read `src/playlist/repair/edge_repair.py` lines 98-230 and 263-344. Confirm: `_candidate_refusal_reasons` calls `identity_keys_for_index` at the `cand_keys = ...` line and inside the `existing_track_keys` loop; calls `_non_seed_artist_counts_after_replacement` for the artist cap; calls `_cap_artist_keys_for_idx` for the candidate and each neighbor in the `min_gap` loop. Confirm `indices` is mutated only **after** the candidate loop (at `indices[replace_pos] = int(new_idx)`), so per-index lookups are stable within a candidate loop.

- [ ] **Step 2: Thread the memo into `_non_seed_artist_counts_after_replacement`**

Add a trailing `memo` parameter and use it when present. Replace the whole function body's `_cap_artist_keys_for_idx(...)` call:

```python
def _non_seed_artist_counts_after_replacement(
    candidate: int,
    current_indices: Sequence[int],
    replace_position: int,
    bundle: ArtifactBundle,
    seed_indices: set[int],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    memo: Optional["_IdentityMemo"] = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pos, idx in enumerate(current_indices):
        track_idx = int(candidate) if int(pos) == int(replace_position) else int(idx)
        if track_idx in seed_indices:
            continue
        keys = (
            memo.cap_keys(bundle, track_idx, artist_identity_cfg)
            if memo is not None
            else _cap_artist_keys_for_idx(bundle, track_idx, artist_identity_cfg)
        )
        for artist_key in keys:
            counts[str(artist_key)] = counts.get(str(artist_key), 0) + 1
    return counts
```

(The signature's `Sequence`/`set`/`Optional` names are already imported in the file; `_IdentityMemo` is referenced as a string annotation to avoid forward-reference issues.)

- [ ] **Step 3: Thread the memo into `_candidate_refusal_reasons`**

Add a trailing `memo` parameter and define two local accessors at the top of the function body (right after `reasons: list[str] = []`), then replace the four identity call sites:

```python
def _candidate_refusal_reasons(
    *,
    candidate: int,
    current_indices: Sequence[int],
    replace_position: int,
    bundle: ArtifactBundle,
    seed_indices: set[int],
    pier_indices: set[int],
    allowed_indices: Optional[set[int]],
    disallowed_artist_keys: set[str],
    metric_context: TransitionMetricContext,
    variety_guard_enabled: bool,
    variety_guard_threshold: float,
    max_non_seed_tracks_per_artist: Optional[int],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    min_gap: int = 0,
    memo: Optional["_IdentityMemo"] = None,
) -> list[str]:
    reasons: list[str] = []
    # Per-index identity lookups routed through the pass memo when present
    # (bit-identical: memo returns the same values as the bare calls).
    if memo is not None:
        _keys_for = lambda i: memo.keys_for_index(bundle, int(i))
        _cap_for = lambda i: memo.cap_keys(bundle, int(i), artist_identity_cfg)
    else:
        _keys_for = lambda i: identity_keys_for_index(bundle, int(i))
        _cap_for = lambda i: _cap_artist_keys_for_idx(bundle, int(i), artist_identity_cfg)
    candidate = int(candidate)
```

Then within the body:
- Replace `cand_keys = identity_keys_for_index(bundle, candidate)` with `cand_keys = _keys_for(candidate)`.
- Replace `existing_track_keys.add(identity_keys_for_index(bundle, int(idx)).track_key)` with `existing_track_keys.add(_keys_for(idx).track_key)`.
- In the `_non_seed_artist_counts_after_replacement(...)` call, add `memo=memo,` as the final argument.
- Replace `cand_artist_keys = _cap_artist_keys_for_idx(bundle, candidate, artist_identity_cfg)` with `cand_artist_keys = _cap_for(candidate)`.
- Replace `other_keys = _cap_artist_keys_for_idx(bundle, int(current_indices[pos]), artist_identity_cfg)` with `other_keys = _cap_for(current_indices[pos])`.

Leave the `detect_title_artifacts(_title_for_idx(...))` and `_edge(...)` calls unchanged — they are not identity lookups.

- [ ] **Step 4: Instantiate and pass the memo in `repair_playlist_edges`**

After `candidates = [int(c) for c in candidate_indices]` (~line 273), add:

```python
    identity_memo = _IdentityMemo()
```

In the `_candidate_refusal_reasons(...)` call inside the candidate loop (~line 329), add as the final keyword argument:

```python
                memo=identity_memo,
```

- [ ] **Step 5: Bit-diff gate (decisive) + fast suite + timing**

Run the Universal verification block.
Expected:
- `test_lossless_speedup_golden.py` → **PASS** (porches `track_ids` identical, ΔT == 0).
- `pytest -q -m "not slow"` → green (and `tests/test_edge_repair_identity_cache.py` still passes).
- Profiler → `repair_playlist_edges` cumulative sharply down (target: 24.9s → single digits); `identity_keys_for_index` / `_cap_artist_keys_for_idx` call counts collapse from ~470k / ~99k to a few thousand. Record the before (24.9s / 96.66s total) and after numbers.

If ΔT != 0: a memo value diverges from the bare call — STOP, diagnose (a missed `memo=` thread or a non-index-pure input), do not commit.

- [ ] **Step 6: Commit**

```bash
git diff --cached --name-only
git add src/playlist/repair/edge_repair.py
git commit --only src/playlist/repair/edge_repair.py -m "perf(edge-repair): memoize per-index identity lookups within a repair pass (bit-identical)

Golden: porches identical track_ids, ΔT==0. edge_repair <before>s -> <after>s
(identity_keys_for_index calls 470k -> <n>). Same T1-a memoization pattern.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat --oneline HEAD | head
```

---

## Task 3 (conditional): Hoist `existing_track_keys` to once-per-edge-position

**Precondition — skip this task entirely unless** Task 2's re-profile leaves `repair_playlist_edges` above ~8s. With the memo in place this only removes the per-candidate rebuild of a ~50-element set; land it only if the measurement says it's worth the churn (two-gate rule).

`existing_track_keys` (built inside `_candidate_refusal_reasons` for the `duplicate_track_key` check) is identical for every candidate at a given `replace_position` — it depends only on `current_indices` (stable across the candidate loop) minus `replace_position`. Compute it once per edge-position in `repair_playlist_edges` and pass it in.

**Files:**
- Modify: `src/playlist/repair/edge_repair.py` (`repair_playlist_edges` edge loop ~324-344; `_candidate_refusal_reasons` ~177-187)

**Interfaces:**
- Produces: `_candidate_refusal_reasons` gains `existing_track_keys: Optional[set] = None`; when provided, the internal rebuild is skipped.

- [ ] **Step 1: Precompute per edge-position in `repair_playlist_edges`**

After `old_worst = _worst_t(old_edges)` and before `for cand in candidates:` (~line 326), add:

```python
        existing_track_keys_for_pos = {
            identity_memo.keys_for_index(bundle, int(idx)).track_key
            for pos, idx in enumerate(indices)
            if int(pos) != int(replace_pos)
        }
```

Pass it into the `_candidate_refusal_reasons(...)` call:

```python
                existing_track_keys=existing_track_keys_for_pos,
```

- [ ] **Step 2: Accept and use it in `_candidate_refusal_reasons`**

Add the parameter `existing_track_keys: Optional[set] = None` to the signature. Replace the `existing_track_keys` construction block (the `for pos, idx in enumerate(current_indices): ... existing_track_keys.add(_keys_for(idx).track_key)` loop) with:

```python
        if existing_track_keys is None:
            existing_track_keys = set()
            for pos, idx in enumerate(current_indices):
                if int(pos) == int(replace_position):
                    continue
                try:
                    existing_track_keys.add(_keys_for(idx).track_key)
                except Exception:
                    continue
        if cand_keys.track_key in existing_track_keys:
            reasons.append("duplicate_track_key")
```

(Keep the `if cand_keys.artist_key and ... in disallowed_artist_keys` check immediately after, unchanged.)

- [ ] **Step 3: Bit-diff gate + fast suite + timing**

Run the Universal verification block. Expected: porches identical, ΔT == 0; `repair_playlist_edges` cumulative down further. If ΔT != 0, revert (the precomputed set must equal the in-loop one — check the `replace_pos` exclusion matches).

- [ ] **Step 4: Commit**

```bash
git diff --cached --name-only
git add src/playlist/repair/edge_repair.py
git commit --only src/playlist/repair/edge_repair.py -m "perf(edge-repair): hoist existing_track_keys to once-per-edge-position (bit-identical)

Golden: porches identical, ΔT==0. edge_repair <before>s -> <after>s.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat --oneline HEAD | head
```

---

## Task 4: Re-profile, record results, update backlog

**Files:**
- Create: `docs/run_audits/lossless_speedup_after_edge_repair_profile.txt`
- Modify: `docs/TIME_OPTIMIZATION.md`

- [ ] **Step 1: Capture the after-profile**

```bash
export PLAYLIST_GOLDEN_ARTIFACT="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"
export PLAYLIST_GOLDEN_DB="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"
python scripts/research/time_golden_replay.py --fixture porches --profile > docs/run_audits/lossless_speedup_after_edge_repair_profile.txt
```

- [ ] **Step 2: Update `docs/TIME_OPTIMIZATION.md`**

Add an "Edge-repair memoization (2026-07-04)" entry under the lossless-wins section: porches before/after total (96.66s → ?), edge_repair before/after (24.9s → ?), which tasks shipped (Task 2 core; Task 3 only if executed), and note that T1-g was **de-prioritized** as a ~1% / high-risk item per the fresh profile (with the flex-cost-is-the-beam finding). One line each — copy the real numbers from Step 1.

- [ ] **Step 3: Full slow suite + commit**

```bash
python -m pytest -q   # include slow so the golden gate runs; expect green
git diff --cached --name-only
git add docs/TIME_OPTIMIZATION.md docs/run_audits/lossless_speedup_after_edge_repair_profile.txt
git commit --only docs/TIME_OPTIMIZATION.md docs/run_audits/lossless_speedup_after_edge_repair_profile.txt -m "docs(perf): record edge_repair memoization result + de-prioritize T1-g

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** core memo → Tasks 1-2; optional `existing_track_keys` hoist → Task 3 (conditional, gated on the re-profile as the spec requires); results/backlog → Task 4. The spec's second stretch idea (incremental artist-cap counts) is intentionally **not** planned — it is the most complex and least-certainly-needed item; add it only if Task 4's profile still shows `_non_seed_artist_counts_after_replacement` as a hotspot, as its own spec+plan.
- **The bit-diff gate is the test.** Perf tasks run the golden harness (parent Task 1) rather than writing a new failing test; the one new unit test (Task 1) is the independent cache-correctness check, exactly as T1-a did.
- **Fail-closed on ΔT != 0.** Every task reverts rather than shipping a bit-changing variant.
- **Only-porches coverage** is a known limit (herbie/multiseed fixtures still TODO — they need the GUI path). Porches exercises both the flex path and a heavy edge_repair pass, so it gates this work; add fixtures if a new path surfaces.
- **Line anchors are approximate** — each task starts by reading the region.
