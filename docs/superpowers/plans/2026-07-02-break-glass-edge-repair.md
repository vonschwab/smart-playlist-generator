# Break-Glass Edge Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing post-order edge-repair pass actually fire on weak edges (`T < edge_repair_t_floor`, default 0.30) with break-glass semantics: best-effort swap on pure transition quality, never worse, leave alone if nothing better exists.

**Architecture:** All logic changes live in `src/playlist/repair/edge_repair.py` (new trigger predicate, worst-first ordering, positional min_gap refusal, logging). A new `PierBridgeConfig.edge_repair_t_floor` knob threads through `pier_bridge_overrides.py` and is passed (with `min_gap`) at the single call site in `pier_bridge_builder.py`. Candidate scoring stays `min(T_in, T_out)` — by construction it ignores anti_center/roam/progress/genre modifiers (the spec's "modifiers get no vote" rule).

**Tech Stack:** Python 3.11+, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-break-glass-edge-repair-design.md`

## Global Constraints

- **Shared checkout:** you are on `master` in a checkout a CONCURRENT session also edits. Never branch/checkout; never `git add -A`/`-u`; stage only this task's explicit paths. **Before editing any file, run `git status --short -- <file>`; if it is ALREADY modified (uncommitted foreign work), STOP and report BLOCKED** — committing the file would sweep another session's work into your commit.
- `tests/unit/test_edge_repair.py` is another session's in-flight file: **read it for fixture patterns, never modify it.** All new tests go in `tests/unit/test_edge_repair_break_glass.py`.
- **`is_broken_transition` (transition_metrics.py) is untouched** — its no-T-gate semantics are a deliberate roam-design decision shared by other callers.
- Break-glass semantics: accept a swap iff `new_worst_T >= old_worst_T + margin`; a lift that still lands below `t_floor` IS accepted (0.003→0.25 is a win); if nothing clears the margin, leave the edge alone. Never raise, never block generation.
- `t_floor=0.0` and `min_gap=0` function defaults = byte-identical to today for any other caller (`replacement.py` worker flow uses `repair_edge_position`).
- Run pytest bounded and NEVER piped through head/tail/Select-Object (hook blocks it): `python -m pytest <paths> -q`.
- mypy must stay clean on every edited `src/` file: `python -m mypy <file>`.

---

## Task 1: Break-glass core in `edge_repair.py`

**Files:**
- Modify: `src/playlist/repair/edge_repair.py`
- Create: `tests/unit/test_edge_repair_break_glass.py`

**Interfaces:**
- Consumes: existing `repair_playlist_edges(...)`, `_candidate_refusal_reasons(...)`, `_cap_artist_keys_for_idx(...)`, `is_broken_transition` (unchanged import).
- Produces (Task 2 relies on these exact names): `repair_playlist_edges(..., t_floor: float = 0.0, min_gap: int = 0)` — two new keyword-only-style params appended to the existing signature; `_candidate_refusal_reasons(..., min_gap: int = 0)`; refusal reason string `"min_gap"`.

- [ ] **Step 1: Write the failing tests.** Create `tests/unit/test_edge_repair_break_glass.py`. Reuse the `_repair_bundle` / `_context` fixture pattern from `tests/unit/test_edge_repair.py` (READ-ONLY reference — copy the helpers into the new file; with `center_transitions=False`, edge T = plain cosine blend, so vectors craft T directly):

```python
"""Break-glass edge repair (spec 2026-07-01): T < t_floor triggers repair;
best-effort/never-worse acceptance; positional min_gap refusal; worst-first order.
The old trigger (centered_cos < -0.5) never fires on mildly-bad edges — these
tests use orthogonal (cos 0) vectors that only the new t_floor catches."""
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.repair.edge_repair import repair_playlist_edges
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge

C25 = [0.25, 0.9682458365518543]   # cos 0.25 with [1,0]
C90 = [0.90, 0.4358898943540674]   # cos 0.90 with [1,0]


def _bundle(X: list[list[float]], artists: list[str] | None = None) -> ArtifactBundle:
    X_arr = np.array(X, dtype=float)
    n = int(X_arr.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array(artists or [f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array(artists or [f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_arr,
        X_sonic_start=X_arr,
        X_sonic_mid=X_arr,
        X_sonic_end=X_arr,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _ctx(bundle: ArtifactBundle):
    return build_transition_metric_context(
        X_sonic=bundle.X_sonic, X_start=bundle.X_sonic_start,
        X_mid=bundle.X_sonic_mid, X_end=bundle.X_sonic_end,
        X_genre=bundle.X_genre_smoothed, center_transitions=False,
    )


def _run(bundle, indices, candidates, **kw):
    defaults = dict(
        final_indices=indices, candidate_indices=candidates,
        metric_context=_ctx(bundle), bundle=bundle,
        seed_indices={indices[0], indices[-1]}, pier_positions={0, len(indices) - 1},
        transition_floor=0.2, centered_cos_floor=-0.5, margin=0.05,
    )
    defaults.update(kw)
    return repair_playlist_edges(**defaults)


def test_t_floor_zero_is_todays_noop():
    # Orthogonal edge (T=0) does NOT trip centered_cos -0.5; with t_floor=0 nothing fires.
    b = _bundle([[1, 0], [0, 1], [1, 0], C90])
    res = _run(b, [0, 1, 2], [3], t_floor=0.0)
    assert res.indices == [0, 1, 2]
    assert not any("new_idx" in e for e in res.swap_log)


def test_weak_edge_fires_and_swaps_to_best():
    # [1,0] -> [0,1] -> [1,0]: both interior edges T=0 < 0.30. Candidate 3 (cos .90
    # to both piers) lifts worst edge 0.0 -> 0.9.
    b = _bundle([[1, 0], [0, 1], [1, 0], C90])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 3, 2]
    swap = next(e for e in res.swap_log if "new_idx" in e)
    assert swap["new_idx"] == 3 and swap["old_idx"] == 1
    worst = min(
        score_transition_edge(_ctx(b), res.indices[i - 1], res.indices[i])["T"]
        for i in range(1, len(res.indices))
    )
    assert worst > 0.8


def test_partial_lift_below_floor_is_accepted():
    # Best available candidate reaches only T=0.25 (< floor 0.30) — break-glass
    # accepts it anyway (0.0 -> 0.25 beats leaving 0.0).
    b = _bundle([[1, 0], [0, 1], [1, 0], C25])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 3, 2]


def test_left_alone_when_no_candidate_clears_margin():
    # Only candidate is ALSO orthogonal to the piers: no improvement >= margin ->
    # leave as-is (never worse, no swap entries).
    b = _bundle([[1, 0], [0, 1], [1, 0], [0, 1]])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 1, 2]
    assert not any("new_idx" in e for e in res.swap_log)


def test_min_gap_refuses_nearby_same_artist():
    # Candidate 5 shares an artist with position 1 (distance 1 < min_gap 3) -> refused;
    # fallback candidate 6 (distinct artist) is used instead.
    X = [[1, 0], C90, [0, 1], [1, 0], [1, 0], C90, C90]
    artists = ["P0", "SameA", "Bad", "P3", "unused", "SameA", "Fresh"]
    b = _bundle(X, artists)
    res = _run(b, [0, 1, 2, 3], [5, 6], t_floor=0.30,
               seed_indices={0, 3}, pier_positions={0, 3}, min_gap=3)
    assert res.indices == [0, 1, 6, 3]
    assert any(e.get("reason") == "min_gap" and e.get("candidate_idx") == 5 for e in res.swap_log)


def test_worst_first_ordering():
    # Two triggered edges; the worse one (T=0 at positions 2->3) must be processed
    # before the milder one (T=0.25 at 1->2): first executed swap targets pos 3.
    # Layout: piers 0,4; interior 1,2,3. Edge 1->2 has cos .25, edge 2->3 cos 0.
    X = [[1, 0], [1, 0], C25, [0, 1], [1, 0], C90]
    b = _bundle(X, ["P", "A", "B", "C", "Q", "R"])
    res = _run(b, [0, 1, 2, 3, 4], [5], t_floor=0.30,
               seed_indices={0, 4}, pier_positions={0, 4})
    swaps = [e for e in res.swap_log if "new_idx" in e]
    assert swaps, "expected at least one executed swap"
    assert swaps[0]["position"] == 3  # worst edge repaired first
```

- [ ] **Step 2: Run to confirm the NEW behaviors fail.** `python -m pytest tests/unit/test_edge_repair_break_glass.py -q` — expect failures on every test except possibly `test_t_floor_zero_is_todays_noop` (TypeError: unexpected keyword `t_floor` counts as the failing state).

- [ ] **Step 3: Implement in `src/playlist/repair/edge_repair.py`.**

3a. Add at module top (after imports): `import logging` and `logger = logging.getLogger(__name__)`.

3b. Add the trigger predicate (below `_worst_t`):

```python
def _needs_repair(
    edge: dict, *, t_floor: float, transition_floor: float, centered_cos_floor: float
) -> bool:
    """Break-glass trigger: weak T (below t_floor) OR catastrophic anti-alignment.

    t_floor=0 disables the weak-T arm (T is always > 0 post-sigmoid), reverting to
    the legacy anti-alignment-only behavior. is_broken_transition itself is untouched.
    """
    t_val = edge.get("T")
    if isinstance(t_val, (int, float)) and float(t_val) < float(t_floor):
        return True
    return is_broken_transition(
        edge,
        transition_floor=float(transition_floor),
        centered_cos_floor=float(centered_cos_floor),
    )
```

3c. In `_candidate_refusal_reasons`, add param `min_gap: int = 0` and, just before the `detect_title_artifacts` check, the positional gap refusal (violation = same identity within distance < min_gap, mirroring `find_min_gap_violations` semantics):

```python
    if int(min_gap) > 0:
        cand_artist_keys = _cap_artist_keys_for_idx(bundle, candidate, artist_identity_cfg)
        if cand_artist_keys:
            lo = max(0, int(replace_position) - (int(min_gap) - 1))
            hi = min(len(current_indices) - 1, int(replace_position) + (int(min_gap) - 1))
            for pos in range(lo, hi + 1):
                if int(pos) == int(replace_position):
                    continue
                other_keys = _cap_artist_keys_for_idx(
                    bundle, int(current_indices[pos]), artist_identity_cfg
                )
                if cand_artist_keys & other_keys:
                    reasons.append("min_gap")
                    break
```

3d. In `repair_playlist_edges`: add params `t_floor: float = 0.0, min_gap: int = 0` (end of signature). Replace the `edge_positions` construction with worst-first ordering:

```python
    if repair_edge_position is not None:
        edge_positions = [int(repair_edge_position)]
    else:
        scored: list[tuple[float, int]] = []
        for pos in range(1, len(indices)):
            e = _edge(metric_context, indices[pos - 1], indices[pos])
            if _needs_repair(
                e, t_floor=float(t_floor),
                transition_floor=float(transition_floor),
                centered_cos_floor=float(centered_cos_floor),
            ):
                t_val = e.get("T")
                scored.append(
                    (float(t_val) if isinstance(t_val, (int, float)) else 1.0, int(pos))
                )
        scored.sort()  # worst-first; a neighboring swap may fix later entries
        edge_positions = [pos for _t, pos in scored]
    edges_triggered = len(edge_positions)
    edges_repaired = 0
```

3e. Inside the per-edge loop, replace the `if not is_broken_transition(current_edge, ...): continue` re-check with the same predicate (this is what skips edges already fixed by an earlier swap):

```python
        if not _needs_repair(
            current_edge,
            t_floor=float(t_floor),
            transition_floor=float(transition_floor),
            centered_cos_floor=float(centered_cos_floor),
        ):
            continue
```

3f. Thread `min_gap=int(min_gap)` into the `_candidate_refusal_reasons(...)` call.

3g. Where the accepted swap is applied (after `indices[replace_pos] = int(new_idx)`), add `edges_repaired += 1` and the per-swap INFO:

```python
        logger.info(
            "Edge repair: pos=%d swapped %s -> %s, worst-T %.3f -> %.3f",
            int(replace_pos), _track_id_for_idx(bundle, old_idx),
            _track_id_for_idx(bundle, int(new_idx)), float(old_worst), float(_new_worst),
        )
```

3h. Before the final `return`, the summary (only when something triggered):

```python
    if repair_edge_position is None and edges_triggered:
        logger.info(
            "Edge repair summary: triggered=%d repaired=%d left_alone=%d (t_floor=%.2f)",
            edges_triggered, edges_repaired, edges_triggered - edges_repaired, float(t_floor),
        )
```

Leave `_all_edges_clear` and the acceptance rule (`new_worst < old_worst + margin: continue`) untouched — `is_broken_transition` already ignores `transition_floor`, so acceptance is anti-alignment + margin only, which IS the break-glass semantics.

- [ ] **Step 4: Run the new tests + the existing repair tests.** `python -m pytest tests/unit/test_edge_repair_break_glass.py tests/unit/test_edge_repair.py -q` → all PASS (existing file exercises the `t_floor=0`/`min_gap=0` defaults — must stay green untouched). Then `python -m mypy src/playlist/repair/edge_repair.py` → clean.

- [ ] **Step 5: Commit.** Verify `git status --short` shows only your two files staged:

```bash
git add src/playlist/repair/edge_repair.py tests/unit/test_edge_repair_break_glass.py
git commit -m "feat(repair): break-glass trigger (T<t_floor), worst-first order, min_gap refusal

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Config knob + call-site wiring

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (add field next to `edge_repair_centered_cos_floor`)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (thread the override)
- Modify: `src/playlist/pier_bridge_builder.py` (call site ~line 2772–2788)
- Modify: `config.example.yaml` and `config.yaml` (add the key beside the existing `edge_repair_*` keys — locate via `grep -n edge_repair config.example.yaml config.yaml`; if the yaml block lacks edge_repair keys entirely, add `edge_repair_t_floor: 0.3` to the `pier_bridge:` block)
- Test: append to `tests/unit/test_edge_repair_break_glass.py`

**Interfaces:**
- Consumes: `repair_playlist_edges(..., t_floor=..., min_gap=...)` from Task 1.
- Produces: `PierBridgeConfig.edge_repair_t_floor: float = 0.30` (live default ON per Layer 4; rollback = set 0).

- [ ] **Step 1: Failing test (append to the Task-1 test file).** Mirror the override-threading pattern used in `tests/unit/test_edge_repair.py` (it imports `apply_pier_bridge_overrides`; copy its exact invocation shape if it differs from this sketch):

```python
def test_edge_repair_t_floor_default_and_override():
    from src.playlist.pier_bridge.config import PierBridgeConfig
    from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

    assert PierBridgeConfig().edge_repair_t_floor == 0.30
    cfg = apply_pier_bridge_overrides(PierBridgeConfig(), {"edge_repair_t_floor": 0.42})
    assert cfg.edge_repair_t_floor == 0.42
```

- [ ] **Step 2: Run to confirm FAIL** (`AttributeError: edge_repair_t_floor`): `python -m pytest tests/unit/test_edge_repair_break_glass.py::test_edge_repair_t_floor_default_and_override -q`

- [ ] **Step 3: Implement.**
  - `pier_bridge/config.py`: next to `edge_repair_centered_cos_floor`, add:
    ```python
    # Break-glass weak-edge trigger: repair edges with T below this (0 = legacy
    # anti-alignment-only). Aligned with variable_bridge_min_edge. Spec 2026-07-01.
    edge_repair_t_floor: float = 0.30
    ```
  - `pipeline/pier_bridge_overrides.py`: find where `edge_repair_margin`/`edge_repair_centered_cos_floor` are threaded (grep `edge_repair`) and add `("edge_repair_t_floor", float)` in the same tuple list / pattern.
  - `pier_bridge_builder.py` call site: add to the `repair_playlist_edges(...)` kwargs:
    ```python
            t_floor=float(getattr(cfg, "edge_repair_t_floor", 0.30)),
            min_gap=int(min_gap),
    ```
    (`min_gap` is the `build_playlist_pier_bridge` parameter, in scope.)
  - Both yaml files: `edge_repair_t_floor: 0.3` beside the other edge_repair keys.

- [ ] **Step 4: Run tests + types.** `python -m pytest tests/unit/test_edge_repair_break_glass.py tests/unit/test_edge_repair.py -q` → PASS. `python -m mypy src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py src/playlist/pier_bridge_builder.py` → clean (pier_bridge_builder has pre-existing looseness; require no NEW errors vs `git stash`-free baseline — just compare error count before/after your edit if any appear).

- [ ] **Step 5: Commit** (config.yaml is gitignored — it will not stage; that's correct):

```bash
git add src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py src/playlist/pier_bridge_builder.py config.example.yaml tests/unit/test_edge_repair_break_glass.py
git commit -m "feat(repair): edge_repair_t_floor knob (0.30 live default) + call-site wiring with min_gap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Validation (suite + live Alvvays rerun)

**Files:** none created (log reading + suite run). Orchestrator-level task — no code edits.

- [ ] **Step 1: Fast unit suite.** `python -m pytest tests/unit -q -m "not slow"` (timeout 600000). Expect green. **If any `tests/unit/goldens/pipeline/*.json` golden test fails**: STOP — do NOT regenerate goldens (they may carry a concurrent session's in-flight state); report which goldens and the T values involved, for coordination.
- [ ] **Step 2: Live generation.** `python main_app.py --artist "Alvvays" --tracks 50` (from repo root; ~2 min). Read the emitted log.
- [ ] **Step 3: Verify in the log:** (a) if any edge fell below 0.30, an `Edge repair summary: triggered=N repaired=M ...` line appears; (b) any executed repair logs old→new worst-T; (c) the final `Weakest transitions` report's min T improved vs. the triggered values (or the summary shows `left_alone` — acceptable break-glass outcome, report it); (d) `repair_applied: true` in the metrics JSON when M>0; (e) mean/p50 T not degraded vs. the 2026-07-01 baseline (mean 0.778 / p50 0.864).
- [ ] **Step 4: Report** actual numbers (triggered/repaired/left_alone, min_transition before-vs-after, weakest-3 edges) — no claims without the log lines.

## Notes / deviations

- Spec's "drop the full-floor clearance on new edges": already true in code — `_all_edges_clear` delegates to `is_broken_transition`, which no longer T-gates. No change needed; documented here so the implementer doesn't "fix" it.
- Piers are never swapped (existing `pier`/`source_before_pier` logic) — the Dives-style landing edge swaps the interior neighbor.
- Phase 2 (beam landing investigation) intentionally absent from this plan.
