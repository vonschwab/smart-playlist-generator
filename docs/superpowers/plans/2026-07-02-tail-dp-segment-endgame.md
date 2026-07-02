# Tail-DP Segment Endgame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each segment's beam+var-bridge finishes, re-open the last 2 interior slots and exactly optimize the window min-edge over the segment pool (never-worse), fixing the beam's landing blindness at the source.

**Architecture:** A pure, unit-testable DP module (`src/playlist/pier_bridge/tail_dp.py`) computes vectorized transition scores from the shared `TransitionMetricContext` and returns the best allowed tail swap. The builder calls it once per finalized segment (immediately before `full_segment = [pier_a] + segment_path + [pier_b]`, currently `pier_bridge_builder.py:2427`) behind `tail_dp_enabled` (live default true; false = byte-identical to today). Two logging-hygiene fixes ride along as separate commits.

**Tech Stack:** Python 3.11+, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-tail-dp-segment-endgame-design.md`

## Global Constraints

- **Shared checkout discipline:** master is shared with a concurrent session. Before editing each file run `git status --short -- <file>`; STOP/BLOCKED if it carries modifications you didn't make. Stage explicit paths only; **commit with an explicit pathspec** (`git commit -m "..." -- <paths>`), never a bare `git commit` (shared index).
- **Never-worse:** a tail swap is applied only if the new window min-edge ≥ current window min-edge + `tail_dp_epsilon` (0.02). On any internal error: log WARNING, keep the original segment, never raise.
- **Pure-T objective:** the DP maximizes `min(T_in, T_mid, T_out)` using the exact shared transition metric; path-shaping modifiers (anti_center, roam, progress, genre) get no vote. Hard constraints (dedup, min_gap, banned artists, used tracks) are enforced via a caller-supplied callback.
- **`tail_dp_enabled: false` must reproduce today's output byte-identically.**
- Vectorized T MUST equal `score_transition_edge(...)["T"]` — there is a unit test asserting this; do not approximate.
- pytest: run directly with `-q`, bounded by tool timeout, never piped through head/tail (hook blocks it). mypy must be clean on all new/edited `src/` files.

---

## Task 1: Pure tail-DP module + tests

**Files:**
- Create: `src/playlist/pier_bridge/tail_dp.py`
- Test: `tests/unit/test_tail_dp.py` (create)

**Interfaces (Task 2 relies on these exact names):**
- `batch_T(ctx, src_indices, dst_indices) -> np.ndarray` — matrix of calibrated T from each src to each dst, exactly matching `score_transition_edge(ctx, s, d)["T"]`.
- `optimize_segment_tail(ctx, *, segment_path: list[int], pier_a: int, pier_b: int, candidates: Sequence[int], epsilon: float, is_allowed_pair, max_pairs_checked: int = 50) -> Optional[TailSwap]` with `@dataclass(frozen=True) class TailSwap: new_tail: tuple[int, ...]; old_min: float; new_min: float`.
  - Window = `min(2, len(segment_path))`; empty path → `None`.
  - 2-slot: maximize `min(T(prefix_end→x), T(x→y), T(y→pier_b))` over ordered pairs x≠y from `candidates`; `prefix_end = segment_path[-3]` if `len(segment_path) >= 3` else `pier_a`.
  - 1-slot: maximize `min(T(prefix_end→x), T(x→pier_b))`; `prefix_end = pier_a` when path length 1.
  - `old_min` = same formula evaluated on the existing tail tracks.
  - Try candidate tails best-first; the first for which `is_allowed_pair(x, y)` (or `is_allowed_pair(x, x)` for 1-slot) returns True wins; give up after `max_pairs_checked`. Return `None` unless `new_min >= old_min + epsilon`.
  - Deterministic: ties broken by (score, then lower candidate index pair).

- [ ] **Step 1: Write the failing tests** (`tests/unit/test_tail_dp.py`; fixture pattern from `tests/unit/test_edge_repair_break_glass.py` — 2-D vectors, `center_transitions=False` so T = plain cosine blend):

```python
"""Tail-DP endgame (spec 2026-07-02): exact max-min re-optimization of the
last 2 interior slots per segment. Pure module tests."""
from pathlib import Path

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.tail_dp import batch_T, optimize_segment_tail
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge

C25 = [0.25, 0.9682458365518543]
C90 = [0.90, 0.4358898943540674]


def _ctx(X):
    X_arr = np.array(X, dtype=float)
    return build_transition_metric_context(
        X_sonic=X_arr, X_start=X_arr, X_mid=X_arr, X_end=X_arr,
        X_genre=np.eye(X_arr.shape[0]), center_transitions=False,
    )


def test_batch_T_matches_score_transition_edge():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4))
    ctx = _ctx(X)
    M = batch_T(ctx, [0, 1, 2], [3, 4, 5])
    for i, a in enumerate([0, 1, 2]):
        for j, b in enumerate([3, 4, 5]):
            assert M[i, j] == pytest.approx(score_transition_edge(ctx, a, b)["T"], abs=1e-9)


def test_two_slot_swap_improves_min():
    # piers 0/1 = [1,0]; existing tail [2,3] orthogonal (min ~ 0);
    # candidates 4,5 = C90 both -> window min jumps to ~0.9-ish.
    X = [[1, 0], [1, 0], [0, 1], [0, 1], C90, C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None
    assert set(res.new_tail) == {4, 5}
    assert res.new_min > res.old_min + 0.02


def test_never_worse_returns_none():
    # candidates are WORSE than the existing decent tail -> None.
    X = [[1, 0], [1, 0], C90, C90, [0, 1], [0, 1]]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is None


def test_disallowed_pairs_are_skipped_for_next_best():
    # best pair (4,5) blocked by the callback -> falls back to (5,4) or mixed.
    X = [[1, 0], [1, 0], [0, 1], [0, 1], C90, C90]
    ctx = _ctx(X)
    blocked = {(4, 5)}
    res = optimize_segment_tail(
        ctx, segment_path=[2, 3], pier_a=0, pier_b=1,
        candidates=[4, 5], epsilon=0.02,
        is_allowed_pair=lambda x, y: (x, y) not in blocked,
    )
    assert res is not None
    assert tuple(res.new_tail) != (4, 5)


def test_one_slot_window():
    # single-interior segment: replace the lone slot.
    X = [[1, 0], [1, 0], [0, 1], C90]
    ctx = _ctx(X)
    res = optimize_segment_tail(
        ctx, segment_path=[2], pier_a=0, pier_b=1,
        candidates=[3], epsilon=0.02, is_allowed_pair=lambda x, y: True,
    )
    assert res is not None and res.new_tail == (3,)


def test_empty_path_and_no_candidates_noop():
    X = [[1, 0], [1, 0], [0, 1]]
    ctx = _ctx(X)
    assert optimize_segment_tail(ctx, segment_path=[], pier_a=0, pier_b=1,
                                 candidates=[2], epsilon=0.02,
                                 is_allowed_pair=lambda x, y: True) is None
    assert optimize_segment_tail(ctx, segment_path=[2], pier_a=0, pier_b=1,
                                 candidates=[], epsilon=0.02,
                                 is_allowed_pair=lambda x, y: True) is None
```

- [ ] **Step 2: Run to confirm FAIL** (import error): `python -m pytest tests/unit/test_tail_dp.py -q`

- [ ] **Step 3: Implement `src/playlist/pier_bridge/tail_dp.py`.**

```python
"""Segment tail re-optimization (tail-DP) — the c-tail landing fix.

The beam picks the last interior slots effectively blind to the landing edge
(spec 2026-07-02: 6/9 segments left >0.2 in-pool min-edge unused). This module
re-opens the last min(2, interior) slots after the segment is finalized and
exactly maximizes the window min-edge over the segment pool. Pure functions —
the builder supplies constraints via a callback. Never-worse by construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TailSwap:
    new_tail: tuple[int, ...]
    old_min: float
    new_min: float


def _calibrate(ctx, arr: np.ndarray) -> np.ndarray:
    """Vectorized twin of vec._calibrate_transition_cos, honoring ctx flags."""
    if not ctx.center_transitions:
        return np.asarray(arr, dtype=np.float64)
    z = float(ctx.calib_gain) * (np.asarray(arr, dtype=np.float64) - float(ctx.calib_center)) / float(ctx.calib_scale)
    return 1.0 / (1.0 + np.exp(-z))


def batch_T(ctx, src_indices: Sequence[int], dst_indices: Sequence[int]) -> np.ndarray:
    """Calibrated blended transition T from each src to each dst.

    Exactly mirrors score_transition_edge: per-component calibration THEN the
    weighted blend; start/mid/end fall back to X_full when absent.
    """
    src = np.asarray(list(src_indices), dtype=int)
    dst = np.asarray(list(dst_indices), dtype=int)
    X_full = ctx.X_full
    X_end = ctx.X_end if ctx.X_end is not None else X_full
    X_start = ctx.X_start if ctx.X_start is not None else X_full
    X_mid = ctx.X_mid if ctx.X_mid is not None else X_full
    es = X_end[src] @ X_start[dst].T
    mm = X_mid[src] @ X_mid[dst].T
    ff = X_full[src] @ X_full[dst].T
    return (
        float(ctx.weight_end_start) * _calibrate(ctx, es)
        + float(ctx.weight_mid_mid) * _calibrate(ctx, mm)
        + float(ctx.weight_full_full) * _calibrate(ctx, ff)
    )


def optimize_segment_tail(
    ctx,
    *,
    segment_path: list[int],
    pier_a: int,
    pier_b: int,
    candidates: Sequence[int],
    epsilon: float,
    is_allowed_pair: Callable[[int, int], bool],
    max_pairs_checked: int = 50,
) -> Optional[TailSwap]:
    """Best allowed re-fill of the last min(2, len(path)) interior slots.

    Returns None when nothing beats the existing tail by >= epsilon (never-
    worse), when the path/candidates are empty, or on internal error (logged).
    """
    try:
        path = [int(i) for i in segment_path]
        cand = [int(c) for c in candidates if int(c) != int(pier_b)]
        if not path or not cand:
            return None
        window = min(2, len(path))
        prefix_end = int(path[-(window + 1)]) if len(path) > window else int(pier_a)

        if window == 1:
            t_in = batch_T(ctx, [prefix_end], cand)[0]      # prefix -> x
            t_out = batch_T(ctx, cand, [pier_b])[:, 0]      # x -> pier_b
            scores = np.minimum(t_in, t_out)
            old_min = float(np.min(batch_T(ctx, [prefix_end, path[-1]],
                                           [path[-1], pier_b])[[0, 1], [0, 1]]))
            order = np.argsort(-scores, kind="stable")
            for rank in order[: int(max_pairs_checked)]:
                x = cand[int(rank)]
                if float(scores[rank]) < old_min + float(epsilon):
                    return None
                if is_allowed_pair(x, x):
                    return TailSwap((x,), old_min, float(scores[rank]))
            return None

        # window == 2
        t_in = batch_T(ctx, [prefix_end], cand)[0]          # prefix -> x
        t_mid = batch_T(ctx, cand, cand)                    # x -> y
        t_out = batch_T(ctx, cand, [pier_b])[:, 0]          # y -> pier_b
        M = np.minimum(np.minimum(t_in[:, None], t_mid), t_out[None, :])
        np.fill_diagonal(M, -np.inf)

        ex, ey = path[-2], path[-1]
        old_edges = [
            batch_T(ctx, [prefix_end], [ex])[0, 0],
            batch_T(ctx, [ex], [ey])[0, 0],
            batch_T(ctx, [ey], [pier_b])[0, 0],
        ]
        old_min = float(min(old_edges))

        flat_order = np.argsort(-M, axis=None, kind="stable")
        for flat in flat_order[: int(max_pairs_checked)]:
            xi, yi = np.unravel_index(int(flat), M.shape)
            score = float(M[xi, yi])
            if not np.isfinite(score) or score < old_min + float(epsilon):
                return None
            x, y = cand[int(xi)], cand[int(yi)]
            if is_allowed_pair(x, y):
                return TailSwap((x, y), old_min, score)
        return None
    except Exception:
        logger.warning("tail_dp: internal error; keeping original segment tail", exc_info=True)
        return None
```

- [ ] **Step 4: Run tests + mypy.** `python -m pytest tests/unit/test_tail_dp.py -q` → 6 passed. `python -m mypy src/playlist/pier_bridge/tail_dp.py` → clean.

- [ ] **Step 5: Commit (pathspec):** `git add src/playlist/pier_bridge/tail_dp.py tests/unit/test_tail_dp.py && git commit -m "feat(tail-dp): pure max-min tail re-optimization module" -- src/playlist/pier_bridge/tail_dp.py tests/unit/test_tail_dp.py` (append the standard `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer).

---

## Task 2: Config knobs + builder integration

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (add `tail_dp_enabled: bool = True`, `tail_dp_epsilon: float = 0.02` near the mini_pier knobs, with a spec-reference comment)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (thread both — mirror the sibling bool/float threading idioms exactly)
- Modify: `src/playlist/pier_bridge_builder.py` (integration; see below)
- Modify: `config.example.yaml` + `config.yaml` (add `tail_dp_enabled: true`, `tail_dp_epsilon: 0.02` in the `pier_bridge:` block — SKIP with a report note if the file carries foreign uncommitted edits)
- Test: append to `tests/unit/test_tail_dp.py`

**Interfaces:**
- Consumes Task 1's `optimize_segment_tail` / `TailSwap` exactly as specified there.

**Integration contract** (behavior is binding; discover exact variable names in the file — it is a ~5k-line god-file under concurrent churn, so anchor by searching, not by line number):

1. Anchor: the per-segment loop, immediately BEFORE `full_segment = [pier_a] + segment_path + [pier_b]` (search that exact string; currently ~line 2427). All downstream consumers (`_compute_edge_scores` diagnostics, assembly, audits) must see the re-optimized path.
2. Gate on `bool(getattr(cfg, "tail_dp_enabled", False)) and segment_path`.
3. Candidates: the chosen attempt's segment candidate pool (the same list whose size is logged as `pool_after_gate`; discover its variable via the `Segment %d attempt` log site or the `_seg_build` / attempt-result dict keys). Prefilter OUT: indices already in `used_track_ids` (or the builder's equivalent cross-segment used set), indices in the current `segment_path` EXCEPT the two being replaced, `pier_a`/`pier_b`, and candidates whose artist identity is in the interior-banned set (reuse the builder's existing banned/pier-artist structures).
4. `is_allowed_pair(x, y)` callback, defined inline in the builder with its existing identity machinery: refuse if x/y share an artist identity with each other (when both slots replaced and min_gap > 0), or with any track within `min_gap` positions in the assembled playlist so far + the kept prefix of this segment (mirror `_enforce_min_gap_global` semantics: distance <= min_gap violates), or (x==y) for the 2-slot case.
5. On an accepted `TailSwap`: replace the tail of `segment_path` in place; UPDATE the used-track and used-artist bookkeeping (remove the displaced tracks, add the new ones) so later segments see correct state; log one INFO: `Tail-DP seg %d: window min %.3f -> %.3f (swapped [%s] -> [%s])`; increment a counter emitted once at the end (`Tail-DP summary: applied=%d/%d segments`). Record `tail_dp_applied`/old/new mins in the segment diagnostics dict if one is easily reachable at the anchor (optional — do not restructure diagnostics for it).
6. On `None`: no change, no log beyond the summary counter.

- [ ] **Step 1: Failing test (append to `tests/unit/test_tail_dp.py`):**

```python
def test_tail_dp_knobs_default_and_override():
    from src.playlist.pier_bridge.config import PierBridgeConfig
    from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

    cfg = PierBridgeConfig()
    assert cfg.tail_dp_enabled is True
    assert cfg.tail_dp_epsilon == 0.02
    # mirror the invocation shape used by test_edge_repair_break_glass.py's
    # knob test (the real apply_pier_bridge_overrides signature).
```

Complete the override assertion by copying the exact invocation shape from `tests/unit/test_edge_repair_break_glass.py::test_edge_repair_t_floor_default_and_override` (READ-ONLY reference) with `{"tail_dp": {"enabled": False, "epsilon": 0.05}}` or the flat keys — whichever matches the sibling threading idiom you implement; assert both values thread.

- [ ] **Step 2: Run → FAIL (AttributeError).**
- [ ] **Step 3: Implement** config fields, override threading, builder integration per the contract, yaml keys.
- [ ] **Step 4: Tests + types.** `python -m pytest tests/unit/test_tail_dp.py tests/unit/test_edge_repair_break_glass.py -q` → all pass. `python -m mypy src/playlist/pier_bridge/tail_dp.py src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py` → clean; `python -c "import src.playlist.pier_bridge_builder"` → OK.
- [ ] **Step 5: Commit (pathspec, only your files):** message `feat(tail-dp): wire segment tail re-optimization into the builder (live default)` + co-author trailer.

---

## Task 3: Logging hygiene (two small commits)

**Files:** `src/playlist/pipeline/pier_bridge_overrides.py`, `src/playlist/pier_bridge/beam.py`

- [ ] **Step 1 (F2):** In `apply_pier_bridge_overrides`, the "Pier-bridge tuning resolved: …" INFO fires before `pb_cfg = pier_bridge_config or PierBridgeConfig(…)` and logs weights that are DISCARDED when `pier_bridge_config` is supplied (artist mode). Move/duplicate the weight portion of the log to AFTER pb_cfg resolution so it reports `pb_cfg.weight_bridge`/`pb_cfg.weight_transition` (the values actually in effect), and when `pier_bridge_config` was supplied append `" (pre-built pier config supplied; resolved tuning weights not applied)"`. Logging-only change; no behavior change; existing tests must stay green.
- [ ] **Step 2 (F3):** In `beam.py`, fix the two stale comments: `# Hard floors: transition + bridge-local` (near the `_transition_gate_failed` call in the expansion loop) and `# Hard floor on final transition` (final-connection block) → `# Anti-alignment safety only (is_broken_transition no longer T-gates; roam design)`.
- [ ] **Step 3:** `python -m pytest tests/unit/test_tail_dp.py tests/unit/test_mode_threshold_resolution.py -q` (threshold tests cover the tuning log path) → pass.
- [ ] **Step 4: Two commits (pathspec):** `fix(log): report the pier-bridge weights actually in effect (artist mode discarded tuning)` and `docs(beam): correct stale hard-floor comments (no T-gate since roam promotion)`.

---

## Task 4: Validation (orchestrator-run)

- [ ] **Step 1:** Fast suite `python -m pytest tests/unit -q -m "not slow"`. Config-snapshot goldens (`tests/unit/goldens/pipeline/*.json`) will gain the two new knobs — if those 4 tests fail with exactly `tail_dp_enabled`/`tail_dp_epsilon` diffs AND `git status` shows the golden dir clean, regenerate via delete+rerun and diff-audit (exactly 2 added keys per file), commit pathspec. Any OTHER golden diff → STOP and report.
- [ ] **Step 2:** Pure-beam live check: Alvvays run with `edge_repair.t_floor=0` override and tail-DP on → expect `Tail-DP seg …` INFO lines and weak-segment window minima near the probe's tail2 column (≈0.54–0.92); worst playlist edge should rise vs the 0.059 baseline.
- [ ] **Step 3:** Composed live check: normal run (repair on) → `Tail-DP summary` + `Edge repair summary` both present; repair's triggered count should DROP vs the earlier validated run (5 triggered) since tail-DP pre-lifts landings; min_transition/below_floor reported.
- [ ] **Step 4:** Report actual numbers; no success claims without log lines.

## Notes

- Do not modify `tests/unit/test_edge_repair.py` (concurrent-session file; read-only reference).
- `batch_T`'s per-component-calibrate-then-blend order is load-bearing (matches `score_transition_edge`); the equality unit test guards it.
- The displaced tracks freed by a swap must leave the used set — forgetting this silently shrinks later segments' pools.
