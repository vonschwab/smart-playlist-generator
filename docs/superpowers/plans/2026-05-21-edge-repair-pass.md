# Edge Repair Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in post-beam repair pass that swaps individual tracks at slots where the adjacent edge is unambiguously broken (T < 0.20 or T_centered_cos < −0.5), with strict "do no harm" guarantees: the replacement is only applied if both resulting edges clear the absolute floor *and* the worst of the two improves by ≥ margin.

**Architecture:** A pure function `repair_playlist_edges` runs after pier-bridge completes and before edge-score reporting. It identifies broken edges using the same T math the reporter uses (centered end-to-start cosine), iterates over the existing eligible candidate pool, and applies a maximin selection on candidates that satisfy hard quality + structural constraints. Default off behind a config flag. Piers are never replaced. Demos/medleys/live cuts are filtered from replacements unconditionally.

**Tech Stack:** Python 3.11+, NumPy, existing pier-bridge artifacts (`bundle.X_sonic_start`, `bundle.X_sonic_end`), existing title-quality module, existing artist identity resolution, pytest.

---

## File Structure

- **Create** `src/playlist/repair/__init__.py` — empty package marker
- **Create** `src/playlist/repair/edge_repair.py` — pure repair function
  - `repair_playlist_edges(...)` — the entry point
  - Internal helpers for T computation, candidate filtering, maximin selection
  - No I/O, no logging at module scope (caller handles logging from the returned `swap_log`)
- **Modify** `src/playlist/pier_bridge/config.py` — add five `edge_repair_*` fields
- **Modify** `src/playlist/pipeline/pier_bridge_overrides.py` — parse `pier_bridge.edge_repair.*` from config dict
- **Modify** `src/playlist/pier_bridge_builder.py` — call repair after final_indices is selected, before per-edge T reporting (~line 1880). Store `swap_log` in pier-bridge result stats.
- **Modify** `src/playlist/reporter.py` — emit the swap log block when present
- **Modify** `config.yaml` — add commented-out opt-in block under `pier_bridge:`
- **Create** `tests/unit/test_edge_repair.py` — focused tests for the pure repair function
- **Modify** `docs/PLAYLIST_ORDERING_TUNING.md` — add the repair pass as Knob 4
- **Modify** `docs/CHANGELOG.md` — add a v4.2 entry

---

## Background: how T is computed

The repair function must use the *same* T computation as the audit reporter (`src/playlist/reporter.py:compute_edge_scores_from_artifact`), not the beam's transition-weighted version. Specifically:

1. Take raw `bundle.X_sonic_start` and `bundle.X_sonic_end` (the per-track audio snippet feature matrices).
2. Mean-center: `X_end_c = X_end - X_end.mean(axis=0)`, `X_start_c = X_start - X_start.mean(axis=0)`.
3. L2-normalize each row of the centered matrices.
4. For edge (a, b): `T_centered_cos = dot(X_end_c_norm[a], X_start_c_norm[b])`.
5. `T = clip((T_centered_cos + 1.0) / 2.0, 0.0, 1.0)` when `center_transitions=True` (the default).

This is the T value shown in the audit table. The repair must trigger on exactly the same metric that defines "broken" in the audit.

---

### Task 1: Pure repair function + unit tests

**Files:**
- Create: `src/playlist/repair/__init__.py`
- Create: `src/playlist/repair/edge_repair.py`
- Create: `tests/unit/test_edge_repair.py`

This is the core of the work. Write tests first, then implement.

- [ ] **Step 1: Create the package marker**

Create `src/playlist/repair/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_edge_repair.py`:

```python
"""Tests for edge repair pass — the post-beam swap-only repair."""
from __future__ import annotations

import numpy as np
import pytest

from src.playlist.repair.edge_repair import repair_playlist_edges


# ----- Fixture builders -----

def _make_bundle(*, n_tracks: int, dim: int = 8, seed: int = 0):
    """Build a minimal stand-in for ArtifactBundle.

    The repair function only uses: X_sonic_start, X_sonic_end, X_sonic,
    track_artists, track_titles. We synthesize matrices where specific
    pairs have controllable cosine similarity by sharing or differing rows.
    """
    rng = np.random.default_rng(seed)
    X_sonic = rng.standard_normal((n_tracks, dim)).astype(np.float64)
    X_sonic_start = rng.standard_normal((n_tracks, dim)).astype(np.float64)
    X_sonic_end = rng.standard_normal((n_tracks, dim)).astype(np.float64)

    class _Bundle:
        pass

    b = _Bundle()
    b.X_sonic = X_sonic
    b.X_sonic_start = X_sonic_start
    b.X_sonic_end = X_sonic_end
    b.track_artists = np.array([f"Artist{i}" for i in range(n_tracks)], dtype=object)
    b.track_titles = np.array([f"Title{i}" for i in range(n_tracks)], dtype=object)
    return b


def _force_edge_T(bundle, *, prev: int, cur: int, centered_cos_target: float):
    """Hack a bundle so the edge (prev → cur) has the requested T_centered_cos.

    After mean-centering and L2-normalizing X_end and X_start, dot(end[prev], start[cur])
    should equal centered_cos_target. Set end[prev] = u, start[cur] = α*u + β*v where
    u, v are orthonormal — but to keep the math simple we set both rows directly to vectors
    whose pre-centering cosine produces approximately the target. The simplest approach
    is to make all *other* rows zero so mean-centering is a no-op, then set the two rows
    directly to vectors with the desired cosine.

    We use only 2 dimensions for simplicity in tests that call this.
    """
    # Set all rows to zero so mean is zero and centering is a no-op.
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Set the target pair so their L2-normalized dot product equals the target.
    bundle.X_sonic_end[prev] = np.array([1.0, 0.0] + [0.0] * (bundle.X_sonic_end.shape[1] - 2))
    theta = np.arccos(np.clip(centered_cos_target, -1.0, 1.0))
    bundle.X_sonic_start[cur] = np.array(
        [np.cos(theta), np.sin(theta)] + [0.0] * (bundle.X_sonic_start.shape[1] - 2)
    )


# ----- Tests -----

def test_no_broken_edges_returns_original():
    bundle = _make_bundle(n_tracks=4, dim=4)
    # Make all edges have T=1.0 by setting end[i] == start[i+1].
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    for i in range(3):
        bundle.X_sonic_end[i] = np.array([1.0, 0.0, 0.0, 0.0])
        bundle.X_sonic_start[i + 1] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]
    assert swap_log == []


def test_repairs_below_floor_edge_with_valid_candidate():
    # Playlist: piers at 0 and 3, interior slots 1 and 2.
    # Make edge (0 -> 1) catastrophic: T_centered_cos = -0.9 → T = 0.05.
    # Provide candidate idx=4 such that edges (0 -> 4) and (4 -> 2) are excellent.
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Edge 0 -> 1 broken
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Candidate 4: high T from 0 and high T to 2
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])  # close to end[0]
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    # Existing slot 1's outgoing edge (1 -> 2) — make it acceptable so post-swap doesn't matter on this side
    bundle.X_sonic_end[1] = np.array([1.0, 0.0, 0.0, 0.0])
    # Pier outgoing closure (2 -> 3) — irrelevant for the swap at slot 1
    bundle.X_sonic_end[2] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[3] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 4, 2, 3]
    assert len(swap_log) == 1
    entry = swap_log[0]
    assert entry["slot"] == 1
    assert entry["old_idx"] == 1
    assert entry["new_idx"] == 4
    assert entry["reason"] == "swapped"
    assert entry["new_worst"] > entry["old_worst"] + 0.05 - 1e-9


def test_refuses_when_no_candidate_clears_floor():
    # Broken edge 0 -> 1. Only candidate (4) has terrible T from 0.
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Candidate 4 is also anti-correlated to 0
    bundle.X_sonic_start[4] = np.array([-0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]
    assert len(swap_log) == 1
    assert swap_log[0]["new_idx"] is None
    assert swap_log[0]["reason"] in ("no_candidate_clears_floor", "no_candidate_meets_margin")


def test_does_not_replace_pier_slot():
    # Edge (1 -> 2) is broken, but slot 2 is a pier. Repair must skip.
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    bundle.X_sonic_end[1] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Make all other edges decent so they don't trigger
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_end[2] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[3] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 2, 3},  # slot 2 is a pier
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]
    assert len(swap_log) == 1
    assert swap_log[0]["new_idx"] is None
    assert swap_log[0]["reason"] == "pier"


def test_centered_cos_trigger_fires_when_t_above_floor():
    # T = (centered_cos + 1) / 2. T_centered_cos = -0.6 -> T = 0.20.
    # The repair triggers because T_centered_cos < -0.5 even though T == floor.
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Centered cosine = -0.6 → T = 0.20 (exactly at floor). Both triggers fire here.
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.6, np.sqrt(1 - 0.36), 0.0, 0.0])
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_end[1] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    # Repair was attempted (slot 1 either swapped or refused; either way there's a log entry)
    assert len(swap_log) >= 1
    assert swap_log[0]["slot"] == 1


def test_title_artifact_filter_excludes_demo_candidate():
    # Broken edge 0 -> 1. Candidate 4 has title "Foo (Demo)" — must be excluded.
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.track_titles[4] = "Foo (Demo)"
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset({"demo"}),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]  # no swap because the only candidate was filtered
    assert swap_log[0]["new_idx"] is None


def test_min_gap_excludes_recent_artist():
    # Broken edge 1 -> 2. Candidate 4 has same artist as slot 0 (within min_gap=6).
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.track_artists[0] = "Shared Artist"
    bundle.track_artists[4] = "Shared Artist"
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    bundle.X_sonic_end[1] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[3] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]
    assert swap_log[0]["new_idx"] is None


def test_picks_maximin_among_eligible():
    # Broken edge 0 -> 1. Two candidates clear the bar; one has higher maximin.
    bundle = _make_bundle(n_tracks=6, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Broken edge
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Candidate 4: T_in=0.95 (excellent), T_out=0.80
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([0.60, np.sqrt(1 - 0.36), 0.0, 0.0])
    # Candidate 5: T_in=0.85, T_out=0.92 — better worst-of-two
    bundle.X_sonic_start[5] = np.array([0.85, np.sqrt(1 - 0.7225), 0.0, 0.0])
    bundle.X_sonic_end[5] = np.array([0.90, np.sqrt(1 - 0.81), 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4, 5],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 5, 2, 3]  # 5 has the better maximin


def test_multiple_broken_edges_repaired_independently():
    # Edges 0->1 and 2->3 both broken; pier at 0 and 4. Eligible candidates for each.
    bundle = _make_bundle(n_tracks=7, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Broken edge 0 -> 1
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Good replacement for slot 1: idx 5
    bundle.X_sonic_start[5] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[5] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_end[1] = np.array([1.0, 0.0, 0.0, 0.0])
    # Broken edge 2 -> 3
    bundle.X_sonic_end[2] = np.array([0.0, 1.0, 0.0, 0.0])
    bundle.X_sonic_start[3] = np.array([0.0, -0.9, np.sqrt(1 - 0.81), 0.0])
    # Good replacement for slot 3: idx 6
    bundle.X_sonic_start[6] = np.array([0.0, 0.95, np.sqrt(1 - 0.9025), 0.0])
    bundle.X_sonic_end[6] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[4] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3, 4],
        pier_positions={0, 4},
        eligible_pool=[5, 6],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    # Both broken slots get repaired.
    assert new_indices[0] == 0 and new_indices[4] == 4  # piers untouched
    assert new_indices[1] == 5
    assert new_indices[3] == 6
    assert len(swap_log) == 2
    assert all(e["reason"] == "swapped" for e in swap_log)


def test_variety_guard_excludes_sonic_clones_when_enabled():
    # Broken edge 0 -> 1. Candidate 4 is a sonic clone of slot 0 (S > 0.85).
    bundle = _make_bundle(n_tracks=5, dim=4)
    bundle.X_sonic[:] = 0.0
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    # Make slot 0 and candidate 4 have S=0.99 (near-identical sonic vector)
    bundle.X_sonic[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic[4] = np.array([0.99, np.sqrt(1 - 0.9801), 0.0, 0.0])
    # Broken transition edge
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])
    # Excellent transition from 4 to context
    bundle.X_sonic_start[4] = np.array([0.95, np.sqrt(1 - 0.9025), 0.0, 0.0])
    bundle.X_sonic_end[4] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[2] = np.array([1.0, 0.0, 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[4],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=True,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]  # variety guard filtered the clone
    assert swap_log[0]["new_idx"] is None


def test_skips_when_pool_empty():
    bundle = _make_bundle(n_tracks=4, dim=4)
    bundle.X_sonic_end[:] = 0.0
    bundle.X_sonic_start[:] = 0.0
    bundle.X_sonic_end[0] = np.array([1.0, 0.0, 0.0, 0.0])
    bundle.X_sonic_start[1] = np.array([-0.9, np.sqrt(1 - 0.81), 0.0, 0.0])

    new_indices, swap_log = repair_playlist_edges(
        track_indices=[0, 1, 2, 3],
        pier_positions={0, 3},
        eligible_pool=[],
        bundle=bundle,
        transition_floor=0.20,
        centered_cos_floor=-0.5,
        margin=0.05,
        min_gap=6,
        title_artifact_filter_set=frozenset(),
        variety_guard_enabled=False,
        variety_guard_threshold=0.85,
        center_transitions=True,
    )

    assert new_indices == [0, 1, 2, 3]
    assert len(swap_log) == 1
    assert swap_log[0]["new_idx"] is None
    assert swap_log[0]["reason"] == "no_candidate_clears_floor"
```

- [ ] **Step 3: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_edge_repair.py -q --basetemp .pytest-tmp-repair -o cache_dir=.pytest-tmp-cache-repair
```
Expected: `ImportError` for `src.playlist.repair.edge_repair`.

- [ ] **Step 4: Implement `repair_playlist_edges`**

Create `src/playlist/repair/edge_repair.py`:

```python
"""Post-beam edge repair pass.

Identifies playlist edges where T < transition_floor or T_centered_cos <
centered_cos_floor (catastrophic by either metric), then attempts to swap
the destination track of each broken edge with a candidate from the
existing eligible pool. Swap is only applied when both resulting edges
clear the absolute transition floor AND the worst-of-two improves by
at least `margin` compared to the original worst-of-two. Piers are never
replaced.

This module computes T using the same math the reporter uses:
   T_centered_cos = dot(centered_X_end[a], centered_X_start[b])    (rows L2-normalized after centering)
   T              = clip((T_centered_cos + 1) / 2, 0, 1)           (when center_transitions=True)

Pure function: no logging at module scope. Callers consume the returned
`swap_log` and emit it where appropriate.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from src.playlist.title_quality import detect_title_artifacts


def _center_and_normalize(M: np.ndarray) -> np.ndarray:
    """Subtract column means then L2-normalize rows."""
    Mc = M - M.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Mc, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Mc / norms


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return M / norms


def _t_for_pair(
    a: int,
    b: int,
    X_end_cn: np.ndarray,
    X_start_cn: np.ndarray,
    center_transitions: bool,
) -> Tuple[float, float]:
    """Returns (T, T_centered_cos) for the directed edge a -> b."""
    cos = float(np.dot(X_end_cn[a], X_start_cn[b]))
    if center_transitions:
        t = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
    else:
        t = cos
    return t, cos


def repair_playlist_edges(
    *,
    track_indices: List[int],
    pier_positions: Set[int],
    eligible_pool: List[int],
    bundle: Any,
    transition_floor: float,
    centered_cos_floor: float,
    margin: float,
    min_gap: int,
    title_artifact_filter_set: FrozenSet[str],
    variety_guard_enabled: bool,
    variety_guard_threshold: float,
    center_transitions: bool,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Run a single repair pass over the playlist.

    Arguments:
        track_indices: ordered list of track indices in the bundle (length N).
        pier_positions: set of positions in track_indices (0-based) that are piers.
        eligible_pool: bundle indices that are valid replacement candidates
            (already filtered by upstream policies like recency, blacklist,
            duration, allowed_ids).
        bundle: ArtifactBundle-like object exposing X_sonic_start, X_sonic_end,
            X_sonic, track_artists, track_titles.
        transition_floor: minimum acceptable T after repair (e.g. 0.20).
        centered_cos_floor: trigger threshold on T_centered_cos (e.g. -0.5).
        margin: minimum improvement in worst-of-two (e.g. 0.05).
        min_gap: window for adjacent-artist exclusion.
        title_artifact_filter_set: flag names (e.g. {"demo", "live"}) that
            cause a candidate to be excluded. Pass empty set to disable.
        variety_guard_enabled: when True, candidates with S(track[i-1], X) >
            variety_guard_threshold are excluded as sonic clones.
        variety_guard_threshold: sonic-cosine threshold for the variety guard.
        center_transitions: pipeline-wide flag controlling T rescaling.

    Returns:
        (new_track_indices, swap_log). new_track_indices is a fresh list
        (input is not mutated). swap_log has one entry per broken edge that
        was considered; entry["new_idx"] is None when the swap was refused.
    """
    n = len(track_indices)
    if n < 2 or not eligible_pool:
        # No edges to repair, or no candidates available.
        swap_log: List[Dict[str, Any]] = []
        # If there are no candidates but there is a broken edge, still emit a refusal log entry per broken edge.
        if n >= 2:
            X_end_cn, X_start_cn = _center_and_normalize(bundle.X_sonic_end), _center_and_normalize(
                bundle.X_sonic_start
            )
            for i in range(1, n):
                t, cc = _t_for_pair(track_indices[i - 1], track_indices[i], X_end_cn, X_start_cn, center_transitions)
                if t < transition_floor or cc < centered_cos_floor:
                    swap_log.append(
                        {
                            "slot": i,
                            "old_idx": track_indices[i],
                            "new_idx": None,
                            "old_T_in": t,
                            "old_T_out": None,
                            "new_T_in": None,
                            "new_T_out": None,
                            "old_worst": t,
                            "new_worst": None,
                            "margin_achieved": None,
                            "reason": "no_candidate_clears_floor",
                            "candidates_evaluated": 0,
                        }
                    )
        return list(track_indices), swap_log

    # Precompute centered, L2-normalized end/start matrices once.
    X_end_cn = _center_and_normalize(bundle.X_sonic_end)
    X_start_cn = _center_and_normalize(bundle.X_sonic_start)

    # Precompute sonic-similarity matrix rows on demand (used for variety guard).
    X_sonic_n = _normalize_rows(getattr(bundle, "X_sonic")) if variety_guard_enabled else None

    swap_log = []
    current_indices = list(track_indices)

    # Identify broken edges from the *original* T values. We process in order.
    broken_slots: List[int] = []
    for i in range(1, n):
        t, cc = _t_for_pair(track_indices[i - 1], track_indices[i], X_end_cn, X_start_cn, center_transitions)
        if t < transition_floor or cc < centered_cos_floor:
            broken_slots.append(i)

    if not broken_slots:
        return current_indices, swap_log

    # Helper: extract artist string for an index.
    def _artist(idx: int) -> str:
        try:
            return str(bundle.track_artists[idx])
        except Exception:
            return ""

    # Helper: extract title for an index.
    def _title(idx: int) -> str:
        try:
            return str(bundle.track_titles[idx])
        except Exception:
            return ""

    # Filter pool to candidates not already in the playlist.
    playlist_set: Set[int] = set(int(j) for j in current_indices)
    base_pool = [int(c) for c in eligible_pool if int(c) not in playlist_set]

    for slot in broken_slots:
        old_idx = current_indices[slot]

        # Rule: never replace a pier.
        if slot in pier_positions:
            t_in, _ = _t_for_pair(
                current_indices[slot - 1], current_indices[slot], X_end_cn, X_start_cn, center_transitions
            )
            t_out = None
            if slot + 1 < n:
                t_out_val, _ = _t_for_pair(
                    current_indices[slot], current_indices[slot + 1], X_end_cn, X_start_cn, center_transitions
                )
                t_out = t_out_val
            swap_log.append(
                {
                    "slot": slot,
                    "old_idx": old_idx,
                    "new_idx": None,
                    "old_T_in": t_in,
                    "old_T_out": t_out,
                    "new_T_in": None,
                    "new_T_out": None,
                    "old_worst": min(t_in, t_out) if t_out is not None else t_in,
                    "new_worst": None,
                    "margin_achieved": None,
                    "reason": "pier",
                    "candidates_evaluated": 0,
                }
            )
            continue

        # Original T_in (broken) and T_out (from slot to next).
        old_T_in, _ = _t_for_pair(
            current_indices[slot - 1], current_indices[slot], X_end_cn, X_start_cn, center_transitions
        )
        old_T_out: Optional[float] = None
        if slot + 1 < n:
            old_T_out_val, _ = _t_for_pair(
                current_indices[slot], current_indices[slot + 1], X_end_cn, X_start_cn, center_transitions
            )
            old_T_out = old_T_out_val
        old_worst = old_T_in if old_T_out is None else min(old_T_in, old_T_out)

        # Adjacent-artist exclusion window: slots [slot - min_gap, slot + min_gap]
        excluded_artists: Set[str] = set()
        lo = max(0, slot - min_gap)
        hi = min(n - 1, slot + min_gap)
        for j in range(lo, hi + 1):
            if j == slot:
                continue
            excluded_artists.add(_artist(current_indices[j]).lower())

        # Filter candidates.
        filtered: List[int] = []
        for cand in base_pool:
            cand_artist = _artist(cand).lower()
            if cand_artist and cand_artist in excluded_artists:
                continue
            cand_title = _title(cand)
            if title_artifact_filter_set and cand_title:
                flags = detect_title_artifacts(cand_title)
                if flags & title_artifact_filter_set:
                    continue
            if variety_guard_enabled and X_sonic_n is not None:
                s_to_prev = float(np.dot(X_sonic_n[current_indices[slot - 1]], X_sonic_n[cand]))
                if s_to_prev > variety_guard_threshold:
                    continue
            filtered.append(cand)

        if not filtered:
            swap_log.append(
                {
                    "slot": slot,
                    "old_idx": old_idx,
                    "new_idx": None,
                    "old_T_in": old_T_in,
                    "old_T_out": old_T_out,
                    "new_T_in": None,
                    "new_T_out": None,
                    "old_worst": old_worst,
                    "new_worst": None,
                    "margin_achieved": None,
                    "reason": "no_candidate_clears_floor",
                    "candidates_evaluated": 0,
                }
            )
            continue

        # Score each candidate: compute new T_in and new T_out, apply hard gates.
        best: Optional[Tuple[float, int, float, float]] = None  # (worst, cand, T_in, T_out)
        for cand in filtered:
            new_T_in, _ = _t_for_pair(
                current_indices[slot - 1], cand, X_end_cn, X_start_cn, center_transitions
            )
            new_T_out: Optional[float] = None
            if slot + 1 < n:
                new_T_out_val, _ = _t_for_pair(
                    cand, current_indices[slot + 1], X_end_cn, X_start_cn, center_transitions
                )
                new_T_out = new_T_out_val
            worst = new_T_in if new_T_out is None else min(new_T_in, new_T_out)
            # Hard gate: both edges above absolute floor.
            if new_T_in < transition_floor:
                continue
            if new_T_out is not None and new_T_out < transition_floor:
                continue
            # Margin gate.
            if worst < old_worst + margin:
                continue
            if best is None or worst > best[0]:
                best = (worst, cand, new_T_in, new_T_out if new_T_out is not None else 1.0)

        candidates_evaluated = len(filtered)

        if best is None:
            # Distinguish "no candidate clears floor" from "no candidate meets margin".
            any_above_floor = False
            for cand in filtered:
                new_T_in_x, _ = _t_for_pair(
                    current_indices[slot - 1], cand, X_end_cn, X_start_cn, center_transitions
                )
                if slot + 1 < n:
                    new_T_out_x, _ = _t_for_pair(
                        cand, current_indices[slot + 1], X_end_cn, X_start_cn, center_transitions
                    )
                    if new_T_in_x >= transition_floor and new_T_out_x >= transition_floor:
                        any_above_floor = True
                        break
                else:
                    if new_T_in_x >= transition_floor:
                        any_above_floor = True
                        break
            reason = "no_candidate_meets_margin" if any_above_floor else "no_candidate_clears_floor"
            swap_log.append(
                {
                    "slot": slot,
                    "old_idx": old_idx,
                    "new_idx": None,
                    "old_T_in": old_T_in,
                    "old_T_out": old_T_out,
                    "new_T_in": None,
                    "new_T_out": None,
                    "old_worst": old_worst,
                    "new_worst": None,
                    "margin_achieved": None,
                    "reason": reason,
                    "candidates_evaluated": candidates_evaluated,
                }
            )
            continue

        new_worst, new_idx, new_T_in_final, new_T_out_final = best
        # Apply the swap to current_indices and update playlist_set / base_pool.
        current_indices[slot] = new_idx
        playlist_set.discard(old_idx)
        playlist_set.add(new_idx)
        # Remove the swapped-in candidate from the pool for subsequent slots.
        base_pool = [c for c in base_pool if c != new_idx]

        swap_log.append(
            {
                "slot": slot,
                "old_idx": old_idx,
                "new_idx": new_idx,
                "old_T_in": old_T_in,
                "old_T_out": old_T_out,
                "new_T_in": new_T_in_final,
                "new_T_out": new_T_out_final if slot + 1 < n else None,
                "old_worst": old_worst,
                "new_worst": new_worst,
                "margin_achieved": new_worst - old_worst,
                "reason": "swapped",
                "candidates_evaluated": candidates_evaluated,
            }
        )

    return current_indices, swap_log
```

- [ ] **Step 5: Run the tests and verify pass**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_edge_repair.py -q --basetemp .pytest-tmp-repair -o cache_dir=.pytest-tmp-cache-repair
```
Expected: all 10 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/repair/__init__.py src/playlist/repair/edge_repair.py tests/unit/test_edge_repair.py
git commit -m "feat: edge repair pure function with strict do-no-harm guarantees"
```

---

### Task 2: Config fields on PierBridgeConfig

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`

- [ ] **Step 1: Add config fields**

Open `src/playlist/pier_bridge/config.py`. Find the section near `emit_selected_edge_audit` (added in v4.1) and add the repair fields adjacent. Inside the `@dataclass class PierBridgeConfig:` block, add:

```python
    # Edge repair pass (opt-in, default off).
    # Post-beam single-pass swap: identifies edges where T < transition_floor
    # OR T_centered_cos < centered_cos_floor, swaps the destination track
    # (never a pier) with a candidate from the eligible pool that strictly
    # improves the worst-of-two adjacent edges by at least margin.
    edge_repair_enabled: bool = False
    edge_repair_centered_cos_floor: float = -0.5
    edge_repair_margin: float = 0.05
    edge_repair_variety_guard_enabled: bool = False
    edge_repair_variety_guard_threshold: float = 0.85
```

- [ ] **Step 2: Run the existing config tests (sanity check)**

```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-cfg -o cache_dir=.pytest-tmp-cache-cfg
```
Expected: most tests pass, but the pipeline smoke goldens may fail because PierBridgeConfig now has new fields serialized in the effective config. Note the failure for the next step.

- [ ] **Step 3: Update the smoke golden baselines**

For each failing golden file under `tests/unit/goldens/pipeline/*.json` (there are 4), open the file and add the new fields to the `pier_config` object alongside `emit_selected_edge_audit`:

```json
"edge_repair_enabled": false,
"edge_repair_centered_cos_floor": -0.5,
"edge_repair_margin": 0.05,
"edge_repair_variety_guard_enabled": false,
"edge_repair_variety_guard_threshold": 0.85,
```

Preserve alphabetical or insertion order to match how the project sorts goldens — look at how `emit_selected_edge_audit` was inserted by prior commits (`108fa1d`, `72616ab`, etc.) and mirror it.

- [ ] **Step 4: Run pipeline smoke goldens to confirm pass**

```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-cfg -o cache_dir=.pytest-tmp-cache-cfg
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/config.py tests/unit/goldens/pipeline/
git commit -m "feat: add edge_repair config fields to PierBridgeConfig"
```

---

### Task 3: Override parsing

**Files:**
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py`

- [ ] **Step 1: Add parsing block**

Open `src/playlist/pipeline/pier_bridge_overrides.py`. Find the parsing block for `emit_selected_edge_audit` (added in v4.1 — search for `emit_selected_edge_audit`). After that block, add:

```python
    edge_repair_raw = pb_overrides.get("edge_repair")
    if isinstance(edge_repair_raw, dict):
        if isinstance(edge_repair_raw.get("enabled"), bool):
            pb_cfg = replace(pb_cfg, edge_repair_enabled=bool(edge_repair_raw.get("enabled")))
        if isinstance(edge_repair_raw.get("centered_cos_floor"), (int, float)):
            pb_cfg = replace(
                pb_cfg,
                edge_repair_centered_cos_floor=float(edge_repair_raw.get("centered_cos_floor")),
            )
        if isinstance(edge_repair_raw.get("margin"), (int, float)):
            pb_cfg = replace(pb_cfg, edge_repair_margin=float(edge_repair_raw.get("margin")))
        variety_raw = edge_repair_raw.get("variety_guard")
        if isinstance(variety_raw, dict):
            if isinstance(variety_raw.get("enabled"), bool):
                pb_cfg = replace(
                    pb_cfg,
                    edge_repair_variety_guard_enabled=bool(variety_raw.get("enabled")),
                )
            if isinstance(variety_raw.get("threshold"), (int, float)):
                pb_cfg = replace(
                    pb_cfg,
                    edge_repair_variety_guard_threshold=float(variety_raw.get("threshold")),
                )
```

- [ ] **Step 2: Run pipeline smoke goldens**

```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-ovr -o cache_dir=.pytest-tmp-cache-ovr
```
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add src/playlist/pipeline/pier_bridge_overrides.py
git commit -m "feat: parse pier_bridge.edge_repair config overrides"
```

---

### Task 4: Integrate repair into pier_bridge_builder

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py`

The repair is called after `final_indices` is selected by the beam, before per-edge T scores are computed for reporting.

- [ ] **Step 1: Locate the integration point**

Open `src/playlist/pier_bridge_builder.py`. Find the section just before the `edge_scores: list[dict[str, Any]] = []` declaration (around line 1882). This is the point after the beam has produced `final_indices` and before the reporter computes per-edge T values.

- [ ] **Step 2: Add the repair call**

Just before the `edge_scores: list[dict[str, Any]] = []` line, add the repair invocation. The repair must build the set of pier positions from the seed track IDs (the piers are at positions where `final_indices[pos]` equals one of the seed indices).

Add this block:

```python
    # Edge repair pass (opt-in via cfg.edge_repair_enabled).
    edge_repair_swap_log: list[dict[str, Any]] = []
    if bool(getattr(cfg, "edge_repair_enabled", False)):
        from src.playlist.repair.edge_repair import repair_playlist_edges

        # Build pier positions: any index in final_indices that matches a seed.
        seed_indices_set = set(int(s) for s in pier_seed_indices)
        pier_positions = {i for i, idx in enumerate(final_indices) if int(idx) in seed_indices_set}

        # Title-artifact flag set: always include the common bad-recording flags.
        # This is independent of the soft penalty knob; repair replacements
        # should never silently introduce demos/live/medleys/etc.
        _repair_title_filter = frozenset({
            "demo", "live", "medley", "remix", "instrumental",
            "take", "outtake", "alternate", "version",
        })

        repaired_indices, edge_repair_swap_log = repair_playlist_edges(
            track_indices=[int(i) for i in final_indices],
            pier_positions=pier_positions,
            eligible_pool=[int(c) for c in candidate_pool_indices],
            bundle=bundle,
            transition_floor=float(cfg.transition_floor),
            centered_cos_floor=float(getattr(cfg, "edge_repair_centered_cos_floor", -0.5)),
            margin=float(getattr(cfg, "edge_repair_margin", 0.05)),
            min_gap=int(getattr(cfg, "min_gap", 6)),  # falls back if not set; min_gap is on outer config
            title_artifact_filter_set=_repair_title_filter,
            variety_guard_enabled=bool(getattr(cfg, "edge_repair_variety_guard_enabled", False)),
            variety_guard_threshold=float(getattr(cfg, "edge_repair_variety_guard_threshold", 0.85)),
            center_transitions=bool(getattr(cfg, "center_transitions", True)),
        )
        # If repair changed any slot, replace final_indices so the reporter
        # uses the repaired playlist for edge_scores.
        if repaired_indices != [int(i) for i in final_indices]:
            final_indices = list(repaired_indices)
```

Note: the variable `pier_seed_indices` is the set of bundle indices for the pier (seed) tracks. If the variable in the local scope at this point is named differently (e.g. `pier_indices` or `seed_indices`), use that name — read the surrounding code to confirm before editing.

- [ ] **Step 3: Expose `edge_repair_swap_log` in the stats**

Further down in the same function, where the final result dict / PierBridgeResult is constructed, add the swap log to the stats:

```python
        "edge_repair_swap_log": edge_repair_swap_log,
```

- [ ] **Step 4: Run the focused tests**

```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_edge_repair.py tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-int -o cache_dir=.pytest-tmp-cache-int
```
Expected: pass. The pipeline goldens should still match because `edge_repair_enabled=False` means repair is skipped entirely.

- [ ] **Step 5: Run the full test suite**

```
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-full -o cache_dir=.pytest-tmp-cache-full
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge_builder.py
git commit -m "feat: integrate edge repair pass into pier_bridge_builder"
```

---

### Task 5: Emit swap log in reporter

**Files:**
- Modify: `src/playlist/reporter.py`
- Modify: `src/playlist_generator.py` (call site)

When the repair runs, the user needs to see what it did. Add a log section that prints one line per swap (and per refusal) immediately after the audit table.

- [ ] **Step 1: Add `emit_edge_repair_log` to reporter.py**

Open `src/playlist/reporter.py`. After `emit_selected_edge_audit` (the function added in v4.1), add:

```python
def emit_edge_repair_log(swap_log: list[dict]) -> None:
    """Log the edge-repair swap log produced by repair_playlist_edges.

    One row per broken edge that was considered: shows whether it was
    swapped, refused-with-reason, or skipped (pier). Quiet when swap_log
    is empty.
    """
    if not swap_log:
        return
    logger.info("=" * 80)
    logger.info("Edge repair pass: %d edge(s) considered", len(swap_log))
    logger.info("=" * 80)
    for entry in swap_log:
        slot = entry.get("slot")
        old_idx = entry.get("old_idx")
        new_idx = entry.get("new_idx")
        reason = entry.get("reason", "?")
        old_worst = entry.get("old_worst")
        new_worst = entry.get("new_worst")
        margin = entry.get("margin_achieved")
        candidates = entry.get("candidates_evaluated", 0)
        old_worst_s = f"{float(old_worst):.3f}" if old_worst is not None else "n/a"
        new_worst_s = f"{float(new_worst):.3f}" if new_worst is not None else "n/a"
        margin_s = f"{float(margin):.3f}" if margin is not None else "n/a"
        if new_idx is None:
            logger.info(
                "Slot %d: REFUSED (%s) — old_idx=%s old_worst=%s candidates_evaluated=%d",
                slot, reason, old_idx, old_worst_s, candidates,
            )
        else:
            logger.info(
                "Slot %d: SWAPPED — old_idx=%s → new_idx=%s, old_worst=%s → new_worst=%s (margin=%s, candidates_evaluated=%d)",
                slot, old_idx, new_idx, old_worst_s, new_worst_s, margin_s, candidates,
            )
```

- [ ] **Step 2: Add the call site in playlist_generator.py**

Open `src/playlist_generator.py`. Find the existing call to `emit_selected_edge_audit` (added in v4.1; search for `emit_selected_edge_audit`). Add the repair log emission just before it, so the repair log appears above the audit table (which then reflects the repaired playlist):

```python
        # Emit edge repair swap log if repair ran.
        _swap_log_data = (
            (self._last_ds_report.get("playlist_stats") or {})
            .get("playlist", {})
            .get("edge_repair_swap_log")
        ) or []
        if _swap_log_data:
            from src.playlist.reporter import emit_edge_repair_log as _emit_repair_log
            _emit_repair_log(_swap_log_data)
```

This goes inside the existing `if bool(getattr(self, "_last_ds_report", {})...emit_selected_edge_audit, False)):` block — or just before that block if you want repair logs to appear unconditionally when repair ran. Place it inside so it only fires when the audit is enabled (clean failure mode: if the audit is off and repair runs, you'd still see the swap_log via the structured stats; only the human-readable INFO log is hidden, which is fine).

Actually, for visibility, put it **outside** the `emit_selected_edge_audit` guard so repair logs appear whenever repair runs, regardless of whether the audit is enabled:

```python
        # Always emit the repair log when repair has run, independent of audit flag.
        _swap_log_data = (
            (self._last_ds_report.get("playlist_stats") or {})
            .get("playlist", {})
            .get("edge_repair_swap_log")
        ) or []
        if _swap_log_data:
            from src.playlist.reporter import emit_edge_repair_log as _emit_repair_log
            _emit_repair_log(_swap_log_data)
```

Place this just above the `if bool(getattr(self, "_last_ds_report", {})...emit_selected_edge_audit, False)):` audit block.

- [ ] **Step 3: Repeat for the seeded-mode call site**

The audit was also wired into the seeded mode generator in `playlist_generator.py` (the call around `Auto: Seeded`). Find the existing audit block there and add the same `_swap_log_data` block above it.

- [ ] **Step 4: Run the full test suite**

```
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-full -o cache_dir=.pytest-tmp-cache-full
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/reporter.py src/playlist_generator.py
git commit -m "feat: emit edge repair swap log in reporter"
```

---

### Task 6: Documentation + commented config example

**Files:**
- Modify: `config.yaml`
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md`
- Modify: `docs/CHANGELOG.md`

- [ ] **Step 1: Add commented opt-in block to config.yaml**

Open `config.yaml`. Find the `pier_bridge:` block. After the existing `emit_selected_edge_audit:` line, add:

```yaml
      # Edge repair pass (opt-in, default OFF).
      # After the beam completes, identifies edges where T < transition_floor
      # OR T_centered_cos < -0.5 (catastrophic by either metric). For each
      # broken edge, attempts to swap the destination track (never a pier)
      # with a candidate from the existing eligible pool. Swap is only
      # applied if BOTH new edges clear transition_floor AND the worst-of-two
      # improves by at least `margin`. Otherwise the original is kept.
      # Always filters demos/live/medleys/etc. from replacements regardless
      # of the title_artifact_penalty knob.
      #
      # edge_repair:
      #   enabled: true
      #   centered_cos_floor: -0.5
      #   margin: 0.05
      #   variety_guard:
      #     enabled: false        # recommended off for narrow/strict modes
      #     threshold: 0.85       # candidates with S > threshold to predecessor excluded
```

- [ ] **Step 2: Add "Knob 4" to the tuning recipe**

Open `docs/PLAYLIST_ORDERING_TUNING.md`. After "Knob 3: Worst-edge lexicographic beam objective", add:

```markdown
---

## Knob 4: Post-beam edge repair (catches catastrophic edges that slip through)

**Use when:** The audit shows occasional below-floor edges (T < 0.20) or
catastrophically anti-correlated edges (T_centered_cos < −0.5) that the
beam couldn't avoid, and the user perceives these specific transitions
as broken.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      edge_repair:
        enabled: true
        centered_cos_floor: -0.5
        margin: 0.05
        variety_guard:
          enabled: false
          threshold: 0.85
```

**What it does:** Runs after the beam, identifies edges where T < `transition_floor`
or T_centered_cos < `centered_cos_floor`. For each broken edge, attempts to swap
the *destination* track (never a pier) with a candidate from the eligible pool.
A swap is only applied when:

1. Both resulting edges clear the absolute transition floor.
2. The worst-of-the-two-edges improves by at least `margin`.
3. The candidate doesn't violate min_gap or share an artist with adjacent slots.
4. The candidate's title doesn't contain artifact flags (demo/live/medley/etc.).
5. (Optional) The candidate isn't a sonic clone of the predecessor (variety guard).

If no candidate clears the bar, the original track stays. Repair is opt-out:
do-no-harm is the default behavior.

**Tuning:**
- `centered_cos_floor: -0.5` catches edges where T might be just above floor
  (e.g. 0.21) but T_centered_cos shows the underlying acoustic edge is severely
  anti-correlated. Raise to -0.3 to fire on more edges; lower to -0.7 for stricter.
- `margin: 0.05` is the worst-of-two improvement required. Raise to 0.10 to make
  repair refuse swaps that produce only marginal improvements.
- `variety_guard.enabled: true` adds a sonic-clone filter — recommended only for
  `dynamic` or `discover` modes where homogenization is a risk. Leave off for
  `narrow`/`strict` modes.

**Reading the repair log:** when repair runs, the playlist log includes a section like:

```
Edge repair pass: 2 edge(s) considered
Slot 5: SWAPPED — old_idx=1234 → new_idx=5678, old_worst=0.112 → new_worst=0.751 (margin=0.639, candidates_evaluated=234)
Slot 17: REFUSED (no_candidate_meets_margin) — old_idx=4321 old_worst=0.183 candidates_evaluated=187
```

Use this to confirm repair only swapped where the math justified it. If you see
many `REFUSED (no_candidate_clears_floor)` entries, the eligible pool may be too
narrow; increase `per_cluster_candidate_pool_size` or `genre_neighbor_pool_size`
(artist mode) to give repair more material.
```

- [ ] **Step 3: Add CHANGELOG entry**

Open `docs/CHANGELOG.md`. Insert at the top of the file, above the existing v4.1.0 entry:

```markdown
## v4.2.0 - Post-beam Edge Repair Pass

**Release Date:** 2026-05-21
**Branch:** `codex-artist-mode-genre-conflict`
**Focus:** Catch catastrophic edges that slip past the beam, with strict do-no-harm swap semantics

### Highlights

- **New opt-in repair pass** (`pier_bridge.edge_repair.enabled`) runs after the beam search and before edge-score reporting. Identifies edges where `T < transition_floor` or `T_centered_cos < -0.5` (catastrophic by either metric) and attempts a single-track swap for each broken slot.
- **Do-no-harm guarantees:** a swap is only applied if both resulting adjacent edges clear the absolute floor *and* the worst-of-two improves by at least `margin` (default 0.05). If no candidate clears the bar, the original track stays.
- **Piers are never replaced.** User-chosen seeds remain anchor points.
- **Title-artifact filter is unconditional** during repair: `demo`, `live`, `medley`, `remix`, `instrumental`, `take`, `outtake`, `alternate`, `version` are excluded from replacements regardless of the soft `title_artifact_penalty` knob.
- **Optional variety guard** refuses candidates that are sonic clones of the predecessor (`S > 0.85` by default). Recommended off for narrow/strict modes; useful for dynamic/discover modes where homogenization is a risk.
- **Swap log is emitted** alongside the playlist report whenever repair runs, showing per-slot decisions (swapped vs. refused with reason) and the achieved margin.

### Configuration

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      edge_repair:
        enabled: true
        centered_cos_floor: -0.5
        margin: 0.05
        variety_guard:
          enabled: false
          threshold: 0.85
```

See `docs/PLAYLIST_ORDERING_TUNING.md` Knob 4 for the full tuning recipe.

### Tests

Adds `tests/unit/test_edge_repair.py` (10 focused tests) covering: no-op on clean playlists, swap with valid candidate, refusal on margin failure, pier protection, centered-cos trigger, title-artifact filter, min_gap exclusion, maximin selection, multi-edge independent repair, variety guard.

---

```

- [ ] **Step 4: Run full test suite**

```
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-full -o cache_dir=.pytest-tmp-cache-full
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add config.yaml docs/PLAYLIST_ORDERING_TUNING.md docs/CHANGELOG.md
git commit -m "docs: edge repair pass tuning recipe + changelog entry"
```

---

## Self-Review

**1. Spec coverage:**
- Trigger `T < 0.20` or `T_centered_cos < -0.5` → Task 1 (`repair_playlist_edges` body), Task 2 (defaults), Task 6 (docs).
- Pier rule (never replace) → Task 1 step 4 (pier check + "pier" reason), Task 1 step 2 test `test_does_not_replace_pier_slot`.
- Variety guard default OFF, opt-in → Task 1 implementation + Task 2 config default + Task 6 docs caveat.
- Title-artifact filter automatic → Task 4 hardcodes `_repair_title_filter` regardless of the existing soft-penalty knob.
- Margin δ = 0.05 → Task 2 config default + Task 1 implementation uses the param.
- Swap log emission → Task 5.
- Do-no-harm semantics (original kept on refusal) → Task 1 implementation explicitly returns original index when no candidate qualifies.
- Backward-compat default-off → Task 2 sets `edge_repair_enabled: bool = False`.

**2. Placeholder scan:** All steps have explicit code blocks or commands. No TBDs, no "implement later", no "similar to Task N", no "handle edge cases." The integration code in Task 4 includes a clear pattern but the variable name `pier_seed_indices` is flagged in step 2 as needing verification against the surrounding code (this is an instruction to the implementer, not a placeholder).

**3. Type consistency:**
- `repair_playlist_edges` returns `Tuple[List[int], List[Dict[str, Any]]]` — used identically in Task 1 tests, Task 4 integration, Task 5 reporter consumption.
- `swap_log` entry shape is consistent: keys `slot`, `old_idx`, `new_idx`, `old_T_in`, `old_T_out`, `new_T_in`, `new_T_out`, `old_worst`, `new_worst`, `margin_achieved`, `reason`, `candidates_evaluated` — used identically in Task 1 (creation), Task 5 (consumption).
- Config field names `edge_repair_enabled`, `edge_repair_centered_cos_floor`, `edge_repair_margin`, `edge_repair_variety_guard_enabled`, `edge_repair_variety_guard_threshold` consistent across Tasks 2, 3, 4.
- The `_repair_title_filter` set in Task 4 includes the same flag names used in `src/playlist/title_quality.py:detect_title_artifacts` (`demo`, `live`, `medley`, `remix`, `instrumental`, `take`, `outtake`, `alternate`, `version`) — these are the flags returned by that function (confirmed by reading `title_quality.py`).
