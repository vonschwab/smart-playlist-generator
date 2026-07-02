# Artist-pier scarcity + mini-pier spacing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In artist mode, (1) re-admit recently-played seed-artist tracks as piers only when the fresh pool can't fill `target_piers`, and (2) distribute mini-pier waypoints evenly across seed-gaps instead of dumping them all in gap 0.

**Architecture:** Change 2 is a self-contained fix to `plan_pier_sequence`'s split-target selection. Change 1 adds a pure helper (`seed_eligibility.py`) that computes the relaxed recency-exclusion set, wired into the artist-style clustering call in `playlist_generator.py`. Tasks are independent and land in this order: 2 (isolated), then 1's helper, then 1's wiring.

**Tech Stack:** Python 3.11+, numpy, pytest.

## Global Constraints

- Scope: artist mode only. No change to seeds/diverse modes, `min_gap`, or the beam.
- Freshness stays a hard gate on the interior/bridge candidate pool — do not touch it.
- New behavior is the live default; `exclude_seed_tracks_from_recency` remains the rollback lever (Layer 4).
- Run pytest bounded, never piped: `python -m pytest <path> -q`.
- Spec: `docs/superpowers/specs/2026-07-01-artist-pier-scarcity-and-spacing-design.md`.

---

## Task 1: Even mini-pier distribution in `plan_pier_sequence`

**Files:**
- Modify: `src/playlist/pier_bridge/mini_pier_select.py` (`plan_pier_sequence`, ~L60-95)
- Test: `tests/unit/test_mini_pier_spacing.py` (create)

**Interfaces:**
- `plan_pier_sequence(ordered_seeds, total_tracks, candidate_indices, X_full_norm, *, max_interior, margin, k_broad, exclude_base=frozenset(), max_waypoints=8) -> list[int]` — signature unchanged; only the split-target selection changes.

- [ ] **Step 1: Write the failing test** (monkeypatch `select_waypoint` so we test *distribution*, not waypoint quality)

```python
# tests/unit/test_mini_pier_spacing.py
import numpy as np
from src.playlist.pier_bridge import mini_pier_select
from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence


def _waypoints_per_gap(seeds, piers):
    """Count inserted (non-seed) piers between each consecutive pair of seeds."""
    seed_pos = [piers.index(s) for s in seeds]
    return [seed_pos[i + 1] - seed_pos[i] - 1 for i in range(len(seed_pos) - 1)]


def test_waypoints_distribute_across_gaps(monkeypatch):
    seeds = list(range(10))               # 10 seeds -> 9 gaps
    total_tracks = 100                    # interior 90, max_interior 5 -> 8 waypoints
    pool = list(range(100, 400))
    X = np.zeros((400, 4), dtype=np.float32)
    counter = {"n": 0}

    def fake_select_waypoint(a, b, cand, Xn, *, margin, k_broad, exclude):
        # deterministic fresh index each call, never a seed/existing pier
        for c in pool:
            if c not in exclude:
                counter["n"] += 1
                return int(c)
        return None

    monkeypatch.setattr(mini_pier_select, "select_waypoint", fake_select_waypoint)
    piers = plan_pier_sequence(
        seeds, total_tracks, pool, X,
        max_interior=5, margin=0.12, k_broad=150, max_waypoints=25,
    )
    per_gap = _waypoints_per_gap(seeds, piers)
    assert sum(per_gap) == 8                     # 8 waypoints inserted
    assert max(per_gap) <= 2                     # never all in one gap (bug was 8 in gap 0)
    assert sum(1 for g in per_gap if g >= 1) >= 7  # spread across >=7 of the 9 gaps
```

- [ ] **Step 2: Run it, confirm it FAILS** — `python -m pytest tests/unit/test_mini_pier_spacing.py -q` → currently `max(per_gap)` is 8 (all in gap 0).

- [ ] **Step 3: Fix the split-target selection.** Replace the loop body's `seg = int(np.argmax(lengths))` selection (and the single-`break`-on-None) with fewest-waypoints-per-gap distribution:

```python
    piers = [int(s) for s in ordered_seeds]
    used = set(piers) | {int(e) for e in exclude_base}
    num_seed_gaps = len(piers) - 1
    if num_seed_gaps < 1:
        return piers
    wp_per_gap = [0] * num_seed_gaps      # waypoints inserted per ORIGINAL seed-gap
    seg_gap = list(range(num_seed_gaps))  # current-segment index -> its origin seed-gap
    for _ in range(int(max_waypoints)):
        num_seg = len(piers) - 1
        interior = int(total_tracks) - len(piers)
        if num_seg < 1 or interior < 1:
            break
        lengths = _even_split_lengths(interior, num_seg)
        if int(max(lengths)) <= int(max_interior):
            break
        # Round-robin across ORIGIN seed-gaps: split the gap with the fewest waypoints
        # so far (ties -> leftmost). Even-split makes every sub-segment ~equal length
        # regardless of which gap we split, so the split-choice only controls seed
        # spacing -- distribute it, don't (as the old argmax did) hammer segment 0.
        # NB: do NOT pre-filter to `lengths[i] > max_interior`. `_even_split_lengths`
        # piles the remainder onto the EARLIEST segments, so at the tail only gap-0
        # segments look "over-long" and waypoints re-concentrate there (the bug found
        # in review). `max(lengths)` above is the stop condition; selection ranges over
        # ALL segments.
        order = sorted(range(num_seg), key=lambda i: (wp_per_gap[seg_gap[i]], i))
        seg = wp = None
        for i in order:
            cand = select_waypoint(
                piers[i], piers[i + 1], candidate_indices, X_full_norm,
                margin=margin, k_broad=k_broad, exclude=frozenset(used),
            )
            if cand is not None:
                seg, wp = i, int(cand)
                break
        if wp is None:
            break
        g = seg_gap[seg]
        piers.insert(seg + 1, wp)
        seg_gap.insert(seg + 1, g)
        wp_per_gap[g] += 1
        used.add(wp)
    return piers
```

- [ ] **Step 4: Run the test + the existing mini-pier tests** — `python -m pytest tests/unit/test_mini_pier_spacing.py tests/unit/ -q -k "mini_pier or pier_sequence"` → PASS.

- [ ] **Step 5: Commit** — `git add src/playlist/pier_bridge/mini_pier_select.py tests/unit/test_mini_pier_spacing.py && git commit` (message: `fix(mini-pier): distribute waypoints across seed-gaps (was all gap 0)`).

---

## Task 2: Pure seed-eligibility helper (scarcity re-admission)

**Files:**
- Create: `src/playlist/seed_eligibility.py`
- Test: `tests/unit/test_seed_eligibility.py` (create)

**Interfaces:**
- Produces: `seed_recency_exclusion_for_presence(artist_track_ids, recency_excluded_ids, target_piers, *, readmit_rank=None) -> set[str]` — returns the recency-exclusion set to actually apply (a subset of `recency_excluded_ids`); only seed-artist tracks are ever re-admitted.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_seed_eligibility.py
from src.playlist.seed_eligibility import seed_recency_exclusion_for_presence


def test_no_relaxation_when_enough_fresh():
    artist = {"a", "b", "c", "d", "e"}
    excluded = {"a"}                       # 4 fresh >= target 4
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=4)
    assert out == {"a"}


def test_relax_up_to_target_only():
    artist = {"a", "b", "c", "d", "e"}     # 5 total
    excluded = {"a", "b", "c"}             # 2 fresh, target 4 -> re-admit 2
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=4)
    assert len(out) == 1                    # exactly one stays excluded
    assert out < excluded


def test_never_exceed_catalog():
    artist = {"a", "b", "c"}
    excluded = {"a", "b", "c"}             # 0 fresh, target 10 -> re-admit all 3
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=10)
    assert out == set()


def test_readmit_rank_prefers_listed_first():
    artist = {"a", "b", "c", "d"}
    excluded = {"a", "b", "c"}             # target 2, 1 fresh (d) -> re-admit 1
    out = seed_recency_exclusion_for_presence(
        artist, excluded, target_piers=2, readmit_rank=["c", "b", "a"])
    assert "c" not in out                   # highest-ranked re-admitted
    assert out == {"a", "b"}


def test_only_touches_seed_artist_tracks():
    artist = {"a", "b"}
    excluded = {"a", "x", "y"}             # x,y are other artists
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=2)
    assert {"x", "y"} <= out                # other artists stay excluded
```

- [ ] **Step 2: Run, confirm FAIL** — `python -m pytest tests/unit/test_seed_eligibility.py -q` → import error.

- [ ] **Step 3: Implement**

```python
# src/playlist/seed_eligibility.py
"""Scarcity-gated freshness for artist-mode seed piers.

Freshness normally removes recently-played tracks from seed selection. For the
SEED artist that starves the piers when the catalog is small (most-played =
most-popular = recently-played). This re-admits the seed artist's own recently-
played tracks, and ONLY as many as needed to keep >= target_piers eligible.
Other artists' recency exclusions are untouched. See
docs/superpowers/specs/2026-07-01-artist-pier-scarcity-and-spacing-design.md.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence


def seed_recency_exclusion_for_presence(
    artist_track_ids: Iterable[str],
    recency_excluded_ids: Iterable[str],
    target_piers: int,
    *,
    readmit_rank: Optional[Sequence[str]] = None,
) -> set[str]:
    """Return the recency-exclusion set to apply, relaxed only enough that at least
    ``target_piers`` of the seed artist's tracks stay eligible.

    ``readmit_rank``: seed-artist track ids in the order to re-admit first (e.g.
    most-popular first when Popular Seeds is on). Ids absent from the excluded-artist
    set are ignored; excluded artist tracks not in the rank are re-admitted last.
    """
    excluded = {str(t) for t in recency_excluded_ids}
    artist = {str(t) for t in artist_track_ids}
    excluded_artist = excluded & artist
    fresh_count = len(artist) - len(excluded_artist)
    shortfall = int(target_piers) - fresh_count
    if shortfall <= 0:
        return excluded
    if readmit_rank:
        ranked = [str(t) for t in readmit_rank if str(t) in excluded_artist]
        ranked += [t for t in excluded_artist if t not in set(ranked)]
    else:
        ranked = sorted(excluded_artist)  # deterministic
    readmit = set(ranked[:shortfall])
    return excluded - readmit
```

- [ ] **Step 4: Run, confirm PASS** — `python -m pytest tests/unit/test_seed_eligibility.py -q`.

- [ ] **Step 5: Commit** — `git add src/playlist/seed_eligibility.py tests/unit/test_seed_eligibility.py && git commit` (`feat(seed): scarcity-gated recency re-admission helper (artist presence)`).

---

## Task 3: Wire the helper into the artist-style clustering path

**Files:**
- Modify: `src/playlist_generator.py` (the `excluded_track_ids=` argument of the `cluster_artist_tracks` call, ~L1786; helper call inserted just before it).

**Interfaces:**
- Consumes: `seed_recency_exclusion_for_presence` (Task 2); `target_pier_count` and `_artist_indices_in_bundle` (already in scope at ~L1731/1735); `popularity_values` (already set ~L1759-1776, aligned to `bundle.track_ids`, or `None`).

- [ ] **Step 1: Insert the relaxation just before the `cluster_artist_tracks` call.** Currently the call passes `excluded_track_ids=seed_recency_excluded_ids if exclude_seed_tracks_from_recency else None`. Replace that argument's value with a pre-computed `_relaxed_excluded`:

```python
                # Scarcity-gated freshness: re-admit the seed artist's recently-played
                # tracks only as needed to fill target_piers (Artist Presence wins over
                # freshness for the seed artist alone). Freshness stays hard elsewhere.
                _relaxed_excluded = None
                if exclude_seed_tracks_from_recency and seed_recency_excluded_ids:
                    from src.playlist.seed_eligibility import seed_recency_exclusion_for_presence
                    _artist_ids = {
                        str(bundle.track_ids[i]) for i in _artist_indices_in_bundle(
                            bundle, artist_name, include_collaborations=include_collaborations)
                    }
                    _rank = None
                    if popularity_values is not None:
                        _idxs = _artist_indices_in_bundle(
                            bundle, artist_name, include_collaborations=include_collaborations)
                        _rank = [
                            str(bundle.track_ids[i]) for i in sorted(
                                _idxs, key=lambda i: float(popularity_values[i]), reverse=True)
                        ]
                    _relaxed_excluded = seed_recency_exclusion_for_presence(
                        _artist_ids, seed_recency_excluded_ids, target_pier_count,
                        readmit_rank=_rank,
                    ) or None
                    _readmitted = len(seed_recency_excluded_ids) - len(_relaxed_excluded or set())
                    if _readmitted:
                        logger.info(
                            "Seed presence: re-admitted %d recently-played %s track(s) to fill "
                            "target_piers=%d (fresh pool was short)",
                            _readmitted, artist_name, target_pier_count,
                        )
```

Then change the clustering call argument from
`excluded_track_ids=seed_recency_excluded_ids if exclude_seed_tracks_from_recency else None`
to
`excluded_track_ids=_relaxed_excluded`.

- [ ] **Step 2: Manual verification via the real generation path** (harness can't do artist-style + Last.fm). Regenerate the Subsonic Eye playlist (restart worker first) and confirm in the log: `Seed presence: re-admitted 4 recently-played Subsonic Eye track(s)`, `total=10` usable for clustering, `Mini-piers: 0 waypoint(s) inserted`, and Subsonic Eye tracks spaced ~every 5 across all 50 (not bunched), with the piers now on the artist's actual top tracks.

- [ ] **Step 3: Commit** — `git add src/playlist_generator.py && git commit` (`feat(seed): fill artist piers under scarcity by relaxing freshness (presence>freshness)`).

---

## Notes / deviations from spec

- **Re-admission fallback ordering:** spec said "stalest-first" when Popular Seeds is off. Per-track recency timestamps aren't threaded to this point (only the excluded *set* is), so the fallback is a deterministic stable sort. Popularity ordering (the case the user cares about) is honored when `popularity_values` is present. Timestamp-based stalest-first is a future refinement if wanted.
- **No config knob added:** reuses `exclude_seed_tracks_from_recency` as the rollback (spec).
