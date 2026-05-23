# Track Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the listener replace any single track in a generated playlist, either by searching for a specific track manually or by picking from one of four flavors of auto-generated suggestions: **Best Match** (closest neighbor under current weights), **Different Pace**, **Different Genre**, or **Different Sound**.

**Architecture:** A replacement is a local, single-position search between two known endpoints (`prev` and `next` from the current playlist). All four auto modes share the same candidate scoring framework — they differ only in a secondary re-ranking step that maximizes divergence on one specific axis (rhythm/BPM, genre signature, or timbre/harmony) while still requiring high transition quality. The candidate pool, transition metric, and artist-identity constraints all come from the bundle the worker already has in memory from the most recent `generate_playlist` call. No artifact reload, no full pipeline rerun.

**Tech Stack:** Python 3.11, numpy, PySide6, pytest. Worker uses the existing NDJSON IPC protocol. New backend module: `src/playlist/replacement.py`. New GUI dialog: `src/playlist_gui/widgets/replace_dialog.py`.

**Out of scope (deferred to v2):**
- Multi-track replacement (consecutive sub-segment re-run, or batch non-consecutive)
- Undo / revert
- Persisted replacement history
- "Hint" or "vibe" text input (e.g. "more uplifting") that biases suggestion scoring with a free-text prompt
- Bulk regenerate ("regenerate worst 3 edges automatically")

---

## The four suggestion modes

All four share the same base: candidates must clear the existing admission gates (sonic / genre / pace / BPM), must not duplicate any track already in the current playlist, must not duplicate the artist of `prev` or `next`, and must produce viable transitions on both sides (`T_prev` and `T_next` ≥ `transition_floor`). Within that eligible set, scoring differs:

| Mode | Primary score | Secondary re-ranking |
|---|---|---|
| **Best Match** | `mean(T_prev, T_next)` (or harmonic mean — same as beam) | none — straight ranking |
| **Different Pace** | `T_prev`, `T_next` must clear floor; filter to top K=50 | rerank by `pace_divergence(c, current)` descending |
| **Different Genre** | same | rerank by `1 − genre_cosine(c, current)` (IDF-weighted) |
| **Different Sound** | same | rerank by `1 − color_cosine(c, current)` where color = timbre + harmony slice of `X_sonic` |

`pace_divergence` is `|log2(c.perceptual_bpm / current.perceptual_bpm)|` when both have BPM data, else falls back to `1 − rhythm_cosine(c, current)`.

This formulation guarantees that "Different X" suggestions still flow well between `prev` and `next` — we never sacrifice transition quality for novelty. The user gets meaningful variation along the chosen axis without breaking the playlist's arc.

Each mode returns top N = 10 candidates with full per-candidate metrics (artist, title, T_prev, T_next, BPM, top-3 genres) so the GUI can show them in a table for the listener to browse before picking.

---

## File structure

**New files:**
- `src/playlist/replacement.py` — scoring engine: `find_replacement_candidates`, mode dispatch, divergence functions
- `src/playlist_gui/widgets/replace_dialog.py` — Qt dialog with Search + 4 suggestion tabs
- `tests/unit/test_replacement_scoring.py`
- `tests/unit/test_replacement_modes.py`
- `tests/unit/test_replace_dialog.py`

**Modified files:**
- `src/playlist_gui/worker.py` — `handle_find_replacement_suggestions` command + last-bundle cache
- `src/playlist_gui/worker_client.py` — client-side method
- `src/playlist_gui/widgets/track_table.py` — context menu entry "Replace track…"
- `src/playlist_gui/main_window.py` — wire context menu → dialog → swap
- `tests/unit/test_worker_protocol.py` — protocol coverage for the new command
- `docs/PLAYLIST_ORDERING_TUNING.md` — short section on the replacement modes

**Read-only references:**
- `src/playlist/transition_metrics.py` — reuse `TransitionMetricContext`, `score_transition_edge`
- `src/playlist/bpm_axis.py` — reuse `bpm_log_distance`
- `src/playlist/sonic_axes.py` — reuse axis slicing for color vector
- `src/playlist/genre_idf.py` — IDF weights for genre divergence

---

## Task 1: Replacement scoring module (pure functions)

**Files:**
- Create: `src/playlist/replacement.py`
- Create: `tests/unit/test_replacement_scoring.py`

The module exposes a single entry point `find_replacement_candidates(...)` that takes the playlist state, position, mode, and a precomputed context object containing the candidate pool indices, metric context, BPM arrays, genre matrix, and IDF weights. Returns a ranked list of `ReplacementCandidate` dicts.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_replacement_scoring.py
import numpy as np
import pytest
from src.playlist.replacement import (
    ReplacementContext,
    find_replacement_candidates,
    SUPPORTED_MODES,
)


def _ctx(N=20, dim=32):
    """Synthetic context with predictable embeddings."""
    rng = np.random.default_rng(42)
    X_sonic = rng.standard_normal((N, dim))
    X_genre = np.eye(N, 8)  # one-hot genre per track
    perceptual_bpm = np.linspace(60.0, 180.0, N)
    return ReplacementContext(
        X_sonic=X_sonic,
        X_full=X_sonic,
        X_start=X_sonic,
        X_end=X_sonic,
        X_mid=X_sonic,
        X_genre_smoothed=X_genre,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
        track_ids=np.array([f"t{i}" for i in range(N)], dtype=object),
        artist_keys=np.array([f"a{i // 2}" for i in range(N)], dtype=object),
        candidate_pool_indices=np.arange(N),
        tower_pca_dims=(8, 16, 8),
        idf_weights=None,
    )


def test_supported_modes_are_four():
    assert SUPPORTED_MODES == ("best", "different_pace", "different_genre", "different_sound")


def test_best_mode_returns_top_k():
    ctx = _ctx()
    result = find_replacement_candidates(
        prev_idx=2, next_idx=10, current_idx=5,
        playlist_indices=[0, 2, 5, 10, 15],
        ctx=ctx, mode="best", top_k=5,
    )
    assert len(result) <= 5
    for c in result:
        assert "track_id" in c
        assert "t_prev" in c
        assert "t_next" in c


def test_excludes_current_track():
    ctx = _ctx()
    result = find_replacement_candidates(
        prev_idx=2, next_idx=10, current_idx=5,
        playlist_indices=[0, 2, 5, 10, 15],
        ctx=ctx, mode="best", top_k=20,
    )
    assert all(c["index"] != 5 for c in result)


def test_excludes_playlist_tracks():
    ctx = _ctx()
    playlist = [0, 2, 5, 10, 15]
    result = find_replacement_candidates(
        prev_idx=2, next_idx=10, current_idx=5,
        playlist_indices=playlist,
        ctx=ctx, mode="best", top_k=20,
    )
    returned_idx = {c["index"] for c in result}
    assert returned_idx.isdisjoint(set(playlist))


def test_excludes_neighbor_artist():
    """Candidate sharing artist with prev or next is filtered."""
    ctx = _ctx()
    # artist_keys: t0,t1 share a0; t2,t3 share a1; t4,t5 share a2; ...
    # prev_idx=2 (a1) and next_idx=4 (a2). Neither a1-sibling (t3) nor a2-sibling (t5) should appear.
    result = find_replacement_candidates(
        prev_idx=2, next_idx=4, current_idx=3,
        playlist_indices=[0, 2, 3, 4, 8],
        ctx=ctx, mode="best", top_k=20,
    )
    returned_idx = {c["index"] for c in result}
    assert 3 not in returned_idx  # current
    assert 5 not in returned_idx  # shares artist with next (idx 4)


def test_different_pace_diverges_from_current_bpm():
    ctx = _ctx()
    # current at idx 5 has perceptual_bpm = ~100 (linspace 60..180, N=20)
    # best mode would pick the highest-T candidate
    # different_pace should pick candidates whose BPM is farthest from current
    current_idx = 10  # bpm ≈ 120
    best = find_replacement_candidates(
        prev_idx=2, next_idx=18, current_idx=current_idx,
        playlist_indices=[2, current_idx, 18],
        ctx=ctx, mode="best", top_k=5,
    )
    pace = find_replacement_candidates(
        prev_idx=2, next_idx=18, current_idx=current_idx,
        playlist_indices=[2, current_idx, 18],
        ctx=ctx, mode="different_pace", top_k=5,
    )
    # On average, "different_pace" should produce greater BPM divergence
    current_bpm = ctx.perceptual_bpm[current_idx]
    best_avg_div = np.mean([abs(np.log2(c["perceptual_bpm"] / current_bpm)) for c in best])
    pace_avg_div = np.mean([abs(np.log2(c["perceptual_bpm"] / current_bpm)) for c in pace])
    assert pace_avg_div > best_avg_div


def test_unknown_mode_raises():
    ctx = _ctx()
    with pytest.raises(ValueError, match="Unknown replacement mode"):
        find_replacement_candidates(
            prev_idx=0, next_idx=5, current_idx=2,
            playlist_indices=[0, 2, 5], ctx=ctx, mode="bogus", top_k=5,
        )


def test_empty_pool_returns_empty():
    ctx = _ctx()
    ctx_empty = ReplacementContext(
        **{**ctx.__dict__, "candidate_pool_indices": np.array([], dtype=int)},
    )
    result = find_replacement_candidates(
        prev_idx=0, next_idx=5, current_idx=2,
        playlist_indices=[0, 2, 5], ctx=ctx_empty, mode="best", top_k=10,
    )
    assert result == []
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python -m pytest tests/unit/test_replacement_scoring.py -v`

- [ ] **Step 3: Implement `src/playlist/replacement.py`**

```python
"""Single-position replacement: find candidates that fit between prev and next.

Four modes:
- best:             rank by transition quality alone
- different_pace:   filter by transition, re-rank by pace divergence from current
- different_genre:  filter by transition, re-rank by genre divergence
- different_sound:  filter by transition, re-rank by timbre+harmony divergence

All filters apply: candidate must not duplicate any playlist track or share
an artist with either neighbor, and must clear minimum transition quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.playlist.sonic_axes import extract_axis_vectors, axis_cosine_similarity
from src.playlist.bpm_axis import bpm_log_distance
from src.playlist.transition_metrics import TransitionMetricContext, score_transition_edge


SUPPORTED_MODES = ("best", "different_pace", "different_genre", "different_sound")
DEFAULT_T_MIN = 0.20  # candidates below this are dropped before re-ranking
DEFAULT_FILTER_K = 50  # for divergence modes, re-rank within top-K by T


@dataclass(frozen=True)
class ReplacementContext:
    X_sonic: np.ndarray
    X_full: np.ndarray
    X_start: Optional[np.ndarray]
    X_end: Optional[np.ndarray]
    X_mid: Optional[np.ndarray]
    X_genre_smoothed: np.ndarray
    perceptual_bpm: Optional[np.ndarray]
    tempo_stability: Optional[np.ndarray]
    track_ids: np.ndarray
    artist_keys: np.ndarray
    candidate_pool_indices: np.ndarray
    tower_pca_dims: Tuple[int, int, int]
    idf_weights: Optional[np.ndarray] = None
    transition_metric_context: Optional[TransitionMetricContext] = None
    transition_floor: float = DEFAULT_T_MIN


def _eligible_candidate_indices(
    ctx: ReplacementContext,
    *,
    prev_idx: int,
    next_idx: int,
    current_idx: int,
    playlist_indices: Sequence[int],
) -> np.ndarray:
    """Return candidate indices passing all hard filters."""
    pool = np.asarray(ctx.candidate_pool_indices, dtype=int)
    if pool.size == 0:
        return pool

    excluded = set(int(i) for i in playlist_indices) | {int(current_idx)}
    prev_artist = str(ctx.artist_keys[int(prev_idx)])
    next_artist = str(ctx.artist_keys[int(next_idx)])

    keep = []
    for idx in pool:
        i = int(idx)
        if i in excluded:
            continue
        artist = str(ctx.artist_keys[i])
        if artist == prev_artist or artist == next_artist:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def _transition_quality(
    ctx: ReplacementContext,
    *,
    prev_idx: int,
    cand_idx: int,
    next_idx: int,
) -> Tuple[float, float]:
    """Return (T_prev, T_next) for the candidate."""
    if ctx.transition_metric_context is not None:
        t_prev = float(score_transition_edge(ctx.transition_metric_context, int(prev_idx), int(cand_idx))["T"])
        t_next = float(score_transition_edge(ctx.transition_metric_context, int(cand_idx), int(next_idx))["T"])
        return t_prev, t_next
    # Fallback: cosine of normalized full vectors
    def _cos(a_idx: int, b_idx: int) -> float:
        a = ctx.X_full[a_idx]
        b = ctx.X_full[b_idx]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)
    return _cos(int(prev_idx), int(cand_idx)), _cos(int(cand_idx), int(next_idx))


def _genre_divergence(
    ctx: ReplacementContext,
    *,
    cand_idx: int,
    current_idx: int,
) -> float:
    """1 - cosine of genre vectors (optionally IDF-weighted)."""
    a = np.asarray(ctx.X_genre_smoothed[int(current_idx)], dtype=float)
    b = np.asarray(ctx.X_genre_smoothed[int(cand_idx)], dtype=float)
    if ctx.idf_weights is not None:
        w = np.asarray(ctx.idf_weights, dtype=float)
        a = a * w
        b = b * w
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)


def _pace_divergence(
    ctx: ReplacementContext,
    *,
    cand_idx: int,
    current_idx: int,
) -> float:
    """|log2 BPM ratio| when both have BPM; falls back to 1 - rhythm cosine."""
    if (
        ctx.perceptual_bpm is not None
        and not np.isnan(ctx.perceptual_bpm[int(cand_idx)])
        and not np.isnan(ctx.perceptual_bpm[int(current_idx)])
    ):
        return float(bpm_log_distance(
            float(ctx.perceptual_bpm[int(cand_idx)]),
            float(ctx.perceptual_bpm[int(current_idx)]),
        ))
    # Fallback: rhythm-cosine divergence
    axes = extract_axis_vectors(ctx.X_sonic, tower_pca_dims=ctx.tower_pca_dims)
    r = axes["rhythm"]
    return float(1.0 - axis_cosine_similarity(r[int(cand_idx)], r[int(current_idx)])[0, 0])


def _sound_divergence(
    ctx: ReplacementContext,
    *,
    cand_idx: int,
    current_idx: int,
) -> float:
    """1 - color cosine (timbre + harmony slice)."""
    axes = extract_axis_vectors(ctx.X_sonic, tower_pca_dims=ctx.tower_pca_dims)
    color = axes["color"]
    return float(1.0 - axis_cosine_similarity(color[int(cand_idx)], color[int(current_idx)])[0, 0])


def find_replacement_candidates(
    *,
    prev_idx: int,
    next_idx: int,
    current_idx: int,
    playlist_indices: Sequence[int],
    ctx: ReplacementContext,
    mode: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown replacement mode: '{mode}'. Supported: {SUPPORTED_MODES}")

    eligible = _eligible_candidate_indices(
        ctx,
        prev_idx=prev_idx, next_idx=next_idx,
        current_idx=current_idx, playlist_indices=playlist_indices,
    )
    if eligible.size == 0:
        return []

    # Score every eligible candidate by transition quality first
    scored: List[Tuple[int, float, float, float]] = []
    for idx in eligible:
        t_prev, t_next = _transition_quality(ctx, prev_idx=prev_idx, cand_idx=int(idx), next_idx=next_idx)
        mean_t = 0.5 * (t_prev + t_next)
        if mean_t < float(ctx.transition_floor):
            continue
        scored.append((int(idx), t_prev, t_next, mean_t))

    if not scored:
        return []

    # Sort by transition quality
    scored.sort(key=lambda r: r[3], reverse=True)

    if mode == "best":
        chosen = scored[:top_k]
    else:
        # Re-rank top-K by axis-specific divergence
        pool = scored[:max(DEFAULT_FILTER_K, top_k)]
        if mode == "different_pace":
            div = _pace_divergence
        elif mode == "different_genre":
            div = _genre_divergence
        else:
            div = _sound_divergence
        pool_with_div = [
            (idx, tp, tn, mt, div(ctx, cand_idx=idx, current_idx=current_idx))
            for (idx, tp, tn, mt) in pool
        ]
        pool_with_div.sort(key=lambda r: r[4], reverse=True)
        chosen = [(idx, tp, tn, mt) for (idx, tp, tn, mt, _) in pool_with_div[:top_k]]

    result: List[Dict[str, Any]] = []
    for idx, t_prev, t_next, mean_t in chosen:
        entry: Dict[str, Any] = {
            "index": int(idx),
            "track_id": str(ctx.track_ids[int(idx)]),
            "artist_key": str(ctx.artist_keys[int(idx)]),
            "t_prev": float(t_prev),
            "t_next": float(t_next),
            "mean_t": float(mean_t),
        }
        if ctx.perceptual_bpm is not None and not np.isnan(ctx.perceptual_bpm[int(idx)]):
            entry["perceptual_bpm"] = float(ctx.perceptual_bpm[int(idx)])
        result.append(entry)
    return result
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/playlist/replacement.py tests/unit/test_replacement_scoring.py
git commit -m "feat: track replacement scoring engine (best + 3 divergence modes)"
```

---

## Task 2: Worker bundle cache

**Files:**
- Modify: `src/playlist_gui/worker.py`

The worker currently does not retain any state between commands. For replacement to be fast we need to keep the last generation's bundle (artifact data + candidate pool indices + tempo arrays + metric context) in memory. Add a module-level `_LAST_GENERATION_CACHE` that `handle_generate_playlist` populates on success, and that `handle_find_replacement_suggestions` reads.

- [ ] **Step 1: Add the cache singleton**

Near the other module-level state in `src/playlist_gui/worker.py`:

```python
@dataclass
class _LastGenerationCache:
    """In-memory snapshot of the most recent generate_playlist run.

    Used by handle_find_replacement_suggestions to avoid reloading the
    artifact or rebuilding the candidate pool. Reset on each new generate.
    """
    playlist_id: Optional[str] = None
    track_ids: Optional[np.ndarray] = None         # full library track_ids from bundle
    artist_keys: Optional[np.ndarray] = None
    X_sonic: Optional[np.ndarray] = None
    X_full: Optional[np.ndarray] = None
    X_start: Optional[np.ndarray] = None
    X_mid: Optional[np.ndarray] = None
    X_end: Optional[np.ndarray] = None
    X_genre_smoothed: Optional[np.ndarray] = None
    perceptual_bpm: Optional[np.ndarray] = None
    tempo_stability: Optional[np.ndarray] = None
    candidate_pool_indices: Optional[np.ndarray] = None
    tower_pca_dims: Optional[tuple] = None
    idf_weights: Optional[np.ndarray] = None
    transition_metric_context: Any = None
    transition_floor: float = 0.20
    playlist_track_ids: Optional[List[str]] = None  # the actual playlist that was emitted


_LAST_GENERATION_CACHE = _LastGenerationCache()
```

- [ ] **Step 2: Populate the cache from `handle_generate_playlist`**

Near the end of `handle_generate_playlist`, after the playlist result has been computed but before emitting the `done` event, capture the relevant arrays. The bundle, embedding, pool, and any computed metric context are all in scope locally at that point — pull them into the cache directly. Include the playlist track ID sequence so replacement knows what's already used.

This is read-only against the bundle (we are NOT mutating it).

- [ ] **Step 3: Add `handle_find_replacement_suggestions`**

```python
def handle_find_replacement_suggestions(cmd_data: Dict[str, Any]) -> None:
    """Find replacement candidates for a single position.

    Request:
        {"cmd":"find_replacement_suggestions","request_id":"...",
         "position": <int>, "mode": "best"|"different_pace"|"different_genre"|"different_sound",
         "top_k": 10}
    """
    request_id = str(cmd_data.get("request_id", ""))
    try:
        position = int(cmd_data["position"])
        mode = str(cmd_data.get("mode", "best"))
        top_k = int(cmd_data.get("top_k", 10))

        cache = _LAST_GENERATION_CACHE
        if cache.playlist_track_ids is None:
            _emit_error(request_id, "No playlist in cache. Generate one first.")
            return

        playlist_track_ids = list(cache.playlist_track_ids)
        if position < 0 or position >= len(playlist_track_ids):
            _emit_error(request_id, f"position {position} out of range 0..{len(playlist_track_ids) - 1}")
            return

        # Convert track_ids in playlist to bundle indices
        id_to_index = {tid: i for i, tid in enumerate(cache.track_ids.tolist())}
        playlist_indices = [id_to_index[tid] for tid in playlist_track_ids if tid in id_to_index]
        current_idx = playlist_indices[position]
        if position == 0 or position == len(playlist_indices) - 1:
            _emit_error(request_id, "Cannot replace pier (first or last) track.")
            return
        prev_idx = playlist_indices[position - 1]
        next_idx = playlist_indices[position + 1]

        from src.playlist.replacement import ReplacementContext, find_replacement_candidates
        ctx = ReplacementContext(
            X_sonic=cache.X_sonic,
            X_full=cache.X_full,
            X_start=cache.X_start,
            X_end=cache.X_end,
            X_mid=cache.X_mid,
            X_genre_smoothed=cache.X_genre_smoothed,
            perceptual_bpm=cache.perceptual_bpm,
            tempo_stability=cache.tempo_stability,
            track_ids=cache.track_ids,
            artist_keys=cache.artist_keys,
            candidate_pool_indices=cache.candidate_pool_indices,
            tower_pca_dims=cache.tower_pca_dims,
            idf_weights=cache.idf_weights,
            transition_metric_context=cache.transition_metric_context,
            transition_floor=cache.transition_floor,
        )
        candidates = find_replacement_candidates(
            prev_idx=prev_idx, next_idx=next_idx, current_idx=current_idx,
            playlist_indices=playlist_indices, ctx=ctx, mode=mode, top_k=top_k,
        )

        # Enrich with display fields from the DB
        enriched = _enrich_replacement_candidates(candidates, cache.track_ids)

        _emit_result(request_id, "replacement_suggestions", {
            "position": position,
            "mode": mode,
            "candidates": enriched,
            "current_track_id": str(cache.track_ids[current_idx]),
            "prev_track_id": str(cache.track_ids[prev_idx]),
            "next_track_id": str(cache.track_ids[next_idx]),
        })
        _emit_done(request_id, "find_replacement_suggestions", ok=True)
    except Exception as e:
        logger.exception("find_replacement_suggestions failed")
        _emit_error(request_id, f"{type(e).__name__}: {e}")
```

`_enrich_replacement_candidates` looks up title/artist/album/duration/top genres for each candidate from `metadata.db`. (Use the same `MetadataClient` pattern other handlers use.)

- [ ] **Step 4: Register the command in the dispatch table**

Wherever `handle_generate_playlist` is dispatched (search for `"cmd":"generate_playlist"` mapping), add an entry for `"find_replacement_suggestions"`.

Update the protocol docstring at the top of `worker.py` to include the new command.

- [ ] **Step 5: Run tests + smoke**

```bash
python -m pytest tests/unit/test_worker_protocol.py -v
python -m pytest tests/ -m "not slow" -q --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/worker.py
git commit -m "feat: worker last-generation cache + find_replacement_suggestions cmd"
```

---

## Task 3: Worker IPC client method

**Files:**
- Modify: `src/playlist_gui/worker_client.py`
- Modify: `tests/unit/test_worker_protocol.py`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_worker_protocol.py`:

```python
def test_find_replacement_suggestions_request_format():
    """Client emits well-formed NDJSON for the replacement command."""
    from src.playlist_gui.worker_client import build_find_replacement_request

    req = build_find_replacement_request(
        request_id="abc-123",
        position=5,
        mode="best",
        top_k=10,
    )
    assert req["cmd"] == "find_replacement_suggestions"
    assert req["request_id"] == "abc-123"
    assert req["position"] == 5
    assert req["mode"] == "best"
    assert req["top_k"] == 10
    assert req["protocol_version"] == 1
```

- [ ] **Step 2: Implement in worker_client.py**

Add a helper:

```python
def build_find_replacement_request(*, request_id: str, position: int, mode: str, top_k: int = 10) -> dict:
    return {
        "cmd": "find_replacement_suggestions",
        "request_id": str(request_id),
        "protocol_version": 1,
        "position": int(position),
        "mode": str(mode),
        "top_k": int(top_k),
    }
```

Then expose a public `request_replacement_suggestions(position, mode, top_k=10)` method on the worker-client class that wraps `build_find_replacement_request`, sends it on stdin, and (asynchronously) yields the result via the existing event signal/callback wiring.

- [ ] **Step 3: Run, commit**

```bash
git add src/playlist_gui/worker_client.py tests/unit/test_worker_protocol.py
git commit -m "feat: GUI client method for find_replacement_suggestions"
```

---

## Task 4: Track-table context menu entry

**Files:**
- Modify: `src/playlist_gui/widgets/track_table.py`
- Modify: `tests/unit/test_generate_panel.py` (if any covers context menu) or new `test_track_table_replace.py`

- [ ] **Step 1: Add new signal**

In `track_table.py`, near `blacklist_requested` and `blacklist_scope_requested`:

```python
replace_track_requested = Signal(int)   # emits playlist position
```

- [ ] **Step 2: Add context-menu entry**

Where the existing blacklist menu items are constructed (in `_on_context_menu` or equivalent), add a new entry:

```python
replace_action = menu.addAction("Replace this track…")
replace_action.setEnabled(
    is_single_selection
    and selected_position is not None
    and 0 < selected_position < (len(playlist) - 1)  # piers protected
)
replace_action.triggered.connect(
    lambda: self.replace_track_requested.emit(int(selected_position))
)
```

The action is disabled (greyed out) when the selection is a pier (first or last position) — those are seeds and replacing them changes the playlist's structural anchors.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/unit/test_generate_panel.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/playlist_gui/widgets/track_table.py
git commit -m "feat: context-menu entry for replacing a single playlist track"
```

---

## Task 5: Replace dialog scaffolding (Manual + 4 Suggestion tabs)

**Files:**
- Create: `src/playlist_gui/widgets/replace_dialog.py`
- Create: `tests/unit/test_replace_dialog.py`

The dialog has a `QTabWidget` with five tabs:
1. **Search** — autocomplete-based manual selection
2. **Best Match** — auto-generated; uses `mode="best"`
3. **Different Pace** — auto; `mode="different_pace"`
4. **Different Genre** — auto; `mode="different_genre"`
5. **Different Sound** — auto; `mode="different_sound"`

Each auto tab shows a small table with columns: `Title | Artist | T_prev | T_next | BPM | Top genres`. When a tab is selected for the first time, the dialog fires off a `request_replacement_suggestions(position, mode)` call to the worker; the result populates the table.

The dialog exposes a single signal `replacement_chosen(int position, str new_track_id)` and emits it on Apply.

- [ ] **Step 1: Failing test for tab layout**

```python
# tests/unit/test_replace_dialog.py
from PySide6.QtWidgets import QApplication
from src.playlist_gui.widgets.replace_dialog import ReplaceTrackDialog


def test_dialog_has_five_tabs(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)
    assert dialog.tab_widget.count() == 5
    expected = ["Search", "Best Match", "Different Pace", "Different Genre", "Different Sound"]
    actual = [dialog.tab_widget.tabText(i) for i in range(5)]
    assert actual == expected


def test_dialog_emits_replacement_chosen_on_apply(qtbot, qapp):
    dialog = ReplaceTrackDialog(position=3, current_track={"title": "Foo", "artist": "Bar"})
    qtbot.addWidget(dialog)
    chosen = []
    dialog.replacement_chosen.connect(lambda pos, tid: chosen.append((pos, tid)))

    # Simulate the dialog having a row selected in the Best Match tab
    dialog._pick_track("new-track-id-xyz")
    assert chosen == [(3, "new-track-id-xyz")]
```

- [ ] **Step 2: Implement the dialog**

Minimum widgets:
- `tab_widget: QTabWidget` (5 tabs)
- Each suggestion tab contains: a `QTableView` bound to a small model (Title / Artist / T_prev / T_next / BPM / Top genres) + a status label ("Loading…" / "No suggestions found")
- The "Search" tab contains: a `QLineEdit` with autocomplete against the library track list + a "Select" button
- Footer: "Apply" button (enabled only when something is selected) + "Cancel"
- `replacement_chosen = Signal(int, str)` — emitted on Apply
- Public method `populate_suggestions(mode, candidates)` — called from main_window when a worker result lands

- [ ] **Step 3: Run, commit**

```bash
git add src/playlist_gui/widgets/replace_dialog.py tests/unit/test_replace_dialog.py
git commit -m "feat: replace track dialog (Search + 4 suggestion tabs)"
```

---

## Task 6: Wire dialog into main_window

**Files:**
- Modify: `src/playlist_gui/main_window.py`

- [ ] **Step 1: Connect track-table signal**

Wherever the playlist track table is created in `main_window.py`, connect:

```python
self.track_table.replace_track_requested.connect(self._open_replace_dialog)
```

- [ ] **Step 2: Implement `_open_replace_dialog`**

```python
def _open_replace_dialog(self, position: int) -> None:
    playlist = self._current_playlist_tracks  # however the GUI stores it
    if position < 0 or position >= len(playlist):
        return
    current_track = playlist[position]

    dialog = ReplaceTrackDialog(
        position=position,
        current_track=current_track,
        parent=self,
    )
    # Wire each tab to request suggestions on demand
    dialog.suggestions_requested.connect(self._request_replacement_suggestions)
    dialog.replacement_chosen.connect(self._apply_replacement)
    dialog.exec()
```

- [ ] **Step 3: Implement `_request_replacement_suggestions`**

```python
def _request_replacement_suggestions(self, position: int, mode: str) -> None:
    self.worker_client.request_replacement_suggestions(
        position=position, mode=mode, top_k=10,
    )
```

- [ ] **Step 4: Handle the worker `replacement_suggestions` event**

Route the worker's `result` event of type `replacement_suggestions` to the currently-open dialog so it can populate the appropriate tab.

- [ ] **Step 5: Implement `_apply_replacement`**

When `replacement_chosen(position, new_track_id)` fires:
- Update the in-memory playlist state to swap the track
- Recompute T_prev and T_next for the two affected edges
- Update the visible track table row
- Log: `Replaced position N: <old_artist - title> → <new_artist - title> (T_prev: X→Y, T_next: X→Y)`

For v1 the recomputation can be a lightweight worker call (`recompute_edges`) or — simpler — bundled into the suggestion result so the GUI already has the new T values for whichever track the user picks.

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/main_window.py
git commit -m "feat: wire replace dialog into main window with worker round-trip"
```

---

## Task 7: Tests for end-to-end replacement protocol

**Files:**
- Create: `tests/unit/test_replacement_modes.py`

- [ ] **Step 1: Integration-flavored test using cached worker state**

```python
# Skeleton — flesh out using existing worker test scaffolding
def test_full_replacement_flow_with_each_mode(tmp_path):
    """Smoke: generate a playlist, then request suggestions in each mode."""
    # 1. Build a small synthetic bundle + library DB
    # 2. Run generate (small playlist)
    # 3. Cache should be populated
    # 4. For each mode, request_replacement_suggestions(position=2)
    # 5. Assert: top_k <= 10, current track excluded, neighbors' artists excluded
    # 6. For "best", T values should be the highest of all modes (control)
    # 7. For "different_*", divergence on that axis should exceed "best"
```

- [ ] **Step 2: Run, commit**

```bash
git add tests/unit/test_replacement_modes.py
git commit -m "test: end-to-end replacement flow across all four modes"
```

---

## Task 8: Docs and logging polish

**Files:**
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md`
- Modify: `README.md`

- [ ] **Step 1: Add a short section under "Diagnostics" in README**

```markdown
### Track replacement
Right-click any non-pier track in the playlist table → **Replace this track…**.
The dialog offers five ways to choose a replacement:

- **Search** — find a specific track by name
- **Best Match** — top alternatives that fit the surrounding tracks under current weights
- **Different Pace** — alternatives that deliberately vary in tempo/energy
- **Different Genre** — alternatives that bridge to a slightly different genre territory
- **Different Sound** — alternatives with different timbre/harmony but similar fit

All four auto modes require the replacement to clear the transition floor against
both neighbors. The "Different X" modes filter to the top 50 by transition quality,
then re-rank by axis-specific divergence from the track being replaced.
```

- [ ] **Step 2: Add a one-paragraph note in `docs/PLAYLIST_ORDERING_TUNING.md`** explaining where replacement modes fit in the toolchain (post-generation refinement, not pre-generation tuning).

- [ ] **Step 3: Commit**

```bash
git add docs/PLAYLIST_ORDERING_TUNING.md README.md
git commit -m "docs: track replacement modes and usage"
```

---

## End-to-end validation

- [ ] **Generate a playlist**, select a track in the middle of it, right-click → Replace.
- [ ] **Search tab**: type a known title, autocomplete should suggest it, picking it should swap the track.
- [ ] **Best Match tab**: should return ≤ 10 suggestions, sorted by mean transition quality.
- [ ] **Different Pace**: BPM column should span a noticeably wider range than Best Match.
- [ ] **Different Genre**: top genres on returned candidates should diverge from the current track's tags.
- [ ] **Different Sound**: timbre/harmony divergence visibly higher; transition quality remains usable.
- [ ] **Pier protection**: right-click on the first or last track in the playlist — "Replace this track…" should be greyed out.
- [ ] **State integrity**: after a replacement, the playlist still has the original length, unique track IDs, and updated T metrics for the two surrounding edges.

---

## Risk notes

- **Cache invalidation.** Generating a new playlist must reset `_LAST_GENERATION_CACHE` cleanly. If the worker process is restarted, the cache is empty and replacement requests must fail gracefully with a clear message ("No playlist in cache. Generate one first.").
- **Pier protection.** Replacing the first or last track changes the playlist's structural endpoints (seeds). For v1 we forbid this in the UI. If a user really wants to swap a seed they should regenerate.
- **Performance.** Scoring ~3,500 candidates against two endpoints uses the existing `score_transition_edge` function — fast (≈ milliseconds). Divergence calculations are similarly cheap. Round-trip target: < 500 ms from right-click to suggestion list.
- **GUI threading.** The dialog must not block on the worker round trip. Each tab issues its request lazily and shows "Loading…" until results arrive. The worker uses its existing async event mechanism.
- **Replacement and locked invariants.** The replacement candidate already passes the global admission gates (sonic, genre, pace, BPM), the artist diversity check against immediate neighbors, and a transition-quality floor. It does NOT re-check the cross-segment artist gap, min_gap, or other playlist-wide constraints — for v1 we accept this gap. Document it and add as a v2 improvement.
- **Multi-track replacement** is deferred. If a user wants to replace 3 adjacent tracks they will do it three times (and each replacement uses the post-previous-replacement state, so transitions stay coherent).
- **Undo** is deferred. The user can either generate again or replace back to the original (a "history" of replaced tracks would solve this in v2).
