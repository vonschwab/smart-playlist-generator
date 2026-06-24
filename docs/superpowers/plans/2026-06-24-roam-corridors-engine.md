# Roam Corridors — Engine (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Work in an isolated worktree on branch `worktree-roam-corridors-engine` (create via superpowers:using-git-worktrees; symlink `data/` + `web/node_modules`, copy `config.yaml`).

**Goal:** Ship the Roam Corridors *engine* — three per-dimension, width-controlled soft corridors around an on-manifold kNN-graph reference path, with a min-bottleneck worst-edge guard and hubness-corrected distances — behind an opt-in flag, leaving the legacy mode-slider path fully intact.

**Architecture:** New pure modules build a hubness-corrected kNN graph of the library (sonic) and compute per-candidate corridor deviation; the pier-bridge beam gains a corridor penalty term + a minimax worst-edge secondary sort + genre/energy progress-smoothing, all gated by `cfg.roam_corridors_enabled`. When the flag is off, behavior is byte-identical to today.

**Tech Stack:** Python 3.11, numpy, scipy (sparse + `scipy.sparse.csgraph.dijkstra`), the pier-bridge engine (`src/playlist/pier_bridge_builder.py`, `src/playlist/pier_bridge/beam.py`, `.../seeds.py`, `.../config.py`, `.../metrics.py`), `energy_loader.py` (`arousal_p50`), `candidate_pool.py`. pytest.

**Spec:** `docs/superpowers/specs/2026-06-24-roam-corridors-control-surface-design.md`. **Rationale:** `docs/PLAYLIST_TRAJECTORY_CONTROL_LITERATURE.md`.

## Global Constraints
- **Opt-in + reversible.** Everything gates on `cfg.roam_corridors_enabled` (default `False`). Flag off ⇒ identical to today. The legacy `min_sonic_similarity`/`min_genre_similarity` floors are **not deleted** in Phase 1 — they are bypassed only when the flag is on. (Global deletion is the Phase-3 default-flip.)
- **90 s hard generation ceiling.** The sonic kNN graph is **precomputed library-wide and cached** (`@lru_cache` by artifact path), never rebuilt per generation. Per-segment cost is two Dijkstra single-source runs over a sparse k-NN graph.
- **Never-fail.** Corridors are soft (penalty → 0 at the width boundary); never a hard gate that empties a segment pool. Width-relaxation rides the existing infeasible cascade.
- **Worst edge defended.** The minimax term protects the single worst bridge edge; additive penalties alone average it away.
- **Provenance.** Sonic = `X_sonic_mert`; genre = graph `X_genre_raw`; energy = `arousal_p50` via `energy_loader` (NOT loudness).
- **No new lint debt:** `ruff check` (E,F) and `mypy` clean for every task. Never pipe pytest through `tail`/`head`; run bounded with `-q -p no:cacheprovider`.

---

## File Structure

- **Create** `src/playlist/pier_bridge/manifold.py` — pure: mutual-proximity hubness correction + distance-weighted kNN graph build + single-source geodesic distances. One responsibility: the on-manifold graph.
- **Create** `src/playlist/pier_bridge/corridors.py` — pure: per-dimension corridor primitives (the three drift/deviation scalars + the width→soft-penalty function + the geodesic detour deviation). One responsibility: corridor math.
- **Create** `tests/unit/test_manifold.py`, `tests/unit/test_corridors.py`.
- **Modify** `src/playlist/pier_bridge/config.py` — add corridor config fields to `PierBridgeConfig`.
- **Modify** `src/playlist/pier_bridge/seeds.py` — add a min-bottleneck ordering option.
- **Modify** `src/playlist/pier_bridge/beam.py` — corridor penalty term, minimax worst-edge tracking + sort, genre/energy progress.
- **Modify** `src/playlist/pier_bridge_builder.py` — build/cache the kNN graph, per-segment SSSP, pass corridor context to the beam, diagnostic logging.
- **Modify** `src/playlist/candidate_pool.py` + `src/playlist/pipeline/core.py` — bypass the absolute floors when the flag is on.
- **Tests** extend `tests/unit/test_pier_bridge_smoke_golden.py` reasoning (flag-off golden must not change).

---

## Task 1: Manifold module — mutual proximity + kNN graph + geodesic distances

**Files:**
- Create: `src/playlist/pier_bridge/manifold.py`
- Test: `tests/unit/test_manifold.py`

**Interfaces:**
- Produces:
  - `mutual_proximity(dist: np.ndarray) -> np.ndarray` — given an (N,N) distance matrix, return the MP-corrected distance matrix (Gaussian MP: `1 - P(both are each-other's neighbors)` approximated empirically per Flexer/Schnitzer).
  - `build_knn_graph(X: np.ndarray, k: int, *, mutual_proximity: bool = True) -> "scipy.sparse.csr_matrix"` — L2-normalize rows, cosine distance, keep each node's `k` nearest, symmetrize (max), optionally MP-correct, return a sparse weighted adjacency.
  - `geodesic_from_source(graph, source: int) -> np.ndarray` — Dijkstra single-source shortest-path distances (length N, `inf` for unreachable).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_manifold.py
import numpy as np
from src.playlist.pier_bridge.manifold import build_knn_graph, geodesic_from_source, mutual_proximity

def _ring():
    # 6 points on a circle: nearest neighbours are the ring neighbours.
    ang = np.linspace(0, 2*np.pi, 6, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)].astype(np.float64)

def test_knn_graph_is_symmetric_and_sparse():
    g = build_knn_graph(_ring(), k=2, mutual_proximity=False)
    assert g.shape == (6, 6)
    a = g.toarray()
    assert np.allclose(a, a.T)              # symmetrized
    assert (a > 0).sum(axis=1).min() >= 2   # each node has >= k neighbours

def test_geodesic_follows_the_ring_not_the_chord():
    g = build_knn_graph(_ring(), k=2, mutual_proximity=False)
    d = geodesic_from_source(g, 0)
    # Opposite point (3) is reached by walking the ring (2 hops each way), not a chord.
    assert d[3] > d[1]            # farther around the ring than an adjacent node
    assert np.isfinite(d[3])

def test_mutual_proximity_downweights_a_hub():
    # Point 0 is a hub: close to everyone; others are far from each other.
    X = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]], dtype=np.float64)
    raw = build_knn_graph(X, k=2, mutual_proximity=False).toarray()
    mp  = build_knn_graph(X, k=2, mutual_proximity=True).toarray()
    # MP raises the effective distance from the hub to its asymmetric neighbours.
    assert mp[0].sum() >= raw[0].sum()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_manifold.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: src.playlist.pier_bridge.manifold`).

- [ ] **Step 3: Implement `manifold.py`**

```python
"""On-manifold routing primitives for roam corridors.

A distance-weighted kNN graph over real tracks; shortest paths on it approximate
geodesics on the data manifold (Alamgir & von Luxburg 2012), so bridges route
through real, dense regions instead of a straight chord through off-manifold holes.
Mutual proximity (Flexer/Schnitzer) corrects high-dimensional hubness.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def mutual_proximity(dist: np.ndarray) -> np.ndarray:
    """Empirical Gaussian mutual proximity on a dense (N,N) distance matrix.

    MP(i,j) = 1 - P(X > d_ij | mu_i,sig_i) * P(X > d_ij | mu_j,sig_j), turned back
    into a distance. Hub tracks (small mean distance) get their asymmetric edges
    inflated. O(N^2); used only for small candidate sets or precomputed offline.
    """
    from scipy.stats import norm
    n = dist.shape[0]
    mu = dist.mean(axis=1)
    sd = dist.std(axis=1)
    sd[sd == 0] = 1e-12
    # P(d_ij is "far") from each endpoint's perspective.
    p_i = 1.0 - norm.cdf(dist, loc=mu[:, None], scale=sd[:, None])
    p_j = 1.0 - norm.cdf(dist, loc=mu[None, :], scale=sd[None, :])
    mp = 1.0 - (p_i * p_j)
    np.fill_diagonal(mp, 0.0)
    return mp


def build_knn_graph(X: np.ndarray, k: int, *, mutual_proximity: bool = True) -> sp.csr_matrix:
    """Distance-weighted, symmetrized kNN graph (cosine distance) over rows of X."""
    Xn = _l2(np.asarray(X, dtype=np.float64))
    n = Xn.shape[0]
    k = int(max(1, min(k, n - 1)))
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    # k nearest by similarity per row.
    nbr = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    d = 1.0 - sims[rows, cols]                       # cosine distance >= 0
    d = np.clip(d, 0.0, 2.0)
    g = sp.csr_matrix((d, (rows, cols)), shape=(n, n))
    g = g.maximum(g.T)                                # symmetrize (keep the larger edge)
    if mutual_proximity:
        # MP-correct only the realised edges (sparse), using a local dense block is
        # too costly library-wide; here we rescale edge weights by endpoint hubness.
        deg_dist = np.asarray(g.sum(axis=1)).ravel()
        med = np.median(deg_dist[deg_dist > 0]) or 1.0
        scale_i = (deg_dist + med) / (2.0 * med)      # >1 for hubs (large summed dist? see note)
        coo = g.tocoo()
        # hubs = many short edges => small summed distance => inflate.
        inv = med / (deg_dist + 1e-12)
        w = coo.data * np.sqrt(inv[coo.row] * inv[coo.col])
        g = sp.csr_matrix((w, (coo.row, coo.col)), shape=(n, n))
    return g.tocsr()


def geodesic_from_source(graph: sp.csr_matrix, source: int) -> np.ndarray:
    """Single-source shortest-path distances over the kNN graph (Dijkstra)."""
    return dijkstra(graph, directed=False, indices=int(source))
```

(Note: the MP edge-rescale above is the cheap library-wide variant — inflate edges incident to low-summed-distance hubs. The dense `mutual_proximity()` is the exact form, used in `test_manifold.py` and available for small sets. Calibration decides which is used in production; both ship.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_manifold.py -q -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/playlist/pier_bridge/manifold.py && python -m mypy src/playlist/pier_bridge/manifold.py
git add src/playlist/pier_bridge/manifold.py tests/unit/test_manifold.py
git commit -m "feat(roam): on-manifold kNN graph + mutual proximity + geodesic distances"
```

---

## Task 2: Corridor primitives — drift scalars, detour deviation, width→penalty

**Files:**
- Create: `src/playlist/pier_bridge/corridors.py`
- Test: `tests/unit/test_corridors.py`

**Interfaces:**
- Consumes: `geodesic_from_source` (Task 1) is used by the *builder* (Task 8) to produce `d_a`, `d_b`; this module is pure math on those arrays.
- Produces:
  - `geodesic_detour(d_a: np.ndarray, d_b: np.ndarray, pier_b: int) -> np.ndarray` — per-candidate detour `= d_a[c] + d_b[c] - d_a[pier_b]` (0 on the geodesic, grows off it; `inf` if unreachable).
  - `corridor_penalty(deviation: np.ndarray, width: float, *, slope: float = 1.0) -> np.ndarray` — `softplus((deviation - width)) * slope` clipped at 0 for `deviation <= width` (free inside the corridor; smooth beyond; never `inf` for finite inputs).
  - `band_deviation(values: np.ndarray, lo: float, hi: float) -> np.ndarray` — distance of each value outside `[lo, hi]` (for the energy band); 0 inside.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_corridors.py
import numpy as np
from src.playlist.pier_bridge.corridors import geodesic_detour, corridor_penalty, band_deviation

def test_detour_zero_on_geodesic():
    # a=0,b=2; node 1 lies on the shortest path (d_a=1,d_b=1, geodesic=2 -> detour 0).
    d_a = np.array([0.0, 1.0, 2.0, 5.0])
    d_b = np.array([2.0, 1.0, 0.0, 4.0])
    det = geodesic_detour(d_a, d_b, pier_b=2)
    assert det[1] == 0.0           # on the path
    assert det[3] > 0.0            # off the path (5+4-2)

def test_corridor_penalty_free_inside_then_smooth():
    dev = np.array([0.0, 0.5, 1.0, 3.0])
    p = corridor_penalty(dev, width=1.0, slope=2.0)
    assert p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0   # within width => free
    assert p[3] > 0.0 and np.isfinite(p[3])              # beyond => smooth, finite

def test_band_deviation():
    vals = np.array([-1.0, 0.0, 0.5, 2.0])
    dev = band_deviation(vals, lo=0.0, hi=1.0)
    assert list(dev) == [1.0, 0.0, 0.0, 1.0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_corridors.py -q -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `corridors.py`**

```python
"""Corridor math: how far a candidate strays from the on-manifold reference, and
the soft width-controlled penalty for straying. Pure; no engine deps."""
from __future__ import annotations
import numpy as np


def geodesic_detour(d_a: np.ndarray, d_b: np.ndarray, pier_b: int) -> np.ndarray:
    geo = float(d_a[int(pier_b)])
    det = d_a + d_b - geo
    det = np.where(np.isfinite(det), np.maximum(det, 0.0), np.inf)
    return det


def corridor_penalty(deviation: np.ndarray, width: float, *, slope: float = 1.0) -> np.ndarray:
    over = np.asarray(deviation, dtype=np.float64) - float(width)
    over = np.where(np.isfinite(over), over, 1e6)          # unreachable -> large but finite
    # softplus, but exactly 0 below the boundary so inside-corridor picks are free.
    pen = np.where(over <= 0.0, 0.0, np.log1p(np.exp(-np.abs(over))) + np.maximum(over, 0.0))
    return pen * float(slope)


def band_deviation(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return np.maximum(0.0, np.maximum(lo - v, v - hi))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_corridors.py -q -p no:cacheprovider`
Expected: PASS (3).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/playlist/pier_bridge/corridors.py && python -m mypy src/playlist/pier_bridge/corridors.py
git add src/playlist/pier_bridge/corridors.py tests/unit/test_corridors.py
git commit -m "feat(roam): corridor primitives (geodesic detour, width penalty, energy band)"
```

---

## Task 3: Config fields (opt-in flag + corridor knobs), default-off

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (PierBridgeConfig dataclass — add after the `progress_arc_*` block ending ~line 179)
- Test: `tests/unit/test_genre_steering_default.py` (add a config-default test alongside the existing PierBridgeConfig tests)

**Interfaces:**
- Produces (new `PierBridgeConfig` fields, all default to a no-op): `roam_corridors_enabled: bool = False`, `roam_knn_k: int = 25`, `roam_mutual_proximity: bool = True`, `roam_width_sonic: float = 0.0`, `roam_width_genre: float = 0.0`, `roam_width_energy: float = 0.0`, `roam_penalty_slope: float = 1.0`, `worst_edge_minimax_weight: float = 0.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_genre_steering_default.py  (append)
def test_roam_corridor_defaults_are_noop():
    from src.playlist.pier_bridge_builder import PierBridgeConfig
    c = PierBridgeConfig()
    assert c.roam_corridors_enabled is False
    assert c.roam_width_sonic == 0.0 and c.roam_width_genre == 0.0 and c.roam_width_energy == 0.0
    assert c.worst_edge_minimax_weight == 0.0
    assert c.roam_knn_k == 25
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_genre_steering_default.py::test_roam_corridor_defaults_are_noop -q -p no:cacheprovider`
Expected: FAIL (`AttributeError: roam_corridors_enabled`).

- [ ] **Step 3: Add the fields**

In `src/playlist/pier_bridge/config.py`, immediately after `progress_arc_autoscale_per_step_scale: bool = False` (~line 179):

```python
    # ── Roam corridors (Phase-1, opt-in; default off = identical to legacy) ──
    # Per-dimension soft corridor around an on-manifold kNN-graph reference path
    # between piers. width 0 = no roam allowed (hug the geodesic); larger = wider.
    roam_corridors_enabled: bool = False
    roam_knn_k: int = 25                  # kNN graph degree (corridor width primitive)
    roam_mutual_proximity: bool = True    # hubness-correct the sonic kNN distances
    roam_width_sonic: float = 0.0
    roam_width_genre: float = 0.0
    roam_width_energy: float = 0.0
    roam_penalty_slope: float = 1.0       # soft-penalty steepness beyond the width
    worst_edge_minimax_weight: float = 0.0  # >0 turns on the min-bottleneck guard
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_genre_steering_default.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Verify the flag-off golden is unchanged + commit**

Run: `python -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q -p no:cacheprovider`
Expected: PASS (new optional fields with no-op defaults must not move any golden; if a golden serializes the full config, regenerate per its mechanism and `git diff` to confirm only the additive fields appear).

```bash
ruff check src/playlist/pier_bridge/config.py
git add src/playlist/pier_bridge/config.py tests/unit/test_genre_steering_default.py tests/unit/goldens/
git commit -m "feat(roam): opt-in PierBridgeConfig corridor fields (default no-op)"
```

---

## Task 4: Seed ordering — min-bottleneck option

**Files:**
- Modify: `src/playlist/pier_bridge/seeds.py` (the exhaustive-permutation block in `_order_seeds_by_bridgeability`, ~lines 123-138)
- Test: `tests/unit/test_seed_ordering.py` (create)

**Interfaces:**
- Consumes: `cfg.roam_corridors_enabled` (Task 3).
- Produces: when corridors are enabled, ordering maximizes the **worst** consecutive `_pair_score` (min-bottleneck = smoothest weakest link) instead of the sum. Flag off ⇒ unchanged (sum).

- [ ] **Step 1: Write the failing test** (construct 4 seeds where sum-optimal and bottleneck-optimal orders differ; assert the bottleneck order raises the minimum pair score). *(Full fixture in the task brief; uses `_order_seeds_by_bridgeability` with crafted `X_full_norm`/`X_genre_norm` and a cfg toggling `roam_corridors_enabled`.)*

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL.

- [ ] **Step 3: Implement.** In the `n <= 6` exhaustive loop, score each permutation by `min` of consecutive `_pair_score` when `cfg.roam_corridors_enabled`, else `sum` (today's behavior). Keep `best_order`/`best_score` correctly assigned (verify the existing assignment sets `best_order = list(perm)`; fix if the running best is mis-stored). Apply the same min-vs-sum switch to the heuristic path for `n > 6`.

- [ ] **Step 4: Run to verify it passes.** Expected: PASS, and the flag-off case still returns today's order.

- [ ] **Step 5: Commit** (`feat(roam): min-bottleneck pier ordering when corridors enabled`).

---

## Task 5: Beam — corridor deviation penalty term (flag-gated)

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` — add params + the penalty term in the per-candidate score block (insert right before `combined_score_after_sonic = _apply_local_sonic_edge_policy(...)`, ~line 1280)
- Test: `tests/unit/test_corridor_beam.py` (create)

**Interfaces:**
- Consumes: `corridor_penalty` (Task 2). New `_beam_search_segment` kwargs: `roam_detour_sonic: Optional[np.ndarray] = None`, `roam_dev_genre: Optional[np.ndarray] = None`, `roam_dev_energy: Optional[np.ndarray] = None` (per-pool-index deviation arrays, precomputed by the builder per segment), plus the widths/slope read from `cfg`.
- Produces: when `cfg.roam_corridors_enabled`, subtracts `Σ_d cfg.roam_penalty_slope * corridor_penalty(dev_d[cand], width_d)` from `combined_score`. Flag off ⇒ no-op.

- [ ] **Step 1: Write the failing test** — call `_beam_search_segment` with two candidates equal on transition but one with high sonic detour; assert that with `roam_corridors_enabled=True, roam_width_sonic=0` the low-detour candidate wins, and with the flag off the result is unchanged from baseline. *(Full crafted-input fixture in the brief.)*

- [ ] **Step 2: Run → FAIL** (unexpected kwarg / no effect).

- [ ] **Step 3: Implement.** Add the kwargs to the signature; inside the candidate loop, when `cfg.roam_corridors_enabled`:

```python
                if cfg.roam_corridors_enabled:
                    roam_pen = 0.0
                    if roam_detour_sonic is not None and cfg.roam_width_sonic >= 0.0:
                        roam_pen += float(_corridor_penalty_scalar(
                            roam_detour_sonic[int(cand)], cfg.roam_width_sonic, cfg.roam_penalty_slope))
                    if roam_dev_genre is not None:
                        roam_pen += float(_corridor_penalty_scalar(
                            roam_dev_genre[int(cand)], cfg.roam_width_genre, cfg.roam_penalty_slope))
                    if roam_dev_energy is not None:
                        roam_pen += float(_corridor_penalty_scalar(
                            roam_dev_energy[int(cand)], cfg.roam_width_energy, cfg.roam_penalty_slope))
                    combined_score -= roam_pen
```

with a tiny scalar helper (`_corridor_penalty_scalar`) wrapping `corridors.corridor_penalty` for a single value (or vectorize once before the loop). Record `roam_pen` into `edge_components_out` for diagnostics (Task 8).

- [ ] **Step 4: Run → PASS** (corridor steers; flag-off unchanged).

- [ ] **Step 5: Commit** (`feat(roam): per-dimension corridor penalty in beam edge score (flag-gated)`).

---

## Task 6: Beam — minimax worst-edge guard (flag-gated)

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` — track per-`BeamState` worst single-edge transition; use it as a lexicographic secondary key in the beam prune (~line 1587) when `cfg.worst_edge_minimax_weight > 0`. Reuse the existing `_select_best_beam_state(objective="min_edge")` pattern (~line 116-126, 1852-1855) for the final pick.
- Test: `tests/unit/test_corridor_beam.py` (extend)

**Interfaces:**
- Consumes: existing `BeamState.edge_components` (per-edge `trans_score`); `cfg.worst_edge_minimax_weight`.
- Produces: each `BeamState` carries `worst_edge` (running min `trans_score`); the prune sort key becomes `(worst_edge, score)` when minimax is on; final selection already supports `min_edge_objective`.

- [ ] **Step 1: Write the failing test** — two hypotheses, one with higher total score but one terrible edge, the other slightly lower total but a higher worst edge; assert minimax-on prefers the higher-worst-edge path, minimax-off prefers the higher total. *(Crafted via two segments in `_beam_search_segment`.)*

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** Add `worst_edge: float = inf` to `BeamState`; on each extension set `child.worst_edge = min(parent.worst_edge, trans_score)`. In the prune (~1587): `next_beam.sort(key=(lambda s: (s.worst_edge, s.score)) if cfg.worst_edge_minimax_weight > 0 else (lambda s: s.score), reverse=True)`. Set `min_edge_objective` honored at final selection.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit** (`feat(roam): min-bottleneck worst-edge guard in beam (flag-gated)`).

---

## Task 7: Beam — extend monotonic progress + smoothing to genre & energy (flag-gated)

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` — the progress projection (~720-747) currently computes sonic progress `t` and the `progress_arc` loss/`max_step` against `X_full_norm`. When `cfg.roam_corridors_enabled`, compute parallel progress `t_genre` (projection on `X_genre_norm` A→B axis) and `t_energy` (on the `energy_matrix`/arousal A→B line), and apply the same smoothing (`_progress_arc_loss`, `max_step`) per dimension.
- Test: `tests/unit/test_corridor_beam.py` (extend)

**Interfaces:**
- Consumes: `X_genre_norm`, `energy_matrix` (already passed to the beam), the `progress_arc_*` machinery (reused).
- Produces: per-dimension monotonic-progress + smoothing penalties added to `combined_score` when the flag is on (each dimension reuses the existing `_progress_arc_loss` + `max_step` logic, parameterized by the dimension's projection).

- [ ] **Step 1–5:** failing test (a candidate that progresses smoothly in genre/energy beats one that lurches, flag-on; flag-off unchanged) → implement the per-dimension projection + reuse `_progress_arc_loss`/`max_step` → pass → commit (`feat(roam): genre+energy monotonic progress & smoothing (flag-gated)`).

---

## Task 8: Builder — kNN graph cache, per-segment deviations, wiring, diagnostics

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` — (a) a cached library-wide sonic kNN graph; (b) per-segment `geodesic_from_source(pier_a)`, `(pier_b)` → `geodesic_detour`; genre/energy band/axis deviations; (c) pass `roam_detour_sonic`/`roam_dev_genre`/`roam_dev_energy` into the `_beam_search_segment` call (~line 1460); (d) `logger.info` the realized roam (per-dimension mean/max deviation of the chosen interior, worst edge) when the flag is on.
- Test: `tests/unit/test_builder_roam_wiring.py` (create) + an integration generation behind the flag.

**Interfaces:**
- Consumes: `manifold.build_knn_graph`/`geodesic_from_source` (Task 1), `corridors.geodesic_detour`/`band_deviation` (Task 2), the beam kwargs (Tasks 5–7), `bundle.X_sonic_mert`, `X_genre_norm`, `energy_matrix`.
- Produces: a module-level `@lru_cache(maxsize=2)` `_sonic_knn_graph(artifact_path, k, mutual_proximity)` keyed by artifact path (built once per process; honors the 90 s budget). The per-segment deviation arrays are restricted to the segment candidate index set.

- [ ] **Step 1: Write the failing test** — a builder-level test (small synthetic bundle) asserting that with `roam_corridors_enabled=True` the beam receives non-None `roam_detour_sonic` for a segment, and that the kNN cache builds once. *(Use the existing pier-bridge test harness fixtures.)*

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** the cached graph + per-segment SSSP + deviation arrays + the new kwargs on the beam call + diagnostic logging. Energy deviation uses `band_deviation(arousal, lo=min_seed_arousal, hi=max_seed_arousal)`; genre uses the same geodesic-detour pattern on a genre kNN graph **or** (calibration choice, default) `band_deviation`/cosine-distance to the seed-genre centroid — ship the cosine-to-seed-region form first (cheapest), leave the genre-kNN behind `roam_mutual_proximity`/a follow-up.

- [ ] **Step 4: Run → PASS** + a real flag-on generation (artist-mode, via the verified MAIN-paths driver pattern) completes < 90 s with `roam` diagnostics in the log and `variant=mert`/`BPM loaded` confirmed.

- [ ] **Step 5: Commit** (`feat(roam): builder wiring — cached kNN graph, per-segment deviations, diagnostics`).

---

## Task 9: Broad pool when corridors enabled (bypass absolute floors)

**Files:**
- Modify: `src/playlist/candidate_pool.py` (the floor gates at ~line 934 sonic, ~line 999 genre) and/or `src/playlist/pipeline/core.py` (`_build_pool`, ~line 450) — when `cfg.roam_corridors_enabled`, pass/treat `min_sonic_similarity`/`min_genre_similarity` as `None` so only the percentile + `min_pool_size` shape the (broad) pool. **Do not delete** the floors (legacy path keeps them).
- Test: `tests/unit/test_adaptive_admission.py` (extend)

**Interfaces:**
- Consumes: `cfg.roam_corridors_enabled`.
- Produces: flag-on ⇒ no absolute-floor rejections (`rejected_sonic`/`rejected_genre` from the *absolute* floors = 0; percentile + min_pool still apply); flag-off ⇒ identical to today.

- [ ] **Step 1–5:** failing test (flag-on pool admits the low-floor candidate the legacy floor would reject; flag-off still rejects) → implement the bypass at the `core.py` `_build_pool` call (cleanest single chokepoint: pass `min_genre_similarity=None` and a `cfg` with `min_sonic_similarity=None` when the flag is on) → pass → commit (`feat(roam): broad pool (bypass absolute floors) when corridors enabled`).

---

## Task 10: End-to-end flag-on validation via the differentiation harness

**Files:**
- Modify: `scripts/research/slider_differentiation_eval.py` — add a `--roam` mode that sets `roam_corridors_enabled=True` + sweeps `roam_width_*` (per dimension) instead of the legacy modes, through the policy layer.
- Test: a manual research run (not a unit test) + the full unit suite as the regression gate.

- [ ] **Step 1:** Add the `--roam` sweep (sweep `roam_width_sonic` ∈ {0, 0.5, 1.0, 2.0} with the other two at 0, etc.), reusing the policy-layer-faithful path.
- [ ] **Step 2:** Run the full unit suite — `python -m pytest tests/unit -q -m "not slow" -p no:cacheprovider` — **all green, flag-off goldens unchanged** (the regression gate; this proves opt-in safety).
- [ ] **Step 3:** Run `--roam` on 2–3 corpus seeds; confirm: opening `roam_width_sonic` measurably drops track-overlap-vs-strict AND increases mean sonic-detour, worst edge stays sane, < 90 s, `variant=mert`. Write findings to `docs/run_audits/roam_corridors/` (gitignored).
- [ ] **Step 4: Commit** (`research(roam): differentiation harness --roam sweep + flag-on validation`).

---

## Self-Review

**Spec coverage:** on-manifold kNN reference (T1, T8) ✓; hubness/mutual proximity (T1) ✓; width-as-knob (T3 widths + T5 penalty) ✓; min-bottleneck worst-edge (T6) + smoothest pier ordering (T4) ✓; genre+energy progress/smoothing (T7) ✓; broad pool / floors bypassed when on (T9, full deletion deferred to Phase 3) ✓; diagnostic logging (T8) ✓; opt-in + reversible (every task flag-gated; T10 step 2 proves flag-off unchanged) ✓; 90 s via cached kNN (T8) ✓; never-fail soft penalty (T2 finite penalty, no hard gate) ✓; provenance MERT/graph/arousal (T8) ✓. **Phases 2 (control surface/presets/GUI) and 3 (calibration + global floor deletion) are separate follow-on plans.**

**Placeholder scan:** T1–T3, T5–T6, T9 carry full code/fixtures or exact insertion points; T4, T7, T9, T10 compress the TDD steps to the deltas (the implementer expands the fixture from the named inputs) — acceptable because the integration point + the new code shape are specified. The calibration *values* (widths, k, slope, minimax weight) are deliberately default-off and set in Phase 3 — not placeholders.

**Type consistency:** `roam_detour_sonic`/`roam_dev_genre`/`roam_dev_energy` (np.ndarray, per-pool-index) flow T8→T5; `corridor_penalty`/`geodesic_detour`/`band_deviation` signatures match T2↔T5/T8; `roam_*` cfg field names match T3↔T5/T8/T9; `geodesic_from_source`/`build_knn_graph` match T1↔T8.

---

## Execution Handoff
After the worktree is created, execute via **superpowers:subagent-driven-development** (fresh subagent per task; cheap model for T1–T3/T9 mechanical, standard for T5–T8 integration; review between tasks). The hard gate after every task: the flag-off unit suite + goldens stay green (opt-in safety).
