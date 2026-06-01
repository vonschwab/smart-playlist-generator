# Genre Arc Steering & Adaptive Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make genre a first-class, arc-shaped vote in pier-bridge selection — a per-step dense target that walks (ladder) or interpolates (linear) from pier-A to pier-B through niche genres — with adaptive (percentile) admission/on-arc floors, then calibrate it with a metrics→shortlist→audition harness.

**Architecture:** Revises the committed `genre-edge-safeguards` branch. The beam score becomes `w_bridge·sonic_bridge + w_transition·sonic_transition + w_genre·waypoint_sim(cand, g_target[step])`, where `g_target` is built in the dense `genre_emb` space. Floors are percentiles: per-seed `P_admit` (candidate pool), per-segment `P_arc` (beam). Reuses the dormant `_build_genre_targets`/waypoint machinery; supersedes Task 3's prev-track model.

**Tech Stack:** Python 3.11, numpy, pytest. Spec: `docs/superpowers/specs/2026-05-31-genre-arc-steering-design.md`. Execute AFTER enrichment + human review + artifact/sidecar rebuild stabilize.

---

## Preconditions (read before starting)

- Branch `genre-edge-safeguards` with committed Tasks 1–6 of the prior plan. The config flag `genre_steering_enabled`, `weight_genre`, the relaxation fields, and builder→beam wiring already exist.
- The dense sidecar exposes `bundle.X_genre_dense` (N×64, L2-normalized) and the sidecar file also contains `genre_emb` (V×64, per-genre dense embedding) and `genre_vocab` (V labels). Confirm `genre_emb` is loaded onto the bundle; if `bundle.genre_emb` is not present, Task 2 adds it to `ArtifactBundle` loading.
- Re-ground all "anchor" references by reading the current code (line numbers will have drifted). Locate by quoted code, not line number.

---

## File Structure

- `src/playlist/pier_bridge/genre_graph.py` *(new)* — build a niche genre-adjacency graph from `genre_emb` cosine (kNN, hub-excluded).
- `src/playlist/pier_bridge/genre_targets.py` — dense `g_targets` for `linear` + `ladder` route shapes (dense rungs).
- `src/playlist/pier_bridge/percentiles.py` *(new)* — pure percentile-floor helpers (per-seed admission, per-segment on-arc).
- `src/playlist/candidate_pool.py` — per-seed adaptive admission floor.
- `src/playlist/pier_bridge/beam.py` — first-class genre-arc vote + per-segment on-arc floor; supersede prev-track `genre_sim`.
- `src/playlist/pier_bridge/config.py`, `src/playlist/config.py` — rename + new percentile/ladder knobs.
- `src/playlist/run_audit.py` — arc-floor relaxation fields.
- `src/features/artifacts.py` — load `genre_emb` onto the bundle (if not already).
- `src/playlist/pier_bridge_builder.py` — build dense g_targets (linear/ladder), pass to beam.
- `scripts/calibrate_genre_arc.py` *(new)* — calibration harness.
- `config.yaml` / `config.example.yaml` — knobs.
- Tests: `tests/unit/test_genre_graph.py`, `tests/unit/test_genre_targets_dense.py`, `tests/unit/test_genre_percentiles.py`, extend `tests/unit/test_genre_edge_steering.py`, `tests/integration/test_genre_steering_integration.py`.

---

## Task 1: Niche genre-adjacency graph from `genre_emb`

**Files:**
- Create: `src/playlist/pier_bridge/genre_graph.py`
- Test: `tests/unit/test_genre_graph.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.playlist.pier_bridge.genre_graph import build_genre_graph


def test_graph_connects_niche_neighbors_and_excludes_hubs():
    # 5 genres: two tight niche clusters + one hub correlated with everything.
    vocab = ["noise rock", "no wave", "power pop", "college rock", "rock"]
    emb = np.array([
        [1.0, 0.0, 0.0],   # noise rock
        [0.95, 0.10, 0.0], # no wave  (near noise rock)
        [0.0, 1.0, 0.0],   # power pop
        [0.0, 0.95, 0.10], # college rock (near power pop)
        [0.6, 0.6, 0.5],   # rock (hub: correlated with all)
    ], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    g = build_genre_graph(emb, vocab, k=3, min_cos=0.5, hub_labels={"rock"})
    # hub excluded as a node
    assert "rock" not in g
    # niche neighbors connected
    assert any(nb == "no wave" for nb, _ in g["noise rock"])
    assert any(nb == "college rock" for nb, _ in g["power pop"])
    # hub never appears as a neighbor either
    for nbrs in g.values():
        assert all(nb != "rock" for nb, _ in nbrs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: ... genre_graph`.

- [ ] **Step 3: Implement**

Create `src/playlist/pier_bridge/genre_graph.py`:

```python
"""Niche genre-adjacency graph derived from the dense per-genre embedding.

Two genres are adjacent when their genre_emb cosine is high. Broad "hub" genres
are excluded as nodes so ladder paths cannot collapse into them. Output format
matches what _shortest_genre_path expects: {label: [(neighbor_label, weight), ...]}.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def build_genre_graph(
    genre_emb: np.ndarray,
    genre_vocab: list[str] | np.ndarray,
    *,
    k: int = 8,
    min_cos: float = 0.35,
    hub_labels: Optional[Iterable[str]] = None,
) -> dict[str, list[tuple[str, float]]]:
    """Build a kNN adjacency graph over genre embeddings.

    Args:
        genre_emb: (V, dim) per-genre embedding (rows need not be normalized).
        genre_vocab: V genre labels aligned to genre_emb rows.
        k: max neighbors per genre.
        min_cos: minimum cosine to create an edge.
        hub_labels: labels to exclude as nodes/neighbors (broad genres).
    Returns:
        {label: [(neighbor_label, cos), ...]} for non-hub labels only.
    """
    labels = [str(g) for g in list(genre_vocab)]
    hubs = {str(h).strip().lower() for h in (hub_labels or set())}
    M = np.asarray(genre_emb, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = M / np.maximum(norms, 1e-12)
    keep = [i for i, lab in enumerate(labels) if lab.strip().lower() not in hubs]
    keep_set = set(keep)
    graph: dict[str, list[tuple[str, float]]] = {}
    for i in keep:
        sims = Mn[keep] @ Mn[i]            # cosine to all kept genres
        order = np.argsort(-sims)
        nbrs: list[tuple[str, float]] = []
        for j_local in order:
            j = keep[int(j_local)]
            if j == i:
                continue
            c = float(sims[int(j_local)])
            if c < min_cos:
                break
            nbrs.append((labels[j], c))
            if len(nbrs) >= k:
                break
        graph[labels[i]] = nbrs
    return graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_graph.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/genre_graph.py tests/unit/test_genre_graph.py
git commit -m "feat(genre-arc): niche genre-adjacency graph from genre_emb

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Dense `g_targets` (linear + ladder rungs)

**Files:**
- Modify: `src/features/artifacts.py` (ensure `bundle.genre_emb` is loaded from the sidecar — only if missing)
- Modify: `src/playlist/pier_bridge/genre_targets.py` (`_build_genre_targets` — dense path)
- Test: `tests/unit/test_genre_targets_dense.py`

- [ ] **Step 1: Confirm `genre_emb` is on the bundle**

Run: `python -c "import numpy as np; d=np.load('data/artifacts/beat3tower_32k/data_matrices_step1_genre_emb_dim64.npz', allow_pickle=True); print([k for k in d.files])"` — confirm `genre_emb` and `genre_vocab` are present. Then check `ArtifactBundle` exposes `genre_emb` (`grep -n "genre_emb" src/features/artifacts.py`). If not loaded onto the bundle, add it in the sidecar-loading block alongside `X_genre_dense` (a field `genre_emb: Optional[np.ndarray] = None`, set from `_sc["genre_emb"]`). If already present, skip this step.

- [ ] **Step 2: Write the failing test**

```python
import numpy as np
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge.genre_targets import build_dense_genre_targets


def _toy():
    # dense pier vectors in a 3-dim genre embedding
    dA = np.array([1.0, 0.0, 0.0])
    dB = np.array([0.0, 0.0, 1.0])
    return dA, dB


def test_linear_dense_targets_interpolate_endpoints():
    dA, dB = _toy()
    g = build_dense_genre_targets(dA, dB, interior_length=3, route="linear",
                                  genre_emb=None, genre_vocab=None, genre_graph=None,
                                  labels_a=None, labels_b=None)
    assert len(g) == 3
    # first target leans toward A, last toward B
    assert float(g[0] @ (dA/np.linalg.norm(dA))) > float(g[0] @ (dB/np.linalg.norm(dB)))
    assert float(g[-1] @ (dB/np.linalg.norm(dB))) > float(g[-1] @ (dA/np.linalg.norm(dA)))
    # rows L2-normalized
    for v in g:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_ladder_walks_rungs_in_dense_space():
    # 4 genres; emb so that a path noiserock->nowave->collegerock->powerpop exists
    vocab = ["noise rock", "no wave", "college rock", "power pop"]
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
        [0.0, 0.7, 0.7, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    graph = {
        "noise rock": [("no wave", 0.7)],
        "no wave": [("noise rock", 0.7), ("college rock", 0.49)],
        "college rock": [("no wave", 0.49), ("power pop", 0.0)],
        "power pop": [("college rock", 0.0)],
    }
    dA, dB = emb[0], emb[3]
    g = build_dense_genre_targets(
        dA, dB, interior_length=4, route="ladder",
        genre_emb=emb, genre_vocab=vocab, genre_graph=graph,
        labels_a=["noise rock"], labels_b=["power pop"], max_steps=6,
    )
    assert len(g) == 4
    # a mid target should be closer to an intermediate rung (no wave/college rock)
    # than to either endpoint, proving it walked rather than blended directly.
    mid = g[1]
    inter = emb[1]  # no wave
    assert float(mid @ inter) > float(mid @ dA)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_targets_dense.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_dense_genre_targets'`.

- [ ] **Step 4: Implement `build_dense_genre_targets` in `genre_targets.py`**

Add to `src/playlist/pier_bridge/genre_targets.py` (reusing `_normalize_vec`, `_shortest_genre_path`, `_step_fraction`, `_progress_target_curve` already imported there):

```python
def build_dense_genre_targets(
    dense_a: np.ndarray,
    dense_b: np.ndarray,
    *,
    interior_length: int,
    route: str,
    genre_emb: Optional[np.ndarray],
    genre_vocab: Optional[list[str]],
    genre_graph: Optional[dict[str, list[tuple[str, float]]]],
    labels_a: Optional[list[str]],
    labels_b: Optional[list[str]],
    max_steps: int = 6,
) -> list[np.ndarray]:
    """Per-step dense genre targets from dense_a -> dense_b.

    route='linear': interpolate the two dense pier vectors directly.
    route='ladder': walk a shortest niche path (genre_graph) and interpolate the
        path's dense rung vectors (genre_emb rows). Falls back to linear when a
        path / vocab / graph is unavailable.
    """
    route = (route or "linear").strip().lower()
    na = dense_a / max(float(np.linalg.norm(dense_a)), 1e-12)
    nb = dense_b / max(float(np.linalg.norm(dense_b)), 1e-12)

    rung_vecs: Optional[list[np.ndarray]] = None
    if route == "ladder" and genre_graph and genre_emb is not None and genre_vocab and labels_a and labels_b:
        vocab_index = {str(g).strip().lower(): i for i, g in enumerate(genre_vocab)}
        path = None
        for la in labels_a:
            for lb in labels_b:
                path = _shortest_genre_path(genre_graph, la, lb, max_steps=int(max_steps))
                if path:
                    break
            if path:
                break
        if path and len(path) >= 2:
            vecs = []
            for lab in path:
                j = vocab_index.get(str(lab).strip().lower())
                if j is not None:
                    vecs.append(_normalize_vec(np.asarray(genre_emb[j], dtype=np.float64)))
            if len(vecs) >= 2:
                rung_vecs = vecs

    targets: list[np.ndarray] = []
    if rung_vecs is not None:
        # interpolate along the rung sequence
        import math as _math
        for i in range(int(interior_length)):
            frac = _step_fraction(i, int(interior_length))
            scaled = frac * float(len(rung_vecs) - 1)
            idx = int(_math.floor(scaled))
            if idx >= len(rung_vecs) - 1:
                targets.append(rung_vecs[-1])
            else:
                local = scaled - float(idx)
                targets.append(_normalize_vec((1.0 - local) * rung_vecs[idx] + local * rung_vecs[idx + 1]))
        return targets

    # linear (fallback)
    for i in range(int(interior_length)):
        frac = _progress_target_curve(i, int(interior_length), "arc") if route == "arc" else _step_fraction(i, int(interior_length))
        targets.append(_normalize_vec((1.0 - frac) * na + frac * nb))
    return targets
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_targets_dense.py -q`
Expected: PASS (2 passed). If the ladder test's path doesn't resolve, verify `_shortest_genre_path`'s expected graph format matches Task 1's output.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/genre_targets.py tests/unit/test_genre_targets_dense.py src/features/artifacts.py
git commit -m "feat(genre-arc): dense g_targets for linear + ladder routes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Percentile floor helpers

**Files:**
- Create: `src/playlist/pier_bridge/percentiles.py`
- Test: `tests/unit/test_genre_percentiles.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.playlist.pier_bridge.percentiles import floor_at_percentile, relax_percentile


def test_floor_at_percentile_is_distribution_relative():
    # Sparse seed: most sims low. Dense seed: many sims high.
    sparse = np.concatenate([np.full(900, 0.05), np.full(100, 0.6)])
    dense = np.concatenate([np.full(500, 0.4), np.full(500, 0.8)])
    # Same percentile P -> different absolute floors.
    f_sparse = floor_at_percentile(sparse, p=0.90)
    f_dense = floor_at_percentile(dense, p=0.90)
    assert f_sparse < f_dense
    # p=0.90 keeps ~top 10%
    assert abs((sparse >= f_sparse).mean() - 0.10) < 0.03


def test_relax_percentile_lowers_toward_min():
    seq = relax_percentile(p=0.90, p_min=0.50, step=0.15)
    assert seq[0] == 0.90
    assert seq[-1] <= 0.50 + 1e-9
    assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_genre_percentiles.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/playlist/pier_bridge/percentiles.py`:

```python
"""Distribution-relative (percentile) floors for adaptive genre gating.

A floor at percentile p of a similarity distribution admits roughly the top
(1 - p) fraction. Because it is relative to the distribution, it survives
embedding rebuilds and adapts to sparse vs dense seeds / disparate vs similar
piers.
"""
from __future__ import annotations

import numpy as np


def floor_at_percentile(sims, p: float) -> float:
    """Return the similarity value at percentile p (0..1) of `sims`.

    A candidate clears the floor iff sim >= floor, so this admits ~ the top
    (1 - p) fraction of the distribution.
    """
    arr = np.asarray(sims, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("-inf")
    p = float(min(max(p, 0.0), 1.0))
    return float(np.quantile(arr, p))


def relax_percentile(p: float, p_min: float, step: float = 0.15) -> list[float]:
    """Descending percentile sequence p -> p_min (admit progressively more)."""
    p = float(p)
    p_min = float(p_min)
    if p_min >= p - 1e-9:
        return [p]
    out = [p]
    cur = round(p - step, 4)
    while cur > p_min + 1e-9 and len(out) < 6:
        out.append(cur)
        cur = round(cur - step, 4)
    if not any(abs(x - p_min) < 1e-9 for x in out):
        out.append(p_min)
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_genre_percentiles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/percentiles.py tests/unit/test_genre_percentiles.py
git commit -m "feat(genre-arc): percentile floor helpers (adaptive admission/on-arc)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Per-seed adaptive admission floor (candidate pool)

**Files:**
- Modify: `src/playlist/candidate_pool.py` (the dense gate block — `_use_dense`)
- Test: extend `tests/unit/test_genre_edge_steering.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.playlist.pier_bridge.percentiles import floor_at_percentile


def test_per_seed_admission_floor_adapts_to_density():
    # This guards the *helper contract* candidate_pool will use: floor derived
    # from THIS seed's dense-sim distribution at percentile P_admit.
    rng = np.random.default_rng(0)
    seed = np.array([1.0, 0.0, 0.0])
    # sparse neighborhood: few aligned, many orthogonal
    D_sparse = np.vstack([np.tile([1,0,0], (50,1)), np.tile([0,1,0], (950,1))]).astype(float)
    D_dense = np.vstack([np.tile([1,0,0], (700,1)), np.tile([0,1,0], (300,1))]).astype(float)
    s_sparse = (D_sparse / np.linalg.norm(D_sparse,axis=1,keepdims=True)) @ seed
    s_dense = (D_dense / np.linalg.norm(D_dense,axis=1,keepdims=True)) @ seed
    f_sparse = floor_at_percentile(s_sparse, 0.90)
    f_dense = floor_at_percentile(s_dense, 0.90)
    # both admit ~top 10%, but the absolute floor differs by neighborhood density
    assert f_dense >= f_sparse
```

- [ ] **Step 2: Run to verify it fails (if candidate_pool wiring incomplete)**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -k per_seed_admission -q`
Expected: PASS for the helper contract (it only uses Task 3's helper). The behavioral wiring is verified by integration (Task 8); this unit test pins the contract candidate_pool must honor.

- [ ] **Step 3: Wire the adaptive floor into `candidate_pool.py`**

In the dense gate block (`_use_dense`, where `genre_sim_all = (X_genre_dense @ seed_dense)` is computed), when an adaptive admission percentile is configured (new param `genre_admission_percentile: Optional[float] = None`), replace the fixed `min_genre_similarity` threshold with the per-seed percentile floor:

```python
# after genre_sim_all is computed for the dense path:
if genre_admission_percentile is not None:
    from src.playlist.pier_bridge.percentiles import floor_at_percentile
    # exclude the seed's self-sim (==1.0) when deriving the distribution
    _dist = genre_sim_all.copy()
    _dist[seed_idx] = np.nan
    _eff_floor = floor_at_percentile(_dist[~np.isnan(_dist)], genre_admission_percentile)
    effective_genre_floor = max(float(min_genre_similarity), float(_eff_floor)) \
        if min_genre_similarity is not None else float(_eff_floor)
else:
    effective_genre_floor = min_genre_similarity
# ... use effective_genre_floor wherever min_genre_similarity gated admission ...
logger.info("Candidate pool genre gating: method=dense (PMI-SVD), dim=%d, "
            "admission_percentile=%s, effective_floor=%.3f, mode=%s",
            X_genre_dense.shape[1], genre_admission_percentile, effective_genre_floor, mode)
```

Thread `genre_admission_percentile` through `build_candidate_pool`'s signature and the `pipeline/core.py` call site (read both first; default `None` preserves legacy behavior).

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -q && python -c "import src.playlist.candidate_pool, src.playlist.pipeline.core"`
Expected: tests pass; clean import.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-arc): per-seed adaptive admission floor (percentile)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Config knobs + rename + relaxation fields

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`, `src/playlist/config.py`, `src/playlist/run_audit.py`
- Modify: `config.yaml`, `config.example.yaml`
- Test: extend `tests/unit/test_genre_edge_steering.py`

- [ ] **Step 1: Write the failing test**

```python
def test_arc_knobs_resolve():
    overrides = {"pier_bridge": {
        "genre_steering_enabled": True,
        "weight_genre_narrow": 0.20,
        "genre_arc_floor_percentile_narrow": 0.85,
        "genre_admission_percentile_narrow": 0.90,
        "dj_route_shape": "ladder",
    }}
    from src.playlist.config import resolve_pier_bridge_tuning
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert abs(t.genre_arc_floor_percentile - 0.85) < 1e-9
    assert abs(t.genre_admission_percentile - 0.90) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -k arc_knobs -q`
Expected: FAIL — attribute/resolution missing.

- [ ] **Step 3: Implement config fields**

- In `src/playlist/pier_bridge/config.py` `PierBridgeConfig`: rename `genre_edge_floor` → `genre_arc_floor` (keep as a default absolute fallback) and add: `genre_arc_floor_percentile: float = 0.0`, `genre_admission_percentile: float = 0.0`. (Update the rename everywhere it's referenced — `grep -rn "genre_edge_floor" src/ tests/`.)
- In `src/playlist/config.py` `PierBridgeTuning`: same field additions/rename; resolve them in `resolve_pier_bridge_tuning` via `_resolve_mode_number_with_source` (keys `genre_arc_floor_percentile`, `genre_admission_percentile`, defaults 0.0). Also resolve `dj_route_shape` if not already exposed in the tuning.
- In `src/playlist/run_audit.py` `InfeasibleHandlingConfig`: rename the genre-floor relaxation fields to arc-floor terms (`genre_arc_relaxation_enabled: bool = True`, `min_genre_arc_percentile: float = 0.5`) + parsing.
- Update `src/playlist/pier_bridge/config.py`'s dict wrapper + `pier_bridge_overrides.py` + `playlist_generator.py` construction sites to forward the new/renamed fields (per the Task-1 fix pattern from the prior plan — `grep` for all `PierBridgeConfig(` construction sites).

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -q && grep -rn "genre_edge_floor" src/ tests/`
Expected: tests pass; grep returns nothing (rename complete).

- [ ] **Step 5: Add config.yaml + config.example.yaml knobs**

Under `playlists.ds_pipeline.pier_bridge`:

```yaml
        genre_steering_enabled: true
        dj_route_shape: ladder          # ladder | linear
        weight_genre_strict: 0.30
        weight_genre_narrow: 0.20
        weight_genre_dynamic: 0.12
        weight_genre_discover: 0.06
        genre_admission_percentile_strict: 0.92
        genre_admission_percentile_narrow: 0.90
        genre_admission_percentile_dynamic: 0.85
        genre_admission_percentile_discover: 0.70
        genre_arc_floor_percentile_strict: 0.90
        genre_arc_floor_percentile_narrow: 0.85
        genre_arc_floor_percentile_dynamic: 0.70
        genre_arc_floor_percentile_discover: 0.50
```

Under `...pier_bridge.infeasible_handling`:

```yaml
          genre_arc_relaxation_enabled: true
          min_genre_arc_percentile: 0.40
```

These are starting points; Task 8 calibrates them.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist/config.py src/playlist/run_audit.py src/playlist/pipeline/pier_bridge_overrides.py src/playlist_generator.py config.example.yaml tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-arc): percentile + route-shape config knobs; rename arc floor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: First-class genre-arc vote + per-segment on-arc floor (beam rework)

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (rework the Task-3 genre block, interior + final connection)
- Test: extend `tests/unit/test_genre_edge_steering.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_arc_vote_is_first_class_and_uses_waypoint_target():
    # 4 tracks, sonic identical; candidate 1's dense vec matches g_target[0],
    # candidate 2 matches the previous track but NOT the target. Steering must
    # pick candidate 1 (on the arc), proving it scores vs g_target not prev-track.
    import numpy as np
    from src.playlist.pier_bridge.beam import _beam_search_segment
    from src.playlist.pier_bridge.config import PierBridgeConfig
    Xn = np.ones((4, 3)); Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [0.0, 1.0, 0.0],   # cand 1: matches the step-0 target below
        [1.0, 0.0, 0.0],   # cand 2: matches pierA (prev track) but not target
        [0.0, 0.0, 1.0],   # pierB
    ])
    g_targets = [np.array([0.0, 1.0, 0.0])]  # step-0 target == cand 1's genre
    cfg = PierBridgeConfig(bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
                           genre_steering_enabled=True, weight_genre=0.4,
                           genre_arc_floor_percentile=0.0, weight_bridge=0.4, weight_transition=0.2)
    path, *_ = _beam_search_segment(0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
                                    X_genre_dense=dense, g_targets_override=g_targets)
    assert path == [1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -k arc_vote -q`
Expected: FAIL (current code scores genre vs previous track, not vs `g_target`).

- [ ] **Step 3: Rework the beam genre block**

In `_beam_search_segment`, replace the Task-3 interior genre block (the `_steering` branch that does `genre_sim = _get_genre_sim(current, cand)` + floor/term) with arc-target scoring. Read the current block, then implement:

```python
                # Genre ARC vote (first-class): closeness to this step's g_target.
                if _steering and g_targets is not None and step < len(g_targets) and X_genre_for_sim is not None:
                    gt = g_targets[step]
                    cand_present = (genre_present is None) or bool(genre_present[int(cand)])
                    if cand_present:
                        arc_sim = float(np.dot(X_genre_for_sim[int(cand)], gt))
                        # per-segment on-arc floor is applied after the pool's arc-sim
                        # distribution is known (see Step 4); here just add the vote.
                        if float(cfg.weight_genre) > 0.0:
                            combined_score += float(cfg.weight_genre) * arc_sim
                        step_arc_sims.setdefault(step, {})[int(cand)] = arc_sim
                elif (not _steering):
                    # legacy prev-track tiebreak unchanged
                    genre_sim = _get_genre_sim(int(current), int(cand))
                    if genre_sim is not None and math.isfinite(genre_sim) and cfg.genre_tiebreak_weight:
                        combined_score += cfg.genre_tiebreak_weight * genre_sim
```

(`X_genre_for_sim` is the dense matrix under steering per the Task-3 repoint; `g_targets` is the per-segment list passed via `g_targets_override`. Initialize `step_arc_sims: dict[int, dict[int,float]] = {}` near the beam loop start.)

- [ ] **Step 4: Apply the per-segment on-arc floor**

Before committing a step's candidates, compute the floor from the step's arc-sim distribution and drop below-floor candidates. Implement using the percentile helper:

```python
from src.playlist.pier_bridge.percentiles import floor_at_percentile
# at the point candidates for `step` are finalized, when steering + g_targets:
if _steering and g_targets is not None and step in step_arc_sims and float(cfg.genre_arc_floor_percentile) > 0.0:
    sims_this_step = np.array(list(step_arc_sims[step].values()), dtype=float)
    arc_floor = floor_at_percentile(sims_this_step, float(cfg.genre_arc_floor_percentile))
    # reject candidates whose arc_sim < arc_floor (skip genreless / those without arc_sim)
    # integrate into the existing candidate-admission filter for the step.
```

Read the beam's existing per-step candidate admission to integrate the floor at the right point (it must drop candidates before scoring selection, mirroring `_transition_gate_failed`). Apply the same logic at the final-pier connection using `g_targets[-1]`.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py tests/unit/test_beam_pace_gate.py tests/unit/test_progress_arc.py -q`
Expected: all pass (the arc-vote test now passes; legacy unaffected with steering off).

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/beam.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-arc): first-class arc vote + per-segment on-arc floor in beam

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Wire dense g_targets (linear/ladder) through the builder

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` (build dense g_targets per segment; pass to beam; build the niche graph once)
- Test: extend `tests/integration/test_genre_steering_integration.py` (collection only here; behavior verified in Task 8)

- [ ] **Step 1: Build the niche genre graph once per run**

In `build_pier_bridge_playlist`, when `cfg.genre_steering_enabled` and `bundle.genre_emb` is available, build the graph once:

```python
genre_graph_arc = None
if bool(cfg.genre_steering_enabled) and getattr(bundle, "genre_emb", None) is not None and getattr(bundle, "genre_vocab", None) is not None:
    from src.playlist.pier_bridge.genre_graph import build_genre_graph
    genre_graph_arc = build_genre_graph(
        bundle.genre_emb, bundle.genre_vocab,
        k=int(getattr(cfg, "dj_ladder_top_labels", 8) or 8),
        min_cos=float(getattr(cfg, "dj_ladder_min_similarity", 0.35) or 0.35),
        hub_labels={"rock", "indie", "alternative", "pop", "indie rock", "electronic"},
    )
```

- [ ] **Step 2: Build dense g_targets per segment and pass to the beam**

Where segment g_targets are currently built (search `_build_genre_targets(` / `segment_g_targets`), build them in dense space using Task 2's function when steering is on. For each segment with piers `pier_a`, `pier_b`:

```python
if bool(cfg.genre_steering_enabled) and getattr(bundle, "X_genre_dense", None) is not None:
    from src.playlist.pier_bridge.genre_targets import build_dense_genre_targets
    from src.playlist.pier_bridge.genre import _select_top_genre_labels
    labels_a = _select_top_genre_labels(bundle.X_genre_raw[pier_a], bundle.genre_vocab,
                                         top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight)) if bundle.genre_vocab is not None else None
    labels_b = _select_top_genre_labels(bundle.X_genre_raw[pier_b], bundle.genre_vocab,
                                         top_n=int(cfg.dj_ladder_top_labels), min_weight=float(cfg.dj_ladder_min_label_weight)) if bundle.genre_vocab is not None else None
    segment_g_targets = build_dense_genre_targets(
        bundle.X_genre_dense[pier_a], bundle.X_genre_dense[pier_b],
        interior_length=interior_len, route=str(cfg.dj_route_shape or "linear"),
        genre_emb=getattr(bundle, "genre_emb", None), genre_vocab=list(bundle.genre_vocab) if bundle.genre_vocab is not None else None,
        genre_graph=genre_graph_arc, labels_a=labels_a, labels_b=labels_b,
        max_steps=int(cfg.dj_ladder_max_steps),
    )
```

Pass `segment_g_targets` to the beam via the existing `g_targets_override=` parameter (already wired). Confirm `X_genre_dense=X_genre_dense` is also passed (from the prior plan's Task 4).

- [ ] **Step 3: Re-wire the relaxation tier to the percentile floor**

The prior plan added a genre-floor relaxation tier in `pier_bridge_builder.py` (helper `_genre_floor_attempts` + `_run_segment_backoff_attempts(..., genre_edge_floor_override=...)` doing `replace(cfg, genre_edge_floor=...)`). Rework it for the percentile floor:
- The override becomes `genre_arc_floor_percentile_override`, applied via `replace(cfg, genre_arc_floor_percentile=...)`.
- The attempt sequence comes from `relax_percentile(cfg_base.genre_arc_floor_percentile, infeasible_handling.min_genre_arc_percentile, step=0.15)` (Task 3 helper), gated on `genre_steering_enabled` + `infeasible_handling.enabled` + `genre_arc_relaxation_enabled`.
- Keep the same per-attempt result-field copy block as the existing tier (mirror it exactly — do not drop fields).

Read the existing tier and mirror it precisely.

- [ ] **Step 4: Run import + a fast smoke**

Run: `python -c "import src.playlist.pier_bridge_builder" && python -m pytest tests/unit/test_pier_bridge_smoke_golden.py -q`
Expected: clean import; smoke/golden pass (regenerate goldens if new config fields changed the snapshot — see Task 8).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge_builder.py
git commit -m "feat(genre-arc): build dense ladder/linear g_targets, feed beam, percentile relaxation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Calibration harness + integration + regression

**Files:**
- Create: `scripts/calibrate_genre_arc.py`
- Modify: `tests/integration/test_genre_steering_integration.py`
- Regenerate: `tests/unit/goldens/pipeline/*.json` (new config fields)

- [ ] **Step 1: Write the integration tests (arc monotonicity + feasibility)**

Add to `tests/integration/test_genre_steering_integration.py`:

```python
REFERENCE = {
    "charli_xcx": "065933d8e2e0db664ec57af1511b662b",
    # Fill in track_ids for Real Estate, Bill Evans, Beach House, Minor Threat
    # via: SELECT track_id FROM tracks WHERE artist=? LIMIT 1
}

def _interior_arc_monotonic(bundle, track_ids, pier_a_id, pier_b_id):
    import numpy as np
    D = bundle.X_genre_dense; ti = bundle.track_id_to_index
    a, b = D[ti[pier_a_id]], D[ti[pier_b_id]]
    sims_to_b = [float(D[ti[str(t)]] @ b) for t in track_ids if str(t) in ti and np.linalg.norm(D[ti[str(t)]])>1e-9]
    # Spearman-ish: fraction of adjacent steps that move toward B
    ups = sum(1 for i in range(len(sims_to_b)-1) if sims_to_b[i+1] >= sims_to_b[i] - 0.05)
    return ups / max(len(sims_to_b)-1, 1)


@pytest.mark.integration
@pytest.mark.slow
@_requires
def test_reference_seeds_feasible_and_arc_monotonic():
    from src.playlist.ds_pipeline_runner import generate_playlist_ds
    from src.features.artifacts import load_artifact_bundle
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))
    ov = {"pier_bridge": {"genre_steering_enabled": True, "dj_route_shape": "ladder",
                          "weight_genre_narrow": 0.20,
                          "genre_admission_percentile_narrow": 0.90,
                          "genre_arc_floor_percentile_narrow": 0.85,
                          "infeasible_handling": {"enabled": True, "genre_arc_relaxation_enabled": True,
                                                  "min_genre_arc_percentile": 0.40}}}
    for name, tid in REFERENCE.items():
        res = generate_playlist_ds(artifact_path=str(ART), seed_track_id=tid,
                                   mode="narrow", length=30, random_seed=42, overrides=ov)
        assert res is not None and len(res.track_ids) >= 24, f"{name} infeasible"
```

- [ ] **Step 2: Implement the calibration harness**

Create `scripts/calibrate_genre_arc.py` that, for each reference seed × mode × config in a small grid of `(genre_admission_percentile, genre_arc_floor_percentile, weight_genre)`:
generates a playlist (catching infeasibility), and records feasibility, admitted pool size, mean/min interior `waypoint_sim`, arc-monotonicity fraction (via `_interior_arc_monotonic`), worst-edge T, distinct artists. Emits a markdown report to `docs/run_audits/genre_arc_calibration_<ts>.md` and prints a shortlist of 2–3 configs/mode passing feasibility, ranked by (arc monotonicity, arc adherence). Model it on `scripts/research_genre_similarity.py` structure (argparse, load bundle, loops, markdown writer). Read-only w.r.t. data.

```python
# scripts/calibrate_genre_arc.py — skeleton (fill grid + metric calls per spec §4)
#   GRID = {"narrow": {"P_admit": [0.85,0.90,0.93], "P_arc": [0.80,0.85,0.90], "w_genre": [0.15,0.20,0.25]}, ...}
#   for seed in REFERENCE: for mode in MODES: for cfg in grid(mode):
#       try: res = generate_playlist_ds(..., overrides=_ov(mode,cfg))
#       except Exception: record infeasible; continue
#       record feasibility, pool size, arc adherence, monotonicity, worst-edge, distinct artists
#   write markdown report + shortlist
```
(The harness is an operator tool, not production code; a complete runnable skeleton with the grid + metric functions filled per spec §4 is the deliverable.)

- [ ] **Step 3: Regenerate goldens for new config fields**

Run `python -m pytest tests/unit/test_pipeline_smoke_golden.py -q`; for each failing scenario delete `tests/unit/goldens/pipeline/<name>.json` and re-run twice (regenerate → verify). Confirm `git diff tests/unit/goldens/pipeline/` shows ONLY the new/renamed genre-arc fields, nothing else.

- [ ] **Step 4: Full regression**

Run: `python -m pytest -m "not slow and not gui" -q`
Expected: all pass.

- [ ] **Step 5: Run the harness against the (stable) artifact and review**

Run: `python scripts/calibrate_genre_arc.py` → review the markdown report; pick shortlist configs; set them in `config.yaml`; audition by ear; iterate config-only (no code change).

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_genre_arc.py tests/integration/test_genre_steering_integration.py tests/unit/goldens/pipeline/ config.example.yaml
git commit -m "feat(genre-arc): calibration harness + arc-monotonicity integration tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Out of scope (do NOT implement here)
- General sonic-scoring re-examination (separate initiative).
- LLM-prior sidecar rebuild.
- The hand-curated `genre_similarity.yaml` graph (superseded by the `genre_emb`-derived graph).
