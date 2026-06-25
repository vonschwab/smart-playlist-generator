# Artist-mode Energy-Aware Spread — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make artist-mode anchor "piers" tile the artist's arousal/energy range, not just timbre, by adding an energy term to the medoid score — so the playlist has an energy arc instead of a flat line.

**Architecture:** Clustering stays pure sonic (MERT). After k-means, each cluster is assigned an evenly-spaced energy *slot* across the artist's robust arousal span (ranked by the cluster's median arousal). A new `energy_slot_proximity` term joins the existing `0.7·sonic + 0.3·duration` medoid score, pulling each cluster's medoid toward its slot. All-zero weight reproduces today's output byte-for-byte. Popularity ("Popular Seeds") is a **separate, later plan** — this plan leaves a `w_pop` hook unused.

**Tech Stack:** Python 3.11+, NumPy. Reuses `src/playlist/energy_loader.py` (Essentia energy sidecar) and `src/playlist/artist_style.py`.

**Spec:** `docs/superpowers/specs/2026-06-23-artist-energy-spread-popular-seeds-design.md` (Component 1 + Component 4 + Component 5).

## Global Constraints

- **Python 3.11+** (pinned in `pyproject.toml`).
- **Opt-in, backward-compatible:** `medoid_energy_weight: 0.0` and absent/empty energy ⇒ **byte-identical to today's medoid selection**. This is a hard acceptance test, not a nicety.
- **A configured knob that can't act is a startup/runtime error, not a silent no-op:** if `medoid_energy_weight > 0` but the energy sidecar is missing or has no finite values, **log a loud WARNING** and fall back to inert (do not crash, do not silently no-op without a log).
- **Config lives at `playlists.ds_pipeline.artist_style.*`** in `config.yaml` (NOT `playlists.artist_style`). `ArtistStyleConfig` is constructed at TWO sites: `src/playlist_generator.py:1625` and `:2554`.
- **Energy is read-only, local-first:** the energy sidecar (`data/artifacts/beat3tower_32k/energy/energy_sidecar.npz`) is only ever read. Never written. `metadata.db` is never touched by this plan.
- **Tests:** `python -m pytest -q` directly (never pipe through `tail`/`head`; use the tool timeout). Markers unaffected.
- **Energy is used ONLY to spread candidate *seed* tracks** (medoid selection). Energy-aware *arc ordering* (reordering piers by energy) is explicitly OUT OF SCOPE here.

## File Structure

- **Modify** `src/playlist/artist_style.py` — add config fields to `ArtistStyleConfig`; add pure helpers (`_robust_energy_span`, `_slot_targets_by_rank`, `_slot_proximity`, `_finite_median`); add `load_artist_energy_values`; extend `_medoids_for_cluster` with an energy term; wire energy slots into `cluster_artist_tracks`.
- **Modify** `src/playlist_generator.py:1625` and `:2554` — map the new config keys into `ArtistStyleConfig`.
- **Modify** `config.example.yaml` — document the new `artist_style` knobs.
- **Modify** `tests/test_artist_style.py` — unit + integration tests.
- **Create** `scripts/research/energy_spread_eval.py` — A/B pier-arousal-span measurement across an artist panel.
- **Create** `tests/research/test_energy_spread_eval.py` — smoke test for the eval metric function.

All clustering changes live in one focused file (`artist_style.py`, ~600 LOC — well within bounds). No new modules needed.

---

### Task 1: Config fields on `ArtistStyleConfig` + wire both generator sites + document

**Files:**
- Modify: `src/playlist/artist_style.py:53-79` (the `ArtistStyleConfig` dataclass)
- Modify: `src/playlist_generator.py:1625` and `src/playlist_generator.py:2554` (both `ArtistStyleConfig(...)` construction sites)
- Modify: `config.example.yaml` (document under `playlists.ds_pipeline.artist_style`)
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Produces: `ArtistStyleConfig.medoid_energy_weight: float = 0.0`, `.energy_feature: str = "arousal_p50"`, `.energy_slot_lo_pct: float = 10.0`, `.energy_slot_hi_pct: float = 90.0`. Later tasks read these off `cfg`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artist_style.py`:

```python
def test_artist_style_config_has_energy_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.medoid_energy_weight == 0.0          # opt-in: off by default
    assert cfg.energy_feature == "arousal_p50"
    assert cfg.energy_slot_lo_pct == 10.0
    assert cfg.energy_slot_hi_pct == 90.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_artist_style.py::test_artist_style_config_has_energy_defaults -v`
Expected: FAIL — `AttributeError: 'ArtistStyleConfig' object has no attribute 'medoid_energy_weight'`

- [ ] **Step 3: Add the fields to the dataclass**

In `src/playlist/artist_style.py`, after the existing medoid weights (line 79, `medoid_duration_weight`), add:

```python
    medoid_duration_weight: float = 0.3    # Weight for duration typicality (avoid outliers)
    # Energy-aware spread (set-level): pull each cluster's medoid toward an
    # evenly-spaced arousal slot so the pier set tiles the artist's energy range.
    # 0.0 => inert (today's behavior). See spec 2026-06-23-artist-energy-spread.
    medoid_energy_weight: float = 0.0
    energy_feature: str = "arousal_p50"    # which energy sidecar column defines the slots
    energy_slot_lo_pct: float = 10.0       # robust span low percentile of artist z-arousal
    energy_slot_hi_pct: float = 90.0       # robust span high percentile
```

- [ ] **Step 4: Wire both generator construction sites**

In `src/playlist_generator.py`, locate the `ArtistStyleConfig(` at line 1625 and add these keyword args alongside the existing `medoid_*` ones (read the block first to place them with the other medoid keys; the raw dict variable is `style_cfg_raw`):

```python
            medoid_energy_weight=float(style_cfg_raw.get("medoid_energy_weight", 0.0)),
            energy_feature=str(style_cfg_raw.get("energy_feature", "arousal_p50")),
            energy_slot_lo_pct=float(style_cfg_raw.get("energy_slot_lo_pct", 10.0)),
            energy_slot_hi_pct=float(style_cfg_raw.get("energy_slot_hi_pct", 90.0)),
```

Repeat the **identical** addition at the second site, `src/playlist_generator.py:2554` (its raw dict is also `style_cfg_raw`). Both sites must stay in sync.

- [ ] **Step 5: Document the knobs in `config.example.yaml`**

Find the `artist_style:` block under `playlists: ds_pipeline:` and add (matching surrounding indentation/comment style):

```yaml
        # Energy-aware spread (artist mode): pull each cluster's medoid toward an
        # evenly-spaced arousal slot so the piers tile the artist's energy range.
        # 0.0 = off (default, identical to legacy). Calibrate upward after A/B.
        medoid_energy_weight: 0.0
        energy_feature: arousal_p50      # arousal_p10 | arousal_p50 | arousal_p90 | danceability
        energy_slot_lo_pct: 10.0         # robust arousal-span percentiles for slot targets
        energy_slot_hi_pct: 90.0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_artist_style.py::test_artist_style_config_has_energy_defaults -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist/artist_style.py src/playlist_generator.py config.example.yaml tests/test_artist_style.py
git commit -m "feat(artist-style): add energy-spread config knobs (default off)"
```

---

### Task 2: Pure helpers — robust span, slot targets, slot proximity

**Files:**
- Modify: `src/playlist/artist_style.py` (add module-level helper functions near the other `_` helpers, e.g. after `_duration_outlier_score`)
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Produces:
  - `_finite_median(values: np.ndarray) -> float` — median of finite entries, `np.nan` if none.
  - `_robust_energy_span(values: np.ndarray, lo_pct: float, hi_pct: float) -> Optional[Tuple[float, float]]` — `(lo, hi)` from finite percentiles, or `None` if fewer than 2 finite values or span below epsilon.
  - `_slot_targets_by_rank(cluster_medians: Sequence[float], span: Tuple[float, float]) -> List[float]` — one target per cluster, evenly spaced across `span`, assigned by ascending median-energy rank. `np.nan` median ⇒ `np.nan` target (that cluster's energy term goes inert).
  - `_slot_proximity(z: np.ndarray, target: float, span_width: float) -> np.ndarray` — per-member proximity `1 − clip(|z−target|/span_width, 0, 1)` in `[0,1]`; `0.0` where `z` is non-finite, all-zeros if `target` non-finite or `span_width <= 0`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artist_style.py`:

```python
from src.playlist.artist_style import (
    _finite_median,
    _robust_energy_span,
    _slot_targets_by_rank,
    _slot_proximity,
)


def test_finite_median_ignores_nan():
    assert _finite_median(np.array([1.0, np.nan, 3.0])) == 2.0
    assert np.isnan(_finite_median(np.array([np.nan, np.nan])))


def test_robust_energy_span_uses_percentiles():
    vals = np.arange(10, dtype=float)  # 0..9
    span = _robust_energy_span(vals, 10.0, 90.0)
    assert span is not None
    lo, hi = span
    assert lo == pytest.approx(0.9)
    assert hi == pytest.approx(8.1)


def test_robust_energy_span_none_when_flat_or_sparse():
    assert _robust_energy_span(np.array([5.0, 5.0, 5.0]), 10.0, 90.0) is None  # zero span
    assert _robust_energy_span(np.array([np.nan, 1.0]), 10.0, 90.0) is None     # <2 finite


def test_slot_targets_even_spacing_by_rank():
    # medians in input order: cluster0=2.0 (highest), cluster1=0.0 (lowest), cluster2=1.0 (mid)
    targets = _slot_targets_by_rank([2.0, 0.0, 1.0], (0.0, 10.0))
    assert targets[1] == pytest.approx(0.0)    # lowest-energy cluster -> low slot
    assert targets[2] == pytest.approx(5.0)    # mid
    assert targets[0] == pytest.approx(10.0)   # highest-energy cluster -> high slot


def test_slot_targets_single_cluster_is_midpoint():
    assert _slot_targets_by_rank([3.0], (0.0, 10.0)) == [pytest.approx(5.0)]


def test_slot_targets_nan_median_stays_nan():
    targets = _slot_targets_by_rank([np.nan, 1.0], (0.0, 10.0))
    assert np.isnan(targets[0])


def test_slot_proximity_peaks_at_target_and_zeros_for_nan():
    z = np.array([5.0, 0.0, 10.0, np.nan])
    prox = _slot_proximity(z, target=5.0, span_width=10.0)
    assert prox[0] == pytest.approx(1.0)     # at target
    assert prox[1] == pytest.approx(0.5)     # half a span away
    assert prox[2] == pytest.approx(0.5)
    assert prox[3] == 0.0                     # NaN energy -> neutral (no bonus)


def test_slot_proximity_inert_when_target_nan():
    z = np.array([1.0, 2.0])
    assert np.all(_slot_proximity(z, target=np.nan, span_width=10.0) == 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -k "finite_median or energy_span or slot_targets or slot_proximity" -v`
Expected: FAIL — `ImportError: cannot import name '_finite_median'`

- [ ] **Step 3: Implement the helpers**

In `src/playlist/artist_style.py`, add after `_duration_outlier_score` (around line 188):

```python
_ENERGY_SPAN_EPS = 1e-6


def _finite_median(values: np.ndarray) -> float:
    """Median of finite entries; NaN if there are none."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def _robust_energy_span(
    values: np.ndarray, lo_pct: float, hi_pct: float
) -> Optional[Tuple[float, float]]:
    """Robust (lo, hi) energy span from finite percentiles, or None if degenerate."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if (hi - lo) < _ENERGY_SPAN_EPS:
        return None
    return (lo, hi)


def _slot_targets_by_rank(
    cluster_medians: Sequence[float], span: Tuple[float, float]
) -> List[float]:
    """Evenly-spaced energy targets across `span`, one per cluster, by median-energy rank.

    Clusters with a NaN median get a NaN target (their energy term is inert).
    Single cluster -> midpoint. Targets are returned aligned to the input order.
    """
    lo, hi = span
    medians = list(cluster_medians)
    n = len(medians)
    targets = [float("nan")] * n
    finite_idx = [i for i, m in enumerate(medians) if np.isfinite(m)]
    k = len(finite_idx)
    if k == 0:
        return targets
    if k == 1:
        targets[finite_idx[0]] = (lo + hi) / 2.0
        return targets
    # rank finite clusters by ascending median energy, space targets across [lo, hi]
    ordered = sorted(finite_idx, key=lambda i: medians[i])
    for rank, i in enumerate(ordered):
        targets[i] = lo + (rank / (k - 1)) * (hi - lo)
    return targets


def _slot_proximity(z: np.ndarray, target: float, span_width: float) -> np.ndarray:
    """Per-member proximity to a slot target in [0,1]; 0 for non-finite members.

    Inert (all zeros) if the target is non-finite or span_width <= 0.
    """
    arr = np.asarray(z, dtype=float)
    if not np.isfinite(target) or span_width <= 0:
        return np.zeros_like(arr)
    dist = np.abs(arr - target) / span_width
    prox = 1.0 - np.clip(dist, 0.0, 1.0)
    prox[~np.isfinite(arr)] = 0.0
    return prox
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -k "finite_median or energy_span or slot_targets or slot_proximity" -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): pure helpers for energy slot targets + proximity"
```

---

### Task 3: Energy term in `_medoids_for_cluster`

**Files:**
- Modify: `src/playlist/artist_style.py:191-275` (`_medoids_for_cluster` signature + score combination at line 234)
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_medoids_for_cluster(..., energy_weight: float = 0.0, energy_proximity: Optional[np.ndarray] = None)` — `energy_proximity` is aligned to `indices`. When provided and `energy_weight > 0`, the combined score becomes `sims·w_sim + dur·w_dur + energy_proximity·energy_weight`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artist_style.py`:

```python
from src.playlist.artist_style import _medoids_for_cluster


def _centroid_for(X, indices):
    c = X[indices].mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


def test_medoid_energy_term_pulls_to_slot():
    # 3 candidates, near-identical sonic centrality; energy proximity favors index 1.
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    rng = np.random.default_rng(0)

    # Baseline (no energy): pick by sonic alone, top_k=1 => deterministic argmax.
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng, 1,
        None, None, 0.7, 0.3,
    )
    # Energy strongly favors index 1.
    rng2 = np.random.default_rng(0)
    energized = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng2, 1,
        None, None, 0.7, 0.3,
        10.0, np.array([0.0, 1.0, 0.0]),   # energy_weight, energy_proximity
    )
    assert energized == ["t1"]
    assert energized != base or base == ["t1"]  # energy moved (or already was) the pick


def test_medoid_energy_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
    )
    with_zero = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
        0.0, np.array([1.0, 0.0, 0.0]),   # weight 0 => proximity ignored
    )
    assert with_zero == base
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -k "medoid_energy" -v`
Expected: FAIL — `_medoids_for_cluster() takes ... positional arguments but 14 were given`

- [ ] **Step 3: Extend the signature and score**

In `src/playlist/artist_style.py`, change the `_medoids_for_cluster` signature (line ~191) to add the two new trailing params:

```python
def _medoids_for_cluster(
    X: np.ndarray,
    indices: List[int],
    centroid: np.ndarray,
    bundle_track_ids: Sequence[str],
    per_cluster: int,
    rng: np.random.Generator,
    top_k: int,
    artist_duration_stats: Optional[Dict[str, float]] = None,
    track_durations_ms: Optional[np.ndarray] = None,
    similarity_weight: float = 0.7,
    duration_weight: float = 0.3,
    energy_weight: float = 0.0,
    energy_proximity: Optional[np.ndarray] = None,
) -> List[int]:
```

Then change the score combination (currently line 234) from:

```python
    # Combined weighted score (configurable weights)
    scores = sims * similarity_weight + duration_weights * duration_weight
```

to:

```python
    # Combined weighted score (configurable weights)
    scores = sims * similarity_weight + duration_weights * duration_weight

    # Energy-aware spread: pull the medoid toward this cluster's arousal slot.
    if energy_proximity is not None and energy_weight > 0:
        prox = np.asarray(energy_proximity, dtype=float)
        if prox.shape[0] == len(indices):
            scores = scores + prox * energy_weight
        else:  # defensive: misaligned proximity must never silently corrupt scores
            logger.warning(
                "artist_style: energy_proximity len %d != cluster size %d; skipping energy term",
                prox.shape[0], len(indices),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -k "medoid_energy" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): energy-slot term in medoid scoring (weight-gated)"
```

---

### Task 4: `load_artist_energy_values` — derive sidecar path, load, warn-if-weighted-but-missing

**Files:**
- Modify: `src/playlist/artist_style.py` (add function; add `from pathlib import Path` if not already imported)
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Produces: `load_artist_energy_values(bundle, cfg: ArtistStyleConfig) -> Optional[np.ndarray]` — z-scored energy vector aligned to `bundle.track_ids`, or `None` when the term is inert (weight ≤ 0, no `artifact_path`, missing sidecar, or no finite values). Loud WARNING on the "weighted but unavailable" paths.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artist_style.py`:

```python
import types
from pathlib import Path
from src.playlist.artist_style import load_artist_energy_values


def _write_energy_sidecar(tmp_path, track_ids, arousal):
    energy_dir = tmp_path / "energy"
    energy_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        energy_dir / "energy_sidecar.npz",
        track_ids=np.array(track_ids, dtype=object),
        arousal_p50=np.array(arousal, dtype=np.float32),
    )


def test_load_artist_energy_values_returns_zscored(tmp_path):
    track_ids = ["a", "b", "c"]
    _write_energy_sidecar(tmp_path, track_ids, [1.0, 3.0, 5.0])
    bundle = types.SimpleNamespace(
        track_ids=np.array(track_ids), artifact_path=tmp_path / "artifact.npz"
    )
    cfg = ArtistStyleConfig(medoid_energy_weight=1.0, energy_feature="arousal_p50")
    vals = load_artist_energy_values(bundle, cfg)
    assert vals is not None and vals.shape == (3,)
    assert vals[0] < vals[1] < vals[2]            # preserves ordering
    assert abs(float(np.mean(vals))) < 1e-6        # z-scored => ~zero mean


def test_load_artist_energy_values_inert_when_weight_zero(tmp_path):
    bundle = types.SimpleNamespace(
        track_ids=np.array(["a"]), artifact_path=tmp_path / "artifact.npz"
    )
    assert load_artist_energy_values(bundle, ArtistStyleConfig()) is None


def test_load_artist_energy_values_warns_when_sidecar_missing(tmp_path, caplog):
    bundle = types.SimpleNamespace(
        track_ids=np.array(["a"]), artifact_path=tmp_path / "artifact.npz"
    )
    cfg = ArtistStyleConfig(medoid_energy_weight=0.5)
    import logging
    with caplog.at_level(logging.WARNING):
        assert load_artist_energy_values(bundle, cfg) is None
    assert any("energy sidecar missing" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -k "load_artist_energy" -v`
Expected: FAIL — `ImportError: cannot import name 'load_artist_energy_values'`

- [ ] **Step 3: Implement the loader**

At the top of `src/playlist/artist_style.py`, ensure `from pathlib import Path` is imported. Then add (near the other module helpers):

```python
def load_artist_energy_values(bundle, cfg: "ArtistStyleConfig") -> Optional[np.ndarray]:
    """Load z-scored energy aligned to bundle.track_ids for energy-aware spread.

    Returns None (inert) when the energy term is off or the sidecar is unavailable.
    Per the configured-knob-must-act rule, a >0 weight with no usable energy WARNs.
    """
    if cfg.medoid_energy_weight <= 0:
        return None
    artifact_path = getattr(bundle, "artifact_path", None)
    if artifact_path is None:
        logger.warning(
            "artist_style: medoid_energy_weight=%.3f but bundle has no artifact_path; "
            "energy spread inert", cfg.medoid_energy_weight,
        )
        return None
    sidecar = Path(artifact_path).parent / "energy" / "energy_sidecar.npz"
    if not sidecar.exists():
        logger.warning(
            "artist_style: medoid_energy_weight=%.3f but energy sidecar missing at %s; "
            "energy spread inert", cfg.medoid_energy_weight, sidecar,
        )
        return None
    from src.playlist.energy_loader import load_energy_matrix

    matrix = load_energy_matrix(
        bundle.track_ids, sidecar_path=str(sidecar), features=(cfg.energy_feature,)
    )
    vals = np.asarray(matrix[:, 0], dtype=float)
    if not np.any(np.isfinite(vals)):
        logger.warning(
            "artist_style: energy sidecar has no finite %s values; energy spread inert",
            cfg.energy_feature,
        )
        return None
    return vals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -k "load_artist_energy" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): load_artist_energy_values with loud-on-missing"
```

---

### Task 5: Wire energy slots into `cluster_artist_tracks`

**Files:**
- Modify: `src/playlist/artist_style.py:278-410` (`cluster_artist_tracks` — signature + the cluster/medoid loop at 352-374)
- Test: `tests/test_artist_style.py`

**Interfaces:**
- Consumes: `load_artist_energy_values`, `_robust_energy_span`, `_finite_median`, `_slot_targets_by_rank`, `_slot_proximity`, extended `_medoids_for_cluster`.
- Produces: `cluster_artist_tracks(..., energy_values: Optional[np.ndarray] = None)` — new optional kwarg for test injection. When `None`, the function loads energy itself via `load_artist_energy_values`. Return tuple unchanged: `(clusters, medoids, medoids_by_cluster, X_norm)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artist_style.py`:

```python
def _two_cluster_bundle():
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],   # cluster 1 (indices 0,1,2)
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],   # cluster 2 (indices 3,4,5)
    ])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def test_cluster_artist_tracks_energy_weight_zero_matches_none():
    bundle = _two_cluster_bundle()
    cfg_off = ArtistStyleConfig(cluster_k_min=2, cluster_k_max=2, enabled=True)
    cfg_zero = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=0.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg_off, random_seed=0)
    zeroed = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg_zero, random_seed=0, energy_values=energy
    )
    assert sorted(base[1]) == sorted(zeroed[1])   # identical medoids


def test_cluster_artist_tracks_energy_runs_and_returns_medoids():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    clusters, medoids, by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=energy
    )
    assert len(clusters) == 2
    assert len(medoids) == 2


def test_cluster_artist_tracks_inert_on_flat_energy():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    flat = np.zeros(6)   # zero span => energy term inert, must not crash
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg, random_seed=0)
    flatted = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=flat
    )
    assert sorted(base[1]) == sorted(flatted[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_artist_style.py -k "cluster_artist_tracks_energy or inert_on_flat" -v`
Expected: FAIL — `cluster_artist_tracks() got an unexpected keyword argument 'energy_values'`

- [ ] **Step 3: Add the kwarg and load fallback**

In `src/playlist/artist_style.py`, add `energy_values` to the `cluster_artist_tracks` signature (after `excluded_track_ids`):

```python
    excluded_track_ids: Optional[set[str]] = None,
    energy_values: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], List[int], List[List[int]], np.ndarray]:
```

Immediately after `artist_indices` is finalized and the `< max(3, cfg.cluster_k_min)` guard passes (just before `k = _select_k(...)`, line ~325), add:

```python
    # Energy-aware spread: load energy if not injected, then derive the artist's
    # robust arousal span (slots are spaced across it). None => term stays inert.
    if energy_values is None:
        energy_values = load_artist_energy_values(bundle, cfg)
    energy_span: Optional[Tuple[float, float]] = None
    if energy_values is not None and cfg.medoid_energy_weight > 0:
        artist_energy = np.asarray(energy_values, dtype=float)[artist_indices]
        energy_span = _robust_energy_span(
            artist_energy, cfg.energy_slot_lo_pct, cfg.energy_slot_hi_pct
        )
```

- [ ] **Step 4: Restructure the medoid loop to assign slots**

Replace the existing cluster/medoid loop (currently lines 352-374, from `clusters: List[List[int]] = []` through `medoids.extend(medoid_list)`) with:

```python
    # First pass: gather non-empty clusters, preserving their centroid index.
    nonempty: List[Tuple[int, List[int]]] = []
    for c in range(centroids.shape[0]):
        members_local = [artist_indices[i] for i, lab in enumerate(labels) if lab == c]
        if members_local:
            nonempty.append((c, members_local))

    # Energy slots: rank clusters by median arousal, space targets across the span.
    slot_targets: Optional[List[float]] = None
    if energy_span is not None:
        ev = np.asarray(energy_values, dtype=float)
        cluster_medians = [_finite_median(ev[members]) for _c, members in nonempty]
        slot_targets = _slot_targets_by_rank(cluster_medians, energy_span)
        logger.info(
            "Artist style energy spread: artist=%s span=(%.3f,%.3f) targets=%s",
            artist_name, energy_span[0], energy_span[1],
            [round(t, 3) if np.isfinite(t) else None for t in slot_targets],
        )

    clusters: List[List[int]] = []
    medoids: List[int] = []
    medoids_by_cluster: List[List[int]] = []
    span_width = (energy_span[1] - energy_span[0]) if energy_span is not None else 0.0
    for ci, (c, members_local) in enumerate(nonempty):
        clusters.append(members_local)
        energy_prox: Optional[np.ndarray] = None
        if slot_targets is not None:
            member_energy = np.asarray(energy_values, dtype=float)[members_local]
            energy_prox = _slot_proximity(member_energy, slot_targets[ci], span_width)
        medoid_list = _medoids_for_cluster(
            X_norm,
            members_local,
            centroids[c],
            track_ids,
            medoid_top_k,
            rng,
            medoid_top_k,
            artist_duration_stats,
            bundle.durations_ms,
            cfg.medoid_similarity_weight,
            cfg.medoid_duration_weight,
            cfg.medoid_energy_weight,
            energy_prox,
        )
        medoids_by_cluster.append(medoid_list)
        medoids.extend(medoid_list)
```

(The downstream `logger.info("Artist style clustering: ...")` and intra/inter diagnostics block are unchanged and still follow this loop.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_artist_style.py -k "cluster_artist_tracks_energy or inert_on_flat" -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full artist_style test file (regression)**

Run: `python -m pytest tests/test_artist_style.py -v`
Expected: PASS (all pre-existing tests + the new ones)

- [ ] **Step 7: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat(artist-style): assign energy slots in cluster_artist_tracks"
```

---

### Task 6: A/B eval script — pier arousal-span measurement

**Files:**
- Create: `scripts/research/energy_spread_eval.py`
- Create: `tests/research/test_energy_spread_eval.py`

**Interfaces:**
- Consumes: `cluster_artist_tracks`, `order_clusters`, `load_energy_matrix`.
- Produces: `pier_arousal_span(medoid_track_ids: Sequence[str], track_ids: Sequence[str], arousal: np.ndarray) -> float` — the z-arousal span (max − min over finite) of a medoid set; `0.0` if <2 finite. The CLI runs energy-off vs energy-on for an artist panel and prints per-artist + mean span delta.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/research/test_energy_spread_eval.py`:

```python
import numpy as np

from scripts.research.energy_spread_eval import pier_arousal_span


def test_pier_arousal_span_basic():
    track_ids = ["a", "b", "c", "d"]
    arousal = np.array([0.0, 1.0, -1.0, np.nan])
    # medoids a,c span [−1, 0] => 1.0
    assert pier_arousal_span(["a", "c"], track_ids, arousal) == 1.0
    # single finite => 0.0
    assert pier_arousal_span(["a"], track_ids, arousal) == 0.0
    # NaN ignored
    assert pier_arousal_span(["a", "d"], track_ids, arousal) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/research/test_energy_spread_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.research.energy_spread_eval'`

- [ ] **Step 3: Implement the eval script**

Create `scripts/research/energy_spread_eval.py`:

```python
"""A/B the energy-aware spread: pier arousal-span, energy-off vs energy-on.

Measures the spec's primary "spread" metric (z-arousal span across the pier set)
for a panel of artists. Guardrails (transition T, diversity, wall-clock) require a
full multi-pier generation run via the gui_fidelity harness — see the
playlist-testing skill; this script only measures the selection-level spread.

Usage:
    python -m scripts.research.energy_spread_eval --artists "Nirvana" "Slowdive" \
        --energy-weight 5.0 --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.playlist.artist_style import ArtistStyleConfig, cluster_artist_tracks
from src.playlist.energy_loader import load_energy_matrix


def pier_arousal_span(
    medoid_track_ids: Sequence[str], track_ids: Sequence[str], arousal: np.ndarray
) -> float:
    """z-arousal span (max-min over finite) of a medoid set; 0.0 if <2 finite."""
    pos = {str(t): i for i, t in enumerate(track_ids)}
    vals = [arousal[pos[str(m)]] for m in medoid_track_ids if str(m) in pos]
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    return float(max(finite) - min(finite))


def _run(bundle, artist: str, energy_weight: float, energy: np.ndarray) -> float:
    cfg = ArtistStyleConfig(enabled=True, medoid_energy_weight=energy_weight)
    try:
        _clusters, medoids, _by_cluster, _X = cluster_artist_tracks(
            bundle=bundle, artist_name=artist, cfg=cfg, random_seed=0,
            energy_values=energy if energy_weight > 0 else None,
        )
    except ValueError as exc:
        print(f"  {artist}: skipped ({exc})")
        return float("nan")
    ids = [str(bundle.track_ids[m]) for m in medoids]
    return pier_arousal_span(ids, [str(t) for t in bundle.track_ids], energy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", nargs="+", required=True)
    ap.add_argument("--energy-weight", type=float, default=5.0)
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    args = ap.parse_args()

    bundle = load_artifact_bundle(Path(args.artifact))
    sidecar = Path(args.artifact).parent / "energy" / "energy_sidecar.npz"
    energy = load_energy_matrix(
        bundle.track_ids, sidecar_path=str(sidecar), features=("arousal_p50",)
    )[:, 0]

    deltas = []
    print(f"{'artist':30s} {'off':>8s} {'on':>8s} {'delta':>8s}")
    for artist in args.artists:
        off = _run(bundle, artist, 0.0, energy)
        on = _run(bundle, artist, args.energy_weight, energy)
        if np.isfinite(off) and np.isfinite(on):
            deltas.append(on - off)
            print(f"{artist:30s} {off:8.3f} {on:8.3f} {on - off:+8.3f}")

    if deltas:
        print(f"\nmean span delta (on - off): {np.mean(deltas):+.3f}  (n={len(deltas)})")
        print("ACCEPTANCE: mean delta should be > 0 (piers spread wider in energy).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the test package marker if missing**

Ensure `tests/research/__init__.py` exists (create empty if not):

```bash
test -f tests/research/__init__.py || touch tests/research/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/research/test_energy_spread_eval.py -v`
Expected: PASS

- [ ] **Step 6: Manual eval on real data (record the numbers)**

Run on a panel with good energy coverage (record output in the PR/commit message — do not skip; this IS the eval-gate's spread metric):

```bash
python -m scripts.research.energy_spread_eval \
  --artists "Nirvana" "Slowdive" "Miles Davis" "Aphex Twin" "Fleetwood Mac" \
  --energy-weight 5.0
```

Expected: a positive mean span delta. Then run the guardrail check — a full multi-pier generation for 2-3 of these artists through the `gui_fidelity` harness per the **playlist-testing** skill (energy-weight set in config vs 0), confirming worst-edge T and distinct-artist count do not regress and wall-clock stays ≤ 90s. Record results.

- [ ] **Step 7: Commit**

```bash
git add scripts/research/energy_spread_eval.py tests/research/test_energy_spread_eval.py tests/research/__init__.py
git commit -m "feat(research): A/B eval for artist energy-spread (pier arousal span)"
```

---

## Final verification

- [ ] Run the full suite (bounded, no pipes): `python -m pytest -q -m "not slow"`
- [ ] `ruff check src/playlist/artist_style.py scripts/research/energy_spread_eval.py`
- [ ] `mypy src/playlist/artist_style.py`
- [ ] Confirm the opt-in invariant by eye: with `medoid_energy_weight: 0.0` (default) and no `energy_values`, `cluster_artist_tracks` takes exactly the legacy code path (energy_span stays None, `_medoids_for_cluster` receives `energy_prox=None`).

## Calibration follow-up (NOT part of this plan)

`medoid_energy_weight` ships at `0.0`. After the eval panel confirms span ↑ with no transition/diversity/time regression, choose a starter weight (the eval uses `5.0` as a probe) and set it as the config default in a separate, eval-gated change. Popular-seeds (the `w_pop` term + Last.fm data path + GUI checkbox/button) is **Plan 2**.
