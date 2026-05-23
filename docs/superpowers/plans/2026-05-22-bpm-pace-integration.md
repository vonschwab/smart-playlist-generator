# BPM-Aware Pace Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Augment pace mode with direct BPM compatibility checks alongside the existing rhythm-axis cosine gate. Catches the half-time/double-time ambiguity that the rhythm PCA cannot resolve (e.g., a 70 BPM slowcore track sharing rhythm-PCA space with a 140 BPM punk track). Rolls into the existing pace mode slider — no new GUI control.

**Architecture:** Every track in the library already has `primary_bpm`, `half_tempo_likely`, `double_tempo_likely`, and `tempo_stability` stored in `sonic_features` (DB blob). We:

1. Resolve each track's **perceptual BPM** from the raw detection: when one of the half/double flags is set unambiguously, fold the detected tempo to its perceptual partner.
2. Score BPM compatibility in **log-space** so a 2:1 mismatch (`|log2(70/140)| = 1.0`) is treated as the catastrophic distance it musically is.
3. Apply two gates driven by `pace_mode`:
   - **Admission floor (Tier 1):** candidate's perceptual BPM must be within `bpm_admission_max_log_distance` of at least one seed.
   - **Bridge floor (Tier 2):** at beam step `s` of segment length `L`, the candidate's perceptual BPM must hug a target that interpolates linearly in log-BPM between pier_A and pier_B (so geometric-mean BPM, not arithmetic-mean — musically correct).
4. Skip the BPM check for tracks with `tempo_stability` below a confidence floor (the detector is uncertain → fall back to the existing rhythm PCA gate alone).

These gates layer on top of the existing rhythm-axis cosine gates from the original pace mode work — the rhythm PCA catches groove/density mismatches, BPM catches tempo mismatches. Both are required for a candidate to clear strict pace.

**Tech Stack:** Python 3.11, numpy, sqlite3 (for BPM extraction from JSON blob), pytest. Reuses the existing `pace_gate.py` infrastructure.

**Out of scope:**
- A new GUI slider (BPM rolls into existing pace_mode)
- Artifact rebuild (Phase 2 — Task 8 below sketches it but doesn't ship)
- Changing the rhythm PCA features themselves
- BPM-aware genre weighting (separate concern)
- Backward compatibility for renamed presets (user is sole operator)

---

## Design parameters (initial values — tunable)

| pace_mode | bpm_admission_max_log_distance | bpm_bridge_max_log_distance |
|---|---:|---:|
| `strict` | 0.30 (≈ ±23% BPM) | 0.40 (≈ ±32% BPM) |
| `narrow` | 0.50 (≈ ±41% BPM) | 0.60 (≈ ±52% BPM) |
| `dynamic` | ∞ (off) | ∞ (off) |

`bpm_stability_min`: 0.5 — below this, skip BPM gate for that candidate (detection too noisy).

Log-distance reference points:
- `|log2(60/60)| = 0.0` — identical
- `|log2(60/72)| ≈ 0.26` — 20% faster
- `|log2(60/80)| ≈ 0.41` — 33% faster
- `|log2(60/120)| = 1.0` — exactly double (octave)
- `|log2(60/180)| ≈ 1.58` — triple

---

## File structure

**New files:**
- `src/playlist/bpm_axis.py` — pure-function module: perceptual BPM resolution, log-distance, log-interpolation
- `src/playlist/bpm_loader.py` — DB → numpy array helper (reads `sonic_features.full.bpm_info` for all artifact track_ids)
- `tests/unit/test_bpm_axis.py`
- `tests/unit/test_bpm_loader.py`
- `tests/unit/test_candidate_pool_bpm_floor.py`
- `tests/unit/test_pier_bridge_bpm_gate.py`

**Modified files:**
- `src/playlist/mode_presets.py` — extend `PACE_MODE_PRESETS` with BPM thresholds
- `src/playlist/config.py` — add BPM fields to `CandidatePoolConfig`
- `src/playlist/pier_bridge/config.py` — add `bpm_bridge_max_log_distance`, `bpm_stability_min` to `PierBridgeConfig`
- `src/playlist/candidate_pool.py` — BPM admission gate alongside existing pace floor
- `src/playlist/pier_bridge/pace_gate.py` — add `compute_step_log_bpm_target` + `filter_candidates_by_bpm_target`
- `src/playlist/pier_bridge/beam.py` — apply BPM gate per step
- `src/playlist/pier_bridge/assemble.py` — load BPM arrays alongside `rhythm_matrix`, pass through to beam
- `src/playlist/pipeline/core.py` — load BPM data, thread through to candidate_pool + pier_bridge
- `docs/PLAYLIST_ORDERING_TUNING.md` — extend Knob 5 (pace_mode) docs with BPM behavior

---

## Task 1: BPM axis utility (perceptual BPM + log-distance)

**Files:**
- Create: `src/playlist/bpm_axis.py`
- Test: `tests/unit/test_bpm_axis.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_bpm_axis.py
import numpy as np
import pytest
from src.playlist.bpm_axis import (
    resolve_perceptual_bpm,
    bpm_log_distance,
    interpolate_log_bpm,
)


def test_resolve_perceptual_bpm_unambiguous_passes_through():
    assert resolve_perceptual_bpm(120.0, half_tempo_likely=False, double_tempo_likely=False) == 120.0


def test_resolve_perceptual_bpm_half_tempo_likely_doubles():
    # Detected at 60, "really feels" at 120
    assert resolve_perceptual_bpm(60.0, half_tempo_likely=True, double_tempo_likely=False) == 120.0


def test_resolve_perceptual_bpm_double_tempo_likely_halves():
    # Detected at 160, "really feels" at 80
    assert resolve_perceptual_bpm(160.0, half_tempo_likely=False, double_tempo_likely=True) == 80.0


def test_resolve_perceptual_bpm_both_flags_falls_back_to_primary():
    # Genuinely ambiguous — trust primary
    assert resolve_perceptual_bpm(70.0, half_tempo_likely=True, double_tempo_likely=True) == 70.0


def test_bpm_log_distance_identical():
    assert bpm_log_distance(120.0, 120.0) == 0.0


def test_bpm_log_distance_octave():
    # 60 vs 120 = exactly 1 octave
    np.testing.assert_allclose(bpm_log_distance(60.0, 120.0), 1.0, atol=1e-9)
    np.testing.assert_allclose(bpm_log_distance(120.0, 60.0), 1.0, atol=1e-9)  # symmetric


def test_bpm_log_distance_handles_zeros_safely():
    # Non-positive BPM should return inf (or some sentinel) — don't crash
    assert bpm_log_distance(0.0, 120.0) == float("inf")
    assert bpm_log_distance(120.0, -5.0) == float("inf")


def test_bpm_log_distance_vector_broadcasts():
    a = np.array([60.0, 120.0, 90.0])
    b = 120.0
    expected = np.array([1.0, 0.0, np.log2(120 / 90)])
    np.testing.assert_allclose(bpm_log_distance(a, b), expected, atol=1e-9)


def test_interpolate_log_bpm_endpoints():
    # At t=0 → bpm_a; at t=1 → bpm_b
    np.testing.assert_allclose(interpolate_log_bpm(60.0, 120.0, t=0.0), 60.0)
    np.testing.assert_allclose(interpolate_log_bpm(60.0, 120.0, t=1.0), 120.0)


def test_interpolate_log_bpm_midpoint_is_geometric_mean():
    # Geometric mean of 60 and 240 is sqrt(60*240) ≈ 120, NOT arithmetic mean 150
    result = interpolate_log_bpm(60.0, 240.0, t=0.5)
    np.testing.assert_allclose(result, 120.0, atol=1e-9)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/unit/test_bpm_axis.py -v`

- [ ] **Step 3: Implement the module**

```python
# src/playlist/bpm_axis.py
"""Perceptual BPM resolution and log-space distance metrics.

The raw `primary_bpm` from beat tracking is ambiguous at 2:1 ratios — a
slow track with strong half-time feel may be detected at twice its "felt"
tempo, and vice versa. The extractor flags this via `half_tempo_likely`
and `double_tempo_likely`. We use those flags to recover the perceptual
tempo before comparing.

Comparison happens in log-space: a 2:1 BPM ratio is a distance of 1.0
("one octave"), reflecting how dramatically different 70 BPM and 140 BPM
feel even though they're harmonically related.
"""
from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def resolve_perceptual_bpm(
    primary_bpm: float,
    *,
    half_tempo_likely: bool,
    double_tempo_likely: bool,
) -> float:
    """Resolve perceived tempo from detected tempo + half/double flags."""
    if bool(half_tempo_likely) and not bool(double_tempo_likely):
        return float(primary_bpm) * 2.0
    if bool(double_tempo_likely) and not bool(half_tempo_likely):
        return float(primary_bpm) / 2.0
    return float(primary_bpm)


def bpm_log_distance(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """|log2(a / b)|. Returns inf for non-positive inputs."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    invalid = (a_arr <= 0) | (b_arr <= 0)
    safe_a = np.where(invalid, 1.0, a_arr)
    safe_b = np.where(invalid, 1.0, b_arr)
    dist = np.abs(np.log2(safe_a / safe_b))
    return np.where(invalid, np.inf, dist)


def interpolate_log_bpm(
    bpm_a: float,
    bpm_b: float,
    *,
    t: float,
) -> float:
    """Log-space interpolation. t=0 → bpm_a, t=1 → bpm_b, t=0.5 → geometric mean."""
    if bpm_a <= 0 or bpm_b <= 0:
        return float(bpm_a if t <= 0.5 else bpm_b)
    log_a = np.log2(bpm_a)
    log_b = np.log2(bpm_b)
    t_clamped = max(0.0, min(1.0, float(t)))
    return float(2.0 ** ((1.0 - t_clamped) * log_a + t_clamped * log_b))
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/playlist/bpm_axis.py tests/unit/test_bpm_axis.py
git commit -m "feat: bpm_axis utility (perceptual BPM, log-distance, log-interpolation)"
```

---

## Task 2: BPM loader (DB → numpy arrays)

**Files:**
- Create: `src/playlist/bpm_loader.py`
- Test: `tests/unit/test_bpm_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_bpm_loader.py
import json
import sqlite3

import numpy as np
import pytest

from src.playlist.bpm_loader import load_bpm_arrays


def _make_db(tmp_path, rows):
    """Build a minimal tracks DB with sonic_features blobs."""
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            sonic_features TEXT
        )
    """)
    for tid, bpm_info in rows.items():
        blob = json.dumps({"full": {"bpm_info": bpm_info}}) if bpm_info else None
        conn.execute("INSERT INTO tracks (track_id, sonic_features) VALUES (?, ?)", (tid, blob))
    conn.commit()
    conn.close()
    return db_path


def test_load_bpm_arrays_basic(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t2": {"primary_bpm": 70.0, "half_tempo_likely": True, "double_tempo_likely": False, "tempo_stability": 0.8},
    })
    track_ids = np.array(["t1", "t2"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    np.testing.assert_allclose(result["primary_bpm"], [120.0, 70.0])
    # t2 has half_tempo_likely → perceptual = 70 * 2 = 140
    np.testing.assert_allclose(result["perceptual_bpm"], [120.0, 140.0])
    np.testing.assert_allclose(result["tempo_stability"], [0.9, 0.8])


def test_load_bpm_arrays_missing_track_returns_nan(tmp_path):
    db_path = _make_db(tmp_path, {"t1": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9}})
    track_ids = np.array(["t1", "tX"], dtype=object)  # tX doesn't exist
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    assert result["perceptual_bpm"][0] == 120.0
    assert np.isnan(result["perceptual_bpm"][1])
    assert np.isnan(result["tempo_stability"][1])


def test_load_bpm_arrays_null_sonic_features_returns_nan(tmp_path):
    db_path = _make_db(tmp_path, {"t1": None})
    track_ids = np.array(["t1"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    assert np.isnan(result["perceptual_bpm"][0])


def test_load_bpm_arrays_preserves_order(tmp_path):
    db_path = _make_db(tmp_path, {
        "t1": {"primary_bpm": 100.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t2": {"primary_bpm": 120.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
        "t3": {"primary_bpm": 140.0, "half_tempo_likely": False, "double_tempo_likely": False, "tempo_stability": 0.9},
    })
    # Request out of insertion order
    track_ids = np.array(["t3", "t1", "t2"], dtype=object)
    result = load_bpm_arrays(track_ids, db_path=str(db_path))
    np.testing.assert_allclose(result["perceptual_bpm"], [140.0, 100.0, 120.0])
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement**

```python
# src/playlist/bpm_loader.py
"""Load BPM arrays from `metadata.db` aligned to an artifact's track_ids.

Reads `sonic_features.full.bpm_info` for each track via JSON extraction.
Returns numpy arrays aligned with the input `track_ids` (NaN for missing).

This is a runtime fallback path until `build_beat3tower_artifacts.py`
bakes BPM arrays directly into the .npz (Task 8).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict

import numpy as np

from src.playlist.bpm_axis import resolve_perceptual_bpm

logger = logging.getLogger(__name__)


def load_bpm_arrays(
    track_ids: np.ndarray,
    *,
    db_path: str,
) -> Dict[str, np.ndarray]:
    """Return arrays aligned with track_ids."""
    n = int(track_ids.shape[0])
    primary = np.full(n, np.nan, dtype=float)
    perceptual = np.full(n, np.nan, dtype=float)
    stability = np.full(n, np.nan, dtype=float)
    half_flags = np.zeros(n, dtype=bool)
    double_flags = np.zeros(n, dtype=bool)

    id_to_pos = {str(tid): i for i, tid in enumerate(track_ids)}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # One query, JSON-extract on the SQLite side
        placeholders = ",".join("?" for _ in id_to_pos)
        if not placeholders:
            return {"primary_bpm": primary, "perceptual_bpm": perceptual,
                    "tempo_stability": stability,
                    "half_tempo_likely": half_flags, "double_tempo_likely": double_flags}

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT track_id,
                   json_extract(sonic_features, '$.full.bpm_info.primary_bpm') AS primary_bpm,
                   json_extract(sonic_features, '$.full.bpm_info.half_tempo_likely') AS half_t,
                   json_extract(sonic_features, '$.full.bpm_info.double_tempo_likely') AS double_t,
                   json_extract(sonic_features, '$.full.bpm_info.tempo_stability') AS stability
            FROM tracks
            WHERE track_id IN ({placeholders})
            """,
            tuple(id_to_pos.keys()),
        )
        missing = 0
        for row in cur.fetchall():
            pos = id_to_pos.get(str(row["track_id"]))
            if pos is None:
                continue
            bpm = row["primary_bpm"]
            if bpm is None:
                missing += 1
                continue
            half = bool(row["half_t"]) if row["half_t"] is not None else False
            dbl = bool(row["double_t"]) if row["double_t"] is not None else False
            stab = float(row["stability"]) if row["stability"] is not None else 0.0
            primary[pos] = float(bpm)
            half_flags[pos] = half
            double_flags[pos] = dbl
            stability[pos] = stab
            perceptual[pos] = resolve_perceptual_bpm(
                float(bpm), half_tempo_likely=half, double_tempo_likely=dbl
            )
    finally:
        conn.close()

    missing_count = int(np.sum(np.isnan(perceptual)))
    if missing_count:
        logger.warning(
            "BPM data missing for %d/%d tracks (will skip BPM gate for them)",
            missing_count, n,
        )

    return {
        "primary_bpm": primary,
        "perceptual_bpm": perceptual,
        "tempo_stability": stability,
        "half_tempo_likely": half_flags,
        "double_tempo_likely": double_flags,
    }
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/playlist/bpm_loader.py tests/unit/test_bpm_loader.py
git commit -m "feat: BPM array loader (DB → numpy, aligned to artifact track_ids)"
```

---

## Task 3: Extend pace mode presets and config with BPM thresholds

**Files:**
- Modify: `src/playlist/mode_presets.py`
- Modify: `src/playlist/config.py` (`CandidatePoolConfig`)
- Modify: `src/playlist/pier_bridge/config.py` (`PierBridgeConfig`)
- Test: extend `tests/unit/test_pace_mode_presets.py`

- [ ] **Step 1: Extend the failing tests**

Add to `tests/unit/test_pace_mode_presets.py`:

```python
def test_pace_mode_presets_include_bpm_thresholds():
    settings = resolve_pace_mode("strict")
    assert "bpm_admission_max_log_distance" in settings
    assert "bpm_bridge_max_log_distance" in settings
    assert settings["bpm_admission_max_log_distance"] == 0.30
    assert settings["bpm_bridge_max_log_distance"] == 0.40


def test_pace_mode_dynamic_disables_bpm_gates():
    settings = resolve_pace_mode("dynamic")
    # inf means "no constraint"
    assert settings["bpm_admission_max_log_distance"] == float("inf")
    assert settings["bpm_bridge_max_log_distance"] == float("inf")


def test_pace_mode_bpm_thresholds_monotonic_strict_tightest():
    strict = resolve_pace_mode("strict")
    narrow = resolve_pace_mode("narrow")
    dynamic = resolve_pace_mode("dynamic")
    assert strict["bpm_admission_max_log_distance"] < narrow["bpm_admission_max_log_distance"] < dynamic["bpm_admission_max_log_distance"]
    assert strict["bpm_bridge_max_log_distance"] < narrow["bpm_bridge_max_log_distance"] < dynamic["bpm_bridge_max_log_distance"]
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Extend `PACE_MODE_PRESETS`**

In `src/playlist/mode_presets.py`, add BPM keys to each pace mode preset:

```python
PACE_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "admission_floor": 0.55,
        "bridge_floor": 0.65,
        "bpm_admission_max_log_distance": 0.30,
        "bpm_bridge_max_log_distance": 0.40,
        # description / use_case ...
    },
    "narrow": {
        "admission_floor": 0.35,
        "bridge_floor": 0.45,
        "bpm_admission_max_log_distance": 0.50,
        "bpm_bridge_max_log_distance": 0.60,
        # ...
    },
    "dynamic": {
        "admission_floor": 0.00,
        "bridge_floor": 0.00,
        "bpm_admission_max_log_distance": float("inf"),
        "bpm_bridge_max_log_distance": float("inf"),
        # ...
    },
}
```

- [ ] **Step 4: Add config fields**

In `src/playlist/config.py:CandidatePoolConfig`:

```python
bpm_admission_max_log_distance: float = float("inf")  # inf = off
bpm_stability_min: float = 0.5  # below this, skip BPM gate for that candidate
```

In `src/playlist/pier_bridge/config.py:PierBridgeConfig`:

```python
bpm_bridge_max_log_distance: float = float("inf")  # inf = off
bpm_stability_min: float = 0.5
```

In `src/playlist/config.py:default_ds_config`, thread the BPM presets onto the candidate config:

```python
pace_admission_floor=float(
    candidate_pool.get("pace_admission_floor", pace_settings["admission_floor"])
),
pace_bridge_floor=float(
    candidate_pool.get("pace_bridge_floor", pace_settings["bridge_floor"])
),
bpm_admission_max_log_distance=float(
    candidate_pool.get("bpm_admission_max_log_distance",
                       pace_settings["bpm_admission_max_log_distance"])
),
bpm_stability_min=float(
    candidate_pool.get("bpm_stability_min", 0.5)
),
```

- [ ] **Step 5: Run, expect pass**

```bash
pytest tests/unit/test_pace_mode_presets.py tests/unit/test_candidate_pool_config_pace.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/playlist/mode_presets.py src/playlist/config.py src/playlist/pier_bridge/config.py tests/unit/test_pace_mode_presets.py
git commit -m "feat: BPM thresholds in pace mode presets + config"
```

---

## Task 4: BPM admission gate in candidate_pool

**Files:**
- Modify: `src/playlist/candidate_pool.py`
- Test: `tests/unit/test_candidate_pool_bpm_floor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_candidate_pool_bpm_floor.py
import numpy as np
from src.playlist.candidate_pool import build_candidate_pool, CandidatePoolConfig


def _make_cfg(**kwargs):
    base = dict(
        similarity_floor=0.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        duration_penalty_enabled=False,
        duration_penalty_weight=0.0,
        duration_cutoff_multiplier=2.5,
        genre_compatibility_enabled=False,
        genre_compatibility_penalty_strength=0.0,
        genre_compatibility_compatible_threshold=0.35,
        genre_compatibility_conflict_threshold=0.15,
        title_hard_exclude_flags=frozenset(),
        genre_idf_enabled=False,
        pace_admission_floor=0.0,
        pace_bridge_floor=0.0,
        bpm_admission_max_log_distance=float("inf"),
        bpm_stability_min=0.5,
    )
    base.update(kwargs)
    return CandidatePoolConfig(**base)


def test_bpm_floor_inf_admits_all():
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([60.0, 120.0, 240.0, 70.0, 180.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=float("inf"))
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    # No BPM filtering — all should be eligible
    assert set(result.pool_indices.tolist()) >= {1, 2, 3, 4}


def test_bpm_floor_rejects_octave_mismatch():
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    # seed at 70 BPM; track 2 at 140 (exactly 1 octave away)
    perceptual_bpm = np.array([70.0, 80.0, 140.0, 72.0, 60.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)  # ~23% BPM
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    pool = set(result.pool_indices.tolist())
    assert 2 not in pool  # 140 BPM rejected (octave away from 70)
    assert 1 in pool      # 80 BPM admitted (close to 70)
    assert 3 in pool      # 72 BPM admitted
    assert result.stats.get("below_bpm_floor", 0) >= 1


def test_bpm_floor_max_over_seeds():
    """Candidate compatible with any seed BPM should pass."""
    rng = np.random.default_rng(0)
    N = 5
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    # seeds at 70 and 140; candidate at 140 should pass (matches seed 1)
    perceptual_bpm = np.array([70.0, 140.0, 140.0, 60.0, 100.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)
    result = build_candidate_pool(
        seed_idx=0, seed_indices=[1],
        embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    assert 2 in set(result.pool_indices.tolist())


def test_bpm_floor_skips_low_stability_candidates():
    """Candidate with unreliable tempo detection should not be filtered by BPM gate."""
    rng = np.random.default_rng(0)
    N = 3
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    # Track 1 is "octave away" but stability is too low to trust
    perceptual_bpm = np.array([70.0, 140.0, 75.0])
    tempo_stability = np.array([0.9, 0.3, 0.9])  # track 1 unreliable
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30, bpm_stability_min=0.5)
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
    )
    # Track 1 not filtered (low stability → bypass BPM gate)
    assert 1 in set(result.pool_indices.tolist())


def test_bpm_floor_skips_nan_candidates():
    """Candidate with missing BPM (NaN) bypasses the gate."""
    rng = np.random.default_rng(0)
    N = 3
    embedding = rng.standard_normal((N, 16))
    artist_keys = np.array([f"a{i}" for i in range(N)])
    perceptual_bpm = np.array([70.0, np.nan, 75.0])
    cfg = _make_cfg(bpm_admission_max_log_distance=0.30)
    result = build_candidate_pool(
        seed_idx=0, embedding=embedding, artist_keys=artist_keys,
        cfg=cfg, random_seed=0,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
    )
    assert 1 in set(result.pool_indices.tolist())
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement the gate**

Extend `build_candidate_pool` signature in `src/playlist/candidate_pool.py`:

```python
perceptual_bpm: Optional[np.ndarray] = None,
tempo_stability: Optional[np.ndarray] = None,
```

After the rhythm-axis pace floor block, add:

```python
# BPM admission gate (Tier 1, complementary to rhythm-axis pace floor).
bpm_seed_distances = None
below_bpm_count = 0
if (
    float(getattr(cfg, "bpm_admission_max_log_distance", float("inf"))) < float("inf")
    and perceptual_bpm is not None
):
    from src.playlist.bpm_axis import bpm_log_distance

    max_log = float(cfg.bpm_admission_max_log_distance)
    stability_min = float(getattr(cfg, "bpm_stability_min", 0.5))
    seed_bpm = perceptual_bpm[seed_list]  # (k,)
    # Distance from each candidate to each seed → (N, k)
    dist_matrix = np.stack(
        [bpm_log_distance(perceptual_bpm, float(sb)) for sb in seed_bpm], axis=1
    )
    # Closest seed
    bpm_seed_distances = np.min(dist_matrix, axis=1)
    # Skip the gate where data is missing or unreliable
    bypass = np.isnan(perceptual_bpm)
    if tempo_stability is not None:
        bypass = bypass | (tempo_stability < stability_min)
    bpm_pass = bypass | (bpm_seed_distances <= max_log)
    bpm_pass[seed_mask] = True  # seeds never rejected by their own gate
    rejected_pre = int(np.sum(~bpm_pass))
    below_bpm_count = rejected_pre
    seed_sim_all[~bpm_pass] = -1.0
    logger.info(
        "BPM admission gate: max_log_distance=%.2f rejected=%d (bypass_low_stability=%d, bypass_nan=%d)",
        max_log, rejected_pre,
        int(np.sum((tempo_stability is None) or (tempo_stability < stability_min)))
            if tempo_stability is not None else 0,
        int(np.sum(np.isnan(perceptual_bpm))),
    )

# In stats:
stats["below_bpm_floor"] = below_bpm_count
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_candidate_pool_bpm_floor.py
git commit -m "feat: BPM admission gate in candidate pool (Tier 1)"
```

---

## Task 5: BPM bridge gate (per-step moving target in log-space)

**Files:**
- Modify: `src/playlist/pier_bridge/pace_gate.py`
- Modify: `src/playlist/pier_bridge/beam.py`
- Modify: `src/playlist/pier_bridge/assemble.py`
- Test: `tests/unit/test_pier_bridge_bpm_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_pier_bridge_bpm_gate.py
import numpy as np
from src.playlist.pier_bridge.pace_gate import (
    compute_step_log_bpm_target,
    filter_candidates_by_bpm_target,
)


def test_compute_step_log_bpm_target_endpoints():
    assert compute_step_log_bpm_target(60.0, 240.0, step=0, segment_length=4) == 60.0
    assert compute_step_log_bpm_target(60.0, 240.0, step=4, segment_length=4) == 240.0


def test_compute_step_log_bpm_target_midpoint_is_geometric_mean():
    # Halfway from 60 to 240 should be sqrt(60*240) = 120 (NOT 150)
    result = compute_step_log_bpm_target(60.0, 240.0, step=2, segment_length=4)
    np.testing.assert_allclose(result, 120.0, atol=1e-9)


def test_filter_candidates_by_bpm_target_inf_floor_keeps_all():
    perceptual_bpm = np.array([60.0, 120.0, 240.0])
    tempo_stability = np.ones(3)
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1, 2],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
        target_bpm=120.0,
        max_log_distance=float("inf"),
        stability_min=0.5,
    )
    assert kept == [0, 1, 2]


def test_filter_candidates_by_bpm_target_drops_octave_away():
    perceptual_bpm = np.array([60.0, 120.0, 240.0, 100.0])
    tempo_stability = np.ones(4)
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1, 2, 3],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
        target_bpm=120.0,
        max_log_distance=0.4,  # ~32% range
        stability_min=0.5,
    )
    # 60 and 240 are 1 octave from 120 → dropped
    # 100 is log2(120/100) ≈ 0.26 → kept
    assert 1 in kept
    assert 3 in kept
    assert 0 not in kept
    assert 2 not in kept


def test_filter_candidates_low_stability_bypasses_bpm():
    perceptual_bpm = np.array([60.0, 240.0])
    tempo_stability = np.array([0.3, 0.9])  # idx 0 unreliable
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
        target_bpm=120.0,
        max_log_distance=0.4,
        stability_min=0.5,
    )
    assert 0 in kept  # low stability → bypass
    assert 1 not in kept  # high stability + far → drop


def test_filter_candidates_nan_bpm_bypasses():
    perceptual_bpm = np.array([np.nan, 240.0])
    tempo_stability = np.ones(2)
    kept = filter_candidates_by_bpm_target(
        candidate_indices=[0, 1],
        perceptual_bpm=perceptual_bpm,
        tempo_stability=tempo_stability,
        target_bpm=120.0,
        max_log_distance=0.4,
        stability_min=0.5,
    )
    assert 0 in kept  # NaN BPM → bypass
    assert 1 not in kept
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement helpers in `pace_gate.py`**

```python
# In src/playlist/pier_bridge/pace_gate.py — add to existing module

import numpy as np
from src.playlist.bpm_axis import interpolate_log_bpm, bpm_log_distance


def compute_step_log_bpm_target(
    bpm_a: float,
    bpm_b: float,
    *,
    step: int,
    segment_length: int,
) -> float:
    """Return target BPM at beam step s (log-space interpolation between piers)."""
    if int(segment_length) <= 0:
        return float(bpm_a)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_log_bpm(float(bpm_a), float(bpm_b), t=t)


def filter_candidates_by_bpm_target(
    *,
    candidate_indices,
    perceptual_bpm: np.ndarray,
    tempo_stability: np.ndarray | None,
    target_bpm: float,
    max_log_distance: float,
    stability_min: float = 0.5,
) -> list[int]:
    """Drop candidates whose perceptual BPM is too far from `target_bpm`.

    Bypasses (keeps) any candidate with NaN BPM or tempo_stability below
    `stability_min` (BPM detection unreliable; defer to other gates).
    """
    if not np.isfinite(max_log_distance):
        return list(candidate_indices)
    indices = list(candidate_indices)
    if not indices:
        return []
    cand_bpm = perceptual_bpm[indices]
    bypass = np.isnan(cand_bpm)
    if tempo_stability is not None:
        cand_stab = tempo_stability[indices]
        bypass = bypass | (cand_stab < stability_min)
    distances = bpm_log_distance(cand_bpm, float(target_bpm))
    pass_mask = bypass | (distances <= float(max_log_distance))
    return [idx for idx, ok in zip(indices, pass_mask) if bool(ok)]
```

- [ ] **Step 4: Wire into beam (matches existing pace gate pattern)**

In `src/playlist/pier_bridge/beam.py`, locate the existing per-step pace gate (around lines 859–873) and add the BPM check **right after**:

```python
if float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf"))) < float("inf") and perceptual_bpm is not None:
    from src.playlist.pier_bridge.pace_gate import compute_step_log_bpm_target

    target_bpm = compute_step_log_bpm_target(
        float(perceptual_bpm[int(pier_a)]),
        float(perceptual_bpm[int(pier_b)]),
        step=step,
        segment_length=interior_length,
    )
    cand_bpm = float(perceptual_bpm[int(cand)])
    cand_stab = float(tempo_stability[int(cand)]) if tempo_stability is not None else 1.0
    if not np.isnan(cand_bpm) and cand_stab >= float(getattr(cfg, "bpm_stability_min", 0.5)):
        from src.playlist.bpm_axis import bpm_log_distance
        if float(bpm_log_distance(cand_bpm, target_bpm)) > float(cfg.bpm_bridge_max_log_distance):
            continue
```

Extend `_beam_search_segment` signature with `perceptual_bpm` and `tempo_stability` kwargs (mirroring the existing `rhythm_matrix` kwarg pattern).

- [ ] **Step 5: Wire from `assemble.py` and `pier_bridge_builder.py`**

In `src/playlist/pier_bridge/assemble.py`, where `rhythm_matrix` is extracted, also pull `perceptual_bpm` and `tempo_stability` (they must arrive as parameters to `build_pier_bridge_playlist` — load them in `pipeline/core.py`).

Pass through to the beam call:

```python
perceptual_bpm=perceptual_bpm,
tempo_stability=tempo_stability,
```

Apply the same backoff scaling as `pace_bridge_floor` — when bridge_floor relaxes, scale `bpm_bridge_max_log_distance` to widen the gate proportionally (i.e., `new = current * (cfg.bridge_floor / configured_bridge_floor)` capped at +inf).

- [ ] **Step 6: Run all pace + bpm tests**

```bash
pytest tests/unit/test_bpm_axis.py tests/unit/test_pier_bridge_bpm_gate.py tests/unit/test_beam_pace_gate.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/playlist/pier_bridge/pace_gate.py src/playlist/pier_bridge/beam.py src/playlist/pier_bridge/assemble.py src/playlist/pier_bridge_builder.py tests/unit/test_pier_bridge_bpm_gate.py
git commit -m "feat: BPM bridge gate with log-space moving target (Tier 2)"
```

---

## Task 6: Wire BPM loading through pipeline/core

**Files:**
- Modify: `src/playlist/pipeline/core.py`
- Test: extend `tests/unit/test_pipeline_pace_plumbing.py`

- [ ] **Step 1: Extend the failing test**

```python
def test_pipeline_loads_bpm_arrays_when_pace_active(tmp_path, monkeypatch):
    """When pace_mode is strict, BPM arrays should be loaded and passed through."""
    # Construct an artifact + minimal DB with BPM data, then run a small playlist
    # and verify the BPM admission gate was actually engaged
    # (Look for "BPM admission gate:" in captured logs.)
    ...
```

- [ ] **Step 2: Load BPM data in `generate_playlist_ds`**

After `embedding` setup, when pace_mode is not dynamic:

```python
from src.playlist.bpm_loader import load_bpm_arrays
from src.playlist.mode_presets import resolve_pace_mode

pace_settings = resolve_pace_mode(pace_mode)
needs_bpm = (
    float(pace_settings["bpm_admission_max_log_distance"]) < float("inf")
    or float(pace_settings["bpm_bridge_max_log_distance"]) < float("inf")
)
perceptual_bpm = None
tempo_stability = None
if needs_bpm:
    db_path = config.get("library", {}).get("database_path", "data/metadata.db")
    try:
        bpm_arrays = load_bpm_arrays(bundle.track_ids, db_path=db_path)
        perceptual_bpm = bpm_arrays["perceptual_bpm"]
        tempo_stability = bpm_arrays["tempo_stability"]
        logger.info(
            "BPM loaded: %d tracks (%d missing)",
            int(np.sum(~np.isnan(perceptual_bpm))),
            int(np.sum(np.isnan(perceptual_bpm))),
        )
    except Exception:
        logger.warning("BPM load failed; BPM gates disabled", exc_info=True)
```

Pass `perceptual_bpm` and `tempo_stability` to `_build_pool(...)` and into the pier-bridge call site.

- [ ] **Step 3: Apply BPM thresholds onto pier-bridge config**

```python
pb_cfg = replace(
    pb_cfg,
    pace_bridge_floor=float(cfg.candidate.pace_bridge_floor),
    bpm_bridge_max_log_distance=float(pace_settings["bpm_bridge_max_log_distance"]),
    bpm_stability_min=float(cfg.candidate.bpm_stability_min),
)
```

- [ ] **Step 4: Run full smoke suite**

```bash
pytest tests/ -m "not slow" -x -q
```

Expected: no regressions. Default pace_mode=dynamic means both BPM thresholds are `inf` — no DB lookup, no behavior change.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pipeline/core.py tests/unit/test_pipeline_pace_plumbing.py
git commit -m "feat: load and thread BPM arrays through pipeline when pace mode active"
```

---

## Task 7: Diagnostics and edge audit

**Files:**
- Modify: `src/playlist/reporter.py` (or wherever `emit_selected_edge_audit` lives)
- Modify: `src/playlist/pier_bridge_builder.py` (pass BPM into reporter)

- [ ] **Step 1: Extend the edge audit**

For each emitted edge, add columns to the per-edge audit table:

```
bpm_a, bpm_b, bpm_log_distance
```

If perceptual BPM is missing or unreliable for either endpoint, show `n/a`.

- [ ] **Step 2: Add a per-playlist BPM summary line**

After the existing "Edge score summaries" section, add:

```
BPM stats: range=70.0–145.0 mean=98.7 std=18.4 (perceptual; n=29/30, 1 missing)
```

- [ ] **Step 3: Add an admission-stats line**

In the candidate pool SUMMARY log, include `below_bpm_floor=N` alongside the existing exclusion counters.

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -m "not slow" -x -q
```

- [ ] **Step 5: Commit**

```bash
git add src/playlist/reporter.py src/playlist/pier_bridge_builder.py
git commit -m "diag: add BPM stats and per-edge BPM distance to audit"
```

---

## Task 8 (deferred — separate session): Bake BPM into artifact

**Why deferred:** The DB-lookup path (Task 2) is acceptable for v5.x — one batch JSON-extract query takes <100ms for 39k tracks. Baking BPM into the `.npz` is a perf optimization, not a correctness fix.

When ready:
- Extend `scripts/build_beat3tower_artifacts.py` to extract `primary_bpm`, `half_tempo_likely`, `double_tempo_likely`, `tempo_stability` per track and save as `perceptual_bpm` and `tempo_stability` arrays in the `.npz`.
- Extend `src/features/artifacts.py:ArtifactBundle` to expose `perceptual_bpm` and `tempo_stability` attributes (default to `None` when artifact lacks them — fall back to DB loader).
- Remove the runtime DB lookup once all live artifacts have the baked arrays.

---

## End-to-end validation

After Tasks 1–7 complete:

- [ ] **Regenerate the problematic shoegaze playlist with `pace_mode=strict`**

Use the same seeds that produced double-time bleed-through. Watch for:
- Log lines `BPM admission gate: max_log_distance=0.30 rejected=N`
- Per-edge audit `bpm_log_distance` column — values should mostly be < 0.4 in strict
- No tracks at ~2x the seed tempo making it into the final playlist

- [ ] **Regenerate with `pace_mode=narrow`**

Wider tolerance; still no octave jumps but more flex in BPM.

- [ ] **Regenerate with `pace_mode=dynamic`**

Should be byte-identical to pre-change runs. No BPM lookup. No BPM gate. Goldens stable.

- [ ] **Multi-tempo seed test**

Construct a playlist with seeds at different tempos (e.g., 70 BPM slowcore + 130 BPM dance). With `pace_mode=narrow`, segments between the two seeds should produce a tempo arc — per-step BPM target slides log-linearly from 70 to 130.

- [ ] **Full unit suite**

```bash
pytest tests/ -m "not slow"
```

All passing. Default behavior unchanged. New tests integrated.

---

## Risk notes

- **Per-seed BPM outliers**: if a seed track itself has a wonky BPM detection (e.g., 38 BPM for a 76 BPM song that the extractor halved), strict mode could reject everything close to the intended tempo. The `tempo_stability` check helps catch this at the candidate end, but a noisy *seed* BPM is a direct problem. Mitigation: log warning if any seed's `tempo_stability < 0.5` and consider auto-relaxing pace to narrow.
- **Log-space midpoint may surprise users**: geometric mean of 60 and 240 is 120, not 150. This is musically correct (tempo perception is logarithmic) but the diagnostic logs should label this clearly so the targets don't look "off."
- **BPM gate stacks with rhythm-axis pace gate** — both must clear in strict mode. If they're too tight together, segments could become infeasible. The existing `infeasible_handling` backoff relaxes `bridge_floor`; we add the same proportional relaxation for `bpm_bridge_max_log_distance` (widening it as bridge_floor shrinks).
- **DB read at runtime** is one query per playlist generation, with a JSON path extract. SQLite handles this in well under 100ms even at 39k rows on an SSD. Negligible vs. beam search cost.
- **`data/metadata.db` is read-only here** — Task 2 only does a `SELECT`. No writes anywhere in this plan.
- **`tempo_stability` thresholds are guesses (0.5)** — should be tuned after a few runs. If many slow tracks have stability ~0.4 because beat tracking is hard for sparse music, raise the bypass band to 0.6+ so we still apply BPM gates more aggressively.
