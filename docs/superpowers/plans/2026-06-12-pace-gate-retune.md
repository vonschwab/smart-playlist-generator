# Pace Gate Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rhythm-cosine hard pace gate with embedding-independent BPM + onset-rate hard bands plus a soft rhythm-cosine penalty, so `pace_mode: narrow` works for beatless/ambient artists, stays meaningful for rhythmic music, respects the 90 s budget, and survives the MERT migration.

**Architecture:** Onset-rate is loaded from `metadata.db` (`sonic_features.full.rhythm.onset_rate`) alongside BPM, then gated with a log-ratio band that mirrors the existing BPM band on both admission and bridge sides. The rhythm-cosine signal moves from a hard `continue` gate to a multiplicative score penalty at the bridge, which auto-skips under no-tower (MERT) variants. New hard bridge bands widen on the segment backoff ladder so they can't blow the budget.

**Tech Stack:** Python 3.11, numpy, pytest. Source under `src/playlist/`. Spec: `docs/superpowers/specs/2026-06-12-pace-gate-retune-design.md`.

---

## File Structure

- `src/playlist/bpm_loader.py` — add `onset_rate` column to `load_bpm_arrays`.
- `src/playlist/pier_bridge/pace_gate.py` — add onset step-target + filter helpers (reuse `bpm_log_distance`).
- `src/playlist/mode_presets.py` — `PACE_MODE_PRESETS`: add onset caps + rhythm-soft keys; zero the rhythm-cosine floors.
- `src/playlist/config.py` — `CandidatePoolConfig` fields + `resolve_thresholds` wiring (admission side).
- `src/playlist/pier_bridge/config.py` — `PierBridgeConfig` fields (bridge side).
- `src/playlist/candidate_pool.py` — onset admission band; remove rhythm-cosine admission gate.
- `src/playlist/pier_bridge/beam.py` — onset bridge band; rhythm-cosine → soft penalty; remove rhythm-cosine bridge gate.
- `src/playlist/pier_bridge_builder.py` — thread `onset_rate`; widen onset/BPM caps on backoff.
- `src/playlist/pipeline/core.py` — load + thread `onset_rate` into pool + pier-bridge.
- Tests under `tests/unit/` and `tests/integration/`.
- `scripts/pace_calibration_sweep.py` — full-pool calibration (outputs to `docs/run_audits/pace_retune/`).

---

## Task 1: Onset-rate loader column

**Files:**
- Modify: `src/playlist/bpm_loader.py:29-109`
- Test: `tests/unit/test_bpm_loader_onset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bpm_loader_onset.py
"""load_bpm_arrays must return an onset_rate array aligned to track_ids."""
import sqlite3
import json
import numpy as np
import pytest

from src.playlist.bpm_loader import load_bpm_arrays


@pytest.fixture
def tiny_db(tmp_path):
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY, sonic_features TEXT)")
    feat = {"full": {"bpm_info": {"primary_bpm": 120.0, "tempo_stability": 0.9,
                                  "half_tempo_likely": False, "double_tempo_likely": False},
                     "rhythm": {"onset_rate": 2.5}}}
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("t1", json.dumps(feat)))
    conn.execute("INSERT INTO tracks VALUES (?, ?)", ("t2", None))  # missing features
    conn.commit()
    conn.close()
    return str(db)


def test_onset_rate_loaded_and_aligned(tiny_db):
    arrs = load_bpm_arrays(np.array(["t1", "t2"]), db_path=tiny_db)
    assert "onset_rate" in arrs
    assert arrs["onset_rate"][0] == pytest.approx(2.5)
    assert np.isnan(arrs["onset_rate"][1])  # NaN for missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_bpm_loader_onset.py -v`
Expected: FAIL with `KeyError: 'onset_rate'`.

- [ ] **Step 3: Implement**

In `src/playlist/bpm_loader.py`, add an `onset` array next to the others and select the column. After line 45 (`double_flags = np.zeros(...)`) add:

```python
    onset = np.full(n, np.nan, dtype=float)
```

In the empty-`id_to_pos` early return dict (around line 49), add the key:

```python
            "onset_rate": onset,
```

Extend the SELECT (around line 63-72) to add the column:

```python
                       json_extract(sonic_features, '$.full.rhythm.onset_rate') AS onset_rate
```
(append `,` after the existing `stability` line, then this line, before `FROM tracks`).

Inside the row loop (after `stability[pos] = stab`, ~line 88) add:

```python
                onset[pos] = float(row["onset_rate"]) if row["onset_rate"] is not None else np.nan
```

Add `"onset_rate": onset,` to the final return dict (around line 103-108).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_bpm_loader_onset.py -v`
Expected: PASS (2 assertions).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_bpm_loader_onset.py src/playlist/bpm_loader.py
git commit -m "feat(pace): load onset_rate array from sonic_features"
```

---

## Task 2: Onset step-target + filter helpers

**Files:**
- Modify: `src/playlist/pier_bridge/pace_gate.py` (append after the BPM helpers, ~line 60)
- Test: `tests/unit/test_pace_gate_onset.py`

Onset rate is a positive rate, so it reuses the BPM log-space math directly (`bpm_log_distance`, `interpolate_log_bpm`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_gate_onset.py
import numpy as np
import pytest
from src.playlist.pier_bridge.pace_gate import (
    compute_step_log_onset_target,
    filter_candidates_by_onset_target,
)


def test_step_onset_target_geometric_midpoint():
    # geometric mean of 1 and 4 is 2
    assert compute_step_log_onset_target(1.0, 4.0, step=1, segment_length=2) == pytest.approx(2.0)


def test_filter_rejects_beyond_cap_keeps_nan():
    onset = np.array([2.0, 8.0, np.nan])  # target=2.0, cap=0.6 log2 (~1.5x)
    kept = filter_candidates_by_onset_target(
        candidate_indices=[0, 1, 2], onset_rate=onset, target_onset=2.0, max_log_distance=0.6,
    )
    assert kept == [0, 2]  # idx1 (8.0, 2 octaves away) rejected; idx2 NaN bypassed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pace_gate_onset.py -v`
Expected: FAIL with `ImportError` (functions not defined).

- [ ] **Step 3: Implement**

Append to `src/playlist/pier_bridge/pace_gate.py`:

```python
def compute_step_log_onset_target(
    onset_a: float,
    onset_b: float,
    *,
    step: int,
    segment_length: int,
) -> float:
    """Log-space interpolated onset-rate target at beam step `step`.

    Onset rate is a positive event-density rate, so it interpolates in
    log-space exactly like BPM (geometric mean at the midpoint).
    """
    if int(segment_length) <= 0:
        return float(onset_a)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_log_bpm(float(onset_a), float(onset_b), t=t)


def filter_candidates_by_onset_target(
    *,
    candidate_indices,
    onset_rate: np.ndarray,
    target_onset: float,
    max_log_distance: float,
) -> list:
    """Drop candidates whose onset-rate log-distance to target exceeds the cap.

    Candidates with NaN onset_rate bypass the gate (graceful coverage gap).
    No tempo-stability bypass: onset density is meaningful regardless of tempo
    tracking stability (unlike BPM).
    """
    if not np.isfinite(float(max_log_distance)):
        return list(candidate_indices)
    indices = list(candidate_indices)
    if not indices:
        return []
    cand_onset = onset_rate[indices]
    bypass = np.isnan(cand_onset)
    distances = _bpm_log_distance(cand_onset, float(target_onset))
    pass_mask = bypass | (distances <= float(max_log_distance))
    return [idx for idx, ok in zip(indices, pass_mask) if bool(ok)]
```

(`interpolate_log_bpm` and `_bpm_log_distance` are already imported at line 43 of this module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_pace_gate_onset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_pace_gate_onset.py src/playlist/pier_bridge/pace_gate.py
git commit -m "feat(pace): onset-rate step-target and band filter helpers"
```

---

## Task 3: Preset keys, dataclass fields, resolution

**Files:**
- Modify: `src/playlist/mode_presets.py:130-163`
- Modify: `src/playlist/config.py:40-65` (CandidatePoolConfig), `:582-595` (resolve)
- Modify: `src/playlist/pier_bridge/config.py:43-45`
- Test: `tests/unit/test_pace_presets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_presets.py
from src.playlist.mode_presets import resolve_pace_mode


def test_narrow_has_onset_and_softpenalty_keys_and_zero_rhythm_floor():
    s = resolve_pace_mode("narrow")
    assert s["onset_admission_max_log_distance"] == 0.50
    assert s["onset_bridge_max_log_distance"] == 0.60
    assert s["rhythm_soft_penalty_threshold"] == 0.25
    assert s["rhythm_soft_penalty_strength"] == 0.15
    # rhythm-cosine hard floors are disabled (soft now)
    assert s["admission_floor"] == 0.0
    assert s["bridge_floor"] == 0.0


def test_off_disables_everything():
    s = resolve_pace_mode("off")
    assert s["onset_admission_max_log_distance"] == float("inf")
    assert s["rhythm_soft_penalty_strength"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pace_presets.py -v`
Expected: FAIL with `KeyError: 'onset_admission_max_log_distance'`.

- [ ] **Step 3: Implement presets**

Replace `PACE_MODE_PRESETS` (`src/playlist/mode_presets.py:130-163`) entries. For each mode set the rhythm-cosine floors to 0.0 and add the four new keys. Final dict:

```python
PACE_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.30,
        "bpm_bridge_max_log_distance": 0.40,
        "onset_admission_max_log_distance": 0.30,
        "onset_bridge_max_log_distance": 0.40,
        "rhythm_soft_penalty_threshold": 0.35,
        "rhythm_soft_penalty_strength": 0.20,
        "description": "Tight tempo fidelity - stay anchored to seed pace",
        "use_case": "Slow/meditative seeds; mood-locked playlists",
    },
    "narrow": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.50,
        "bpm_bridge_max_log_distance": 0.60,
        "onset_admission_max_log_distance": 0.50,
        "onset_bridge_max_log_distance": 0.60,
        "rhythm_soft_penalty_threshold": 0.25,
        "rhythm_soft_penalty_strength": 0.15,
        "description": "Moderate tempo anchoring",
        "use_case": "Consistent energy with some flex",
    },
    "dynamic": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": 0.75,
        "bpm_bridge_max_log_distance": 0.85,
        "onset_admission_max_log_distance": 0.75,
        "onset_bridge_max_log_distance": 0.85,
        "rhythm_soft_penalty_threshold": 0.15,
        "rhythm_soft_penalty_strength": 0.10,
        "description": "Gentle pace anchoring - catches double-time, allows natural drift",
        "use_case": "General-purpose default; varied playlists with sensible tempo coherence",
    },
    "off": {
        "admission_floor": 0.0,
        "bridge_floor": 0.0,
        "bpm_admission_max_log_distance": float("inf"),
        "bpm_bridge_max_log_distance": float("inf"),
        "onset_admission_max_log_distance": float("inf"),
        "onset_bridge_max_log_distance": float("inf"),
        "rhythm_soft_penalty_threshold": 0.0,
        "rhythm_soft_penalty_strength": 0.0,
        "description": "No pace constraint - rhythm contributes via sonic embedding only",
        "use_case": "Multi-tempo playlists; no explicit tempo gating",
    },
}
```

- [ ] **Step 4: Implement dataclass fields (CandidatePoolConfig)**

In `src/playlist/config.py`, add to `CandidatePoolConfig` (after line 64 `bpm_stability_min`):

```python
    onset_admission_max_log_distance: float = float("inf")  # inf = disabled
```

In `resolve_thresholds` candidate config build (after line 594 `bpm_stability_min=...`), add:

```python
        onset_admission_max_log_distance=float(
            candidate_pool.get(
                "onset_admission_max_log_distance",
                pace_settings["onset_admission_max_log_distance"],
            )
        ),
```

- [ ] **Step 5: Implement dataclass fields (PierBridgeConfig)**

In `src/playlist/pier_bridge/config.py`, after line 45 (`bpm_stability_min`):

```python
    onset_bridge_max_log_distance: float = float("inf")  # inf = disabled
    rhythm_soft_penalty_threshold: float = 0.0  # below this rhythm cosine, demote
    rhythm_soft_penalty_strength: float = 0.0   # multiplicative penalty (0 = off)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_pace_presets.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_pace_presets.py src/playlist/mode_presets.py src/playlist/config.py src/playlist/pier_bridge/config.py
git commit -m "feat(pace): preset keys + config fields for onset band and rhythm soft penalty"
```

---

## Task 4: Onset admission band + remove rhythm-cosine admission gate

**Files:**
- Modify: `src/playlist/candidate_pool.py:502-528` (signature), `:631-642` + `:872-882` (rhythm gate removal), `:644-675` (add onset band after BPM band)
- Test: `tests/unit/test_candidate_pool_onset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_candidate_pool_onset.py
"""Onset admission band rejects far-density candidates; rhythm-cosine no longer gates."""
import numpy as np
from dataclasses import replace
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _cfg():
    return CandidatePoolConfig(
        similarity_floor=-1.0, min_sonic_similarity=None, max_pool_size=50,
        target_artists=50, candidates_per_artist=5, seed_artist_bonus=0,
        max_artist_fraction_final=1.0, onset_admission_max_log_distance=0.6,
    )


def test_onset_band_rejects_far_density():
    # 4 tracks; seed idx0 onset=2.0; idx1 close (2.5), idx2 far (16.0), idx3 NaN bypass
    N = 4
    emb = np.eye(N, 8, dtype=float)  # arbitrary, similarity_floor=-1 admits all sonically
    onset = np.array([2.0, 2.5, 16.0, np.nan])
    pool = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=emb,
        artist_keys=np.array(["s", "a", "b", "c"]),
        track_ids=np.array(["s", "a", "b", "c"]),
        track_titles=np.array(["s", "a", "b", "c"]),
        track_artists=np.array(["s", "a", "b", "c"]),
        durations_ms=np.array([200000] * N),
        cfg=_cfg(), random_seed=0, X_sonic=emb,
        onset_rate=onset,
    )
    members = set(pool.indices) if hasattr(pool, "indices") else set(pool)
    assert 1 in members        # close density admitted
    assert 2 not in members    # far density rejected
    assert 3 in members        # NaN bypassed
```

> Note: confirm `build_candidate_pool`'s return shape during implementation; adapt the `members` extraction to the actual returned object (it returns a pool object with selected indices — match the existing accessor used elsewhere, e.g. `tests/unit/test_*candidate*`).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_candidate_pool_onset.py -v`
Expected: FAIL — `build_candidate_pool` has no `onset_rate` kwarg (TypeError).

- [ ] **Step 3: Add `onset_rate` param**

In `src/playlist/candidate_pool.py:502`, add to the signature (after `tempo_stability` at line 528):

```python
    onset_rate: Optional[np.ndarray] = None,
```

- [ ] **Step 4: Add onset admission band**

After the BPM admission gate block (ends ~line 675), add a parallel onset band:

```python
    # ── Onset-rate admission band ────────────────────────────────────────────
    if (
        not np.isinf(float(getattr(cfg, "onset_admission_max_log_distance", float("inf"))))
        and onset_rate is not None
    ):
        from src.playlist.bpm_axis import bpm_log_distance

        max_log_onset = float(cfg.onset_admission_max_log_distance)
        seed_onset_vals = onset_rate[seed_list]
        onset_dist_cols = np.stack(
            [bpm_log_distance(onset_rate, float(so)) for so in seed_onset_vals], axis=1
        )
        onset_seed_min_dist = np.min(onset_dist_cols, axis=1)

        onset_bypass = np.isnan(onset_rate)  # no stability bypass for onset
        onset_fail = ~onset_bypass & (onset_seed_min_dist > max_log_onset)
        onset_fail[seed_list] = False

        seed_sim_all[onset_fail] = -2.0
        logger.info(
            "Onset admission band: max_log_distance=%.2f rejected=%d",
            max_log_onset, int(np.sum(onset_fail)),
        )
```

- [ ] **Step 5: Remove rhythm-cosine admission gate**

Delete the `rhythm_seed_sim` computation block (`src/playlist/candidate_pool.py:631-642`), the gate check (`:872-874`, the `if rhythm_seed_sim is not None and rhythm_seed_sim[i] < float(cfg.pace_admission_floor)` block), and the log block (`:877-882`). Also delete the now-unused `below_pace_floor` initialization. (`pace_admission_floor` defaults to 0.0 in presets so any leftover reference is inert, but remove the dead code per the configured-knob-must-act rule.)

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_candidate_pool_onset.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_candidate_pool_onset.py src/playlist/candidate_pool.py
git commit -m "feat(pace): onset admission band; remove rhythm-cosine admission gate"
```

---

## Task 5: Thread onset_rate through the builders

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py:241-242` (signature)
- Modify: `src/playlist/pier_bridge_builder.py:268-287` (signature), `:1274-1276` (beam call)
- Modify: `src/playlist/pipeline/core.py:281-301` (load), `:396-397` (pool call), `:616-617` (pier-bridge call)
- Test: covered by Task 6 integration; this task is pure wiring.

- [ ] **Step 1: Add `onset_rate` to `_beam_search_segment`**

`src/playlist/pier_bridge/beam.py:242`, after `tempo_stability`:

```python
    onset_rate: Optional[np.ndarray] = None,
```

- [ ] **Step 2: Add `onset_rate` to `build_pier_bridge_playlist`**

`src/playlist/pier_bridge_builder.py:287`, after `tempo_stability_arr`:

```python
    onset_rate: Optional[np.ndarray] = None,
```

And pass it into the `_beam_search_segment` call (`:1276`, after `rhythm_matrix=rhythm_matrix,`):

```python
                        onset_rate=onset_rate,
```

- [ ] **Step 3: Load + thread in core.py**

In `src/playlist/pipeline/core.py`, after `tempo_stability_bpm = _bpm_arrays["tempo_stability"]` (line 294) add:

```python
            onset_rate_arr = _bpm_arrays["onset_rate"]
```

Add `onset_rate_arr: Optional[np.ndarray] = None` initialization next to `perceptual_bpm` at line 281 (so it's defined when the BPM block is skipped):

```python
    onset_rate_arr: Optional[np.ndarray] = None
```

Pass into `build_candidate_pool` (`_build_pool`, after line 397 `tempo_stability=tempo_stability_bpm,`):

```python
            onset_rate=onset_rate_arr,
```

Pass into `_run_pier_bridge` → `build_pier_bridge_playlist` (after line 617 `tempo_stability_arr=tempo_stability_bpm,`):

```python
                    onset_rate=onset_rate_arr,
```

> Note: the BPM block only loads when a BPM **or** onset band is active. Update the guard at `core.py:285` so onset alone triggers the load:
> ```python
> _onset_adm = float(getattr(cfg.candidate, "onset_admission_max_log_distance", float("inf")))
> if not (np.isinf(_bpm_adm) and np.isinf(_bpm_brd) and np.isinf(_onset_adm) and np.isinf(_onset_brd)):
> ```
> where `_onset_brd = float(pace_settings.get("onset_bridge_max_log_distance", float("inf")))`. Define `_onset_brd` next to `_bpm_brd` at line 284.

- [ ] **Step 4: Run the existing suite to confirm no regression**

Run: `python -m pytest tests/unit/test_artifact_builder_graph.py tests/unit/test_pace_presets.py -q`
Expected: PASS (signatures accept the new kwarg; nothing else changed).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/beam.py src/playlist/pier_bridge_builder.py src/playlist/pipeline/core.py
git commit -m "feat(pace): thread onset_rate through pool + pier-bridge builders"
```

---

## Task 6: Onset bridge band + rhythm-cosine soft penalty

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py:1035-1047` (rhythm gate → capture pace_sim, no continue), `:1049-1076` (add onset band after BPM band), `:1138-1141` (apply soft penalty)
- Modify: `src/playlist/pier_bridge_builder.py:471-473` (copy onset/soft fields into `pb_cfg`)
- Test: `tests/unit/test_beam_onset_softpenalty.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_beam_onset_softpenalty.py
"""Onset bridge band gates; rhythm-cosine soft penalty demotes; both skip cleanly
when their inputs are absent (MERT no-tower path)."""
import numpy as np
from src.playlist.pier_bridge.pace_gate import filter_candidates_by_onset_target


def test_onset_bridge_band_rejects_far_density():
    onset = np.array([4.0, 4.5, 32.0])
    kept = filter_candidates_by_onset_target(
        candidate_indices=[0, 1, 2], onset_rate=onset, target_onset=4.0, max_log_distance=0.6,
    )
    assert kept == [0, 1]  # 32.0 (3 octaves) rejected


def test_soft_penalty_multiplier_below_threshold():
    # Pure arithmetic guard for the multiplier the beam applies.
    strength = 0.15
    base = 1.0
    demoted = base * (1.0 - strength)
    assert demoted == 0.85
```

> The full gate behavior is verified end-to-end in Task 8 (the Green-House integration test). This unit test pins the band helper and the penalty arithmetic; the beam wiring is exercised by integration.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_beam_onset_softpenalty.py -v`
Expected: PASS for the helper test if Task 2 done, but keep this file — its purpose is regression pinning. (If run before Task 2, FAIL on import.)

- [ ] **Step 3: Convert the rhythm-cosine hard gate to a captured similarity**

In `src/playlist/pier_bridge/beam.py`, replace the hard pace gate (lines 1035-1047) with a capture that does NOT `continue`:

```python
                pace_sim_for_penalty = None
                if rhythm_matrix is not None:
                    from src.playlist.pier_bridge.pace_gate import compute_step_rhythm_target
                    from src.playlist.sonic_axes import axis_cosine_similarity

                    _pace_target = compute_step_rhythm_target(
                        rhythm_matrix[int(pier_a)],
                        rhythm_matrix[int(pier_b)],
                        step=step,
                        segment_length=interior_length,
                    )
                    pace_sim_for_penalty = float(
                        axis_cosine_similarity(rhythm_matrix[int(cand)], _pace_target).reshape(-1)[0]
                    )
```

- [ ] **Step 4: Add the onset bridge band**

Immediately after the BPM bridge gate block (ends ~line 1076), add:

```python
                # Onset-rate bridge band (hard; embedding-independent)
                if (
                    float(getattr(cfg, "onset_bridge_max_log_distance", float("inf"))) < float("inf")
                    and onset_rate is not None
                ):
                    from src.playlist.pier_bridge.pace_gate import compute_step_log_onset_target
                    from src.playlist.bpm_axis import bpm_log_distance as _old

                    _onset_target = compute_step_log_onset_target(
                        float(onset_rate[int(pier_a)]),
                        float(onset_rate[int(pier_b)]),
                        step=step,
                        segment_length=interior_length,
                    )
                    _cand_onset = float(onset_rate[int(cand)])
                    if not np.isnan(_cand_onset):
                        if float(_old(_cand_onset, _onset_target)) > float(cfg.onset_bridge_max_log_distance):
                            continue
```

- [ ] **Step 5: Apply the soft rhythm penalty to combined_score**

After `combined_score` is first assembled (`src/playlist/pier_bridge/beam.py:1138-1141`), add:

```python
                if (
                    pace_sim_for_penalty is not None
                    and float(getattr(cfg, "rhythm_soft_penalty_strength", 0.0)) > 0.0
                    and pace_sim_for_penalty < float(cfg.rhythm_soft_penalty_threshold)
                ):
                    combined_score *= (1.0 - float(cfg.rhythm_soft_penalty_strength))
```

> Apply the same penalty in the tie-break scoring path (the `apply_tie_break` loop, ~line 1356-1373) for parity: recompute `pace_sim` there from `rhythm_matrix` or thread the captured value via `cand_entries`. Keep it behind the same `rhythm_soft_penalty_strength > 0` guard. If `rhythm_matrix is None` (MERT), `pace_sim_for_penalty` stays None and the penalty is skipped entirely.

- [ ] **Step 6: Copy new fields into pb_cfg**

In `src/playlist/pier_bridge_builder.py`, the `replace(pb_cfg, ...)` does not exist here — it's in `core.py:468-473`. Update that block:

```python
            pb_cfg = replace(
                pb_cfg,
                pace_bridge_floor=float(cfg.candidate.pace_bridge_floor),
                bpm_bridge_max_log_distance=float(pace_settings.get("bpm_bridge_max_log_distance", float("inf"))),
                bpm_stability_min=float(cfg.candidate.bpm_stability_min),
                onset_bridge_max_log_distance=float(pace_settings.get("onset_bridge_max_log_distance", float("inf"))),
                rhythm_soft_penalty_threshold=float(pace_settings.get("rhythm_soft_penalty_threshold", 0.0)),
                rhythm_soft_penalty_strength=float(pace_settings.get("rhythm_soft_penalty_strength", 0.0)),
            )
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/unit/test_beam_onset_softpenalty.py tests/unit/test_pace_gate_onset.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tests/unit/test_beam_onset_softpenalty.py src/playlist/pier_bridge/beam.py src/playlist/pipeline/core.py
git commit -m "feat(pace): onset bridge band + rhythm-cosine soft penalty (MERT-safe)"
```

---

## Task 7: Widen onset + BPM bridge caps on backoff

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py:869-878` (`_run_segment_backoff_attempts`)
- Test: `tests/unit/test_backoff_band_widen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backoff_band_widen.py
def test_band_widen_factor():
    # The backoff applies 1.5x per attempt index to finite caps.
    base = 0.60
    assert base * (1.5 ** 0) == 0.60
    assert round(base * (1.5 ** 1), 4) == 0.90
    assert round(base * (1.5 ** 2), 4) == 1.35
```

> This pins the chosen widen factor. The integration of widening into `cfg_attempt` is exercised by Task 8 (the Green-House case relaxes rather than throwing).

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/unit/test_backoff_band_widen.py -v`
Expected: PASS (pure arithmetic).

- [ ] **Step 3: Implement widening in the backoff loop**

In `src/playlist/pier_bridge_builder.py`, change the per-attempt cfg construction (line 878) to also widen finite onset/BPM bridge caps:

```python
            _widen = 1.5 ** floor_attempt_idx

            def _widened(cap: float) -> float:
                cap = float(cap)
                return cap if not np.isfinite(cap) else cap * _widen

            cfg_attempt = replace(
                cfg,
                bridge_floor=float(bridge_floor),
                onset_bridge_max_log_distance=_widened(
                    getattr(cfg, "onset_bridge_max_log_distance", float("inf"))
                ),
                bpm_bridge_max_log_distance=_widened(
                    getattr(cfg, "bpm_bridge_max_log_distance", float("inf"))
                ),
            )
```

- [ ] **Step 4: Run targeted suite**

Run: `python -m pytest tests/unit/test_backoff_band_widen.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_backoff_band_widen.py src/playlist/pier_bridge_builder.py
git commit -m "feat(pace): widen onset/BPM bridge caps on segment backoff (budget guard)"
```

---

## Task 8: Integration regression — Green-House narrow generates

**Files:**
- Test: `tests/integration/test_gui_fidelity_regressions.py` (add a case)

Uses the production fidelity harness (playlist-testing skill). The Green-House artist run is a DB-clustering path, which `generate_like_gui` does not cover; instead reproduce the **failure mechanism** with multi-pier seeds drawn from Green-House's bundle track_ids under `pace_mode="narrow"`, which is what threw `Segment infeasible`.

- [ ] **Step 1: Write the integration test**

```python
# Append to tests/integration/test_gui_fidelity_regressions.py
import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_pace_narrow_feasible_for_ambient_piers(artifact_or_skip):
    """Regression: pace_mode=narrow threw 'Segment infeasible under bridge_floor
    backoff' for Green-House (beatless) because the rhythm-cosine hard gate was
    unsatisfiable. After the BPM+onset band retune it must generate.
    Fixing commit: pace-gate-retune (2026-06-12)."""
    import sys
    sys.path.insert(0, "tests")
    from support.gui_fidelity import generate_like_gui, resolved_artifact_path
    from src.features.artifacts import load_artifact_bundle

    bundle = load_artifact_bundle(resolved_artifact_path())
    ti = bundle.track_id_to_index
    # Green-House pier track_ids from the failing run (skip any absent).
    gh = [
        "dc7a45bf0c0dbf6ebd574343df4e0159",  # Produce Aisle
        "1d73b404fc6e0de8e4628e64ae9dc982",  # Dragline Silk
        "1e86f3e9cae613f43ece846b71c9f7d5",  # Sanibel
        "981e59a511d15e23109f5a3bcf8f4f8c",  # Hinterland I
        "37dc61f3c8f6ba0c3742979deef6af96",  # Farewell, Little Island
    ]
    seeds = [t for t in gh if t in ti]
    if len(seeds) < 4:
        pytest.skip("Green-House piers not in this artifact build")

    res = generate_like_gui(
        seeds=seeds, cohesion_mode="dynamic", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow", length=30, random_seed=0,
    )
    assert res is not None
    assert len(res.track_ids) == 30  # adapt to DsRunResult's track accessor
```

> During implementation, confirm the `artifact_or_skip` fixture name and the `DsRunResult` track accessor from the existing tests in this file; adapt the two `assert`/skip lines to match.

- [ ] **Step 2: Run it to verify it fails on `master` baseline behavior**

Run: `python -m pytest tests/integration/test_gui_fidelity_regressions.py::test_pace_narrow_feasible_for_ambient_piers -v`
Expected (BEFORE the retune is wired through generation): FAIL with `ValueError: Segment ... infeasible`. AFTER Tasks 1-7: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_gui_fidelity_regressions.py
git commit -m "test(pace): regression — narrow feasible for ambient piers"
```

---

## Task 9: Full-pool calibration sweep

**Files:**
- Create: `scripts/pace_calibration_sweep.py`
- Output: `docs/run_audits/pace_retune/` (gitignored run artifacts + a committed `findings.md`)

Per evaluation-methodology: full pool (N≈40k), distributions not means, ambient AND rhythmic seeds, never write production paths.

- [ ] **Step 1: Write the sweep script**

```python
# scripts/pace_calibration_sweep.py
"""Full-pool onset/BPM band calibration. Read-only. Writes docs/run_audits/pace_retune/."""
import json
from pathlib import Path
import numpy as np
from src.features.artifacts import load_artifact_bundle
from src.playlist.bpm_loader import load_bpm_arrays
from src.playlist.bpm_axis import bpm_log_distance

ART = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
OUT = Path("docs/run_audits/pace_retune"); OUT.mkdir(parents=True, exist_ok=True)

bundle = load_artifact_bundle(ART)
arrs = load_bpm_arrays(bundle.track_ids, db_path="data/metadata.db")
onset = arrs["onset_rate"]; bpm = arrs["perceptual_bpm"]
artists = np.array([str(a) for a in bundle.track_artists])

CAPS = [0.30, 0.40, 0.50, 0.60, 0.75, 0.85, 1.00]
AMBIENT = ["Green-House", "Hiroshi Yoshimura", "Stars of the Lid", "Brian Eno"]
RHYTHMIC = ["J Dilla", "De La Soul", "Beastie Boys", "Kendrick Lamar"]

def passrate(metric_arr, targets, cap):
    rates = []
    for t in targets:
        if np.isnan(metric_arr[t]):
            continue
        d = bpm_log_distance(metric_arr[t], metric_arr)
        rates.append(np.nanmean(d <= cap))
    return float(np.mean(rates)) if rates else float("nan")

report = {"N": int(len(onset)), "onset_coverage": float(np.mean(~np.isnan(onset)))}
rng = np.random.default_rng(0)
rand = rng.choice(len(onset), 200, replace=False)
for label, arr in (("onset", onset), ("bpm", bpm)):
    report[label] = {}
    for cap in CAPS:
        row = {"library": passrate(arr, rand, cap)}
        for a in AMBIENT + RHYTHMIC:
            idx = np.where(artists == a)[0]
            row[a] = passrate(arr, idx, cap) if len(idx) else None
        report[label][f"cap_{cap}"] = row

(OUT / "index.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
```

- [ ] **Step 2: Run the sweep**

Run: `python scripts/pace_calibration_sweep.py`
Expected: prints per-cap pass-rate table; writes `docs/run_audits/pace_retune/index.json`. Verify `onset_coverage` ≈ 1.0 (assumption check from spec §Risks).

- [ ] **Step 3: Write findings**

Create `docs/run_audits/pace_retune/findings.md`: for each mode, the chosen onset/BPM caps with the pass-rate evidence (library p50 + ambient/rhythmic split), and an explicit statement that narrow admits a workable pool for Green-House while staying tighter than dynamic for J Dilla. State N and that these are structural-feasibility numbers (perceptual validation is a follow-up blind audition).

- [ ] **Step 4: Commit**

```bash
git add scripts/pace_calibration_sweep.py docs/run_audits/pace_retune/findings.md
git commit -m "chore(pace): full-pool calibration sweep + findings"
```

---

## Task 10: Finalize preset caps + verify full suite

**Files:**
- Modify: `src/playlist/mode_presets.py` (lock caps from Task 9 findings, if they differ from initial)
- Modify: spec doc note that values are finalized.

- [ ] **Step 1: Update caps** if Task 9 findings indicate the initial mirror-BPM values starve or over-admit. If the initial values hold, record that in the commit message and skip the edit.

- [ ] **Step 2: Run the targeted pace suite**

Run: `python -m pytest tests/unit/test_bpm_loader_onset.py tests/unit/test_pace_gate_onset.py tests/unit/test_pace_presets.py tests/unit/test_candidate_pool_onset.py tests/unit/test_beam_onset_softpenalty.py tests/unit/test_backoff_band_widen.py -q`
Expected: all PASS.

- [ ] **Step 3: Run the broader unit suite (bound the run, no pipe)**

Run: `python -m pytest tests/unit -q -m "not slow"`
Expected: no new failures vs the pre-change baseline (record the baseline counts before Task 1). Quote real pass/fail numbers.

- [ ] **Step 4: Exercise the real path**

Run: `python main_app.py --artist "Green-House" --tracks 50 --dry-run --pace-mode narrow`
Expected: generates 50 tracks within the 90 s budget (was: `Segment infeasible`). Also re-run `--pace-mode dynamic` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/mode_presets.py docs/superpowers/specs/2026-06-12-pace-gate-retune-design.md
git commit -m "feat(pace): finalize calibrated pace-mode caps"
```

---

## Self-Review notes

- **Spec coverage:** §1 onset signal → Task 1; §2 onset band both sides → Tasks 4 (admission) + 6 (bridge); §3 soft penalty + MERT skip → Task 6; §4 backoff widening → Task 7; §5 presets → Task 3 (initial) + Task 10 (final); §6 calibration → Task 9; §7 testing → Tasks 1-8 + 10.
- **MERT durability** verified structurally: soft penalty guarded by `rhythm_matrix is not None` / `pace_sim_for_penalty is not None`; hard bands read DB arrays only.
- **Budget:** Task 7 ensures the new hard bands relax on backoff instead of cascading past 90 s; Task 10 step 4 confirms wall-clock.
- **Open adaptation points flagged inline** (return-object accessors, fixture names) — these are existing-codebase details to confirm at implementation time, not placeholders for undecided design.
