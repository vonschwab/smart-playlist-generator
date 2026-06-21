# Pace as a Co-equal Axis (energy admission-rescue) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the energy/arousal arc a seat in candidate-pool admission so on-arc tracks exist in the pool even in strict/narrow, making pace a co-equal axis (sonic ⊗ genre ⊗ pace).

**Architecture:** An **additive energy-rescue** step in `build_candidate_pool`: after the existing gates, tracks rejected *only* by the onset/BPM rhythm bands but still clearing the genre floor AND the sonic floor are eligible for rescue. From that set, an arousal-spanning subset (up to per-mode `k_energy`) is force-admitted back into the pool. The already-merged beam energy penalty then places them on the per-segment arc. Purely additive (never removes/gates), genre + sonic floors always respected, default no-op (`k_energy=0`).

**Tech Stack:** Python 3.11, numpy, pytest; `src/playlist/candidate_pool.py` (admission), `src/playlist/pipeline/core.py` (threading), `src/playlist/mode_presets.py` (presets), `tests/support/gui_fidelity.py` (`generate_like_gui`), React/TS GUI.

## Global Constraints

- **Never hard-fail on pace.** The rescue ONLY unions candidates in — it never removes a track, never adds a hard gate, never raises. Missing energy data ⇒ rescue skipped, behavior unchanged. (`feedback_never_fail_three_axes`)
- **Genre authority preserved.** Rescued tracks must clear the genre floor (`effective_genre_floor`). Touch no genre gate/weight/arc.
- **Sonic safety floor preserved.** Rescued tracks must clear `min_sonic_similarity`; the beam's per-edge `bridge_floor` still applies downstream. Pace may admit a rhythmically-dissimilar track, NEVER a sonically-disconnected one.
- **Default is a no-op.** `pace_rescue_k_energy` defaults to `0` in every mode and in `CandidateConfig` ⇒ byte-identical to current behavior (golden-safe) until calibrated values ship.
- **90s budget.** `k_energy` is bounded and small; rescue is O(pool) over a 1-D arousal array. (`feedback_generation_time_budget`)
- **Worst-edge eval-gate before default-on.** A mode ships rescue-on only if its weakest sonic edge stays above threshold (rescue-on vs -off, diverse seeds). A failing mode ships `k_energy=0`.
- **Data access in this worktree:** `data/` is not symlinked here; run generations against the MAIN-checkout absolute paths (`C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/...`) and verify `BPM loaded: N/N` in the log (the confound that wasted a day).

---

## File Structure

- `src/playlist/energy_rescue.py` (Create) — pure selection helper `select_energy_rescue(arousal, source_indices, k_energy)`. One responsibility, fully unit-testable, no pipeline deps.
- `src/playlist/candidate_pool.py` (Modify) — capture the rhythm-fail mask in the BPM/onset blocks; after the genre gate, compute the rescue source mask, call the helper, restore `seed_sim_all` for the chosen, extend `eligible`, log. New `build_candidate_pool` param `X_energy`; new `CandidateConfig.pace_rescue_k_energy`.
- `src/playlist/pipeline/core.py` (Modify) — thread `X_energy` (already loaded for the beam) + `cfg.candidate.pace_rescue_k_energy` into `_build_pool`/`build_candidate_pool`; extend the lazy energy-load gate to also fire when `pace_rescue_k_energy>0`.
- `src/playlist/mode_presets.py` (Modify) — per-mode `pace_rescue_k_energy` in `PACE_MODE_PRESETS` (default 0); surfaced into `CandidateConfig` in core.
- `web/src/components/GenerateControls.tsx` (Modify) — add `off` to the pace dropdown + TS union.
- `tests/unit/test_energy_rescue.py` (Create) — helper + admission-integration tests.
- `scripts/research/pace_cede_eval.py` (Modify, exists) — worst-edge sonic + per-mode calibration runner.
- `docs/run_audits/pace_cedes_sonic/` (gitignored) — calibration report.

---

### Task 1: Pure arousal-spanning selection helper

**Files:**
- Create: `src/playlist/energy_rescue.py`
- Test: `tests/unit/test_energy_rescue.py`

**Interfaces:**
- Produces: `select_energy_rescue(arousal: np.ndarray, source_indices: Sequence[int], k_energy: int) -> list[int]` — returns up to `k_energy` indices from `source_indices`, evenly spaced across their sorted arousal (so the picks span the source's arousal range). Skips NaN-arousal indices. `k_energy<=0` or empty source ⇒ `[]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_energy_rescue.py
import numpy as np
from src.playlist.energy_rescue import select_energy_rescue

def test_spans_arousal_range():
    arousal = np.array([0.0, -2.0, 2.0, 1.0, -1.0, 0.5])
    src = [0, 1, 2, 3, 4, 5]
    picked = select_energy_rescue(arousal, src, k_energy=3)
    vals = sorted(arousal[i] for i in picked)
    assert len(picked) == 3
    assert vals[0] == -2.0 and vals[-1] == 2.0   # endpoints of the range are represented

def test_returns_all_when_source_small():
    arousal = np.array([0.0, 1.0, -1.0])
    assert sorted(select_energy_rescue(arousal, [0, 1, 2], k_energy=5)) == [0, 1, 2]

def test_zero_k_and_empty_source():
    arousal = np.array([0.0, 1.0])
    assert select_energy_rescue(arousal, [0, 1], k_energy=0) == []
    assert select_energy_rescue(arousal, [], k_energy=3) == []

def test_skips_nan_arousal():
    arousal = np.array([0.0, np.nan, 2.0])
    picked = select_energy_rescue(arousal, [0, 1, 2], k_energy=3)
    assert 1 not in picked and set(picked) == {0, 2}
```

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/unit/test_energy_rescue.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the helper**

```python
# src/playlist/energy_rescue.py
"""Pure selection helper for energy admission-rescue (pace as a co-equal axis)."""
from __future__ import annotations
from typing import Sequence
import numpy as np


def select_energy_rescue(
    arousal: np.ndarray,
    source_indices: Sequence[int],
    k_energy: int,
) -> list[int]:
    """Pick up to k_energy indices spanning the source's arousal range.

    `arousal` is a 1-D (z-scored) array indexed library-wide. `source_indices`
    are the rescue-eligible tracks (rhythm-rejected but genre+sonic-OK). Returns
    indices evenly spaced across their sorted arousal so low/mid/high are present.
    NaN-arousal indices are skipped. k_energy<=0 or empty source -> [].
    """
    if int(k_energy) <= 0:
        return []
    src = [int(i) for i in source_indices if np.isfinite(arousal[int(i)])]
    if not src:
        return []
    src.sort(key=lambda i: float(arousal[i]))
    if len(src) <= int(k_energy):
        return src
    pos = np.linspace(0, len(src) - 1, int(k_energy)).round().astype(int)
    return [src[j] for j in sorted(set(int(p) for p in pos))]
```

- [ ] **Step 4: Run, verify it passes**

Run: `python -m pytest tests/unit/test_energy_rescue.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/energy_rescue.py tests/unit/test_energy_rescue.py
git commit -m "feat(pace): arousal-spanning energy-rescue selection helper"
```

---

### Task 2: Wire the rescue into `build_candidate_pool`

**Files:**
- Modify: `src/playlist/candidate_pool.py` (BPM block ~632-663, onset block ~665-687, genre gate ~939-949, eligible already built ~884; `CandidateConfig` dataclass; `build_candidate_pool` signature ~502)
- Test: `tests/unit/test_energy_rescue.py` (extend)

**Interfaces:**
- Consumes: `select_energy_rescue` (Task 1).
- Produces: `CandidateConfig.pace_rescue_k_energy: int = 0`; `build_candidate_pool(..., X_energy: Optional[np.ndarray] = None)` keyword param. When `cfg.pace_rescue_k_energy>0` and `X_energy` is not None, rhythm-rejected-but-genre+sonic-OK tracks are rescued into `eligible`.

- [ ] **Step 1: Capture the rhythm-fail mask (no behavior change)**

In the BPM block (`candidate_pool.py` ~653) after `bpm_fail` is computed, and the onset block (~680) after `onset_fail`, accumulate a combined mask. Initialize before the BPM block:

```python
    rhythm_fail = np.zeros(len(seed_sim_all), dtype=bool)
```
In the BPM block, after `bpm_fail[seed_list] = False`:
```python
        rhythm_fail = rhythm_fail | bpm_fail
```
In the onset block, after `onset_fail[seed_list] = False`:
```python
        rhythm_fail = rhythm_fail | onset_fail
```

- [ ] **Step 2: Add the `CandidateConfig` field + `build_candidate_pool` param**

Add to the `CandidateConfig` dataclass (find it in this file or its import; mirror an existing int field like `max_pool_size`):
```python
    pace_rescue_k_energy: int = 0
```
Add to `build_candidate_pool(...)` signature (keyword, after `onset_rate`):
```python
    X_energy: Optional[np.ndarray] = None,
```

- [ ] **Step 3: Write the failing integration test**

```python
# tests/unit/test_energy_rescue.py  (add)
import numpy as np
from dataclasses import replace as _replace
from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig

def _toy(n=40):
    rng = np.random.default_rng(0)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    track_ids = [f"t{i}" for i in range(n)]
    artist_keys = [f"a{i}" for i in range(n)]
    return X_sonic, track_ids, artist_keys

def test_rescue_admits_rhythm_rejected_but_sonic_ok():
    # Two seeds with high onset; make most tracks fail the onset band, but keep
    # them sonically close to a seed; rescue should re-admit arousal-spanning ones.
    X_sonic, track_ids, artist_keys = _toy()
    n = len(track_ids)
    onset = np.full(n, 5.0); onset[:2] = 0.1            # seeds slow, others fast -> onset band rejects others
    arousal = np.linspace(-2, 2, n)
    base = CandidateConfig(
        similarity_floor=-1.0, min_sonic_similarity=None, max_pool_size=1000,
        target_artists=1000, onset_admission_max_log_distance=0.30,
    )
    seeds = [0, 1]
    no_rescue = build_candidate_pool(
        seed_idx=0, seed_indices=seeds, embedding=None, artist_keys=artist_keys,
        track_ids=track_ids, cfg=base, X_sonic=X_sonic, onset_rate=onset,
        X_energy=arousal,
    )
    with_rescue = build_candidate_pool(
        seed_idx=0, seed_indices=seeds, embedding=None, artist_keys=artist_keys,
        track_ids=track_ids, cfg=_replace(base, pace_rescue_k_energy=6),
        X_sonic=X_sonic, onset_rate=onset, X_energy=arousal,
    )
    n0 = len(no_rescue.pool_indices); n1 = len(with_rescue.pool_indices)
    assert n1 >= n0                                   # additive: never shrinks
    assert n1 > n0                                    # rescue actually admitted some
    # rescued set spans arousal (min and max arousal both present among rescued)
    rescued = set(int(i) for i in with_rescue.pool_indices) - set(int(i) for i in no_rescue.pool_indices)
    a_res = sorted(arousal[i] for i in rescued)
    assert a_res and a_res[0] < -0.5 and a_res[-1] > 0.5
```
Adjust the `build_candidate_pool` call kwargs to match the real signature (read it first; pass the minimum required args — seeds, cfg, X_sonic, track_ids, artist_keys — plus `onset_rate`, `X_energy`). The assertions (additive, non-empty rescue, arousal span) are the contract.

- [ ] **Step 4: Run, verify it fails**

Run: `python -m pytest tests/unit/test_energy_rescue.py -k rescue_admits -q`
Expected: FAIL (rescue not implemented; `n1 == n0`).

- [ ] **Step 5: Implement the rescue (after the genre gate, ~line 950)**

Insert after the genre hard-gate block and before the layered-genre block (`candidate_pool.py` ~951), so `eligible`, `sonic_seed_sim`, `genre_sim_all`, `effective_genre_floor`, `rhythm_fail`, `seed_sim_all` are all in scope:

```python
    # ── Energy admission-rescue (pace as a co-equal axis) ────────────────────
    # Re-admit tracks rejected ONLY by the rhythm bands (onset/BPM) that still
    # clear the genre AND sonic floors, choosing an arousal-spanning subset so
    # the pool carries on-arc-energy candidates even in tight modes. Additive;
    # never removes. Genre + sonic floors fully respected. No-op when k_energy=0.
    k_energy = int(getattr(cfg, "pace_rescue_k_energy", 0))
    if k_energy > 0 and X_energy is not None and np.any(rhythm_fail):
        from src.playlist.energy_rescue import select_energy_rescue
        eligible_set = set(eligible)
        sonic_ok = (
            sonic_seed_sim is not None and sonic_floor is not None
        )
        source = []
        for i in np.nonzero(rhythm_fail)[0]:
            i = int(i)
            if i in eligible_set or seed_mask[i]:
                continue
            if sonic_ok and (sonic_seed_sim[i] + epsilon) < sonic_floor:
                continue  # sonic safety floor — never rescue a disconnected track
            if (
                genre_sim_all is not None
                and effective_genre_floor is not None
                and genre_sim_all[i] < effective_genre_floor
            ):
                continue  # genre authority preserved
            source.append(i)
        rescued = select_energy_rescue(np.asarray(X_energy, dtype=float), source, k_energy)
        for i in rescued:
            # Restore a genuine rank score so the track survives similarity_floor
            # and ranks sensibly (it was set to the rhythm sentinel -2.0).
            seed_sim_all[i] = float(sonic_seed_sim[i]) if sonic_ok else float(cfg.similarity_floor)
            eligible.append(i)
        if rescued:
            logger.info(
                "Energy rescue: admitted=%d from rhythm-rejected (k_energy=%d, source=%d)",
                len(rescued), k_energy, len(source),
            )
```

- [ ] **Step 6: Run, verify it passes**

Run: `python -m pytest tests/unit/test_energy_rescue.py -q`
Expected: PASS (helper + integration). If the toy `build_candidate_pool` call needs more required args, add them minimally; do not change the assertions.

- [ ] **Step 7: Default-no-op guard (golden safety)**

Run: `python -m pytest tests/unit -k "golden" -q`
Expected: PASS. With `pace_rescue_k_energy=0` (default) and `X_energy=None`, the rescue block is skipped entirely — no behavior change. If a golden fails, STOP and investigate (the default path must be untouched), do not re-snapshot.

- [ ] **Step 8: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_energy_rescue.py
git commit -m "feat(pace): energy admission-rescue in build_candidate_pool (additive, floors preserved)"
```

---

### Task 3: Per-mode presets + thread X_energy/k_energy through core

**Files:**
- Modify: `src/playlist/mode_presets.py` (`PACE_MODE_PRESETS` 4 entries)
- Modify: `src/playlist/pipeline/core.py` (energy lazy-load gate ~324-328; the `cfg.candidate` build / `_build_pool` ~420-460)
- Test: `tests/unit/test_pace_mode_energy_presets.py` (extend) + a generation smoke

**Interfaces:**
- Consumes: `CandidateConfig.pace_rescue_k_energy`, `build_candidate_pool(X_energy=...)` (Task 2); `pace_settings` (already resolved in core).
- Produces: `PACE_MODE_PRESETS[mode]["pace_rescue_k_energy"]` (default 0 all modes); core passes `X_energy` + sets `cfg.candidate.pace_rescue_k_energy` from `pace_settings`.

- [ ] **Step 1: Write the failing preset test**

```python
# tests/unit/test_pace_mode_energy_presets.py  (add)
from src.playlist.mode_presets import PACE_MODE_PRESETS
def test_presets_have_rescue_k_default_zero():
    for mode in ("strict", "narrow", "dynamic", "off"):
        assert PACE_MODE_PRESETS[mode].get("pace_rescue_k_energy", 0) == 0, \
            f"{mode}.pace_rescue_k_energy must default 0 (no-op until calibrated)"
```

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/unit/test_pace_mode_energy_presets.py -k rescue_k -q`
Expected: FAIL (key absent → `.get` returns 0 → actually PASSES vacuously). To make it a real failing test first, assert presence: change to `assert "pace_rescue_k_energy" in PACE_MODE_PRESETS[mode]`. Run → FAIL (key missing).

- [ ] **Step 3: Add the preset key (default 0) to all four modes**

In each of `strict`, `narrow`, `dynamic`, `off` dicts in `PACE_MODE_PRESETS`:
```python
        "pace_rescue_k_energy": 0,
```

- [ ] **Step 4: Run, verify it passes**

Run: `python -m pytest tests/unit/test_pace_mode_energy_presets.py -k rescue_k -q`
Expected: PASS.

- [ ] **Step 5: Thread into core.py**

(a) Extend the lazy energy-load gate (`core.py` ~324) so energy loads when rescue is active too:
```python
    _energy_active = any(
        float(pace_settings.get(k, 0.0)) > 0.0
        for k in ("energy_step_strength", "energy_arc_strength")
    ) or int(pace_settings.get("pace_rescue_k_energy", 0)) > 0
```
(b) `_build_pool` is `def _build_pool(candidate_cfg, genre_gate)` and is called `pool = _build_pool(cfg.candidate, min_genre_similarity)` (~460). Do NOT assume `cfg` supports `replace`. Instead build a ceded candidate-config local and pass it in:
```python
    from dataclasses import replace
    _candidate_cfg = replace(
        cfg.candidate,
        pace_rescue_k_energy=int(pace_settings.get("pace_rescue_k_energy", 0)),
    )
    pool = _build_pool(_candidate_cfg, min_genre_similarity)   # was: _build_pool(cfg.candidate, ...)
```
(c) Thread `X_energy` into the `build_candidate_pool(...)` call inside `_build_pool`. `energy_matrix` is the already-loaded `(n,1)` arousal matrix (loaded earlier in core for the beam). `_build_pool` is a closure, so it can read `energy_matrix` directly; add to the call:
```python
            X_energy=(energy_matrix.reshape(-1) if energy_matrix is not None else None),
```
Verify `energy_matrix` is assigned *before* the `_build_pool` definition/first call (the lazy load at ~329). If it is defined after, move the energy load above the pool build so the closure captures the loaded array.

- [ ] **Step 6: Generation smoke (no-op default + rescue-on path runs)**

Run a strict generation via `generate_like_gui` against MAIN-checkout data (verify `BPM loaded: N/N`), once with default presets (rescue off → unchanged vs master) and once with `pace_rescue_k_energy` temporarily set (e.g. 40) to confirm the rescue log line fires and the pool grows. Confirm completion < 90s. (This is a manual smoke; record numbers in the report, do not commit a script change unless trivial.)

- [ ] **Step 7: Commit**

```bash
git add src/playlist/mode_presets.py src/playlist/pipeline/core.py tests/unit/test_pace_mode_energy_presets.py
git commit -m "feat(pace): per-mode pace_rescue_k_energy preset + thread X_energy into pool (default 0)"
```

---

### Task 4: Add `off` to the GUI pace dropdown

**Files:**
- Modify: `web/src/components/GenerateControls.tsx` (dropdown ~line 343; TS cast ~line 151)

**Interfaces:**
- Consumes: existing `axes.pace_mode` + server `VALID_PACE_MODES` (already includes `off`).
- Produces: pace options `["off","dynamic","narrow","strict"]`; `pace_mode` union includes `"off"`.

- [ ] **Step 1: Widen the dropdown options** (line ~343)

```tsx
            {["off", "dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
```

- [ ] **Step 2: Widen the TS union** (line ~151)

```tsx
      pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic" | "off",
```

- [ ] **Step 3: Type-check + build**

Run: `npm --prefix web run build`
Expected: build succeeds (no TS error; `off` flows through existing request types).

- [ ] **Step 4: Commit**

```bash
git add web/src/components/GenerateControls.tsx
git commit -m "feat(web): add 'off' to the pace dropdown"
```

---

### Task 5: Calibration + eval-gate (worst-edge kill criterion)

**Files:**
- Modify: `scripts/research/pace_cede_eval.py` (worst-edge sonic metric + per-mode calibration runner)
- Modify: `src/playlist/mode_presets.py` (set passing `pace_rescue_k_energy` + energy strengths per mode — PASSING modes only)
- Modify: `tests/unit/test_pace_mode_energy_presets.py` (pin the shipped values)
- Output: `docs/run_audits/pace_cedes_sonic/CALIBRATION_COEQUAL.md`

**Interfaces:**
- Consumes: the rescue (Tasks 2-3) + the merged beam energy penalty; `generate_like_gui`.
- Produces: final per-mode `pace_rescue_k_energy` + `energy_arc_band/arc_strength/step_cap/step_strength`; eval report with per-mode PASS/FAIL.

- [ ] **Step 1: Worst-edge sonic metric**

```python
# scripts/research/pace_cede_eval.py
import numpy as np
def worst_edge_sonic(track_ids, bundle):
    """Min adjacent MERT cosine over the playlist (weakest sonic transition)."""
    X = bundle.X_full_norm
    idx = [bundle.track_id_to_index[t] for t in track_ids if t in bundle.track_id_to_index]
    sims = [float(np.dot(X[idx[i]], X[idx[i+1]])) for i in range(len(idx)-1)]
    return min(sims) if sims else float("nan")
```
Confirm `bundle.X_full_norm` / `bundle.track_id_to_index` attribute names against `tests/support/gui_fidelity.py`; adjust if different.

- [ ] **Step 2: Calibration runner (ramp UP from minimal)**

For each mode in (strict, narrow): from a small `pace_rescue_k_energy` + small energy strengths, generate (multi-pier, ≥3 diverse seed sets incl. high-arousal + wide-swing, BPM active), measure `arc_dev`, `max_step`, `worst_edge_sonic` rescue-on vs -off. Increase `k_energy`/strengths only while `worst_edge_sonic(on) >= worst_edge_sonic(off) - DELTA` (e.g. DELTA=0.05) AND wall-time < 90s. dynamic: confirm energy already steers (k_energy may stay 0). off: rescue 0, arc disabled, keep step-cap; verify no whiplash.

- [ ] **Step 3: Run calibration, write the report**

Run: `python scripts/research/pace_cede_eval.py --calibrate-coequal` (MAIN-checkout data).
Write `docs/run_audits/pace_cedes_sonic/CALIBRATION_COEQUAL.md`: per-mode chosen values, arc improvement, worst-edge cost, timing, genre-cohesion-unchanged check, PASS/FAIL.

- [ ] **Step 4: Set passing values + pin them in a test**

For each PASSING mode, set its `pace_rescue_k_energy` (>0) and energy strengths (>0) in `PACE_MODE_PRESETS`. FAILING modes stay at 0/0.0 (disabled). Update `tests/unit/test_pace_mode_energy_presets.py` to assert the shipped values equal the calibration report (pin against drift); update the `==0` default test to reference only the modes that ship disabled.

- [ ] **Step 5: Full regression + budget + genre-preservation**

Run: `python -m pytest -q -m "not slow"` (note any pre-existing failures vs master).
Then generation smoke across all 4 pace_modes (multi-pier, MAIN-checkout data, `BPM loaded` verified): all < 90s (never-hard-fail); genre distinct-artist/cohesion metric unchanged across pace_modes (proves genre untouched).

- [ ] **Step 6: Commit**

```bash
git add scripts/research/pace_cede_eval.py src/playlist/mode_presets.py tests/unit/test_pace_mode_energy_presets.py
git commit -m "feat(pace): calibrate per-mode energy rescue + arc, eval-gated (worst-edge)"
```

---

## Notes for the executor
- **Tasks 1-4 are correct regardless of calibration** (everything ships no-op at `k_energy=0`/strength 0.0). Task 5 sets the live values and is the gate before any default-on.
- **Genre-preservation is a required gate in Task 5** — if any genre cohesion metric shifts across pace_modes, the rescue has leaked into genre; fix before shipping.
- **Read the logs, not just metrics** (CLAUDE.md / playlist-testing skill): verify `BPM loaded: N/N`, the gate tally, and the `Energy rescue: admitted=N` line in every calibration run.
- The energy soft-penalty (arc_band/step_cap) is already on master; this plan only adds the admission rescue + calibration + GUI off. Do not re-implement the beam penalty.
