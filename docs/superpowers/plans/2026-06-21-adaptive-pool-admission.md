# Adaptive Candidate-Pool Admission — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-absolute candidate-pool admission floors with per-seed **adaptive percentile** floors (sonic + genre) plus a **minimum-pool guarantee**, so the pool never starves for low-cosine niches (Charli XCX: sonic floor 0.18 > median 0.156) while genre/sonic stay soft.

**Architecture:** Mirror the existing genre-percentile pattern (`floor_at_percentile`) onto sonic; **decouple** the genre percentile from the dead `X_genre_dense` block so it runs on the sparse genre vectors that are actually live; add a post-artist-walk `min_pool_size` backstop that relaxes the percentile / falls back to top-K while respecting artist caps. Deprecated dense plumbing is left in place (already inert, `X_genre_dense=None`) and removed in **Plan 2** — this plan does not touch it.

**Tech Stack:** Python 3.11, numpy, pytest; `src/playlist/candidate_pool.py`, `src/playlist/config.py`, `src/playlist/mode_presets.py`, `src/playlist/pipeline/core.py`, `src/playlist/pier_bridge/percentiles.py` (reuse), the fidelity harness (`tests/support/gui_fidelity.py`).

## Global Constraints

- **Never-fail on sonic/genre.** Admission is soft/relaxable; the `min_pool_size` backstop makes starvation impossible. Diversity (min_gap, per-artist cap) stays the only hard constraint. (`feedback_never_fail_three_axes`)
- **90s budget.** O(pool) over already-computed 1-D distributions; the backstop is bounded. (`feedback_generation_time_budget`)
- **MERT-live + graph genre.** Design against the current artifact (`X_sonic_variant=mert`, graph-sourced `X_genre_raw/smoothed`). Verify `Using precomputed sonic variant 'mert'` + `BPM loaded: N/N` in every generation log.
- **No legacy shadow default.** Percentile replaces the absolute floor as the operative gate; do not keep the absolute floor as a parallel live path. (Its config field may remain as the `min_pool_size` backstop input only, logged.)
- **No silent no-ops.** Every effective floor (sonic + genre) is logged with the percentile that produced it.
- **Data access in this worktree:** `data/` is symlinked, but generation tests must point at MAIN-checkout absolute paths and verify `BPM loaded: N/N` (the confound that wasted a day). `config.yaml` is copied in (gitignored).
- **Test every change on a real playlist with real logs** (gate tally before/after, pool never starves, <90s) — per `docs/WIRING_STATUS.md` verification protocol. Update `WIRING_STATUS.md` as each task lands.
- **Calibration last.** Percentile + `min_pool_size` values are eval-gated on a **diverse seed corpus** (hyperpop, metal, jazz, ambient, hip-hop, folk, Top-40), distributions not means.

---

## File Structure

- `src/playlist/pier_bridge/percentiles.py` — REUSE `floor_at_percentile(sims, p) -> float` (returns `np.quantile`, empty → `-inf`). No change.
- `src/playlist/candidate_pool.py` — add `sonic_admission_percentile` param; compute adaptive sonic floor after `sonic_seed_sim` (line 634); decouple the genre percentile onto the sparse `genre_sim_all`; add the `min_pool_size` backstop after the artist-walk (lines 1067-1083). One file, one responsibility (admission).
- `src/playlist/config.py` — `sonic_admission_percentile` + `min_pool_size` fields + per-mode resolution (mirror `genre_admission_percentile` at config.py:122/350-353).
- `src/playlist/mode_presets.py` — per-`sonic_mode` `sonic_admission_percentile`, per-mode `min_pool_size` (initial conservative values; calibration sets finals).
- `src/playlist/pipeline/core.py` — thread `sonic_admission_percentile` + `min_pool_size` into `build_candidate_pool` (mirror `_genre_admission_percentile` extraction at core.py:405-413).
- `config.yaml` / `config.example.yaml` — the new per-mode keys.
- Tests: `tests/test_candidate_filters.py`, `tests/unit/test_candidate_pool_max_over_seeds.py`, `tests/unit/test_candidate_pool_idf.py` (migrate 10), plus a new `tests/unit/test_adaptive_admission.py`.
- `scripts/research/adaptive_admission_eval.py` (create) — diverse-seed calibration runner.

---

### Task 1: Sonic admission percentile (mirror the genre pattern)

**Files:**
- Modify: `src/playlist/candidate_pool.py` (signature ~523-535; sonic floor assignment :628; gate :891; add compute after `sonic_seed_sim` at :634)
- Modify: `src/playlist/config.py` (add field + per-mode resolver, mirror `genre_admission_percentile` at :122, :350-353)
- Modify: `src/playlist/mode_presets.py` (per-`sonic_mode` value)
- Modify: `src/playlist/pipeline/core.py` (extract + pass, mirror `_genre_admission_percentile` at :405-413)
- Create: `tests/unit/test_adaptive_admission.py`

**Interfaces:**
- Consumes: `floor_at_percentile(sims, p: float) -> float` (`src/playlist/pier_bridge/percentiles.py:13`).
- Produces: `build_candidate_pool(..., sonic_admission_percentile: Optional[float] = None)`. When set and `> 0`, `sonic_floor = floor_at_percentile(sonic_seed_sim[non-seed, finite], p)` replacing `cfg.min_sonic_similarity` as the operative gate; logged.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_adaptive_admission.py
import numpy as np
from dataclasses import replace as _replace
from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig

def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    track_ids = [f"t{i}" for i in range(n)]
    artist_keys = [f"a{i}" for i in range(n)]
    return X_sonic, track_ids, artist_keys

def test_sonic_percentile_admits_top_fraction():
    # With sonic_admission_percentile=0.80, ~top 20% by sonic sim to the seed are admitted,
    # regardless of any absolute floor. Adapts to the seed's own distribution.
    X_sonic, track_ids, artist_keys = _toy()
    base = CandidateConfig(
        similarity_floor=-1.0, min_sonic_similarity=0.99,  # absolute floor would reject almost all
        max_pool_size=10_000, target_artists=10_000,
    )
    res = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=X_sonic, artist_keys=artist_keys,
        track_ids=track_ids, cfg=_replace(base, sonic_admission_percentile=0.80),
        X_sonic=X_sonic,
    )
    n_admitted = len(res.pool_indices)
    # ~20% of 59 non-seed ≈ 12; assert it's in a sane band and NOT gutted by the 0.99 absolute floor
    assert 6 <= n_admitted <= 20, n_admitted

def test_sonic_percentile_none_uses_absolute_floor():
    # Default (no percentile) preserves legacy absolute-floor behavior.
    X_sonic, track_ids, artist_keys = _toy()
    base = CandidateConfig(similarity_floor=-1.0, min_sonic_similarity=0.0,
                           max_pool_size=10_000, target_artists=10_000)
    res = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=X_sonic, artist_keys=artist_keys,
        track_ids=track_ids, cfg=base, X_sonic=X_sonic,
    )
    assert len(res.pool_indices) > 0
```
Adjust kwargs to the real `build_candidate_pool` signature (read it first; pass the minimum required args). The two assertions (percentile admits ~top fraction despite a high absolute floor; None → legacy) are the contract.

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -k sonic_percentile -q`
Expected: FAIL (`sonic_admission_percentile` not a param / not applied).

- [ ] **Step 3: Add the config field + per-mode resolver**

In `src/playlist/config.py`, mirror `genre_admission_percentile` (field at :122, resolution at :350-353). Add to the same tuning struct:
```python
    sonic_admission_percentile: float = 0.0
```
and in `resolve_pier_bridge_tuning` (~:350), resolve it the same way (`_resolve_mode_number_with_source` with per-mode key `sonic_admission_percentile_<mode>`).

- [ ] **Step 4: Add the `build_candidate_pool` param + compute the adaptive floor**

In `src/playlist/candidate_pool.py`, add to the signature (near :535, beside `genre_admission_percentile`):
```python
    sonic_admission_percentile: Optional[float] = None,
```
After `sonic_seed_sim` is computed (line 634) and before the gate at :891, replace the fixed assignment at :628 path with:
```python
    sonic_floor = cfg.min_sonic_similarity
    if sonic_admission_percentile is not None and float(sonic_admission_percentile) > 0.0:
        from src.playlist.pier_bridge.percentiles import floor_at_percentile
        _sdist = np.asarray(sonic_seed_sim, dtype=np.float64).copy()
        for _si in seed_indices:
            if 0 <= int(_si) < _sdist.shape[0]:
                _sdist[int(_si)] = np.nan
        _sfin = _sdist[np.isfinite(_sdist)]
        sonic_floor = floor_at_percentile(_sfin, float(sonic_admission_percentile))
        logger.info(
            "Sonic admission percentile active: p=%.2f -> effective sonic_floor=%.3f (was abs=%s)",
            float(sonic_admission_percentile), float(sonic_floor),
            cfg.min_sonic_similarity,
        )
```
(The existing gate at :891 `sonic_seed_sim[i] + ε < sonic_floor` now uses the adaptive value.)

- [ ] **Step 5: Thread through `pipeline/core.py`**

Mirror the `_genre_admission_percentile` extraction (`core.py:405-413`): extract `sonic_admission_percentile` from `pb_overrides` (base + mode-specific key) and pass it into the `build_candidate_pool` call.

- [ ] **Step 6: Run, verify it passes**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -k sonic -q`
Expected: PASS.

- [ ] **Step 7: Preset (initial conservative value) + config keys**

In `src/playlist/mode_presets.py` `SONIC_MODE_PRESETS`, add `sonic_admission_percentile` per mode (initial: strict 0.75 / narrow 0.60 / dynamic 0.40 / discover 0.20 / off 0.0 — calibration sets finals). Add the per-mode keys to `config.example.yaml` near the existing `genre_admission_percentile` block (config.example.yaml:305-308) and to `config.yaml`.

- [ ] **Step 8: Real-playlist gate-tally check + commit**

Generate Charli XCX narrow via `generate_like_gui` against MAIN-checkout data (verify `BPM loaded: N/N`, `variant 'mert'`); capture the `Sonic floor applied` / `Candidate pool: admitted=N` lines. Expected: sonic floor now adapts (logged `Sonic admission percentile active`), admitted count rises vs the 0.18-absolute baseline. Record numbers in the report. Then:
```bash
git add src/playlist/candidate_pool.py src/playlist/config.py src/playlist/mode_presets.py src/playlist/pipeline/core.py config.yaml config.example.yaml tests/unit/test_adaptive_admission.py
git commit -m "feat(admission): per-seed adaptive sonic percentile floor (mirrors genre percentile)"
```

---

### Task 2: Decouple the genre percentile onto the sparse vectors

**Files:**
- Modify: `src/playlist/candidate_pool.py` (the sparse genre gate path around :812-956; the percentile compute currently lives in the dense block :764-800)
- Test: `tests/unit/test_adaptive_admission.py` (extend)

**Interfaces:**
- Consumes: `genre_admission_percentile` (already a `build_candidate_pool` param, :535), `floor_at_percentile`.
- Produces: when `genre_admission_percentile > 0`, `effective_genre_floor` is computed from the **sparse** `genre_sim_all` distribution (no `X_genre_dense` required), applied at the sparse hard gate (:956). The dead dense block is untouched (Plan 2 deletes it).

**Why:** the genre percentile path today is inside the `X_genre_dense` block (:740-810); with `X_genre_dense=None` it never runs, so genre admission falls to the absolute `min_genre_similarity=0.4` (sparse gate :956). This task makes the percentile compute on the sparse distribution so genre admission is adaptive on the live path.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_adaptive_admission.py  (add)
import numpy as np
from dataclasses import replace as _replace
from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig

def test_genre_percentile_runs_without_dense():
    # genre_admission_percentile must compute an effective genre floor from the SPARSE
    # genre vectors (X_genre_dense=None), not fall back to the absolute min_genre_similarity.
    n = 60
    rng = np.random.default_rng(1)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    # sparse genre: 5-dim one/two-hot
    X_genre = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        X_genre[i, rng.integers(0, 5)] = 1.0
    tids = [f"t{i}" for i in range(n)]; aks = [f"a{i}" for i in range(n)]
    cfg = CandidateConfig(similarity_floor=-1.0, min_sonic_similarity=None,
                          max_pool_size=10_000, target_artists=10_000)
    res = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=X_sonic, artist_keys=aks, track_ids=tids,
        cfg=_replace(cfg, genre_admission_percentile=0.50),
        X_sonic=X_sonic, X_genre_raw=X_genre, X_genre_smoothed=X_genre,
        X_genre_dense=None, min_genre_similarity=0.4,
    )
    # With percentile active on sparse vectors, the effective floor is data-derived, NOT the abs 0.4.
    # Assert the pool reflects percentile admission (not the degenerate all-or-nothing of abs 0.4).
    assert len(res.pool_indices) > 0
    assert res.stats.get("effective_genre_floor") is not None
```
Confirm the real param names for `X_genre_raw`/`X_genre_smoothed`/`min_genre_similarity` and that `stats` exposes `effective_genre_floor` (add it to stats if absent — it's logged at :802-810 already).

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -k genre_percentile_runs -q`
Expected: FAIL (percentile not applied on sparse path → falls to abs 0.4).

- [ ] **Step 3: Implement — compute the genre percentile on the sparse distribution**

In `candidate_pool.py`, in the sparse genre path (where `genre_sim_all` is computed, before the hard gate at :956), add — mirroring the dense block's centroid logic (:794-800):
```python
    if genre_admission_percentile is not None and float(genre_admission_percentile) > 0.0 \
       and genre_sim_all is not None:
        from src.playlist.pier_bridge.percentiles import floor_at_percentile
        _gdist = np.asarray(genre_sim_all, dtype=np.float64).copy()
        for _si in seed_indices:
            if 0 <= int(_si) < _gdist.shape[0]:
                _gdist[int(_si)] = np.nan
        _gfin = _gdist[np.isfinite(_gdist)]
        effective_genre_floor = floor_at_percentile(_gfin, float(genre_admission_percentile))
        logger.info(
            "Genre admission percentile (sparse) active: p=%.2f -> effective_genre_floor=%.3f",
            float(genre_admission_percentile), float(effective_genre_floor),
        )
```
Ensure `effective_genre_floor` is surfaced in `result.stats` (it is logged; add to the stats dict if not present).

- [ ] **Step 4: Run, verify it passes**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -k genre -q`
Expected: PASS.

- [ ] **Step 5: Golden-safety check**

Run: `python -m pytest tests/unit -k golden -q`
Expected: PASS (with `genre_admission_percentile` unset, behavior unchanged). If a golden fails, STOP and investigate — do not re-snapshot.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_adaptive_admission.py
git commit -m "feat(admission): compute genre percentile floor on sparse vectors (decouple from dead dense)"
```

---

### Task 3: Minimum-pool guarantee (never-starve backstop)

**Files:**
- Modify: `src/playlist/candidate_pool.py` (after the artist-walk stop condition, :1067-1083)
- Modify: `src/playlist/config.py` (`min_pool_size` field + per-mode resolution)
- Modify: `src/playlist/mode_presets.py` (per-mode `min_pool_size`)
- Modify: `src/playlist/pipeline/core.py` (thread it through)
- Test: `tests/unit/test_adaptive_admission.py` (extend)

**Interfaces:**
- Produces: `build_candidate_pool(..., min_pool_size: int = 0)`. After percentile admission + artist-walk, if `len(pool_indices) < min_pool_size`, admit the highest-`sonic_seed_sim` candidates not already admitted until `min_pool_size` is reached **or candidates exhausted**, respecting the per-artist cap (`candidates_per_artist`) and never admitting a seed-artist interior. Logged.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_adaptive_admission.py  (add)
def test_min_pool_guarantee_never_starves():
    import numpy as np
    from dataclasses import replace as _replace
    from src.playlist.candidate_pool import build_candidate_pool, CandidateConfig
    n = 40
    rng = np.random.default_rng(2)
    X_sonic = rng.normal(size=(n, 8)).astype(np.float64)
    tids = [f"t{i}" for i in range(n)]
    aks = [f"a{i%6}" for i in range(n)]  # 6 distinct artists
    # Tight percentile would admit very few; min_pool_size must backfill.
    cfg = CandidateConfig(similarity_floor=-1.0, min_sonic_similarity=None,
                          max_pool_size=10_000, target_artists=10_000)
    res = build_candidate_pool(
        seed_idx=0, seed_indices=[0], embedding=X_sonic, artist_keys=aks, track_ids=tids,
        cfg=_replace(cfg, sonic_admission_percentile=0.98, min_pool_size=15),
        X_sonic=X_sonic,
    )
    assert len(res.pool_indices) >= 15            # never starves
    # diversity respected: not a single-artist pool
    pooled_artists = {aks[i] for i in res.pool_indices}
    assert len(pooled_artists) >= 2
```

- [ ] **Step 2: Run, verify it fails**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -k min_pool -q`
Expected: FAIL (`min_pool_size` not a param; pool < 15 under p=0.98).

- [ ] **Step 3: Add field + resolver + threading**

`config.py`: `min_pool_size: int = 0` (per-mode resolvable). `mode_presets.py`: initial per mode (strict 12 / narrow 16 / dynamic 20 / discover 24 / off 0 — calibration tunes). `pipeline/core.py`: extract + pass into `build_candidate_pool`.

- [ ] **Step 4: Implement the backstop**

In `candidate_pool.py` after the artist-walk that builds `pool_indices` (:1067-1083), add:
```python
    if int(min_pool_size) > 0 and len(pool_indices) < int(min_pool_size):
        from collections import Counter
        already = set(int(i) for i in pool_indices)
        per_artist = Counter(artist_keys[i] for i in pool_indices)
        cap = int(getattr(cfg, "candidates_per_artist", 6) or 6)
        # rank remaining by sonic_seed_sim (desc), respect per-artist cap, skip seeds
        ranked = sorted(
            (i for i in range(len(track_ids))
             if i not in already and i not in set(seed_indices)),
            key=lambda i: float(sonic_seed_sim[i]) if sonic_seed_sim is not None else 0.0,
            reverse=True,
        )
        added = 0
        for i in ranked:
            if len(pool_indices) >= int(min_pool_size):
                break
            ak = artist_keys[i]
            if per_artist[ak] >= cap:
                continue
            pool_indices.append(int(i)); already.add(int(i)); per_artist[ak] += 1; added += 1
        if added:
            logger.info(
                "Min-pool backstop: pool %d below min %d; admitted %d more (top sonic-sim, artist-cap respected)",
                len(pool_indices) - added, int(min_pool_size), added,
            )
```
(Place after `pool_indices` is final but before `pool_indices` is used to build the result.)

- [ ] **Step 5: Run, verify it passes**

Run: `python -m pytest tests/unit/test_adaptive_admission.py -q`
Expected: PASS (all adaptive-admission tests).

- [ ] **Step 6: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/config.py src/playlist/mode_presets.py src/playlist/pipeline/core.py
git commit -m "feat(admission): min_pool_size never-starve backstop (top sonic-sim, artist-cap respected)"
```

---

### Task 4: Migrate the 10 absolute-floor tests

**Files:**
- Modify: `tests/test_candidate_filters.py` (6 sonic + genre floor tests: lines 8, 49, 122, 158, 245, 291, 492/581)
- Modify: `tests/unit/test_candidate_pool_max_over_seeds.py` (lines 22, 56)
- Modify: `tests/unit/test_candidate_pool_idf.py` (line 49)

**Interfaces:** consumes Tasks 1-3. These tests assert absolute-floor rejection; migrate them to assert the **percentile** floor behavior (or keep absolute where the test is specifically about the legacy path being gone). Migrate — do not silence or delete the coverage.

- [ ] **Step 1: Inventory the 10 tests** (see `sdd/trace-absolute-floors.md` §4 for the exact list + what each asserts).

- [ ] **Step 2: For each, decide: keep-as-legacy-guard or migrate-to-percentile.** Tests asserting "absolute floor X rejects candidate at X-ε" become "percentile p rejects below the p-th percentile." Tests of the relaxation retry (`pool_calls[1].min_sonic_similarity < pool_calls[0]`) become assertions on the percentile/min-pool relaxation path.

- [ ] **Step 3: Rewrite each test** to exercise the adaptive path with a concrete distribution + expected admitted set. (Full per-test code is written during execution from the trace inventory — each is a small, self-contained edit.)

- [ ] **Step 4: Run the migrated tests**

Run: `python -m pytest tests/test_candidate_filters.py tests/unit/test_candidate_pool_max_over_seeds.py tests/unit/test_candidate_pool_idf.py -q`
Expected: PASS (no skipped/xfail to dodge the migration).

- [ ] **Step 5: Full unit suite**

Run: `python -m pytest tests/unit -q -m "not slow"`
Expected: PASS (note any pre-existing failures vs master).

- [ ] **Step 6: Commit**

```bash
git add tests/test_candidate_filters.py tests/unit/test_candidate_pool_max_over_seeds.py tests/unit/test_candidate_pool_idf.py
git commit -m "test(admission): migrate absolute-floor tests to adaptive percentile + min-pool"
```

---

### Task 5: Calibration + eval-gate on a diverse seed corpus

**Files:**
- Create: `scripts/research/adaptive_admission_eval.py`
- Modify: `src/playlist/mode_presets.py` (set calibrated per-mode `sonic_admission_percentile` / `genre_admission_percentile` / `min_pool_size`)
- Modify: `tests/unit/test_adaptive_admission.py` (pin the shipped values)
- Output: `docs/run_audits/adaptive_admission/CALIBRATION.md`
- Modify: `docs/WIRING_STATUS.md` (flip the admission rows to ✅ with evidence)

**Interfaces:** consumes Tasks 1-4 + the merged generation path; `generate_like_gui` + the worst-edge sonic metric (reuse `worst_edge_sonic` from `scripts/research/pace_cede_eval.py`).

- [ ] **Step 1: Build the diverse seed corpus** — ≥7 seed sets across niches (hyperpop, metal, jazz, ambient, hip-hop, folk, Top-40), multi-pier, from the MAIN DB. Document the track_ids.

- [ ] **Step 2: Per-mode sweep** — for each mode, ramp `sonic_admission_percentile` / `genre_admission_percentile` from conservative toward looser, measure per seed set: `admitted` pool size, distinct artists, `worst_edge_sonic`, arousal/genre spread, wall-time. Report distributions (min/p10/p50/p90), never means alone.

- [ ] **Step 3: Apply the eval-gate** — a (mode, value) ships only if: pool never starves (admitted ≥ min_pool_size, distinct artists ≥ a floor) AND `worst_edge_sonic` does not drop below the master baseline by more than DELTA (e.g. 0.05) AND wall-time < 90s, across ALL seed niches. Charli XCX is one data point, not the target.

- [ ] **Step 4: Set the shipped values** in `PACE`/`SONIC`/`GENRE` presets; pin them in `test_adaptive_admission.py`; write `CALIBRATION.md` (per-mode chosen values, the distributions, the eval-gate pass/fail per niche).

- [ ] **Step 5: Full regression + all-modes real-playlist smoke**

Run: `python -m pytest -q -m "not slow"`; then `generate_like_gui` across all modes × the diverse corpus (MAIN data, `BPM loaded` verified): all < 90s, pools never starve, genre cohesion not regressed vs master.

- [ ] **Step 6: Commit + flip WIRING_STATUS rows**

```bash
git add scripts/research/adaptive_admission_eval.py src/playlist/mode_presets.py tests/unit/test_adaptive_admission.py docs/WIRING_STATUS.md
git commit -m "feat(admission): calibrate adaptive percentile floors + min_pool, eval-gated on diverse corpus"
```

---

## Notes for the executor
- **Tasks 1-3 ship inert at the default** (`*_admission_percentile=0.0`, `min_pool_size=0` → legacy absolute-floor behavior) until Task 5 sets calibrated values. The presets in Steps 7/3 set initial *conservative* values; Task 5 finalizes them past the eval-gate. Keep golden tests green throughout.
- **This plan does NOT touch the dense plumbing** (`X_genre_dense` param, the dense block, the dim64 sidecar). That is Plan 2 (`sdd/trace-dense-steering.md` is its blueprint). The dense block is already inert; leaving it temporarily is safe.
- **Read the logs, not just metrics** (CLAUDE.md / playlist-testing): verify `BPM loaded`, `variant 'mert'`, the `Sonic/Genre admission percentile active` lines, and `Candidate pool: admitted=N` in every calibration run.
- Genre percentile values in the live config today (narrow 0.90 = top 10%) are TIGHT and part of the starvation — Task 5 must re-derive them, not assume the current values are right.
