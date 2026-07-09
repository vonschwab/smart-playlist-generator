# Bridge-side Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make on-tag-but-sonically-peripheral tracks reachable as bridges, by fixing the stage-D segment-pool admission (`min`→`max`), force-including on-tag tracks past the floor, and activating the (already-built) beam term + worst-edge band.

**Architecture:** Three coordinated changes in the segment-pool + beam layer, ALL gated on tag steering being active so non-steered playlists stay byte-identical. Changes 1-2 live in `segment_pool_builder.py` (pure-ish, unit-testable); Change 3 is config activation of existing machinery.

**Tech Stack:** Python 3.11, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-bridge-side-phase-a.md`. **Gate map / background:** `docs/superpowers/specs/2026-07-08-tag-steering-architecture.md` (this fixes stage D).

## Global Constraints

- **Every change gated on tag steering active** (`on_tag_guarantee_ids` present). Non-steered → byte-identical.
- **On-tag = authority membership** (`_on_tag_guarantee_ids`, already computed `pipeline/core.py` ~636-641). No new genre read.
- **Live default with rollback** (#22): relaxed admission + guarantee default ON when steering; config knobs to disable.
- **Keep genre a weighted term, never a gate** (research; project Layer-4). The guarantee force-includes past a *sonic* floor; genre still only ranks.
- **No silent no-op:** if a knob is set but the data path is absent, warn — never silently skip.
- **Tests through the real `PlaylistGenerator`** for generation (per the pier-fix precedent; `generate_like_gui` only reaches seeds mode).
- **Sub-agent models:** Tasks 1-2 sonnet (judgment in the segment-pool edits), Task 3 sonnet (integration wiring), Task 4 sonnet (calibration/validation). Never inherit the session model.
- **Shared checkout:** commit explicit paths only; never `git add -A/-u/.`; verify `git diff --cached --name-only`.

## File Structure

- `src/playlist/segment_pool_builder.py` — `SegmentPoolConfig` new fields; `_compute_bridge_scores` (min→max + on-tag force-include); `_select_final_candidates` (guarantee priority-insert).
- `src/playlist/pier_bridge_builder.py` — thread the new params into `build_pier_bridge_playlist` + resolve ids→indices + set them on each `SegmentPoolConfig`.
- `src/playlist/pipeline/core.py` — read knobs, pass to the builder reusing existing `_on_tag_guarantee_ids`/`_guar_*`.
- `config.example.yaml` — new knobs + activate the beam term/band.
- `tests/unit/test_segment_pool_guarantee.py` — new unit tests (Tasks 1-2).
- `tests/integration/test_gui_fidelity_regressions.py` — integration cases (Task 4).

---

### Task 1: Relaxed bridge admission (`min` → `max`, gated)

**Files:** Modify `src/playlist/segment_pool_builder.py`. Test: `tests/unit/test_segment_pool_guarantee.py`.

**Interfaces:** `SegmentPoolConfig` gains `bridge_admission_relaxed: bool = False`. When True, `_compute_bridge_scores` admits a candidate if `max(sim_a, sim_b) >= bridge_floor` (else the legacy `min`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_segment_pool_guarantee.py
import numpy as np
from src.playlist.segment_pool_builder import SegmentPoolConfig, SegmentCandidatePoolBuilder


def _cfg(**kw):
    # 4 tracks: 0=pier_a, 1=pier_b, 2=near-a-far-from-b, 3=near both
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.98, 0.05], [0.7, 0.7]], dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    class _B:  # minimal bundle stand-in
        track_ids = np.array(["t0", "t1", "t2", "t3"])
    base = dict(
        pier_a=0, pier_b=1, X_full_norm=X, universe_indices=[2, 3],
        used_track_ids=set(), bundle=_B(), bridge_floor=0.30, segment_pool_max=10,
    )
    base.update(kw)
    return SegmentPoolConfig(**base)


def test_relaxed_admits_one_pier_candidate():
    b = SegmentCandidatePoolBuilder()
    # track 2 is near pier_a (~0.98) but far from pier_b (~0.05): min<0.30 fails, max>=0.30 passes
    strict = b._compute_bridge_scores(_cfg(bridge_admission_relaxed=False), [2, 3], {})
    assert 2 not in strict.passing_candidates and 3 in strict.passing_candidates
    relaxed = b._compute_bridge_scores(_cfg(bridge_admission_relaxed=True), [2, 3], {})
    assert 2 in relaxed.passing_candidates and 3 in relaxed.passing_candidates
```

(If `SegmentCandidatePoolBuilder()` needs constructor args or `_compute_bridge_scores` isn't directly callable, read the class and adapt — keep the test to the admission behavior. `_BridgeScoreResult.passing_candidates` is the field to assert on.)

- [ ] **Step 2: Run to verify it fails**
Run: `python -m pytest tests/unit/test_segment_pool_guarantee.py -q`
Expected: FAIL (`bridge_admission_relaxed` unknown kwarg, or track 2 excluded under relaxed).

- [ ] **Step 3: Implement**
Add to `SegmentPoolConfig` (near `bridge_floor`):
```python
    bridge_admission_relaxed: bool = False
    """When True, admit a candidate if max(sim_a, sim_b) >= bridge_floor (a stepping-stone
    near EITHER pier) instead of min(...) >= floor. Gated on tag steering; the beam's
    sequential worst-edge + destination-pull enforce actual bridge quality downstream."""
```
In `_compute_bridge_scores`, replace the gate (~541):
```python
            gate_sim = max(sim_a, sim_b) if config.bridge_admission_relaxed else min(sim_a, sim_b)
            if gate_sim < float(config.bridge_floor):
                below_bridge_floor += 1
                continue
```

- [ ] **Step 4: Run to verify it passes**
Run: `python -m pytest tests/unit/test_segment_pool_guarantee.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/playlist/segment_pool_builder.py tests/unit/test_segment_pool_guarantee.py
git commit --only -- src/playlist/segment_pool_builder.py tests/unit/test_segment_pool_guarantee.py -m "feat(segment-pool): relaxed bridge admission (min->max) gated flag"
```

---

### Task 2: On-tag guarantee at stage D (force-include past floor + priority-insert)

**Files:** Modify `src/playlist/segment_pool_builder.py`. Test: append to `tests/unit/test_segment_pool_guarantee.py`.

**Interfaces:** `SegmentPoolConfig` gains `on_tag_guarantee_indices: Optional[Set[int]] = None`, `on_tag_guarantee_max: int = 0`, `on_tag_guarantee_per_artist: int = 0`. On-tag candidates bypass the admission floor and are priority-inserted ahead of the `segment_pool_max` truncation.

- [ ] **Step 1: Write the failing test**
```python
def test_guarantee_forces_track_past_floor_and_into_final():
    b = SegmentCandidatePoolBuilder()
    # track 2 fails BOTH the strict AND relaxed floor if we set it far from both:
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6], [0.05, 0.02]], dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    class _B:
        track_ids = np.array(["t0", "t1", "t2", "t3"])
    cfg = SegmentPoolConfig(
        pier_a=0, pier_b=1, X_full_norm=X, universe_indices=[2, 3], used_track_ids=set(),
        bundle=_B(), bridge_floor=0.30, segment_pool_max=1,   # tiny cap
        on_tag_guarantee_indices={3}, on_tag_guarantee_max=4, on_tag_guarantee_per_artist=4,
    )
    res = b._compute_bridge_scores(cfg, [2, 3], {})
    assert 3 in res.passing_candidates          # forced past the floor despite ~0 sim to both
    # and it must survive final selection even though the cap is 1 and track 2 outranks it:
    # (call the builder's full build() or _select_final_candidates per the class's API)
```
(Adapt the second assertion to the class's actual `build()` / `_select_final_candidates` signature after reading it — the requirement: a guaranteed track survives `segment_pool_max` truncation.)

- [ ] **Step 2: Run to verify it fails** → FAIL (unknown kwargs / track 3 excluded).

- [ ] **Step 3: Implement**
Add the three fields to `SegmentPoolConfig`. In `_compute_bridge_scores`, force-include on-tag candidates past the floor (respecting a per-segment/per-artist cap), before the floor gate:
```python
        _guar = config.on_tag_guarantee_indices or set()
        _guar_used = 0
        _guar_per_artist: Dict[str, int] = {}
        for i in candidates:
            sim_a = float(sim_to_a[i]); sim_b = float(sim_to_b[i])
            _is_guar = False
            if i in _guar and _guar_used < int(config.on_tag_guarantee_max):
                _ak = identity_keys_for_index(config.bundle, int(i)).artist_key
                if _guar_per_artist.get(_ak, 0) < int(config.on_tag_guarantee_per_artist):
                    _is_guar = True
            if not _is_guar:
                gate_sim = max(sim_a, sim_b) if config.bridge_admission_relaxed else min(sim_a, sim_b)
                if gate_sim < float(config.bridge_floor):
                    below_bridge_floor += 1
                    continue
            else:
                _guar_used += 1
                _guar_per_artist[_ak] = _guar_per_artist.get(_ak, 0) + 1
            score = self._compute_bridge_score(sim_a, sim_b, config)
            ... (genre blend unchanged) ...
            bridge_sim[i] = score
            passing.append(i)
```
Track the forced set so `_select_final_candidates` can prioritize it. Simplest: return the guarantee-selected indices on `_BridgeScoreResult` (add a `guaranteed: List[int]` field) OR recompute `passing ∩ on_tag_guarantee_indices` there. In `_select_final_candidates`, priority-insert the guaranteed indices FIRST (mirroring `internal_connectors`), before the `passing_sorted` external fill, so the `segment_pool_max` cut can't drop them:
```python
        _guar_final = [i for i in passing_sorted if config.on_tag_guarantee_indices
                       and i in config.on_tag_guarantee_indices][: int(config.on_tag_guarantee_max)]
        # prepend _guar_final ahead of the external fill (dedupe preserved by dict.fromkeys)
```
(Place it so the final `combined` = internal_connectors + guaranteed + external, deduped; do not let the `segment_pool_max` break in the external-fill loop drop the guaranteed set.)

- [ ] **Step 4: Run to verify it passes** → PASS (all Task 1+2 tests).

- [ ] **Step 5: Commit**
```bash
git add src/playlist/segment_pool_builder.py tests/unit/test_segment_pool_guarantee.py
git commit --only -- src/playlist/segment_pool_builder.py tests/unit/test_segment_pool_guarantee.py -m "feat(segment-pool): on-tag guarantee — force-include past floor + priority-insert"
```

---

### Task 3: Wire through `build_pier_bridge_playlist` + `pipeline/core.py` + config

**Files:** `src/playlist/pier_bridge_builder.py`, `src/playlist/pipeline/core.py`, `config.example.yaml`. Test: covered by Task 4.

**Interfaces:** Consumes Tasks 1-2. `build_pier_bridge_playlist` gains `tag_steering_relax_bridge_admission: bool = False`, `on_tag_guarantee_ids: Optional[set[str]] = None`, `on_tag_segment_guarantee_max: int = 0`, `on_tag_segment_guarantee_per_artist: int = 0`.

- [ ] **Step 1: Config knobs** (`config.example.yaml`, under `pier_bridge:`, near the other `tag_steering_*`):
```yaml
      tag_steering_relax_bridge_admission: true   # stage-D: admit stepping-stones near EITHER pier (min->max) when steering; false = legacy min()
      tag_steering_segment_guarantee_max: 8       # per-segment on-tag tracks force-admitted past the bridge floor (0 = off)
      tag_steering_segment_guarantee_per_artist: 2
```

- [ ] **Step 2: `build_pier_bridge_playlist`** — add the 4 params (near `sonic_tag_beam_weight`/`tag_steering_worst_edge_band`). Resolve `on_tag_guarantee_ids` → a `Set[int]` once (mirror `allowed_set_indices`, `pier_bridge_builder.py` ~806-814). When building each `SegmentPoolConfig` (the call ~1254-1267), set:
  - `bridge_admission_relaxed = bool(tag_steering_relax_bridge_admission and on_tag_guarantee_ids)` (steering-gated — only relax when steering resolved on-tag ids),
  - `on_tag_guarantee_indices = <resolved set or None>`,
  - `on_tag_guarantee_max = on_tag_segment_guarantee_max`,
  - `on_tag_guarantee_per_artist = on_tag_segment_guarantee_per_artist`.

- [ ] **Step 3: `pipeline/core.py`** — read the knobs from `pb_overrides` (near `_beam_edge_band`, ~604), pass to `build_pier_bridge_playlist` (the call ~992-1016) reusing the existing `_on_tag_guarantee_ids` (already computed ~636-641):
```python
    _relax_bridge = bool(pb_overrides.get("tag_steering_relax_bridge_admission", True))
    _seg_guar_max = int(pb_overrides.get("tag_steering_segment_guarantee_max", 8))
    _seg_guar_pa = int(pb_overrides.get("tag_steering_segment_guarantee_per_artist", 2))
    # ... in the build_pier_bridge_playlist(...) call:
    #   tag_steering_relax_bridge_admission=_relax_bridge,
    #   on_tag_guarantee_ids=_on_tag_guarantee_ids,
    #   on_tag_segment_guarantee_max=_seg_guar_max,
    #   on_tag_segment_guarantee_per_artist=_seg_guar_pa,
```

- [ ] **Step 4: Verify off-path byte-identical + imports**
Run: `python -c "import src.playlist.pier_bridge_builder, src.playlist.pipeline.core, src.playlist.segment_pool_builder"` → clean.
Run: `python -m pytest tests/unit/test_segment_pool_guarantee.py tests/test_gui_fidelity.py tests/unit/test_beam_contract.py -q` → PASS.
`ruff check` the three files (fix only new E/F). With no tag (no `_on_tag_guarantee_ids`), `bridge_admission_relaxed=False` and `on_tag_guarantee_indices=None` → segment pool byte-identical.

- [ ] **Step 5: Commit**
```bash
git add src/playlist/pier_bridge_builder.py src/playlist/pipeline/core.py config.example.yaml
git commit --only -- src/playlist/pier_bridge_builder.py src/playlist/pipeline/core.py config.example.yaml -m "feat(bridge-side): wire relaxed admission + on-tag segment guarantee through to the segment pool"
```

---

### Task 4: Activate/calibrate the beam term + band, and validate

**Files:** `config.example.yaml`, `tests/integration/test_gui_fidelity_regressions.py`.

- [ ] **Step 1: Integration cases** (real `PlaylistGenerator`, `@pytest.mark.integration @pytest.mark.slow`, skip if artifact absent — mirror the pier-fix Task-6 helpers that re-read authority membership for realized track_ids):
  - BoC + ["hauntology"], off: assert the realized **playlist** contains ≥1 non-BoC authority-hauntology bridge. If 0 → `pytest.xfail`("Phase A insufficient for the extreme case — Phase B (anchors) needed"), recorded, not a silent pass.
  - Bowie + ["krautrock"], off: authority-krautrock bridge count ≥ the pre-change count (compare with `tag_steering_relax_bridge_admission: false`), worst-edge min-T ≥ pre-change − 0.1.
  - Non-steered artist (no tag): realized pool + playlist byte-identical to `tag_steering_relax_bridge_admission`-off (Change-1 rollback guard).
  - Eno + ["neoclassical"] or Real Estate + ["jangle pop"]: distinct-artist + worst-edge within one notch (no regression on the already-good cases).

- [ ] **Step 2: Calibrate the beam term + band (manual, record numbers).** Using `scratchpad/verify_hauntology.py` (BEAMW/BAND envs already wired) + a Bowie/krautrock variant, sweep `tag_steering_sonic_beam_weight` ∈ {0.5, 1.0, 2.0} × `tag_steering_worst_edge_band` ∈ {0.05, 0.10, 0.15}. Pick the pair that maximizes on-tag bridge count while keeping worst-edge min-T within ~0.1 of the relaxed-admission-only baseline. Then set those as the `config.example.yaml` defaults:
```yaml
      tag_steering_sonic_beam_weight: <calibrated>   # was 0.0; activated now that on-tag tracks reach the beam
      tag_steering_worst_edge_band: <calibrated>     # was 0.0
```

- [ ] **Step 3: Run integration + manual verify (record real numbers, do NOT skip).**
Run: `python -m pytest tests/integration/test_gui_fidelity_regressions.py -q -k "krautrock or hauntology or relax"` (bounded; NO head/tail pipe).
Regenerate BoC/hauntology + Bowie/krautrock through the worker path; count on-genre bridges + worst-edge; quote vs the pre-Phase-A runs. **Report the BoC result explicitly** — it decides whether Phase B is built.

- [ ] **Step 4: Commit**
```bash
git add config.example.yaml tests/integration/test_gui_fidelity_regressions.py
git commit --only -- config.example.yaml tests/integration/test_gui_fidelity_regressions.py -m "feat(bridge-side): activate+calibrate beam term/band; Phase A integration validation"
```

---

## Self-Review (completed)

- **Spec coverage:** Change 1 (T1), Change 2 (T2), wiring+config (T3), activate/calibrate+validate (T4). Covered.
- **Placeholders:** none — real code for the segment-pool edits; the calibration values are a documented sweep (T4), not a TBD.
- **Type consistency:** `SegmentPoolConfig.bridge_admission_relaxed: bool` + `on_tag_guarantee_indices: Set[int]` (T1/T2) ← resolved from `on_tag_guarantee_ids: set[str]` in the builder (T3) ← `_on_tag_guarantee_ids` in core.py (existing). Gate = `relax AND on_tag_guarantee_ids` (steering-active) consistent across T2/T3.
- **Gating:** every effect requires `on_tag_guarantee_ids` present (steering active) → non-steered byte-identical (asserted in T4).
