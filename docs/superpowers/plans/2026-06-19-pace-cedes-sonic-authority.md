# Pace Cedes Sonic Authority — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `pace_mode` slider actually steer the energy arc in real gated playlists by letting pace cede MERT *sonic* authority (genre preserved), behind per-mode knobs that default to a no-op and ship only past a worst-edge eval-gate.

**Architecture:** Per-`pace_mode` "sonic cede" multipliers (default `1.0` = current behavior) scale two reachable, never-fail-safe sonic levers — the candidate-pool admission floor (`candidate_cfg.min_sonic_similarity`) and the beam sonic-bridge weight (`pb_cfg.weight_bridge`) — so the on-arc candidates already in the pool (ceiling probe: ~80/step) stop being out-gated/out-scored on sonic grounds. The already-shipped energy `arc_band`/`step_cap` soft terms then shape the pick. The hard sonic safety floor (`bridge_floor`, per-edge minimum) is never touched. Genre authority is untouched at every level.

**Tech Stack:** Python 3.11, pytest; existing pier-bridge beam (`src/playlist/pier_bridge/`), pipeline (`src/playlist/pipeline/core.py`), mode presets (`src/playlist/mode_presets.py`); React/TS web GUI (`web/src`); `generate_like_gui` fidelity harness (`tests/support/gui_fidelity.py`).

## Global Constraints

- **Never hard-fail on pace.** The cede only ever *loosens* sonic authority (lower admission floor / lower sonic weight) and energy stays a soft penalty. No new hard gate; cannot strand a segment or trigger the relaxation cascade. (`feedback_never_fail_three_axes`)
- **Genre authority is preserved** at every level (admission floor, weighting, genre-arc floor). Pace spends *sonic* cohesion only — never genre.
- **Preserve the hard sonic safety floor** in every mode incl. strict: the cede never touches `bridge_floor` or the per-edge sonic minimum. Pace can demote sonic; it can never admit a sonically-disconnected edge.
- **Default is a no-op.** All cede factors default `1.0` and all energy strengths default `0.0` → byte-identical to current behavior until calibrated values ship (golden-safe).
- **90s generation budget.** (`feedback_generation_time_budget`)
- **Worst-edge kill criterion.** A mode ships energy-on only if its weakest sonic edge stays above threshold in the eval-gate; otherwise it ships at cede `1.0`/energy `0.0` (disabled). "Do more harm than good" is a measured fail state.
- **Data access in this worktree:** `data/` artifacts are not symlinked here; run generations with the artifact + sidecar paths pointing at the main checkout (`C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/...`), as the research probes do.

---

## File Structure

- `scripts/research/pace_cede_eval.py` (Create) — measurement harness: per-mode arousal curve (arc-deviation + max adjacent step), worst-edge sonic T, never-fail/budget check; ablation mode that isolates each sonic lever. Used by Task 1 (ablation) and Task 5 (calibration).
- `src/playlist/mode_presets.py` (Modify) — add `sonic_cede_admission` + `sonic_cede_bridge` (default `1.0`) to all four `PACE_MODE_PRESETS` entries.
- `src/playlist/pipeline/core.py` (Modify) — read the two cede factors from `pace_settings`; scale `cfg.candidate.min_sonic_similarity` (preserving `None`) before pool build, and `pb_cfg.weight_bridge` before the beam; never touch `bridge_floor`.
- `web/src/components/GenerateControls.tsx` (Modify) — add `"off"` to the pace dropdown + widen the TS union.
- `tests/unit/test_pace_sonic_cede.py` (Create) — preset defaults, the cede helper math, floor-preservation.
- `docs/run_audits/pace_cedes_sonic/` (Create, gitignored) — ablation + calibration reports.

---

### Task 1: Measurement harness + sonic-lever ablation

**Files:**
- Create: `scripts/research/pace_cede_eval.py`
- Output: `docs/run_audits/pace_cedes_sonic/ABLATION.md` (gitignored dir; the decision is also echoed into the progress ledger)

**Interfaces:**
- Consumes: `tests/support/gui_fidelity.generate_like_gui`, `src/playlist/energy_loader.load_energy_matrix`.
- Produces: `compute_pace_metrics(track_ids, *, sidecar_path) -> dict` with keys `arc_dev: float`, `max_step: float`, `arousal_curve: list[float]`; reused by Task 5. Ablation decision: which lever(s) of {`min_sonic_similarity`, `weight_bridge`} unblock energy with genre held on, and starting per-mode magnitudes.

This is a measurement task (no production code). It de-risks Task 3/5 by confirming which lever to calibrate non-`1.0`.

- [ ] **Step 1: Write `compute_pace_metrics` + a tiny self-test**

```python
# scripts/research/pace_cede_eval.py
import numpy as np
from src.playlist.energy_loader import load_energy_matrix

def compute_pace_metrics(track_ids, *, sidecar_path, bundle_track_ids=None):
    """Realized arousal curve + arc-deviation (RMS vs first->last line) + max adjacent step."""
    e = load_energy_matrix(track_ids, sidecar_path=sidecar_path, features=("arousal_p50",)).reshape(-1)
    arr = e[np.isfinite(e)]
    if len(arr) < 2:
        return {"arc_dev": 0.0, "max_step": 0.0, "arousal_curve": [round(float(x), 2) for x in e]}
    line = np.linspace(arr[0], arr[-1], len(arr))
    return {
        "arc_dev": float(np.sqrt(np.mean((arr - line) ** 2))),
        "max_step": float(np.max(np.abs(np.diff(arr)))),
        "arousal_curve": [round(float(x), 2) for x in e],
    }
```

- [ ] **Step 2: Add an ablation runner that isolates each lever**

Mutate the merged config *after* `apply_mode_presets` (presets overwrite naive overrides, so config-only isolation is impossible — mutate post-preset). Replicate `generate_like_gui`'s resolution (`resolve_gui_overrides` → but inject a per-arm mutation of the resolved `candidate_pool.min_sonic_similarity` or `pier_bridge.weight_bridge`). Arms, all with genre=dynamic, pace=dynamic, energy strong (`arc_strength=10, arc_band=0.1, step_strength=10, step_cap=0.1`):
- `BASELINE`: nothing relaxed (expect inert — energy-on == energy-off).
- `ADMISSION`: force `min_sonic_similarity=None`.
- `BRIDGE`: force `weight_bridge=0.1`.

For each arm, generate energy-off and energy-on; report `compute_pace_metrics` for both + position diff. A lever "unblocks" if energy-on differs from energy-off AND `arc_dev` drops.

- [ ] **Step 3: Run the ablation against the main-checkout data**

Run: `python scripts/research/pace_cede_eval.py --ablation` (artifact/sidecar paths → main checkout).
Expected: BASELINE inert; at least one of ADMISSION/BRIDGE shows energy-on diverging + lower `arc_dev`. Record which lever(s) and the relaxation magnitude that worked.

- [ ] **Step 4: Write the decision doc**

Write `docs/run_audits/pace_cedes_sonic/ABLATION.md`: per-arm metrics table, the winning lever(s), and recommended starting per-mode cede magnitudes (strict cedes most → off none). If *neither* admission nor bridge unblocks energy (both inert), STOP and escalate — it means the hybrid `sonic_weight` is the lever and Task 3 must add a third factor (documented extension).

- [ ] **Step 5: Commit**

```bash
git add scripts/research/pace_cede_eval.py
git commit -m "research(pace): sonic-lever ablation harness + pace metrics"
```

---

### Task 2: Per-pace-mode sonic-cede config fields (default no-op)

**Files:**
- Modify: `src/playlist/mode_presets.py` (the four `PACE_MODE_PRESETS` entries, ~lines 131-206)
- Test: `tests/unit/test_pace_sonic_cede.py` (Create)

**Interfaces:**
- Produces: `PACE_MODE_PRESETS[mode]["sonic_cede_admission"]: float` and `["sonic_cede_bridge"]: float`, both `1.0` in every mode (calibrated later). `resolve_pace_mode(mode)` already passes preset keys through unchanged (`mode_presets.py:344`), so these surface in `pace_settings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pace_sonic_cede.py
from src.playlist.mode_presets import PACE_MODE_PRESETS, resolve_pace_mode

CEDE_KEYS = ["sonic_cede_admission", "sonic_cede_bridge"]

def test_presets_have_cede_keys_default_one():
    for mode in ("strict", "narrow", "dynamic", "off"):
        for k in CEDE_KEYS:
            assert k in PACE_MODE_PRESETS[mode], f"{mode} missing {k}"
            assert PACE_MODE_PRESETS[mode][k] == 1.0, f"{mode}.{k} must default 1.0 (no-op until calibrated)"

def test_resolve_pace_mode_surfaces_cede_keys():
    s = resolve_pace_mode("strict")
    assert s["sonic_cede_admission"] == 1.0 and s["sonic_cede_bridge"] == 1.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/unit/test_pace_sonic_cede.py -q`
Expected: FAIL (KeyError / missing keys).

- [ ] **Step 3: Add the keys to all four presets**

In each of `strict`, `narrow`, `dynamic`, `off` dicts in `PACE_MODE_PRESETS`, add:

```python
        "sonic_cede_admission": 1.0,
        "sonic_cede_bridge": 1.0,
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python -m pytest tests/unit/test_pace_sonic_cede.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the existing preset + golden tests (no regression)**

Run: `python -m pytest tests/unit/test_pace_mode_energy_presets.py -q && python -m pytest tests/unit -k golden -q`
Expected: PASS. The cede keys live only in `PACE_MODE_PRESETS` (surfaced via `pace_settings`), not in `PierBridgeConfig`, so pb_cfg golden snapshots are unaffected. If any golden unexpectedly fails, stop and investigate before regenerating — do not blindly re-snapshot.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/mode_presets.py tests/unit/test_pace_sonic_cede.py
git commit -m "feat(pace): per-mode sonic-cede config fields (default 1.0 no-op)"
```

---

### Task 3: Apply the sonic cede in the pipeline (preserve the hard floor)

**Files:**
- Modify: `src/playlist/pipeline/core.py` — two focused helpers, applied at their natural (different) sites: admission cede on `cfg.candidate` *before* `_build_pool` (~line 460); bridge cede inside the `pb_cfg = replace(pb_cfg, ...)` block (~line 531). The two levers are needed at different points, so they are two helpers, not one combined call.
- Test: `tests/unit/test_pace_sonic_cede.py` (extend)

**Interfaces:**
- Consumes: `pace_settings["sonic_cede_admission"]`, `pace_settings["sonic_cede_bridge"]` (Task 2); `candidate_cfg.min_sonic_similarity` (Optional[float]); `pb_cfg.weight_bridge` (float).
- Produces: `cede_admission_floor(candidate_cfg, factor: float) -> candidate_cfg` (scales `min_sonic_similarity`, `None` stays `None`, `1.0` is no-op); `cede_bridge_weight(pb_cfg, factor: float) -> pb_cfg` (scales `weight_bridge`, `1.0` no-op). Neither touches `bridge_floor`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_pace_sonic_cede.py
from src.playlist.pipeline.core import cede_admission_floor, cede_bridge_weight
from src.playlist.pier_bridge.config import PierBridgeConfig

class _Cand:  # minimal stand-in carrying just the field the helper touches
    def __init__(self, min_sonic_similarity): self.min_sonic_similarity = min_sonic_similarity

def test_cede_admission_scales_and_preserves_none():
    assert abs(cede_admission_floor(_Cand(0.20), 0.5).min_sonic_similarity - 0.10) < 1e-9
    assert cede_admission_floor(_Cand(0.20), 1.0).min_sonic_similarity == 0.20  # no-op
    assert cede_admission_floor(_Cand(None), 0.5).min_sonic_similarity is None  # None stays None

def test_cede_bridge_scales_weight_preserves_floor():
    pb = PierBridgeConfig(weight_bridge=0.6, bridge_floor=0.05)
    pb2 = cede_bridge_weight(pb, 0.5)
    assert abs(pb2.weight_bridge - 0.30) < 1e-9   # sonic weight halved
    assert pb2.bridge_floor == 0.05               # SAFETY FLOOR untouched
    assert cede_bridge_weight(pb, 1.0).weight_bridge == 0.6  # no-op
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/unit/test_pace_sonic_cede.py -k cede -q`
Expected: FAIL (`cede_admission_floor` / `cede_bridge_weight` not defined).

- [ ] **Step 3: Implement the two helpers**

```python
# src/playlist/pipeline/core.py (module-level, near _relaxed_one_each_candidate_attempts)
from dataclasses import replace

def cede_admission_floor(candidate_cfg, factor: float):
    """Lower the sonic admission floor by `factor` (pace cedes sonic authority).
    None stays None; 1.0 is a no-op. NEVER touches bridge_floor (hard safety floor)."""
    if candidate_cfg.min_sonic_similarity is None or factor == 1.0:
        return candidate_cfg
    return replace(candidate_cfg, min_sonic_similarity=float(candidate_cfg.min_sonic_similarity) * float(factor))

def cede_bridge_weight(pb_cfg, factor: float):
    """Lower the beam sonic-bridge weight by `factor`. 1.0 is a no-op.
    NEVER touches bridge_floor / per-edge minimum (hard safety floor)."""
    if factor == 1.0:
        return pb_cfg
    return replace(pb_cfg, weight_bridge=float(pb_cfg.weight_bridge) * float(factor))
```

- [ ] **Step 4: Run it, verify it passes**

Run: `python -m pytest tests/unit/test_pace_sonic_cede.py -k cede -q`
Expected: PASS.

- [ ] **Step 5: Wire the admission cede before pool build**

In `core.py`, where `pace_settings` is resolved (it is in scope before the pool is built), read both factors once, and apply the admission cede to the candidate cfg passed into `_build_pool`. The pool is built via `pool = _build_pool(cfg.candidate, min_genre_similarity)` (~line 460) where `_build_pool(candidate_cfg, genre_gate)` forwards `candidate_cfg` to `build_candidate_pool`. Pass a ceded local instead of `cfg.candidate`:

```python
        _cede_admission = float(pace_settings.get("sonic_cede_admission", 1.0))
        _cede_bridge = float(pace_settings.get("sonic_cede_bridge", 1.0))
        _ceded_candidate = cede_admission_floor(cfg.candidate, _cede_admission)
        pool = _build_pool(_ceded_candidate, min_genre_similarity)   # was: _build_pool(cfg.candidate, ...)
```

(Do not mutate `cfg` itself — only the candidate cfg handed to `_build_pool`, so nothing else downstream sees a changed `cfg.candidate`.)

- [ ] **Step 6: Apply the bridge cede in the `pb_cfg` block**

Immediately after the existing `pb_cfg = replace(pb_cfg, ...)` energy block (~line 531), add:

```python
            pb_cfg = cede_bridge_weight(pb_cfg, _cede_bridge)
```

- [ ] **Step 7: Default no-op sanity + a generation smoke**

Run: `python -m pytest tests/unit/test_pace_sonic_cede.py tests/unit/test_pace_mode_energy_presets.py -q`
Expected: PASS. Then a generation smoke (main-checkout data) with default presets confirming output is unchanged vs master (cede=1.0 no-op).

- [ ] **Step 8: Commit**

```bash
git add src/playlist/pipeline/core.py tests/unit/test_pace_sonic_cede.py
git commit -m "feat(pace): apply per-mode sonic cede (admission + bridge), floor preserved"
```

---

### Task 4: Add `off` to the GUI pace dropdown

**Files:**
- Modify: `web/src/components/GenerateControls.tsx` (dropdown ~line 343; TS cast ~line 151)

**Interfaces:**
- Consumes: existing `axes.pace_mode` state + `VALID_PACE_MODES` (already includes `"off"` server-side in `policy.py`).
- Produces: pace dropdown options `["off", "dynamic", "narrow", "strict"]`; the `pace_mode` union type includes `"off"`.

- [ ] **Step 1: Widen the dropdown options**

Change (line ~343):
```tsx
            {["dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
```
to:
```tsx
            {["off", "dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
```

- [ ] **Step 2: Widen the TS union (line ~151)**

Change `pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic",` to include `"off"`:
```tsx
      pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic" | "off",
```

- [ ] **Step 3: Type-check + build**

Run: `npm --prefix web run build`
Expected: build succeeds (no TS error). Confirms `off` flows through the existing request types (server already accepts it).

- [ ] **Step 4: Commit**

```bash
git add web/src/components/GenerateControls.tsx
git commit -m "feat(web): add 'off' to the pace dropdown"
```

---

### Task 5: Calibration + eval-gate (worst-edge kill criterion)

**Files:**
- Modify: `scripts/research/pace_cede_eval.py` (add worst-edge sonic T metric + per-mode calibration runner)
- Modify: `src/playlist/mode_presets.py` (set the calibrated non-`1.0` cede + non-`0.0` energy values per mode — only for modes that PASS the gate)
- Output: `docs/run_audits/pace_cedes_sonic/CALIBRATION.md`

**Interfaces:**
- Consumes: `compute_pace_metrics` (Task 1); the cede mechanism (Tasks 2-3); `generate_like_gui`.
- Produces: final per-mode `sonic_cede_admission` / `sonic_cede_bridge` + energy `arc_band`/`arc_strength`/`step_cap`/`step_strength` values; an eval report with per-mode pass/fail.

- [ ] **Step 1: Add the worst-edge sonic metric**

```python
# scripts/research/pace_cede_eval.py
import numpy as np
def worst_edge_sonic(track_ids, bundle):
    """Min adjacent MERT cosine over the playlist (the weakest sonic transition)."""
    idx = [bundle.track_id_to_index[t] for t in track_ids if t in bundle.track_id_to_index]
    X = bundle.X_full_norm if hasattr(bundle, "X_full_norm") else None
    sims = [float(np.dot(X[idx[i]], X[idx[i+1]])) for i in range(len(idx)-1)]
    return min(sims) if sims else float("nan")
```
Confirm the bundle's sonic matrix + index map attribute names against `tests/support/gui_fidelity.py` / `ArtifactBundle`; adjust if different.

- [ ] **Step 2: Calibration runner (ramp UP from minimal)**

For each mode in (strict, narrow, dynamic): starting from the ablation's lever + smallest magnitudes, generate (multi-pier, diverse seeds incl. high-arousal/wide-swing), compute `arc_dev`, `max_step`, `worst_edge_sonic` energy-on vs -off. Increase the cede only while `worst_edge_sonic(on) >= worst_edge_sonic(off) - DELTA` (DELTA preset, e.g. 0.05) AND total time < 90s. `off`: arc disabled, keep a small step-cap; verify no whiplash and `worst_edge_sonic` ~unchanged.

- [ ] **Step 3: Run calibration, write the report**

Run: `python scripts/research/pace_cede_eval.py --calibrate` (main-checkout data, ≥3 seed sets).
Write `docs/run_audits/pace_cedes_sonic/CALIBRATION.md`: per-mode chosen values, arc improvement, worst-edge cost, timing, and PASS/FAIL per the kill criterion.

- [ ] **Step 4: Set the passing values in presets**

For each mode that PASSES, set its `sonic_cede_*` (<1.0) and energy strengths (>0.0) in `PACE_MODE_PRESETS`. Modes that FAIL stay at cede `1.0` / energy `0.0` (disabled). Update `tests/unit/test_pace_sonic_cede.py`: assert the shipped values match the calibration report (so the values are pinned by a test, not drift).

- [ ] **Step 5: Full regression + budget verification**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS (note any pre-existing failures vs master). Then generation smoke across all 4 pace_modes (multi-pier, main-checkout data): all complete <90s (never-hard-fail), and genre distinct-artist/genre-cohesion metric unchanged across pace_modes (proves genre preserved).

- [ ] **Step 6: Commit**

```bash
git add scripts/research/pace_cede_eval.py src/playlist/mode_presets.py tests/unit/test_pace_sonic_cede.py
git commit -m "feat(pace): calibrate per-mode sonic cede + energy, eval-gated (worst-edge)"
```

---

## Notes for the executor
- **Task 1 informs Tasks 3/5 but does not block coding them:** Task 3 builds *both* cede levers (admission + bridge) at default `1.0`, so it is correct regardless of the ablation outcome; Task 1/5 only decide which factor becomes non-`1.0` and by how much. If Task 1 escalates (neither lever unblocks → it's the hybrid `sonic_weight`), pause and extend Task 3 with a third factor before Task 5.
- **Genre-preservation check is a required gate in Task 5** — if any genre cohesion metric moves across pace_modes, the cede has leaked into genre and must be fixed.
- **Everything ships at no-op default** until Task 5 sets passing values; a failing mode ships disabled. This is the safety contract.
