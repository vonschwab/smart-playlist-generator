# Sonic Centered-Transition Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the transition score's `(x+1)/2` rescale with a calibrated sigmoid so bad sonic edges score meaningfully worse than good ones, recalibrate the floors that consume `T`, and remove the dead/duplicate rescale wiring — shipped activated by default.

**Architecture:** The live beam scorer is `transition_metrics.py::score_transition_edge`. We swap its per-component `(x+1)/2` rescale for a calibrated logistic (`σ(gain·(x−center)/scale)`) whose three params are fixed constants derived once from the library cosine distribution and stored in config. We collapse the three duplicate rescale implementations to one shared function, delete verified-dead wiring, recalibrate the `T`-consuming floors against the new distribution, and regenerate goldens.

**Tech Stack:** Python 3.11+, numpy, pytest. No new dependencies.

## Global Constraints

- Python 3.11+ (pinned in `pyproject.toml`).
- Generation must NEVER exceed 90s (hard ceiling) — floors are tuned against this; prefer soft penalties over hard gates that detonate the relaxation cascade.
- Backward-compat default: ship the sigmoid **activated** (no `mode: legacy` switch). A configured calibration that fails to load is a **startup error**, not a silent fallback.
- A playlist can never *fail* on sonic — sonic is a soft, relaxable axis; floors relax, they don't crash.
- `pytest` is run directly and bounded by the tool timeout — **never piped through `tail`/`head`** (hangs sessions).
- Work in a dedicated git worktree on its own branch (create via `superpowers:using-git-worktrees` at execution start; copy `config.yaml` in manually — it's gitignored). Subagents launch in the MAIN checkout — any subagent step must be inline-in-worktree or cd-guarded + branch-verified.
- Preserve (do NOT touch): `T_centered_cos` + the `-0.5` anti-alignment gate (`beam.py:638`); the `S` field (feeds `_layered_transition_delta`); `build_hybrid_embedding` in `build_transition_metric_context` (reporter `H` display); the `center_transitions` flag + raw branch.
- Merge order: this sonic fix merges FIRST, before the genre soft-cosine session's calibration.

---

## Pre-flight (execution start, before Task 1)

- [ ] Create the worktree (`superpowers:using-git-worktrees`), copy `config.yaml` in, commit the spec (`docs/superpowers/specs/2026-06-25-sonic-centered-transition-design.md`) and this plan as the first commit.
- [ ] **Confirm production centers.** Run one real multi-pier generation (the `gui_fidelity` harness — see `playlist-testing` skill) at INFO and grep the log for `center_transitions=` from `pier_bridge_builder.py:655,666`. Expected: `center_transitions=True`. If it resolves `False`, STOP — the fix target is wrong; reconcile config resolution before proceeding.

---

## File Structure

- `src/playlist/transition_metrics.py` — replace `_rescale_centered_cos` with the shared calibrated `_calibrate_transition_cos`; add calibration params to `TransitionMetricContext`; remove `T_used` + gamma local storage. (The single rescale source of truth lives here.)
- `src/playlist/pier_bridge/config.py` — add `transition_calibration` fields to `PierBridgeConfig`; build them in the config resolver (~line 660); delete the dead `_compute_transition_score` wrapper.
- `src/config_loader.py` — surface the `transition_calibration` block (~line 519 mapping).
- `config.yaml` / `config.example.yaml` — add the `transition_calibration` block; updated floor values.
- `src/playlist/pier_bridge_builder.py` — pass calibration params into the directly-constructed `TransitionMetricContext` (~line 730); repoint the DEBUG-only `_compute_transition_score` use.
- `src/playlist/pier_bridge/vec.py` — route the remaining blend helper's rescale through the shared function; delete the non-raw `_compute_transition_score`.
- `src/playlist/pier_bridge/beam.py` — delete the dead `transition_metric_context is None` fallback (~643-654).
- `src/playlist/scoring/transition_scoring.py` — DELETE (production-dead).
- `tests/unit/test_scoring.py` — repoint onto `score_transition_edge` (or retire the dead-module tests).
- `scripts/research/calibrate_transition_sigmoid.py` — NEW: derive `center/scale/gain` + sweep floors.
- Golden fixtures under `tests/unit/goldens/` — regenerate deliberately.

---

## Task 1: Calibrated sigmoid rescale + config block (the fix)

**Files:**
- Modify: `src/playlist/transition_metrics.py` (`_rescale_centered_cos` ~61-64; `TransitionMetricContext` ~15-30; `score_transition_edge` ~165-170; `build_transition_metric_context` ~130-143)
- Modify: `src/playlist/pier_bridge/config.py` (`PierBridgeConfig` fields ~70; resolver ~660-668)
- Modify: `src/config_loader.py` (~457-519)
- Modify: `config.yaml`, `config.example.yaml` (constraints block ~96)
- Modify: `src/playlist/pier_bridge_builder.py` (~730 direct `TransitionMetricContext` construction)
- Test: `tests/unit/test_transition_calibration.py` (new)

**Interfaces:**
- Produces: `_calibrate_transition_cos(value: float, *, center: float, scale: float, gain: float) -> float` in `transition_metrics.py`.
- Produces: `TransitionMetricContext` gains `calib_center: float`, `calib_scale: float`, `calib_gain: float` (defaults 0.32 / 0.0625 / 1.0 — provisional; Task 3 sets final).
- Consumes (later tasks): `vec.py` and the audit call `_calibrate_transition_cos` with the same params.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_transition_calibration.py
import math
from src.playlist.transition_metrics import _calibrate_transition_cos

P = dict(center=0.32, scale=0.0625, gain=1.0)  # gain/scale = 16

def test_band_is_spread_across_unit_interval():
    lo = _calibrate_transition_cos(0.138, **P)   # p1 of the real band
    hi = _calibrate_transition_cos(0.501, **P)   # p99 of the real band
    assert lo < 0.12 and hi > 0.88               # band maps to ~[0.05, 0.95]
    assert 0.0 < lo < hi < 1.0                    # stays strictly inside (0,1), no clip ties

def test_restores_good_vs_bad_gap():
    bad = _calibrate_transition_cos(0.151, **P)   # the Yuji edge cosine
    good = _calibrate_transition_cos(0.260, **P)  # the Beach House edge cosine
    rel_gap = (good - bad) / bad
    assert rel_gap > 0.40                          # legacy (x+1)/2 gives ~0.08

def test_monotonic_and_finite():
    xs = [-0.2, 0.0, 0.14, 0.27, 0.50, 0.71]
    ys = [_calibrate_transition_cos(x, **P) for x in xs]
    assert all(ys[i] < ys[i+1] for i in range(len(ys)-1))
    assert all(math.isfinite(y) for y in ys)

def test_nan_passthrough():
    assert math.isnan(_calibrate_transition_cos(float("nan"), **P))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_transition_calibration.py -q`
Expected: FAIL — `ImportError: cannot import name '_calibrate_transition_cos'`.

- [ ] **Step 3: Implement the calibrated rescale** (replace `_rescale_centered_cos`, `transition_metrics.py:61-64)

```python
def _calibrate_transition_cos(
    value: float, *, center: float, scale: float, gain: float
) -> float:
    """Calibrated logistic remap of a transition cosine into (0,1).

    Standardize the cosine to its operating band, then squash with a logistic
    (Platt-style). Replaces the legacy `(x+1)/2`, which wasted its output range
    on negative cosines that real edges never produce. Monotonic; no hard clip.
    """
    if not _finite(value):
        return float("nan")
    z = gain * (float(value) - center) / scale
    # numerically stable logistic
    if z >= 0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))
```

- [ ] **Step 4: Add calibration params to the context** (`TransitionMetricContext`, after `transition_gamma` field)

```python
    calib_center: float = 0.32
    calib_scale: float = 0.0625
    calib_gain: float = 1.0
```

- [ ] **Step 5: Use the calibrated rescale in `score_transition_edge`** (replace the centered branch ~165-170)

```python
    if context.center_transitions:
        def _r(x: float) -> float:
            return _calibrate_transition_cos(
                x, center=context.calib_center, scale=context.calib_scale, gain=context.calib_gain
            )
        t_val = (
            float(context.weight_end_start) * _r(sim_end_start_raw)
            + float(context.weight_mid_mid) * _r(sim_mid_raw)
            + float(context.weight_full_full) * _r(sim_full_raw)
        )
    else:
        t_val = t_raw
```

- [ ] **Step 6: Thread params through construction.** In `build_transition_metric_context` add `calib_center/scale/gain` params (default 0.32/0.0625/1.0) and pass into the returned `TransitionMetricContext`. In `pier_bridge_builder.py` (~730) where `TransitionMetricContext(...)` is built directly, pass `calib_center=cfg.transition_calib_center` etc. Add `transition_calib_center/scale/gain` fields to `PierBridgeConfig` (`pier_bridge/config.py:70` area) and build them from config in the resolver (~660): `transition_calib_center=float(constraints.get("transition_calibration", {}).get("center", 0.32))`, etc. Surface the block in `config_loader.py` and add to `config.yaml`/`config.example.yaml` under the constraints block:

```yaml
      transition_calibration:
        center: 0.32   # provisional — Task 3 replaces with calibrated value
        scale: 0.0625
        gain: 1.0
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_transition_calibration.py -q`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add src/playlist/transition_metrics.py src/playlist/pier_bridge/config.py src/config_loader.py src/playlist/pier_bridge_builder.py config.yaml config.example.yaml tests/unit/test_transition_calibration.py
git commit -m "feat(sonic): calibrated sigmoid transition rescale (provisional params)"
```

---

## Task 2: Calibration + floor-sweep script

**Files:**
- Create: `scripts/research/calibrate_transition_sigmoid.py`

**Interfaces:**
- Produces: a CLI that prints recommended `center/scale/gain` and a floor-vs-percentile table. Read-only on artifacts.

- [ ] **Step 1: Write the script** (adapt `scripts/research/remap_eval.py` from the design session + `sonic_centering_probe.py`). It must:
  1. Build the real centered context via `build_transition_metric_context(center_transitions=True, transition_weights=(0.2,0.5,0.3), sonic_variant="mert", ...)`.
  2. Over ~60 random valid destinations, restrict to each destination's top-2000 realistic pool (exclude self + same-artist), collect the centered end→start cosine band; report p1/p50/p99.
  3. Emit `center = (p1+p99)/2`, `gain/scale` such that p1→~0.05 and p99→~0.95 (`k = 2*ln(19)/(p99−p1)`; report as `center`, `scale=1/k`, `gain=1.0`).
  4. Apply the calibrated rescale; recompute the blended `T` distribution over the pools; print p1/p10/p50/p90/p99.
  5. Sweep candidate `transition_floor` / `bridge_floor` values and report, for each, the fraction of realistic edges rejected — so a floor can be chosen at a deliberate operating percentile.

- [ ] **Step 2: Run it**

Run: `python scripts/research/calibrate_transition_sigmoid.py`
Expected: prints the band, recommended `center/scale/gain`, the new `T` distribution, and the floor sweep table. Sanity: recommended params near 0.32 / 0.06 / 1.0; new `T` median ≈ 0.30.

- [ ] **Step 3: Commit**

```bash
git add scripts/research/calibrate_transition_sigmoid.py
git commit -m "research(sonic): transition-sigmoid calibration + floor-sweep script"
```

---

## Task 3: Apply calibrated params + recalibrated floors

**Files:**
- Modify: `config.yaml`, `config.example.yaml` (`transition_calibration` + floor values)
- Reference: existing floor keys — `transition_floor` (`config.py:665` ← `pier_tuning.transition_floor`), `bridge_floor`, per-mode `bridge_floor_<mode>` (enumerate via grep before editing).

- [ ] **Step 1: Set calibrated params.** Copy the Task-2 recommended `center/scale/gain` into the `transition_calibration` block in both yaml files.

- [ ] **Step 2: Set recalibrated floors.** From the Task-2 floor sweep, set `transition_floor` and `bridge_floor`(`_<mode>`) so each gates at its intended operating percentile of the new `T` distribution (start: reject roughly the bottom decile of realistic edges; tighten only with audition evidence). Record the chosen values + rationale as a comment (tuning recipe, principle #23).

- [ ] **Step 3: Verify on real data with the probe.** Re-run `scripts/research/sonic_centering_probe.py` (it reads the live config rescale path). Expected: good-vs-bad gap ≥ ~70% (was 8%); field median `T` clearly off ~0.5; the Yuji edge sits below the recalibrated `transition_floor` or is demoted well below the kin.

- [ ] **Step 4: Commit**

```bash
git add config.yaml config.example.yaml
git commit -m "tune(sonic): calibrated transition params + recalibrated floors"
```

---

## Task 4: Consolidate the rescale to one source of truth (cleanup 1/3)

**Files:**
- Modify: `src/playlist/pier_bridge/vec.py` (`_compute_transition_score_raw_and_transformed` rescale ~118-120; delete `_compute_transition_score` ~34-79)
- Modify: `src/playlist/pier_bridge/config.py` (delete the `_compute_transition_score` wrapper ~373-389)
- Modify: `src/playlist/pier_bridge/beam.py` (delete the dead `transition_metric_context is None` fallback ~643-654)
- Modify: `src/playlist/pier_bridge_builder.py` (DEBUG block ~2530 that calls `_compute_transition_score` — repoint to the `_raw_and_transformed` helper or remove)
- Test: `tests/unit/test_audit_matches_beam.py` (new)

**Interfaces:**
- Consumes: `_calibrate_transition_cos` (Task 1).

- [ ] **Step 1: Write the failing test** — the audit's transformed `T` must equal the beam scorer's `T` for the same edge (the latent-divergence guard).

```python
# tests/unit/test_audit_matches_beam.py
import numpy as np
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge
from src.playlist.pier_bridge.config import _compute_transition_score_raw_and_transformed
from src.playlist.pier_bridge.config import PierBridgeConfig

def test_audit_transformed_equals_beam_T():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 8)).astype(np.float32)
    ctx = build_transition_metric_context(
        X_sonic=X, X_start=X, X_mid=X, X_end=X, center_transitions=True,
        weight_end_start=0.7, weight_mid_mid=0.15, weight_full_full=0.15,
        calib_center=0.32, calib_scale=0.0625, calib_gain=1.0,
    )
    cfg = PierBridgeConfig(center_transitions=True, weight_end_start=0.7,
                           weight_mid_mid=0.15, weight_full_full=0.15,
                           transition_calib_center=0.32, transition_calib_scale=0.0625,
                           transition_calib_gain=1.0)
    _raw, transformed = _compute_transition_score_raw_and_transformed(
        0, 1, ctx.X_full, ctx.X_start, ctx.X_mid, ctx.X_end, cfg
    )
    beam = score_transition_edge(ctx, 0, 1)["T"]
    assert abs(transformed - beam) < 1e-9
```

- [ ] **Step 2: Run it — expect FAIL** (vec.py still uses `(x+1)/2`, params not threaded).

Run: `python -m pytest tests/unit/test_audit_matches_beam.py -q`
Expected: FAIL (mismatch, or missing `transition_calib_*` fields).

- [ ] **Step 3: Route the audit helper's rescale through the shared function.** In `vec.py::_compute_transition_score_raw_and_transformed`, replace the three `(sim_* + 1.0)/2.0` lines with `_calibrate_transition_cos(sim_*, center=..., scale=..., gain=...)`; add `calib_center/scale/gain` keyword params to the function; update the `config.py:392` wrapper to pass `cfg.transition_calib_center` etc. Add the `transition_calib_*` fields to `PierBridgeConfig` if not already (Task 1 added them).

- [ ] **Step 4: Delete the dead non-raw variant + fallback.** Delete `vec.py::_compute_transition_score` (~34-79) and the `config.py` `_compute_transition_score` wrapper (~373-389). Delete the dead `beam.py` fallback branch (~643-654) so the contract is "context is required" (it is always constructed at `pier_bridge_builder.py:730`). Repoint or remove the DEBUG-only call in `pier_bridge_builder.py` (~2530).

- [ ] **Step 5: Run the new test + the pier-bridge suite**

Run: `python -m pytest tests/unit/test_audit_matches_beam.py tests/unit/test_pier_bridge_smoke_golden.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/vec.py src/playlist/pier_bridge/config.py src/playlist/pier_bridge/beam.py src/playlist/pier_bridge_builder.py tests/unit/test_audit_matches_beam.py
git commit -m "refactor(sonic): single calibrated rescale; drop dead transition-score fallbacks"
```

---

## Task 5: Delete the production-dead scoring module (cleanup 2/3)

**Files:**
- Delete: `src/playlist/scoring/transition_scoring.py`
- Modify: `tests/unit/test_scoring.py` (repoint onto `score_transition_edge`, or remove the tests of the deleted functions)

- [ ] **Step 1: Confirm no production importer.**

Run: `python -m pytest -q tests/unit/test_scoring.py` then `git grep -n "scoring.transition_scoring\|from .*transition_scoring import\|compute_transition_score" -- src/` 
Expected: only `tests/unit/test_scoring.py` references it in non-`src` paths; no `src/` importer (the live path uses `transition_metrics`).

- [ ] **Step 2: Delete the module and repoint tests.** Remove `transition_scoring.py`. In `test_scoring.py`, replace tests of `compute_transition_score(_raw_and_transformed)` with equivalent assertions on `score_transition_edge` / `_calibrate_transition_cos`, or delete the now-redundant tests if `test_transition_calibration.py` + `test_audit_matches_beam.py` already cover the behavior.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS (no import errors from the deletion).

- [ ] **Step 4: Commit**

```bash
git add -A src/playlist/scoring/ tests/unit/test_scoring.py
git commit -m "cleanup(sonic): delete production-dead scoring/transition_scoring.py"
```

---

## Task 6: Delete vestigial edge-dict wiring (cleanup 3/3)

**Files:**
- Modify: `src/playlist/transition_metrics.py` (`T_used` ~182; `transition_gamma` context field ~30, 77, 142; `edge["gamma"]` ~188)

**Interfaces:**
- Note: `reporter.py:350` sets `edge["gamma"]` itself for display; that path is untouched. We only remove gamma's *dead storage inside the live scorer*.

- [ ] **Step 1: Confirm `T_used` has no reader.**

Run: `git grep -n "T_used"`
Expected: only the write in `transition_metrics.py:182`. (If any reader appears, STOP and reassess.)

- [ ] **Step 2: Remove `T_used` and gamma local storage.** Delete `"T_used": float(t_val),` from the edge dict. Remove `transition_gamma` from `TransitionMetricContext`, from `build_transition_metric_context`'s signature/body, and the `"gamma": context.transition_gamma` edge entry. Leave `reporter.py` and `constructor.py` (deferred) untouched.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/playlist/transition_metrics.py
git commit -m "cleanup(sonic): drop vestigial T_used and dead gamma storage from live scorer"
```

---

## Task 7: Regenerate goldens

**Files:**
- Modify: golden fixtures under `tests/unit/goldens/` (config snapshots that encode `transition_calibration`/floors; any playlist-`T` goldens)

- [ ] **Step 1: Run the suite to surface golden diffs.**

Run: `python -m pytest -q -m "not slow"`
Expected: golden mismatches localized to the `transition_calibration` block, retuned floor values, and `T`/edge-dict shape (no `T_used`, no `gamma` from the scorer).

- [ ] **Step 2: Regenerate the affected goldens deliberately** (use the repo's golden-update mechanism for the failing fixtures only — inspect each diff). Confirm every diff is an *intended* change: new config block, new floor values, removed `T_used`/`gamma`, calibrated `T` values. NO unexpected drift in unrelated fields.

- [ ] **Step 3: Run the full fast suite green**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS (full counts quoted, not a subset).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/goldens
git commit -m "test(sonic): regenerate goldens for calibrated transition score"
```

---

## Task 8: Verification gate (no success claim without these)

- [ ] **Step 1: Probe** — `python scripts/research/sonic_centering_probe.py`; confirm good-vs-bad gap ≥ ~70%, median `T` off ~0.5, and a rank-fidelity check (blended-`T` order vs raw blended order Spearman ≥ ~0.95). Add the Spearman check to the probe if absent.
- [ ] **Step 2: Full fast suite** — `python -m pytest -q -m "not slow"`, run directly, bounded by the tool timeout. Quote real pass/fail counts.
- [ ] **Step 3: Real generation + READ THE LOG** — generate a multi-pier playlist via the `gui_fidelity` harness with `pier_bridge.emit_selected_edge_audit: true` at INFO. Confirm from the per-edge audit (not a summary metric) that weak sonic edges are now ranked/penalized correctly and the selected-edge `T` distribution is de-compressed. Confirm generation stays < 90s.
- [ ] **Step 4: Perceptual audition** — audition 3–5 seeds; confirm the worst edge sounds better than before. Record notes.
- [ ] **Step 5: Finish the branch** — use `superpowers:finishing-a-development-branch`. Update memory (`project_centered_transition_rescale_flaw` → shipped) and flag the deferred `transition_gamma`/`constructor.py` removal to the dead-code program. Coordinate the merge BEFORE the genre session's calibration.

---

## Self-Review (completed by plan author)

- **Spec coverage:** §4 fix → Tasks 1-3; §5 floors → Tasks 2-3, 7; §6.1 cleanup → Tasks 4-6; §6.2 preserve → Global Constraints; §7 verification → Task 8; §8 process → Pre-flight + Global Constraints. Covered.
- **Placeholder scan:** calibration `center/scale/gain` and floor values are intentionally execution-derived (Task 2 emits them, Task 3 applies them) — the *method* and *acceptance criteria* are concrete, not "TBD". Provisional defaults (0.32/0.0625/1.0) keep the system runnable between Task 1 and Task 3.
- **Type consistency:** `_calibrate_transition_cos(value, *, center, scale, gain)` and the `calib_center/scale/gain` / `transition_calib_center/scale/gain` field names are used consistently across Tasks 1, 4. (Note the two naming surfaces: `TransitionMetricContext.calib_*` vs `PierBridgeConfig.transition_calib_*` — the resolver maps one to the other.)
- **Open risk:** Task 4's audit-equivalence assumes the audit feeds the same centered matrices the beam uses; the `test_audit_matches_beam` test is the guard. If the audit's `_tr_norm` matrices turn out NOT centered, the executor must center them (or pass the context) — flagged in the test failure.
