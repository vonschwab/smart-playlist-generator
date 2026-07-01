# SP-B: Retire MERT + Beat3Tower Sonic Embedding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the deprecated MERT and Beat3Tower tower-blend sonic paths (code + artifact bake), leaving MuQ as the sole registered variant behind a thin baked-variant seam, then rebuild the artifact muq-native.

**Architecture:** Bottom-up removal along the dependency spine: first flip defaults and make the loader override-aware (both work against the CURRENT artifact), then simplify each consumer of `sonic_variant.py`/`sonic_axes.py` to the plain-cosine behavior it already exhibits under muq, then delete the modules, then the MERT analyze path, then the tower bake. Only after all code lands and the suite is green does the artifact get rebuilt (backup first). Archive of MERT data is last and gated on Dylan's explicit confirmation.

**Tech Stack:** Python 3.11, numpy, pytest, existing analyze pipeline (`scripts/analyze_library.py`).

**Spec:** `docs/superpowers/specs/2026-07-01-sp-b-remove-mert-towers-design.md` (approved 2026-07-01).

## Global Constraints

- **Execution context:** work on `master` in the MAIN checkout `C:\Users\Dylan\Desktop\PLAYLIST_GENERATOR_V3` (the artifact rebuild needs real `data/`). Every subagent must first run `git branch --show-current` and STOP if it does not print `master`. Another session may be committing concurrently: **stage explicit paths only** (never `git add -A`/`-u`), re-check `git status` immediately before each commit, and leave files you didn't touch alone (`data/layered_genre_taxonomy.yaml`, `docs/ARCHITECTURE.md`, `docs/CLEANUP_LIST.md`, `docs/LOGGING.md`, `docs/PLAYLIST_ORDERING_TUNING.md`, `docs/TROUBLESHOOTING.md` are known in-flight).
- **Data safety (HARD):** nothing under `data/` is deleted, ever. `metadata.db` is never written. Music files are read-only. `muq_sidecar.npz`/`muq_failed.json` untouched. The artifact is rewritten ONLY in Task 10, after a timestamped backup. The MERT archive move (Task 11) requires Dylan's explicit confirmation at that moment.
- **Behavior preservation:** under muq, every simplified call site must be behaviorally identical (the tower transforms already degrade to raw passthrough for 512-dim input). The Task 10 before/after generation must produce an IDENTICAL tracklist.
- **Tests:** run `python -m pytest -q -m "not slow"` directly with the tool timeout — NEVER pipe pytest through `tail`/`head`/`grep`. `ruff check <touched files>` after each task.
- **Muq stays intact:** `src/analyze/muq_runner.py`, `stage_muq`, `scripts/fold_muq_into_artifact.py`, `tests/unit/test_muq_runner.py` are load-bearing — no deletions there.
- **Keep `stage_sonic`** (`SonicFeaturePipeline`, `beat3tower_extractor.py`, `tracks.sonic_features`) — it is the live BPM/pace source and the artifact universe gate (SP-C removes it later).
- Calibration constants, verbatim: muq calib = `(0.594, 0.092)`, gain `1.0`. Tower weights being deleted: `0.20/0.50/0.30`; transition-weights example bug being deleted: `0.40/0.35/0.25`.

## File Structure (net effect)

- **Deleted:** `scripts/extract_mert_sidecar.py`, `scripts/fold_mert_into_artifact.py`, `scripts/calibrate_mert_transform.py`, `scripts/fold_2dftm_into_artifact.py`, `scripts/extract_harmony_2dftm_sidecar.py`, `src/similarity/sonic_variant.py`, `src/playlist/sonic_axes.py`, 10 test files (listed in Tasks 6–7).
- **Created:** `src/analyze/track_paths.py` (relocated `load_paths`), `scripts/research/spb_artifact_checks.py` (Task 10 assertions), `tests/unit/test_artifact_required_keys.py`, `tests/unit/test_replacement_divergence.py`.
- **Modified (major):** `scripts/analyze_library.py`, `src/features/artifacts.py`, `src/playlist/transition_metrics.py`, `src/playlist/pier_bridge_builder.py`, `src/playlist/pier_bridge/config.py`, `src/playlist/pipeline/core.py`, `src/playlist/replacement.py`, `src/playlist_gui/worker.py`, `scripts/build_beat3tower_artifacts.py`, `src/config_loader.py`, `config.example.yaml`.

---

### Task 1: Flip transition-calibration defaults to muq

**Files:**
- Modify: `src/playlist/transition_metrics.py:23-29` (calib table + default variant), `:81-83` and `:130-132` (dataclass + builder defaults)
- Modify: `src/playlist/pier_bridge/config.py:88-90` (dataclass defaults)
- Test: `tests/unit/test_transition_calibration.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `resolve_transition_calib(variant)` with muq-only table; `TRANSITION_CALIB_BY_VARIANT == {"muq": (0.594, 0.092)}`; `_DEFAULT_CALIB_VARIANT == "muq"`. Later tasks rely on `resolve_transition_calib` keeping its exact signature `(variant, *, override=None) -> tuple[float, float, float]`.

Rationale: crash point #2 in the spec — the `"mert"` entry, the `None`→`"mert"` default, and the `0.32/0.0625` dataclass defaults must all flip **in one commit**, or `resolve_transition_calib(None)` raises. Live behavior is unchanged: the beam/reporter/worker all resolve from `bundle.sonic_variant == "muq"` already; the dataclass defaults are only reachable fallbacks.

- [ ] **Step 1: Update the tests first**

In `tests/unit/test_transition_calibration.py`: delete every test that asserts the mert band (the tests around lines 76–98 asserting `resolve_transition_calib("mert") == (0.32, 0.0625, 1.0)` and the `None`→mert mapping). Replace with:

```python
def test_none_variant_maps_to_muq_default():
    # Post-SP-B: muq is the only registered variant; legacy/no-variant
    # artifacts get the muq band (pre-variant artifacts no longer exist).
    assert resolve_transition_calib(None) == (0.594, 0.092, 1.0)


def test_mert_variant_now_raises():
    # The mert band was removed with the MERT path (SP-B).
    with pytest.raises(ValueError, match="No transition calibration"):
        resolve_transition_calib("mert")


def test_muq_band_unchanged():
    assert resolve_transition_calib("muq") == (0.594, 0.092, 1.0)
```

Keep all existing muq-band and override-priority tests as-is.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest -q tests/unit/test_transition_calibration.py`
Expected: FAIL — `test_none_variant_maps_to_muq_default` gets `(0.32, 0.0625, 1.0)`; `test_mert_variant_now_raises` gets no raise.

- [ ] **Step 3: Implement**

In `src/playlist/transition_metrics.py`, replace lines 23-29 with:

```python
TRANSITION_CALIB_BY_VARIANT: dict[str, tuple[float, float]] = {
    "muq": (0.594, 0.092),
}
# muq is the sole registered variant (SP-B removed MERT/towers); a None
# variant (defensive only — the loader always stamps one) maps to muq.
_DEFAULT_CALIB_VARIANT = "muq"
```

Update the `resolve_transition_calib` docstring's "maps to the historical MERT band" sentence to "maps to the muq band". In the same file flip the `TransitionMetricContext` defaults (lines 81-83) and the `build_transition_metric_context` keyword defaults (lines 130-132) from `0.32`/`0.0625` to `0.594`/`0.092` (gain stays `1.0`). Update the comment above them (lines 78-80) to name muq. Also update the module-top comment block (lines 15-22): drop the mert band line.

In `src/playlist/pier_bridge/config.py` lines 88-90, flip:

```python
    transition_calib_center: float = 0.594
    transition_calib_scale: float = 0.092
    transition_calib_gain: float = 1.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/unit/test_transition_calibration.py tests/unit/test_reporter_variant_calib.py tests/unit/test_transition_metric_alignment.py`
Expected: PASS. If `test_reporter_variant_calib.py` or `test_transition_metric_alignment.py` reference the mert band/defaults, update those assertions to the muq numbers in this same task (the `4e1136d` regression test `test_muq_does_not_get_the_mert_default` should now assert muq-vs-raise rather than muq-vs-mert — keep it, adjusted).

- [ ] **Step 5: Ruff + commit**

```bash
ruff check src/playlist/transition_metrics.py src/playlist/pier_bridge/config.py tests/unit/test_transition_calibration.py
git add src/playlist/transition_metrics.py src/playlist/pier_bridge/config.py tests/unit/test_transition_calibration.py tests/unit/test_reporter_variant_calib.py tests/unit/test_transition_metric_alignment.py
git commit -m "feat(sp-b): transition calib is muq-only — mert band removed, defaults flipped in all three sites"
```

---

### Task 2: Override-aware artifact required-keys (loader contract)

**Files:**
- Modify: `src/features/artifacts.py:141-158` (required keys + base-matrix load), `:224-232` (raw fallback paths)
- Test: `tests/unit/test_artifact_required_keys.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `load_artifact_bundle(path, sonic_variant_override)` that loads an npz with NO plain `X_sonic` when the override's `X_sonic_{override}` key is present. Task 10's rebuilt artifact depends on this.

The existing variant-window resolution (lines 238-260) already prefers `X_sonic_{variant}_{seg}` and tolerates absent legacy keys — untouched. This task only makes the *base* key override-aware. Must remain compatible with the CURRENT artifact (which has both `X_sonic` and `X_sonic_muq`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_artifact_required_keys.py`:

```python
"""SP-B loader contract: per-variant sonic keys only (no plain X_sonic)."""
import numpy as np
import pytest

from src.features.artifacts import load_artifact_bundle


def _write_npz(path, *, with_plain=False, with_muq=True):
    n, d = 4, 8
    arrs = {
        "track_ids": np.array([f"t{i}" for i in range(n)], dtype=object),
        "artist_keys": np.array([f"a{i}" for i in range(n)], dtype=object),
        "X_genre_raw": np.zeros((n, 3), dtype=np.float32),
        "X_genre_smoothed": np.zeros((n, 3), dtype=np.float32),
        "genre_vocab": np.array(["g1", "g2", "g3"], dtype=object),
        "X_sonic_variant": np.array("muq"),
    }
    if with_plain:
        arrs["X_sonic"] = np.random.rand(n, d).astype(np.float32)
    if with_muq:
        arrs["X_sonic_muq"] = np.random.rand(n, d).astype(np.float32)
    np.savez(path, **arrs)
    return path


def test_muq_only_artifact_loads_under_override(tmp_path):
    p = _write_npz(tmp_path / "art.npz", with_plain=False, with_muq=True)
    bundle = load_artifact_bundle(p, sonic_variant_override="muq")
    assert bundle.X_sonic.shape == (4, 8)
    assert bundle.sonic_variant == "muq"


def test_missing_variant_key_still_raises(tmp_path):
    p = _write_npz(tmp_path / "art2.npz", with_plain=True, with_muq=False)
    with pytest.raises(ValueError, match="X_sonic_muq"):
        load_artifact_bundle(p, sonic_variant_override="muq")


def test_legacy_artifact_without_override_still_requires_plain(tmp_path):
    p = _write_npz(tmp_path / "art3.npz", with_plain=False, with_muq=True)
    # no override: legacy contract, plain X_sonic required
    with pytest.raises(ValueError, match="X_sonic"):
        load_artifact_bundle(p, sonic_variant_override=None)
```

Note: `load_artifact_bundle` is lru-cached by `(path, override)` — distinct tmp filenames per test avoid cache hits.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q tests/unit/test_artifact_required_keys.py`
Expected: `test_muq_only_artifact_loads_under_override` FAILS with `ValueError: Artifact missing required keys: ['X_sonic']`. (The other two may already pass — fine.)

- [ ] **Step 3: Implement**

In `_load_artifact_bundle_cached` (`src/features/artifacts.py`), replace lines 141-149 with:

```python
    # SP-B contract: the sonic space is per-variant. With an override the
    # variant key is required and plain X_sonic is optional (rebuilt
    # artifacts no longer carry it); without an override (legacy artifacts,
    # unit fixtures) plain X_sonic remains required.
    required_keys = {
        "track_ids",
        "artist_keys",
        "X_genre_raw",
        "X_genre_smoothed",
        "genre_vocab",
    }
    required_keys.add(
        f"X_sonic_{sonic_variant_override}" if sonic_variant_override else "X_sonic"
    )
    _require_keys(data, required_keys)
```

Then make the plain-key load tolerant (line 155):

```python
    X_sonic_raw = data["X_sonic"] if "X_sonic" in data else None
```

And harden the two fallback paths that assign `X_sonic = X_sonic_raw` (lines 224-232) so a None raw can never leak out as the bundle's sonic matrix:

```python
        else:
            if X_sonic_raw is None:
                raise ValueError(
                    f"Artifact {artifact_path} declares sonic_variant={declared_variant!r} "
                    f"but has neither X_sonic_{declared_variant} nor a plain X_sonic key."
                )
            X_sonic = X_sonic_raw
            logger.warning(
                "Artifact declared sonic_variant=%s but missing key %s; falling back to X_sonic raw.",
                declared_variant,
                variant_key,
            )
    else:
        X_sonic = X_sonic_raw  # guaranteed non-None: no override => X_sonic was required
```

Grep the remainder of the function for other unconditional `data["X_sonic"]` accesses and guard the same way: `grep -n 'data\["X_sonic"\]' src/features/artifacts.py` — expected: only the line edited above.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/unit/test_artifact_required_keys.py tests/unit/test_artifact_tower_weighted_load.py`
Expected: PASS (the tower-load test still passes — it uses artifacts with plain `X_sonic`; it is deleted later in Task 6).

- [ ] **Step 5: Ruff + commit**

```bash
ruff check src/features/artifacts.py tests/unit/test_artifact_required_keys.py
git add src/features/artifacts.py tests/unit/test_artifact_required_keys.py
git commit -m "feat(sp-b): override-aware artifact required keys — per-variant sonic key required, plain X_sonic optional under override"
```

---

### Task 3: Simplify the transition-metric context (drop tower transforms)

**Files:**
- Modify: `src/playlist/transition_metrics.py:115-160` (builder), `src/playlist/reporter.py:239-320`, `src/playlist_gui/worker.py:732-790`, `scripts/research/calibrate_transition_sigmoid.py:26-60`, `scripts/research/sonic_centering_probe.py`, `scripts/research/verify_roam_transition.py`
- Test: `tests/unit/test_transition_metric_alignment.py` (existing, must stay green)

**Interfaces:**
- Consumes: Task 1's muq defaults.
- Produces: `build_transition_metric_context(*, X_sonic, X_start=None, X_mid=None, X_end=None, X_genre=None, center_transitions=False, transition_gamma=None, embedding_random_seed=None, weight_end_start=0.70, weight_mid_mid=0.15, weight_full_full=0.15, calib_center=0.594, calib_scale=0.092, calib_gain=1.0)` — the `transition_weights` and `sonic_variant` parameters are GONE. Tasks 4–5 rely on this signature.

Under muq, `resolve_sonic_variant`→`compute_sonic_variant_norm` degrades to plain L2 normalization and `apply_transition_weights` to identity (512 ≠ 137-dim slice) — so this is a pure simplification.

- [ ] **Step 1: Simplify the builder**

In `build_transition_metric_context`: delete the `transition_weights` and `sonic_variant` parameters; delete the `from src.similarity.sonic_variant import ...` block (lines 136-140); replace lines 142-154 with:

```python
    # One sonic space (muq): plain L2-normalized cosine, no tower transforms.
    X_sonic_norm = _l2_normalize_rows(X_sonic)

    X_full_tr = X_sonic
    X_start_tr = X_start
    X_mid_tr = X_mid
    X_end_tr = X_end
```

(The `center_transitions` block below already handles centering these.) Verify `_l2_normalize_rows` is imported/defined in this module — it is used elsewhere in the file; if it lives in another module, import from there.

- [ ] **Step 2: Update the five callers**

For each, remove the `transition_weights=...` and/or `sonic_variant=...` kwargs from the `build_transition_metric_context(...)` call (nothing else):
1. `src/playlist/reporter.py:311-319` — also delete the now-unused `resolve_sonic_variant` import (line 17) and the `sonic_variant = resolve_sonic_variant(...)` resolution (line 302). The `config_sonic_variant`/`sonic_variant` **parameters** of the enclosing report function stay for now (Task 5 removes the whole thread); just stop using them here.
2. `src/playlist_gui/worker.py:775-786` — drop both kwargs (`transition_weights=transition_weights,` and `sonic_variant=ds_report.get("sonic_variant"),`). Leave `transition_weights` variable assignment (line 766) — Task 5 cleans the report keys.
3. `scripts/research/calibrate_transition_sigmoid.py:46` — this script must KEEP working (it derives future variants' bands; the calib error message points at it). Drop the removed kwargs; if it passed `sonic_variant`, it now operates on the raw matrix it loads — which is what it measures anyway.
4. `scripts/research/sonic_centering_probe.py:24` — same treatment.
5. `scripts/research/verify_roam_transition.py:56` — same treatment.

Verification: `grep -rn "transition_weights=\|sonic_variant=" src/playlist/transition_metrics.py` → zero matches inside `build_transition_metric_context` calls; `grep -rn "build_transition_metric_context(" src scripts | wc -l` unchanged caller count.

- [ ] **Step 3: Run the covering tests**

Run: `python -m pytest -q tests/unit/test_transition_metric_alignment.py tests/unit/test_transition_calibration.py tests/unit/test_reporter_variant_calib.py`
Expected: PASS. If alignment tests passed `transition_weights=`/`sonic_variant=` kwargs, update them to the new signature in this task.

- [ ] **Step 4: Ruff + commit**

```bash
ruff check src/playlist/transition_metrics.py src/playlist/reporter.py src/playlist_gui/worker.py scripts/research/calibrate_transition_sigmoid.py scripts/research/sonic_centering_probe.py scripts/research/verify_roam_transition.py
git add <the six files above + any edited tests>
git commit -m "refactor(sp-b): transition context is plain-cosine — tower transforms and variant resolution removed from the builder + 5 callers"
```

---

### Task 4: Beam side — plain sonic space, BPM-only pace, no tower knobs

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py:549-650` (sim space, rhythm axis block, transition space), `src/playlist/pier_bridge/config.py:62-63,71-72` (dead fields), `src/playlist/pier_bridge/pace_gate.py` (axis functions), `src/playlist/pier_bridge/beam.py:1127-1136` (rhythm-axis penalty block), `src/playlist/constructor.py:141-155`, `src/playlist/pipeline/core.py:454-466,718-719,730-740`
- Test: existing pier-bridge/beam suite (must stay green); `tests/unit/test_tower_knob_guard.py` deleted here (its subject is deleted)

**Interfaces:**
- Consumes: Task 3's builder signature.
- Produces: `PierBridgeConfig` WITHOUT `transition_weights`, `sonic_variant`, `rhythm_soft_penalty_threshold`, `rhythm_soft_penalty_strength`; `validate_tower_knobs` has no callers (deleted in Task 6). `apply_pier_bridge_overrides` no longer returns `transition_weights` (returns 3-tuple `pb_cfg, tuning, tuning_sources`).

All these paths are verified inert/fallback under muq today; this makes the fallback the only path.

- [ ] **Step 1: pier_bridge_builder.py — sim space + transition space**

Replace lines 549-555 (variant sim space) with:

```python
    # Similarity space for bridge gating (full vectors): plain L2-normalized
    # cosine on the loaded sonic matrix (muq) — matches DS admission.
    X_full_norm = _l2_normalize_rows(X_full_raw)
    logger.debug("Pier+Bridge sonic sim space: dim=%d", int(X_full_norm.shape[1]))
```

Delete the whole rhythm-axis attempt (the `_needs_rhythm` block from `rhythm_matrix: Optional[np.ndarray] = None` through the end of the tower_dims branch) and keep the BPM path as primary. The result (replacing lines ~557-627):

```python
    # Pace gating is BPM/onset-band based (the rhythm tower axis was removed
    # in SP-B; under muq it had already fallen back to BPM permanently).
    rhythm_matrix: Optional[np.ndarray] = None
    if float(getattr(cfg, "pace_bridge_floor", 0.0)) > 0.0:
        if perceptual_bpm is not None:
            from src.playlist.pier_bridge.pace_gate import bpm_fallback_max_log_distance

            _bpm_cap = float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf")))
            if not np.isfinite(_bpm_cap):
                _bpm_cap = bpm_fallback_max_log_distance(float(cfg.pace_bridge_floor))
                cfg = replace(cfg, bpm_bridge_max_log_distance=_bpm_cap)
            logger.info(
                "Pace bridge gate: perceptual-BPM band (bpm_bridge_max_log_distance=%.2f)",
                float(cfg.bpm_bridge_max_log_distance),
            )
        else:
            logger.warning(
                "Pace bridge gate DISABLED: pace_bridge_floor=%.2f is set but no "
                "perceptual-BPM data is available (a configured knob that can't act).",
                float(cfg.pace_bridge_floor),
            )
```

`rhythm_matrix` stays as an always-None local ONLY if the beam call signature still takes it — preferred: remove the `rhythm_matrix=` argument from the beam invocation and the beam's parameter (see Step 3). Replace the transition-space block (lines 629-640):

```python
    # Transition space: the raw sonic matrices (optional mean-centering below).
    X_full_tr = X_full_raw
    X_start_tr = X_start_raw
    X_mid_tr = X_mid_raw
    X_end_tr = X_end_raw
```

Delete the `from src.similarity.sonic_variant import apply_transition_weights` import.

- [ ] **Step 2: config fields + core threading**

`src/playlist/pier_bridge/config.py`: delete fields `rhythm_soft_penalty_threshold` (62), `rhythm_soft_penalty_strength` (63), `transition_weights` (71), `sonic_variant` (72).

`src/playlist/pipeline/core.py`:
- Delete the transition-weights extraction block (lines 454-466, `transition_weights = None` through the dict branch).
- Delete the two `rhythm_soft_penalty_*` lines from the `pb_cfg = replace(...)` pace block (lines 718-719).
- Delete the `validate_tower_knobs(...)` call and its comment (lines ~730-740) and the `validate_tower_knobs` import.
- `apply_pier_bridge_overrides(...)`: change its signature/return to drop `transition_weights` (find it: `grep -n "def apply_pier_bridge_overrides" src/playlist/` — update the function body to stop reading `transition_weights` from overrides/config and return `pb_cfg, tuning, tuning_sources`); update the unpacking at line ~695.
- Grep for remaining threading: `grep -rn "rhythm_soft_penalty" src/ config.example.yaml` — remove every remaining site (pace presets in `src/config_loader.py` or `src/playlist_gui/policy.py` if present, plus the `config.example.yaml` lines). A preset key for a deleted knob is exactly the silent-no-op trap.

- [ ] **Step 3: pace_gate + beam rhythm-axis code**

`src/playlist/pier_bridge/pace_gate.py`: delete the `from src.playlist.sonic_axes import ...` import and every function that consumes `rhythm_matrix`/axis vectors (`axis_cosine_similarity`, `interpolate_axis_vector` users, line 39 region). KEEP `bpm_fallback_max_log_distance` and all BPM/onset band logic.

`src/playlist/pier_bridge/beam.py`: delete the rhythm-axis soft-penalty block at lines 1127-1136 (the `from src.playlist.sonic_axes import axis_cosine_similarity` local import and its penalty computation) and remove the now-unused `rhythm_matrix` parameter end-to-end: `grep -n "rhythm_matrix" src/playlist/pier_bridge/beam.py src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/pace_gate.py` → remove every remaining reference (parameters, call-site arguments, None-checks).

`src/playlist/constructor.py` lines 141-155: replace the `apply_transition_weights` calls:

```python
    if bundle.X_sonic_start is not None and bundle.X_sonic_end is not None:
        X_start_orig = get_sonic_matrix(bundle, "start")
        X_end_orig = get_sonic_matrix(bundle, "end")
        X_start = X_start_orig
        X_end = X_end_orig
        transition_weight_stats = {}
```

Delete the `apply_transition_weights` import and the two `start_stats`/`end_stats` debug-log lines.

- [ ] **Step 4: Delete the tower-knob guard test + run the suite**

```bash
git rm tests/unit/test_tower_knob_guard.py
python -m pytest -q -m "not slow"
```
Expected: PASS (full fast suite; use timeout 600000). Any failure here is a missed reference — fix it, don't skip it.

- [ ] **Step 5: Ruff + commit**

```bash
ruff check src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/config.py src/playlist/pier_bridge/pace_gate.py src/playlist/pier_bridge/beam.py src/playlist/constructor.py src/playlist/pipeline/core.py
git add <the six files + config_loader/policy/config.example if rhythm_soft_penalty removal touched them>
git commit -m "refactor(sp-b): beam runs plain-cosine sonic + BPM-band pace — tower transition weights, rhythm axis, and tower knobs removed"
```

---

### Task 5: Purge the sonic_variant thread (embedding, artist_style, generator, reporter, worker, diagnostics)

**Files:**
- Modify: `src/playlist/pipeline/embedding_setup.py:21,56,70-71,86,100`, `src/playlist/artist_style.py:535-553`, `src/playlist_generator.py` (sites: 60-66, 475-478, 805, 1783, 1926-1986, 2193-2195, 2760-2880), `src/playlist/reporter.py:239-261`, `src/playlist/pipeline/core.py` (the `sonic_variant` param of `generate_playlist_ds` and `setup_embedding` call), `src/playlist_gui/worker.py` (report keys), `scripts/diagnose_artist_style.py`, `scripts/diagnose_candidate_scores.py`, `scripts/diagnose_sonic_floor.py`
- Test: existing suite green; `tests/unit/test_sonic_variant_resolution.py` deleted in Task 6

**Interfaces:**
- Consumes: Tasks 3–4 signatures.
- Produces: NO import of `src.similarity.sonic_variant` anywhere outside the module itself (Task 6's deletion precondition). `generate_playlist_ds` and `setup_embedding` no longer take `sonic_variant`; `build_ds_overrides` no longer emits `tower_weights`/`transition_weights`/`tower_pca_dims`.

The `sim_variant` config knob (`playlists.sonic.sim_variant: tower_pca`) is the source of this thread; it resolves to a tower transform that degrades to raw under muq — the whole thread is a no-op carrying dead names.

- [ ] **Step 1: embedding_setup + artist_style**

`src/playlist/pipeline/embedding_setup.py`: delete the import (21); delete the `sonic_variant` parameter (56) and the `resolved_variant = resolve_sonic_variant(...)` (70-71); the embed matrix (100) becomes `X_sonic_for_embed = bundle.X_sonic` with `variant_stats = {"variant": getattr(bundle, "sonic_variant", None), "fallback": False}` (keep whatever stats-dict shape downstream reads — grep `variant_stats` consumers in the file and preserve keys). Line 86's report field: use `getattr(bundle, "sonic_variant", None)`.

`src/playlist/artist_style.py`: delete the `sonic_variant` parameter (535) and the import/resolve (550-552); line 553 becomes plain L2: `X_norm = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-12)` and `variant_stats = {}` (or drop the variable if only logged). Fix its callers: `grep -rn "artist_style" src/ --include="*.py" | grep "sonic_variant"` → remove the kwarg at each.

- [ ] **Step 2: generate_playlist_ds + generator threading**

`src/playlist/pipeline/core.py`: remove the `sonic_variant` parameter from `generate_playlist_ds` (and `resolved_variant` — the audit context (687) and `apply_pier_bridge_overrides` should use `getattr(bundle, "sonic_variant", None)` instead). `setup_embedding(...)` call drops `sonic_variant=sonic_variant` (472).

`src/playlist_generator.py`:
- `build_ds_overrides` (60-66): delete the `"tower_weights"`, `"transition_weights"`, `"tower_pca_dims"` entries.
- Remove every `sonic_variant=sonic_variant_cfg` / `sonic_variant=sonic_variant` kwarg at the `generate_playlist_ds`/report call sites (475-478, 805, 1783, 1985-1986, 2760, 2879-2880) and the `config_sonic_variant=self.sonic_variant` (476). Remove the local `tw_raw = ds_cfg.get("transition_weights")` blocks (1926, 2834) and the `transition_weights=transition_weights` kwargs they feed (1985, 2879). Keep purely-reporting dict reads like line 862/2193-2195 ONLY if the report key still exists — otherwise delete the lines; grep-verify each.
- Find and remove the `self.sonic_variant` definition (it reads the `sim_variant` config): `grep -n "sim_variant\|self.sonic_variant" src/playlist_generator.py src/config_loader.py` — delete the property/attribute and the `config_loader` accessor for `playlists.sonic.sim_variant`.

`src/playlist/reporter.py`: delete the `config_sonic_variant` and `sonic_variant` parameters (239-261) and their docstring lines; the calib resolve (310) already uses `bundle.sonic_variant` — keep.

`src/playlist_gui/worker.py`: line 766's `transition_weights = ...` local and line 786's report read — delete the local (its consumer went in Task 3); keep `ds_report.get("sonic_variant")` only where a report field is still populated (grep the producer; if `generate_playlist_ds` no longer emits `sonic_variant` in its report, delete the reads).

- [ ] **Step 3: diagnostics scripts**

`scripts/diagnose_artist_style.py`, `scripts/diagnose_candidate_scores.py`, `scripts/diagnose_sonic_floor.py`: replace their `sonic_variant` imports/uses with plain L2-normalized `bundle.X_sonic` (same pattern as Step 1). These are debug tools; behavior under muq is identical.

- [ ] **Step 4: Zero-import verification + suite**

```bash
grep -rln "similarity.sonic_variant\|from src.similarity import sonic_variant" src/ scripts/ tools/ --include="*.py"
```
Expected: EXACTLY `src/similarity/sonic_variant.py` (the module itself) — plus `scripts/analyze_library.py` (its `_ALLOWED` import goes in Task 7). Anything else is a miss.

Run: `python -m pytest -q -m "not slow"` (timeout 600000). Expected: PASS except `tests/unit/test_sonic_variant_resolution.py`-style tests that import removed parameters — those are deleted in Task 6; if they fail here on *signatures you changed*, update or defer per file ownership (deletions belong to Task 6; signature updates to this task).

- [ ] **Step 5: Ruff + commit**

```bash
ruff check <all files touched>
git add <explicit file list>
git commit -m "refactor(sp-b): purge the sonic_variant/tower thread from embedding, artist_style, generator, reporter, worker, diagnostics"
```

---

### Task 6: Delete sonic_variant.py, sonic_axes.py, tower guards, replacement re-base

**Files:**
- Delete: `src/similarity/sonic_variant.py`, `src/playlist/sonic_axes.py`
- Modify: `src/features/artifacts.py` (tower guards + `tower_dims`), `src/playlist/replacement.py:28-40,104-119`, `src/playlist_gui/worker.py:203,579-616,794,811,909`, `src/config_loader.py:347-375,495-497`
- Delete tests: `tests/unit/test_sonic_variant_resolution.py`, `tests/unit/test_sonic_axes.py`, `tests/unit/test_worker_tower_pca_dims.py`, `tests/unit/test_artifact_tower_weighted_load.py`
- Test: `tests/unit/test_replacement_divergence.py` (create)

**Interfaces:**
- Consumes: Tasks 3–5 (zero remaining imports of the two modules except `analyze_library.py`, handled in Task 7 — to keep this task green, this task ALSO retargets that one import, see Step 3).
- Produces: `ReplacementContext` without `tower_pca_dims`; `_sound_divergence(ctx, cand_idx, current_idx)` = full-sonic cosine divergence.

- [ ] **Step 1: TDD the replacement re-base**

Create `tests/unit/test_replacement_divergence.py`:

```python
"""SP-B: replacement divergences are BPM (pace) and full-sonic cosine (sound)."""
import numpy as np

from src.playlist.replacement import ReplacementContext, _pace_divergence, _sound_divergence


def _ctx(X_sonic, perceptual_bpm=None):
    return ReplacementContext(
        track_ids=np.array(["a", "b"], dtype=object),
        artist_keys=np.array(["x", "y"], dtype=object),
        X_sonic=X_sonic,
        X_genre_smoothed=None,
        idf_weights=None,
        perceptual_bpm=perceptual_bpm,
    )


def test_sound_divergence_is_full_sonic_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # orthogonal
    ctx = _ctx(X)
    assert abs(_sound_divergence(ctx, cand_idx=1, current_idx=0) - 1.0) < 1e-6
    ctx2 = _ctx(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))  # identical
    assert abs(_sound_divergence(ctx2, cand_idx=1, current_idx=0) - 0.0) < 1e-6


def test_pace_divergence_uses_bpm_and_degrades_to_zero():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ctx = _ctx(X, perceptual_bpm=np.array([120.0, 60.0]))
    assert _pace_divergence(ctx, cand_idx=1, current_idx=0) > 0.0
    ctx_nobpm = _ctx(X, perceptual_bpm=None)
    # No BPM data: no pace signal (the tower rhythm axis is gone) — 0.0, not garbage.
    assert _pace_divergence(ctx_nobpm, cand_idx=1, current_idx=0) == 0.0
```

Adjust the `ReplacementContext` constructor kwargs to its actual field list (read `replacement.py:20-40` first; drop `tower_pca_dims`, keep required fields — fill any other required fields with minimal values).

- [ ] **Step 2: Run to verify failure, then implement**

Run: `python -m pytest -q tests/unit/test_replacement_divergence.py` — expected FAIL (constructor still requires `tower_pca_dims`; `_sound_divergence` still carves axes).

Implement in `src/playlist/replacement.py`: delete the `tower_pca_dims` field (32) and the `extract_axis_vectors`/`axis_cosine_similarity` import; replace lines 104-119:

```python
def _pace_divergence(ctx: ReplacementContext, *, cand_idx: int, current_idx: int) -> float:
    if ctx.perceptual_bpm is not None:
        cand_bpm = float(ctx.perceptual_bpm[int(cand_idx)])
        current_bpm = float(ctx.perceptual_bpm[int(current_idx)])
        if np.isfinite(cand_bpm) and np.isfinite(current_bpm) and cand_bpm > 0 and current_bpm > 0:
            return float(bpm_log_distance(cand_bpm, current_bpm))
    # No usable BPM: no pace signal (the tower rhythm axis was removed in SP-B).
    return 0.0


def _sound_divergence(ctx: ReplacementContext, *, cand_idx: int, current_idx: int) -> float:
    # Full-sonic cosine divergence on the loaded (muq) matrix. The old tower
    # "color" carving was meaningless on a no-tower embedding.
    a = np.asarray(ctx.X_sonic[int(current_idx)], dtype=float)
    b = np.asarray(ctx.X_sonic[int(cand_idx)], dtype=float)
    return _safe_cosine_divergence(a, b)
```

`src/playlist_gui/worker.py`: delete `_infer_tower_pca_dims` (579-587) and `_resolve_tower_pca_dims` (590-616); delete the cache field (203), the resolve+assign (794, 811), the required-list entry (909), and the `ReplacementContext(... tower_pca_dims ...)` argument where the context is built.

`src/features/artifacts.py`: delete `DEFAULT_TOWER_TRANSITION_WEIGHTS` (20), `_is_default_transition_weights` (365-379), `_variant_lacks_tower_split` (382-401), `validate_tower_knobs` (404-439); delete the `tower_dims` parse block (180-187) and the bundle field + any `bundle.tower_dims` remaining readers (`grep -rn "tower_dims" src/ scripts/ tests/` → remaining hits must be only in files deleted by this task or `build_beat3tower_artifacts.py`/`fold_2dftm` (Task 7-8 territory)).

`src/config_loader.py`: delete `ds_tower_weights` (347-355), `ds_transition_weights` (357-365), `ds_tower_pca_dims` (367-375), and their `get_ds_tuning_dict` entries (495-497).

- [ ] **Step 3: Delete the modules + retarget analyze_library's `_ALLOWED` import**

```bash
git rm src/similarity/sonic_variant.py src/playlist/sonic_axes.py
git rm tests/unit/test_sonic_variant_resolution.py tests/unit/test_sonic_axes.py tests/unit/test_worker_tower_pca_dims.py tests/unit/test_artifact_tower_weighted_load.py
```

In `scripts/analyze_library.py` (lines 152-158): replace the `_ALLOWED` import + union with the muq-only set (full MERT surgery is Task 7; this keeps the tree importable now):

```python
# Recognized artifacts.sonic_variant_override values. muq is the sole baked
# variant (SP-B removed MERT and the tower/transform variants).
_KNOWN_SONIC_VARIANTS = frozenset({"muq"})
```

Update `tests/unit/test_variant_gate.py`: the `test_known_variants_do_not_warn` case using `tower_weighted` now expects a WARNING — rewrite it:

```python
def test_tower_weighted_override_now_warns(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "tower_weighted"), "muq")
    assert any("not a recognized sonic variant" in r.getMessage() for r in caplog.records)
```

(Keep `muq` not-warning coverage.)

- [ ] **Step 4: Full fast suite**

Run: `python -m pytest -q -m "not slow"` (timeout 600000). Expected: PASS. Then the import check:
`grep -rln "sonic_variant\b" src/ --include="*.py" | xargs grep -ln "import"` → no file imports a deleted module (mentions of the *string/field* `sonic_variant` — bundle field, config override — are correct and stay).

- [ ] **Step 5: Ruff + commit**

```bash
ruff check src/ tests/unit/test_replacement_divergence.py tests/unit/test_variant_gate.py
git add <explicit list incl. deletions>
git commit -m "feat(sp-b): delete sonic_variant/sonic_axes modules + tower guards; replacement divergences re-based on BPM + full-sonic cosine"
```

---

### Task 7: Remove the MERT analyze path (+ registries, scripts, config, tests)

**Files:**
- Create: `src/analyze/track_paths.py`
- Modify: `scripts/analyze_library.py` (regions listed below), `src/playlist/request_models.py:25,51`, `web/src/components/ToolsPanel.tsx:10-14`, `config.example.yaml:5-15`, `pyproject.toml:8`
- Delete: `scripts/extract_mert_sidecar.py`, `scripts/fold_mert_into_artifact.py`, `scripts/calibrate_mert_transform.py`, `scripts/fold_2dftm_into_artifact.py`, `scripts/extract_harmony_2dftm_sidecar.py`
- Delete tests: `tests/unit/test_extract_mert_sidecar.py`, `tests/unit/test_analyze_mert_stage.py`, `tests/unit/test_mert_extraction_cancel.py`, `tests/unit/test_fold_mert.py`, `tests/unit/test_calibrate_mert_transform.py`, `tests/unit/test_fold_2dftm.py`
- Edit tests: `tests/unit/test_variant_gate.py`, `tests/unit/test_stage_fingerprint_variant.py`, `tests/unit/test_muq_stage_registration.py`

**Interfaces:**
- Consumes: Task 6's `_KNOWN_SONIC_VARIANTS = {"muq"}`.
- Produces: `src/analyze/track_paths.py::load_paths(db_path) -> dict[str, str]` (exact body relocated from `extract_mert_sidecar.py:433-440`); `_sonic_fold_settings(config_path) -> Tuple[bool, str]` (renamed from `_mert_fold_settings`, default `"muq"`); unconditional muq fold in `stage_artifacts`; verify guard keyed to `muq_sidecar.npz`. Task 8 relies on `stage_artifacts`' post-build fold sequence being: build → fold_muq → done (no 2DFTM step).

- [ ] **Step 1: TDD the seam changes (edit the three test files first)**

`tests/unit/test_variant_gate.py`: change `test_default_variant_is_mert` to:

```python
def test_default_variant_is_muq(tmp_path):
    # No override key at all -> the active variant defaults to muq (SP-B).
    _, active = _sonic_fold_settings(_cfg(tmp_path, None))
    assert active == "muq"
```

(Adapt `_cfg` so passing `None` writes a config without the override key; update the module import from `_mert_fold_settings` to `_sonic_fold_settings`.)

`tests/unit/test_stage_fingerprint_variant.py`: delete the mert-fingerprint test (line ~45); keep/rename the muq one; add:

```python
def test_mert_stage_no_longer_registered():
    from scripts.analyze_library import STAGE_FUNCS
    assert "mert" not in STAGE_FUNCS
```

`tests/unit/test_muq_stage_registration.py`: replace the "right after mert" assertion:

```python
def test_muq_registered_and_mert_gone():
    from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER
    assert "muq" in ANALYZE_LIBRARY_STAGE_ORDER
    assert "mert" not in ANALYZE_LIBRARY_STAGE_ORDER
```

Also port the verify-guard regression test: the deleted `tests/unit/test_analyze_mert_stage.py::test_verify_flags_sonic_variant_mismatch` (recover its body from git before deleting the file: `git show HEAD:tests/unit/test_analyze_mert_stage.py`) moves into `tests/unit/test_variant_gate.py`, re-keyed so the guard trigger is `muq_sidecar.npz` existing (create an empty stub file in tmp) and a stamped `X_sonic_variant != "muq"` yields the `sonic_variant_mismatch` issue.

Run: `python -m pytest -q tests/unit/test_variant_gate.py tests/unit/test_stage_fingerprint_variant.py tests/unit/test_muq_stage_registration.py` — expected FAIL (rename + registrations not done yet).

- [ ] **Step 2: Relocate `load_paths`**

Create `src/analyze/track_paths.py`:

```python
"""Track-id -> file-path map from metadata.db (read-only helper for analyze stages)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def load_paths(db_path: Path | str) -> dict[str, str]:
    """track_id -> file_path from metadata.db (read-only, URI mode=ro)."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    finally:
        con.close()
    return {str(t): p for t, p in rows if p}
```

Update `stage_muq`'s import in `scripts/analyze_library.py` (line ~2425): `from src.analyze.track_paths import load_paths`. Grep for other importers: `grep -rn "extract_mert_sidecar import" src/ scripts/ tests/` — every remaining import must be in a file this task deletes.

- [ ] **Step 3: analyze_library surgery**

In `scripts/analyze_library.py` (current line refs, adjust for drift):
1. Rename `_mert_fold_settings` → `_sonic_fold_settings`; body reads `analyze.muq.fold_into_artifact` (default `True`) and defaults the override to `"muq"`:

```python
def _sonic_fold_settings(config_path: str) -> Tuple[bool, str]:
    """(fold_enabled, active_variant) for the post-artifact sonic fold.

    - ``analyze.muq.fold_into_artifact`` (default True) toggles the auto-fold at
      the end of the artifacts stage.
    - ``artifacts.sonic_variant_override`` (default 'muq') names the active variant.
    """
    import yaml

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    muq_cfg = (cfg.get("analyze") or {}).get("muq") or {}
    enabled = bool(muq_cfg.get("fold_into_artifact", True))
    override = ((cfg.get("artifacts") or {}).get("sonic_variant_override") or "muq")
    return enabled, str(override).strip() or "muq"
```

Extend `_variant_gate`'s docstring with the seam's future-embedding recipe (spec §4): "To add a future embedding: add an extraction stage writing `<name>_sidecar.npz`, a fold writing `X_sonic_<name>*`, a `TRANSITION_CALIB_BY_VARIANT` entry, register `<name>` in `_KNOWN_SONIC_VARIANTS` + the stage registries, then flip `artifacts.sonic_variant_override`. Both variants coexist in the artifact during A/B."

Update ALL callers (`grep -n "_mert_fold_settings" scripts/analyze_library.py tests/`).
2. Delete the `"mert"` fingerprint branch (422-444) from `compute_stage_fingerprint`.
3. Delete the 2DFTM fold block in `stage_artifacts` (2074-2090, `_sidecar = out_dir / "harmony_2dftm_sidecar.npz"` through its except).
4. Replace the branched fold (2099-2143) with the unconditional muq fold — keep the muq branch's body exactly (fold_enabled check, missing-sidecar `logger.error`, fold-failure `logger.error`), drop the `elif fold_enabled:` MERT arm entirely.
5. Re-key the verify guard (2240-2260): `mert_sidecar = artifact_path.parent / "mert_sidecar.npz"` → `sidecar = artifact_path.parent / f"{active_variant}_sidecar.npz"`; condition `if fold_enabled and sidecar.exists():`; update the comment (it protects the ACTIVE variant now) and log text.
6. Delete `stage_mert` (2283-2412) and `_build_mert_embedder` (2270-2280); remove `"mert": stage_mert` from `STAGE_FUNCS` (2635).
7. `src/playlist/request_models.py`: remove `"mert"` from the `AnalyzeLibraryStage` Literal (25) and `ANALYZE_LIBRARY_STAGE_ORDER` (51).
8. `web/src/components/ToolsPanel.tsx`: remove `"mert"` from `ALL_STAGES` (10-14). Rebuild dist: `npm --prefix web run build`.

- [ ] **Step 4: Delete scripts + MERT tests, config lines**

```bash
git rm scripts/extract_mert_sidecar.py scripts/fold_mert_into_artifact.py scripts/calibrate_mert_transform.py scripts/fold_2dftm_into_artifact.py scripts/extract_harmony_2dftm_sidecar.py
git rm tests/unit/test_extract_mert_sidecar.py tests/unit/test_analyze_mert_stage.py tests/unit/test_mert_extraction_cancel.py tests/unit/test_fold_mert.py tests/unit/test_calibrate_mert_transform.py tests/unit/test_fold_2dftm.py
```

`config.example.yaml`: delete the `analyze.mert` block (lines 5-12); extend the `analyze.muq` block comment to document `fold_into_artifact: true` (the new home of the fold toggle). `pyproject.toml` line 8: description → `"Local playlist generator: MuQ-MuLan sonic embedding + genre-graph fusion over your own library"`.

- [ ] **Step 5: Suite + zero-reference verification + commit**

```bash
grep -rn "stage_mert\|fold_mert\|extract_mert\|fold_2dftm\|2dftm\|calibrate_mert" src/ scripts/ web/src/ tests/ --include="*.py" --include="*.tsx"
```
Expected: zero matches (docs may still mention them — Task 9). Run `python -m pytest -q -m "not slow"` (timeout 600000) — PASS. Then:

```bash
ruff check scripts/analyze_library.py src/analyze/track_paths.py src/playlist/request_models.py
git add <explicit list incl. deletions + web/dist if tracked>
git commit -m "feat(sp-b): remove the MERT analyze path — stage, folds, scripts, registries; verify guard re-keyed to the active variant sidecar"
```

---

### Task 8: Strip the tower bake from the artifact build

**Files:**
- Modify: `scripts/build_beat3tower_artifacts.py` (tower extraction/normalizer/bake: 241-274, 847-911, 960-1028; module docstring)
- Edit test: `tests/test_beat3tower_fallback.py` (keep `stage_sonic`-side coverage, delete bake-side)

**Interfaces:**
- Consumes: Task 2 (loader tolerates no plain `X_sonic`), Task 7 (`stage_artifacts` fold sequence).
- Produces: `build_artifacts` writing ONLY: `X_genre_raw`, `X_genre_smoothed`, `genre_vocab`, `track_ids`, `track_artists`, `track_titles`, `artist_keys`, `durations_ms`, `build_config`. NO `X_sonic*`, no `tower_*`, no `normalizer_params`, no `bpm_array`, no `sonic_feature_names`, no `X_sonic_variant` stamp (the muq fold stamps it).

**KEEP untouched:** `load_tracks_with_beat3tower` (152-224), `_is_beat3tower_features` (227-238), `refresh_genre_matrices` (749-792), the whole genre matrix pipeline, durations/metadata assembly. The track universe MUST be identical to today's.

- [ ] **Step 1: Read then cut**

Read `build_artifacts` (795-1044) end-to-end first. Delete: `extract_tower_vectors` (241-274) and its call; the normalizer fit/transform block (847-882); `compute_tower_calibration_stats` (884-898); the concatenation + robust_whiten precompute (900-911) including the `compute_sonic_variant_matrix` import; the per-tower/start/mid/end matrix assembly feeding the bake; `bpm_list` derivation. Replace the `np.savez(...)` (972-1028) with:

```python
    np.savez(
        out_path,
        # Genre matrices
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=np.array(vocab, dtype=object),
        # Track metadata
        track_ids=np.array(track_ids, dtype=object),
        track_artists=np.array(track_artists, dtype=object),
        track_titles=np.array(track_titles, dtype=object),
        artist_keys=np.array(artist_keys, dtype=object),
        durations_ms=durations_ms,
        # Build metadata. The sonic space is NOT baked here: stage_artifacts
        # folds the active variant sidecar (X_sonic_muq*) immediately after,
        # which also stamps X_sonic_variant. (SP-B)
        build_config={
            'random_seed': args.random_seed,
            'extraction_method': 'universe_gate_beat3tower_features',
            'genre_normalization': genre_stats["normalization_applied"],
            'genre_stats': genre_stats,
            'genre_source': genre_source.value,
        },
    )
```

Keep any variables the retained code still needs; delete newly-unused args/imports (`--no-pca`, `--pca-variance`, `--clip-sigma` argparse entries and their `stage_artifacts` `args_ns` fields in `scripts/analyze_library.py` if they fed only the tower path — grep each name; update `args_ns` in the same commit). Update the module docstring: this script builds the genre+metadata artifact skeleton over the `sonic_features` universe; the sonic space comes from the variant fold.

- [ ] **Step 2: Split `tests/test_beat3tower_fallback.py`**

Read it. Keep tests exercising `SonicFeaturePipeline`/`LibrosaAnalyzer`/`use_beat3tower` extraction modes (that subsystem stays); delete tests exercising the artifact bake/tower matrices. If all three are extraction-side, keep the file whole.

- [ ] **Step 3: Suite + smoke-import**

Run: `python -m pytest -q -m "not slow"` (timeout 600000) — PASS.
Run: `python -c "from scripts.build_beat3tower_artifacts import build_artifacts; print('import ok')"` — no ImportError.

- [ ] **Step 4: Ruff + commit**

```bash
ruff check scripts/build_beat3tower_artifacts.py scripts/analyze_library.py
git add scripts/build_beat3tower_artifacts.py scripts/analyze_library.py tests/test_beat3tower_fallback.py
git commit -m "feat(sp-b): artifact build bakes genre+metadata only — tower extraction, normalizer, sonic bake, bpm_array removed"
```

---

### Task 9: Config + docs sweep

**Files:**
- Modify: `config.example.yaml` (153-170, 700), live `config.yaml`, `CLAUDE.md`, `docs/CLEANUP_LIST.md` (append-only)

- [ ] **Step 1: config.example.yaml**

Delete `playlists.ds_pipeline.tower_weights` (153-156), `transition_weights` (158-163), `tower_pca_dims` (165-170), `playlists.sonic.sim_variant` (700). Update the `artifacts.sonic_variant_override` commented example (815-825): value `muq`, text noting muq is the default and the seam's future use.

- [ ] **Step 2: live config.yaml (gitignored — edit in place, no git)**

Remove the same keys (`transition_weights` at ~342-345, any `tower_weights`/`tower_pca_dims`/`sim_variant`/`analyze.mert` blocks); KEEP `artifacts.sonic_variant_override: muq`. Print the removed lines in the task report.

- [ ] **Step 3: CLAUDE.md (shared file — explicit staging, leave unrelated hunks alone)**

- Key-paths: mark the MERT shard/sidecar/calibration entries as archived-by-SP-B (final wording lands in Task 11 when the move actually happens; here, note "rollback data; archived under `data/archive/mert_2026/` after SP-B").
- Delete the gotcha "Don't change `transition_weights` without also changing `tower_weights`..." block.
- Rewrite the "0.20/0.50/0.30 tower weighting is baked into the `tower_weighted` artifact" gotcha → a short "Sonic space is MuQ (`X_sonic_muq`, 512-dim, `sonic_variant_override: muq`); towers/MERT removed by SP-B 2026-07-01; future embeddings slot in via the variant seam" note.
- Design-principles Layer 3 items 17/18 (tower weights / transition_weights alignment): mark as historical — replaced by the single MuQ space (keep the lesson one-line, drop the live guidance).
- [ ] **Step 4: docs/CLEANUP_LIST.md — append-only (in-flight file!)**

`git status` first; append (do not reflow) a completion note under the SP-B entry + a new SP-C entry (retire Beat3Tower extraction: dedicated BPM/onset/pace extractor, migrate off `tracks.sonic_features`, rewrite universe gate to muq coverage, delete `stage_sonic`+extractor, re-validate pace gates).

- [ ] **Step 5: Commit**

```bash
git add config.example.yaml CLAUDE.md docs/CLEANUP_LIST.md
git commit -m "docs(sp-b): config keys + CLAUDE.md + cleanup list — tower/mert knobs gone, muq is the documented sonic space"
```

---

### Task 10: Rebuild the artifact + acceptance gates

**Files:**
- Create: `scripts/research/spb_artifact_checks.py`
- Data: `data/artifacts/beat3tower_32k/data_matrices_step1.npz` (backup + rebuild)

**Precondition:** Tasks 1-9 committed, full fast suite green. This task runs in the MAIN checkout only.

- [ ] **Step 1: Baseline generation (new code + OLD artifact)**

Pick three stable seeds (SP3's validation pair works):

```bash
python - <<'PY'
import sqlite3, json
con = sqlite3.connect("file:data/metadata.db?mode=ro", uri=True)
rows = con.execute("""SELECT track_id, artist, norm_title FROM tracks
  WHERE (artist='Beach House' AND norm_title IN ('myth','space song'))
     OR (artist='Slowdive' AND norm_title='alison') ORDER BY artist, norm_title""").fetchall()
print(json.dumps(rows, indent=1))
PY
```

Then generate and save the baseline (INFO log captured per the playlist-testing skill):

```python
# .superpowers/sdd/spb_baseline_gen.py  (run: python .superpowers/sdd/spb_baseline_gen.py old)
import sys, json, logging
sys.path.insert(0, "tests")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from support.gui_fidelity import generate_like_gui
SEEDS = ["<the three track_ids from the query>"]
res = generate_like_gui(seeds=SEEDS, cohesion_mode="dynamic", genre_mode="dynamic",
                        sonic_mode="dynamic", pace_mode="dynamic",
                        artist_spacing="strong", length=30, random_seed=0)
out = f".superpowers/sdd/spb_tracklist_{sys.argv[1]}.json"
json.dump(list(res["track_ids"]), open(out, "w"), indent=1)
print("saved", out, len(res["track_ids"]))
```

(Adapt the result-key access to `generate_like_gui`'s actual return shape — read `tests/support/gui_fidelity.py` first.) Verify the log shows `BPM loaded: N/N` and the muq sonic space.

- [ ] **Step 2: Backup, then rebuild**

```bash
python - <<'PY'
import shutil, time
src = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
dst = src.replace(".npz", f".bak_spb_{time.strftime('%Y%m%d_%H%M%S')}.npz")
shutil.copy2(src, dst); print("backup:", dst)
PY
python scripts/analyze_library.py --stages artifacts,verify --force
```

Expected: build (genre+metadata) → "Folding MuQ sidecar..." → "MuQ fold complete; X_sonic_variant=muq" → verify green. **Assert verify's variant check EXECUTED**: its log line must appear (or, on mismatch, `sonic_variant_mismatch` would be an issue) — quote the log line in the report.

- [ ] **Step 3: Artifact assertions**

Create + run `scripts/research/spb_artifact_checks.py`:

```python
"""SP-B acceptance: rebuilt artifact carries per-variant keys only, same universe."""
import sys
import numpy as np
import zipfile

new_p = "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
bak_p = sys.argv[1]  # the .bak_spb_* path printed in Step 2

def keys(p):
    with zipfile.ZipFile(p) as z:
        return {n[:-4] for n in z.namelist() if n.endswith(".npy")}

new_keys = keys(new_p)
DEAD = {k for k in new_keys if k.startswith(("X_sonic_mert", "X_sonic_rhythm", "X_sonic_timbre",
        "X_sonic_harmony", "X_sonic_tower", "X_sonic_raw", "X_sonic_pre", "X_sonic_robust",
        "mert_", "tower_", "normalizer_params", "bpm_array"))}
assert not DEAD, f"dead keys survived: {sorted(DEAD)}"
assert "X_sonic" not in new_keys, "plain X_sonic must be gone (per-variant contract)"
for k in ("X_sonic_muq", "X_sonic_variant", "X_genre_raw", "X_genre_smoothed", "track_ids"):
    assert k in new_keys, f"missing required key {k}"

with np.load(new_p, allow_pickle=True) as znew, np.load(bak_p, allow_pickle=True) as zbak:
    assert np.array_equal(znew["track_ids"], zbak["track_ids"]), "TRACK UNIVERSE CHANGED"
    assert str(znew["X_sonic_variant"].item() if znew["X_sonic_variant"].shape == () else znew["X_sonic_variant"]) == "muq"
    n = znew["track_ids"].shape[0]
    assert znew["X_sonic_muq"].shape == (n, 512), znew["X_sonic_muq"].shape
    assert np.array_equal(znew["X_sonic_muq"], zbak["X_sonic_muq"]), "muq matrix changed — must be byte-identical"
print(f"OK: {n} tracks, muq-only artifact, universe + muq matrix identical to backup")
```

Expected: `OK: ...`.

- [ ] **Step 4: Post-rebuild generation identity + GUI smoke**

Run the Step 1 script again with arg `new`; then:

```bash
python - <<'PY'
import json
a = json.load(open(".superpowers/sdd/spb_tracklist_old.json"))
b = json.load(open(".superpowers/sdd/spb_tracklist_new.json"))
assert a == b, f"TRACKLIST CHANGED:\nold={a}\nnew={b}"
print(f"identical tracklists ({len(a)} tracks)")
PY
```

Expected: identical. Any diff is a defect — STOP and diagnose (per spec §11), do not rationalize.
GUI smoke: restart `serve_web.py`, confirm the Tools stage list has no `mert`, and one generation completes end-to-end.

- [ ] **Step 5: Commit the checks script**

```bash
git add scripts/research/spb_artifact_checks.py
git commit -m "test(sp-b): artifact acceptance checks — muq-only keys, identical universe + muq matrix, tracklist identity verified"
```

---

### Task 11: Archive MERT data (Dylan-gated) + wrap-up

**Precondition:** Task 10 acceptance passed. **STOP and ask Dylan for explicit confirmation before moving anything** (data-safety second-confirmation rule). Nothing is deleted at any point.

- [ ] **Step 1: With Dylan's confirmation, move (same volume, `shutil.move`) to `data/archive/mert_2026/`:**

`mert_shards/`, `mert_sidecar.npz` + every `mert_sidecar.npz.bak.*`, `mert_transform_calibration.npz` (+bak), `harmony_2dftm_sidecar.npz` (+`.tmp.npz`), `mert_shards_a/`, `mert_shards_b/`, `mert_layers_seeds.npz` — all from `data/artifacts/beat3tower_32k/`. Print a before/after listing; verify byte counts match per file.

- [ ] **Step 2: Post-archive verification**

Re-run: `python scripts/analyze_library.py --stages verify` (green — the guard keys off `muq_sidecar.npz`, which stayed). Re-run one generation. Finalize the CLAUDE.md key-paths wording (Task 9 left it provisional).

- [ ] **Step 3: Ledger + close**

Append Task-by-task completion to `.superpowers/sdd/progress.md`; commit CLAUDE.md wording fix. The old artifact backup (`.bak_spb_*`) may move to the archive dir too — ask Dylan in the same confirmation.
