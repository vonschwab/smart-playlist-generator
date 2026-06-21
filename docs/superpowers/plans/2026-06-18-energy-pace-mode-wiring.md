# Energy pace_mode wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the validated energy signal into the beam as two SOFT pace terms (always-on adjacent-step cap + per-mode arc-band), so `pace_mode` actually steers pace, with a configurable arousal-led representation and a hard guarantee that pace never causes infeasibility.

**Architecture:** A new `energy_loader` builds a z-scored per-track energy matrix from the sidecar (configurable features, default `[arousal_p50]`). `pace_gate` gains two pure helpers — an interpolated pier→pier target and a penalty function. The beam calls the penalty helper per candidate, accumulating into the existing `_pace_penalty` (subtracted from `combined_score`). Penalties only down-rank — never exclude — so a segment is always buildable in every mode.

**Tech Stack:** Python 3.11+, numpy, sqlite (none new). Energy read from the prebuilt sidecar npz (no essentia at runtime). pytest/ruff/mypy.

## Global Constraints

- **NEVER hard-fail on pace.** Both terms are SOFT penalties added to `_pace_penalty` — NO `continue`/candidate-reject, NO admission-pool filter. A candidate is only down-ranked, never excluded → a segment is buildable in EVERY mode (incl. strict). This is the onset/BPM hard-gate lesson (stranded segments → cascade → "took FOREVER"; 90s ceiling).
- **NaN-safe:** a track missing energy (sidecar gap, or a pier) → that term contributes **0** for that edge (never NaN into the score, never a reject).
- **Backward-compatible:** all new config fields default `0.0` (strengths) → terms are no-ops → current behavior preserved until a `pace_mode` preset enables them.
- **Representation:** configurable feature list `analyze.pace.energy_features`, **default `["arousal_p50"]`** (arousal-led, per the head-probe findings — danceability off by default, 0.89 MERT-redundant; texture/mood heads excluded). z-scored library-wide; Euclidean distance.
- **Preset caps/bands are z-std units calibrated to the arousal_p50 distribution** (Pass-1: adjacent p50 ≈ 0.40, cross-register ≈ 1.22). Values below are STARTING points; the eval-gate tunes them.
- **No essentia at runtime.** Energy comes from `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz`.
- **Eval-gate before default-on:** per-mode multi-pier generation through the playlist-testing harness (completes in all modes = never-hard-fail; energy curve tightens strict→off) + blind A/B (energy-on vs -off). Ship behind the knobs.
- **pytest:** `python -m pytest <path> -q`; never pipe through tail/head. ruff (E,F) + mypy clean. Worktree branch `worktree-pace-energy-steering`; commit per task.

## File Structure
- **Create** `src/playlist/energy_loader.py` — `load_energy_matrix`. [Task 1]
- **Modify** `src/playlist/pier_bridge/pace_gate.py` — `compute_step_energy_target` + `compute_energy_pace_penalty`. [Task 2]
- **Modify** `src/playlist/pier_bridge/config.py` — 4 energy fields. [Task 3]
- **Modify** `src/playlist/mode_presets.py` — per-mode energy values. [Task 3]
- **Modify** `config.example.yaml` — `analyze.pace.energy_features`. [Task 3]
- **Modify** `src/playlist/pier_bridge/beam.py` — `energy_matrix` kwarg + call the penalty helper. [Task 4]
- **Modify** `src/playlist/pier_bridge_builder.py` + `src/playlist/pipeline/core.py` — load + thread `energy_matrix`. [Task 5]
- **Tests:** `tests/unit/test_energy_loader.py`, `test_pace_gate_energy.py`, `test_beam_energy_pace.py`, `test_pace_mode_energy_presets.py`. [Tasks 1–4]

---

### Task 1: `energy_loader.py`

**Files:**
- Create: `src/playlist/energy_loader.py`
- Test: `tests/unit/test_energy_loader.py`

**Interfaces:**
- Produces: `load_energy_matrix(track_ids, *, sidecar_path, features=("arousal_p50",)) -> np.ndarray` shape `(len(track_ids), len(features))`, z-scored library-wide per feature; NaN row for a track absent from the sidecar or with a NaN feature.

- [ ] **Step 1: Write the failing test** `tests/unit/test_energy_loader.py`:

```python
import numpy as np
from src.playlist.energy_loader import load_energy_matrix


def _make_sidecar(tmp_path):
    p = tmp_path / "energy_sidecar.npz"
    np.savez(
        p,
        track_ids=np.array(["a", "b", "c", "d"], dtype=object),
        arousal_p50=np.array([2.0, 4.0, 6.0, np.nan], dtype=np.float32),
        danceability=np.array([0.1, 0.5, 0.9, 0.5], dtype=np.float32),
    )
    return str(p)


def test_zscored_shape_and_values(tmp_path):
    side = _make_sidecar(tmp_path)
    m = load_energy_matrix(["a", "b", "c"], sidecar_path=side, features=("arousal_p50",))
    assert m.shape == (3, 1)
    # arousal [2,4,6] over the library (a,b,c finite; d is NaN, ignored) -> mean 4, so b -> 0
    assert abs(float(m[1, 0])) < 1e-6
    assert float(m[0, 0]) < 0 < float(m[2, 0])


def test_missing_track_and_nan_feature_are_nan_rows(tmp_path):
    side = _make_sidecar(tmp_path)
    m = load_energy_matrix(["a", "zzz", "d"], sidecar_path=side, features=("arousal_p50", "danceability"))
    assert m.shape == (3, 2)
    assert np.all(np.isfinite(m[0]))          # a present
    assert np.all(np.isnan(m[1]))             # zzz absent -> NaN row
    assert np.isnan(m[2, 0]) and np.isfinite(m[2, 1])  # d: arousal NaN, dance finite
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_energy_loader.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** `src/playlist/energy_loader.py`:

```python
"""Load a z-scored per-track energy matrix from the energy sidecar.

Runtime-only consumer of the Essentia energy sidecar (no essentia import).
Mirrors bpm_loader: returns arrays aligned to the requested track_ids, NaN for
gaps. Library-wide z-score so distances are in std units (matches the eval).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _zscore_params(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    mean = float(finite.mean())
    std = float(finite.std())
    return (mean, std if std > 0 else 1.0)


def load_energy_matrix(
    track_ids: Sequence[str],
    *,
    sidecar_path: str,
    features: Sequence[str] = ("arousal_p50",),
) -> np.ndarray:
    """Return (len(track_ids), len(features)) z-scored energy matrix; NaN rows for gaps."""
    track_ids = [str(t) for t in track_ids]
    n = len(track_ids)
    feats = list(features)
    out = np.full((n, len(feats)), np.nan, dtype=float)

    z = np.load(sidecar_path, allow_pickle=True)
    side_ids = [str(t) for t in z["track_ids"]]
    pos = {t: i for i, t in enumerate(side_ids)}
    for fi, feat in enumerate(feats):
        if feat not in z:
            logger.warning("energy_loader: feature %r absent from sidecar; column left NaN", feat)
            continue
        col = np.asarray(z[feat], dtype=float)
        mean, std = _zscore_params(col)            # library-wide
        for ti, tid in enumerate(track_ids):
            j = pos.get(tid)
            if j is not None and np.isfinite(col[j]):
                out[ti, fi] = (col[j] - mean) / std
    return out
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_energy_loader.py -q` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/energy_loader.py tests/unit/test_energy_loader.py
git commit -m "feat(pace): energy_loader (z-scored energy matrix from sidecar)"
```

---

### Task 2: `pace_gate` energy helpers

**Files:**
- Modify: `src/playlist/pier_bridge/pace_gate.py`
- Test: `tests/unit/test_pace_gate_energy.py`

**Interfaces:**
- Consumes: `src.playlist.sonic_axes.interpolate_axis_vector` (already imported in pace_gate via `compute_step_rhythm_target`).
- Produces:
  - `compute_step_energy_target(e_a, e_b, *, step, segment_length) -> np.ndarray` (linear interp; `t = step/segment_length`).
  - `compute_energy_pace_penalty(energy_matrix, *, current, cand, pier_a, pier_b, step, segment_length, step_cap, step_strength, arc_band, arc_strength) -> float` — the SOFT penalty (>= 0). Returns 0.0 when `energy_matrix is None` or any needed row is non-finite. NEVER raises, NEVER signals exclusion.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_pace_gate_energy.py`:

```python
import numpy as np
from src.playlist.pier_bridge.pace_gate import (
    compute_step_energy_target,
    compute_energy_pace_penalty,
)


def test_step_energy_target_linear_midpoint():
    t = compute_step_energy_target(np.array([0.0, 0.0]), np.array([4.0, 2.0]), step=1, segment_length=2)
    assert np.allclose(t, [2.0, 1.0])


def _em(rows):
    return np.array(rows, dtype=float)


def test_step_cap_fires_above_cap_only():
    # current row0, cand row1 distance = 2.0 ; cap 1.0, strength 0.5 -> 0.5*(2-1)=0.5
    em = _em([[0.0], [2.0], [0.0], [0.0]])
    pen = compute_energy_pace_penalty(em, current=0, cand=1, pier_a=2, pier_b=3,
                                      step=0, segment_length=1, step_cap=1.0,
                                      step_strength=0.5, arc_band=99.0, arc_strength=0.0)
    assert abs(pen - 0.5) < 1e-9
    # within cap -> no penalty
    pen2 = compute_energy_pace_penalty(em, current=0, cand=2, pier_a=2, pier_b=3,
                                       step=0, segment_length=1, step_cap=1.0,
                                       step_strength=0.5, arc_band=99.0, arc_strength=0.0)
    assert pen2 == 0.0


def test_arc_band_penalizes_distance_from_target():
    # piers 0 and 4 ; step0/len1 target=0 ; cand=4 -> arc dist 4 ; band 1, strength 0.5 -> 1.5
    em = _em([[0.0], [4.0], [0.0], [4.0]])  # pier_a=row0(0), pier_b=row3(4)? use explicit rows
    em = _em([[0.0], [4.0]])
    pen = compute_energy_pace_penalty(em, current=0, cand=1, pier_a=0, pier_b=1,
                                      step=0, segment_length=2, step_cap=99.0,
                                      step_strength=0.0, arc_band=1.0, arc_strength=0.5)
    # target at step0 = pier_a (0); cand=row1=4 -> arc dist 4 -> 0.5*(4-1)=1.5
    assert abs(pen - 1.5) < 1e-9


def test_nan_and_none_are_zero_never_raise():
    assert compute_energy_pace_penalty(None, current=0, cand=1, pier_a=0, pier_b=1,
                                       step=0, segment_length=1, step_cap=0.1,
                                       step_strength=1.0, arc_band=0.1, arc_strength=1.0) == 0.0
    em = _em([[np.nan], [2.0]])
    assert compute_energy_pace_penalty(em, current=0, cand=1, pier_a=0, pier_b=1,
                                       step=0, segment_length=1, step_cap=0.1,
                                       step_strength=1.0, arc_band=0.1, arc_strength=1.0) == 0.0
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_gate_energy.py -q` → FAIL.

- [ ] **Step 3: Implement** — append to `src/playlist/pier_bridge/pace_gate.py`:

```python
def compute_step_energy_target(
    e_a: np.ndarray,
    e_b: np.ndarray,
    *,
    step: int,
    segment_length: int,
) -> np.ndarray:
    """Linear pier->pier energy target at beam step `step` (energy is a linear scale)."""
    if int(segment_length) <= 0:
        return np.asarray(e_a, dtype=float)
    t = max(0.0, min(1.0, float(step) / float(segment_length)))
    return interpolate_axis_vector(e_a, e_b, t)


def compute_energy_pace_penalty(
    energy_matrix,
    *,
    current: int,
    cand: int,
    pier_a: int,
    pier_b: int,
    step: int,
    segment_length: int,
    step_cap: float,
    step_strength: float,
    arc_band: float,
    arc_strength: float,
) -> float:
    """SOFT pace penalty (>= 0) for placing `cand` after `current`.

    Two terms: adjacent-step cap (energy distance current->cand) and arc-band
    (distance from the interpolated pier->pier target). NaN/None -> 0.0.
    NEVER raises and NEVER signals exclusion — callers only subtract this.
    """
    if energy_matrix is None:
        return 0.0
    penalty = 0.0
    e_cand = energy_matrix[int(cand)]
    if not np.all(np.isfinite(e_cand)):
        return 0.0
    # adjacent-step cap
    if step_strength > 0.0:
        e_cur = energy_matrix[int(current)]
        if np.all(np.isfinite(e_cur)):
            d_step = float(np.linalg.norm(e_cand - e_cur))
            if d_step > step_cap:
                penalty += step_strength * (d_step - step_cap)
    # arc-band
    if arc_strength > 0.0:
        e_a, e_b = energy_matrix[int(pier_a)], energy_matrix[int(pier_b)]
        if np.all(np.isfinite(e_a)) and np.all(np.isfinite(e_b)):
            target = compute_step_energy_target(e_a, e_b, step=step, segment_length=segment_length)
            d_arc = float(np.linalg.norm(e_cand - target))
            if d_arc > arc_band:
                penalty += arc_strength * (d_arc - arc_band)
    return penalty
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_gate_energy.py -q` → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/pace_gate.py tests/unit/test_pace_gate_energy.py
git commit -m "feat(pace): energy target + soft penalty helpers in pace_gate"
```

---

### Task 3: config fields + presets + config.example

**Files:**
- Modify: `src/playlist/pier_bridge/config.py`
- Modify: `src/playlist/mode_presets.py`
- Modify: `config.example.yaml`
- Test: `tests/unit/test_pace_mode_energy_presets.py`

**Interfaces:**
- Produces: `PierBridgeConfig` fields `energy_step_cap`, `energy_step_strength`, `energy_arc_band`, `energy_arc_strength` (all `float = 0.0`). `PACE_MODE_PRESETS[mode]` carries the 4 keys for `strict|narrow|dynamic|off`. `config.example.yaml` has `analyze.pace.energy_features`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_pace_mode_energy_presets.py`:

```python
from src.playlist.mode_presets import PACE_MODE_PRESETS
from src.playlist.pier_bridge.config import PierBridgeConfig

KEYS = ["energy_step_cap", "energy_step_strength", "energy_arc_band", "energy_arc_strength"]


def test_config_defaults_off():
    c = PierBridgeConfig()
    for k in KEYS:
        assert getattr(c, k) == 0.0


def test_presets_have_energy_keys_and_off_disables_arc():
    for mode in ("strict", "narrow", "dynamic", "off"):
        for k in KEYS:
            assert k in PACE_MODE_PRESETS[mode], f"{mode} missing {k}"
    # always-on step cap (anti-whiplash) even at off
    assert PACE_MODE_PRESETS["off"]["energy_step_strength"] > 0.0
    # off disables the arc term
    assert PACE_MODE_PRESETS["off"]["energy_arc_strength"] == 0.0
    # strict is the tightest step cap
    assert PACE_MODE_PRESETS["strict"]["energy_step_cap"] < PACE_MODE_PRESETS["off"]["energy_step_cap"]
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_mode_energy_presets.py -q` → FAIL.

- [ ] **Step 3a: Add fields to `PierBridgeConfig`** (`src/playlist/pier_bridge/config.py`) — alongside the existing `bpm_bridge_*` / `onset_bridge_*` fields, add:

```python
    energy_step_cap: float = 0.0
    energy_step_strength: float = 0.0
    energy_arc_band: float = 0.0
    energy_arc_strength: float = 0.0
```

- [ ] **Step 3b: Add per-mode values in `PACE_MODE_PRESETS`** (`src/playlist/mode_presets.py`) — add these keys to each pace mode dict (starting values, z-std units from the arousal_p50 distribution; tuned in the eval-gate):

```python
# strict
    "energy_step_cap": 0.4, "energy_step_strength": 0.5,
    "energy_arc_band": 0.4, "energy_arc_strength": 0.5,
# narrow
    "energy_step_cap": 0.6, "energy_step_strength": 0.4,
    "energy_arc_band": 0.7, "energy_arc_strength": 0.3,
# dynamic
    "energy_step_cap": 0.9, "energy_step_strength": 0.3,
    "energy_arc_band": 1.1, "energy_arc_strength": 0.15,
# off  (anti-whiplash floor only; arc disabled)
    "energy_step_cap": 1.2, "energy_step_strength": 0.2,
    "energy_arc_band": 0.0, "energy_arc_strength": 0.0,
```

- [ ] **Step 3c: Add to `config.example.yaml`** under the existing `analyze:` block, a `pace:` sibling:

```yaml
analyze:
  pace:
    energy_features: ["arousal_p50"]   # configurable; danceability/arousal_p10/p90 available
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_mode_energy_presets.py -q` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist/mode_presets.py config.example.yaml tests/unit/test_pace_mode_energy_presets.py
git commit -m "feat(pace): energy_* config fields + per-mode presets + config.example"
```

---

### Task 4: beam wiring + beam test

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py`
- Test: `tests/unit/test_beam_energy_pace.py`

**Interfaces:**
- Consumes: `pace_gate.compute_energy_pace_penalty`; the loop vars `current`, `cand`, `pier_a`, `pier_b`, `step`, `interior_length`, `_pace_penalty`.
- Produces: `_beam_search_segment(..., energy_matrix: Optional[np.ndarray] = None)`; the energy penalty folded into `_pace_penalty`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_beam_energy_pace.py` — mirror the existing `tests/unit/test_beam_pace_soft_penalty.py` setup (use its `_flat()`/segment scaffolding). Core assertions:

```python
# (Reuse the segment-setup helper from test_beam_pace_soft_penalty.py: build a
#  small two-pier segment with N interior candidates, identical sonic vectors so
#  only the energy term differentiates them.)
import numpy as np
from tests.unit._beam_pace_helpers import run_one_segment  # shared scaffold (see note)

def test_energy_step_cap_demotes_big_jump_but_never_excludes():
    # candidates c_lo (energy near piers) and c_hi (huge energy jump)
    # with energy_step_strength>0, c_lo must be chosen over c_hi...
    res_lo, res_hi_only = run_one_segment(...)
    assert res_lo.picked == "c_lo"
    # ...but if c_hi is the ONLY candidate, the segment STILL builds (never hard-fail)
    assert res_hi_only.path is not None and res_hi_only.beam_failure_reason is None
```

If `test_beam_pace_soft_penalty.py` has no reusable scaffold, inline the same `_beam_search_segment(...)` call it uses (copy its fixture: `X_full_norm`, `candidates`, `pier_a/pier_b`, a `PierBridgeConfig` with `energy_step_cap`/`energy_step_strength` set), passing a synthetic `energy_matrix`. The two assertions are the contract: (1) the big-energy-jump candidate is demoted; (2) with only the big-jump candidate available, the segment still returns a path (never-hard-fail).

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_beam_energy_pace.py -q` → FAIL (`energy_matrix` kwarg unknown / not applied).

- [ ] **Step 3a: Add the kwarg** to `_beam_search_segment` (`beam.py` ~line 241, next to `onset_rate`):

```python
    energy_matrix: Optional[np.ndarray] = None,
```

- [ ] **Step 3b: Apply the penalty** in the candidate loop, immediately after the onset-band block and BEFORE `combined_score` is built (i.e., right after the onset block that ends ~line 1130; `_pace_penalty` is already initialized at ~1052 and subtracted at ~1199). Insert:

```python
                if energy_matrix is not None:
                    from src.playlist.pier_bridge.pace_gate import compute_energy_pace_penalty
                    _pace_penalty += compute_energy_pace_penalty(
                        energy_matrix,
                        current=int(current), cand=int(cand),
                        pier_a=int(pier_a), pier_b=int(pier_b),
                        step=step, segment_length=interior_length,
                        step_cap=float(getattr(cfg, "energy_step_cap", 0.0)),
                        step_strength=float(getattr(cfg, "energy_step_strength", 0.0)),
                        arc_band=float(getattr(cfg, "energy_arc_band", 0.0)),
                        arc_strength=float(getattr(cfg, "energy_arc_strength", 0.0)),
                    )
```
(Purely additive into `_pace_penalty` — no `continue`, so it can never exclude a candidate.)

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_beam_energy_pace.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/beam.py tests/unit/test_beam_energy_pace.py
git commit -m "feat(pace): apply soft energy step-cap + arc-band in beam (never excludes)"
```

---

### Task 5: load + thread `energy_matrix` (core + builder)

**Files:**
- Modify: `src/playlist/pipeline/core.py`
- Modify: `src/playlist/pier_bridge_builder.py`
- Test: `tests/unit/test_energy_threading.py`

**Interfaces:**
- Consumes: `energy_loader.load_energy_matrix`; the builder's `_beam_search_segment` call (~line 1369) and signature (~line 344).
- Produces: builder param `energy_matrix=None` threaded to `_beam_search_segment`; `core.py` loads `energy_matrix` when any energy strength > 0 in `pace_settings` and passes it to the builder.

- [ ] **Step 1: Write the failing test** `tests/unit/test_energy_threading.py` (verifies the builder accepts + forwards `energy_matrix` without invoking a full generation):

```python
import inspect
from src.playlist import pier_bridge_builder as b


def test_builder_accepts_energy_matrix_param():
    sig = inspect.signature(b.build_pier_bridge_playlist)  # the builder entrypoint
    assert "energy_matrix" in sig.parameters
```
(If the builder entrypoint has a different name, use it; the assertion is that `energy_matrix` is a parameter.)

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_energy_threading.py -q` → FAIL.

- [ ] **Step 3a: Builder** (`pier_bridge_builder.py`): add `energy_matrix: Optional[np.ndarray] = None,` to the builder function signature next to `onset_rate` (~line 346); in the `_beam_search_segment(...)` call (~line 1369) add `energy_matrix=energy_matrix,` next to `onset_rate=onset_rate`.

- [ ] **Step 3b: core** (`pipeline/core.py`): after the BPM arrays load block (~line 320), add energy loading gated on an active energy strength in `pace_settings`:

```python
    energy_matrix = None
    _energy_active = any(
        float(pace_settings.get(k, 0.0)) > 0.0
        for k in ("energy_step_strength", "energy_arc_strength")
    )
    if _energy_active:
        try:
            from src.playlist.energy_loader import load_energy_matrix
            _energy_feats = tuple(
                ((overrides or {}).get("analyze", {}).get("pace", {}) or {}).get(
                    "energy_features", ["arousal_p50"]
                )
            )
            _sidecar = str(bundle_dir / "energy" / "energy_sidecar.npz") \
                if (bundle_dir := __import__("pathlib").Path(bundle.artifact_path).parent) else ""
            energy_matrix = load_energy_matrix(bundle.track_ids, sidecar_path=_sidecar, features=_energy_feats)
            logger.info("energy loaded: %d/%d tracks", int(np.sum(np.all(np.isfinite(energy_matrix), axis=1))), energy_matrix.shape[0])
        except Exception:
            logger.warning("energy load failed; pace energy terms disabled", exc_info=True)
            energy_matrix = None
```
Then pass `energy_matrix=energy_matrix` in BOTH the builder call (~line 643) and any direct beam call. (Resolve `bundle.artifact_path`/`bundle_dir` to whatever the bundle exposes for the artifact directory; the sidecar is `<artifact_dir>/energy/energy_sidecar.npz`.) Also add `energy_step_cap`/`energy_step_strength`/`energy_arc_band`/`energy_arc_strength` to the `replace(pb_cfg, ...)` block (~line 492) from `pace_settings` (mirroring the `bpm_bridge_soft_penalty_strength` line).

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_energy_threading.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pipeline/core.py src/playlist/pier_bridge_builder.py tests/unit/test_energy_threading.py
git commit -m "feat(pace): load + thread energy_matrix (core -> builder -> beam)"
```

---

### Task 6: gates + generation smoke + learnings note (controller)

**Files:** none new (verification).

- [ ] **Step 1: Lint + types + new tests** — `ruff check src/playlist/energy_loader.py src/playlist/pier_bridge/pace_gate.py src/playlist/pier_bridge/config.py src/playlist/mode_presets.py && mypy src/playlist/energy_loader.py src/playlist/pier_bridge/pace_gate.py && python -m pytest tests/unit/test_energy_loader.py tests/unit/test_pace_gate_energy.py tests/unit/test_beam_energy_pace.py tests/unit/test_pace_mode_energy_presets.py tests/unit/test_energy_threading.py -q` → all green.
- [ ] **Step 2: Regression** — `python -m pytest tests/unit/test_beam_pace_soft_penalty.py tests/unit/test_beam_bpm_trust.py -q` → still green (energy additions didn't disturb the existing pace bands).
- [ ] **Step 3: Generation smoke (playlist-testing skill — mandatory; never-hard-fail check).** Multi-pier artist seeds through the `generate_like_gui`/`gui_fidelity` harness, once per `pace_mode` (strict/narrow/dynamic/off), with `analyze.pace.energy_features: ["arousal_p50"]` and the energy presets active. Confirm: (a) **every mode completes** within the 90s budget (never-hard-fail); (b) the realized energy curve (arousal_p50 of the ordered tracks) tightens strict→off and shows no whiplash even at `off`. Capture the per-mode curves to `docs/run_audits/pace_energy_wiring/`.
- [ ] **Step 4: Update the learnings log** `docs/PACE_AXIS_LEARNINGS.md` (master): wiring shipped behind knobs (default off); the eval-gate (blind A/B energy-on vs -off) is the remaining gate before default-on; representation default `arousal_p50` pending the Pass-2 blind session.
- [ ] **Step 5: Commit** the research artifacts + learnings note (run_audits is gitignored; commit the learnings log on master separately).

---

## Self-Review

**Spec coverage:** two soft terms (step-cap always-on + arc-band per-mode) → Tasks 2+4; configurable arousal-led representation → Tasks 1+3; never-hard-fail (soft-only, NaN-safe, no pool gate) → Task 2 helper (additive, returns 0 on NaN) + Task 4 (no `continue`) + Task 6 Step 3 (all modes complete); reuse pace_gate interp + edge-penalty site → Tasks 2+4; energy_loader mirrors bpm_loader → Task 1; presets/config/threading → Tasks 3+5; eval-gate → Task 6. ✓

**Placeholder scan:** Task 4's test references the existing `test_beam_pace_soft_penalty.py` scaffold rather than re-deriving the heavy `_beam_search_segment` fixture — the implementer copies that file's setup (the one genuinely large fixture in the repo); the contract (demote-not-exclude + builds-with-only-bad-candidate) is concrete. Task 5 Step 3b leaves the exact `bundle` artifact-dir attribute to resolve against the real bundle — flagged explicitly as the one lookup to confirm in-repo. All other steps have complete code.

**Type consistency:** `load_energy_matrix(track_ids, *, sidecar_path, features)`, `compute_step_energy_target(e_a,e_b,*,step,segment_length)`, `compute_energy_pace_penalty(energy_matrix,*,current,cand,pier_a,pier_b,step,segment_length,step_cap,step_strength,arc_band,arc_strength)`, the 4 `energy_*` config fields, and `energy_matrix` kwarg — names/signatures match across Tasks 1–5.
