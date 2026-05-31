# Genre Edge Safeguards & Steering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reintroduce a hard genre floor (safeguard) plus a first-class genre steering term in the pier-bridge beam edge scoring, computed on the dense PMI-SVD embedding, so genre-appropriate tracks displace stylistically-off-but-sonically-plausible tracks in segment interiors.

**Architecture:** All changes live in pier-bridge config resolution + the beam edge scoring; the monotonic-progress sonic mechanism is untouched. A master flag (`genre_steering_enabled`, code-default OFF, config.yaml ON) gates new behavior. When ON: the beam's genre signal switches from sparse `X_genre_smoothed` to `X_genre_dense`; edges below `genre_edge_floor` are rejected (with progressive relaxation when a segment is infeasible); genre becomes a renormalized third weight `weight_genre`; the old soft penalty is bypassed.

**Tech Stack:** Python 3.11, numpy, pytest. Spec: `docs/superpowers/specs/2026-05-30-genre-edge-safeguards-design.md`.

---

## File Structure

- `src/playlist/pier_bridge/config.py` — `PierBridgeConfig`: add 3 fields (`genre_steering_enabled`, `weight_genre`, `genre_edge_floor`).
- `src/playlist/config.py` — `PierBridgeTuning`: add the same 3 fields; `resolve_pier_bridge_tuning`: resolve + renormalize weights.
- `src/playlist/pipeline/pier_bridge_overrides.py` — pass the 3 resolved fields into `PierBridgeConfig`.
- `src/playlist/run_audit.py` — `InfeasibleHandlingConfig`: add `genre_floor_relaxation_enabled`, `min_genre_edge_floor` + parsing.
- `src/playlist/pier_bridge/beam.py` — new `X_genre_dense` param; dense signal repoint; floor gate; steering term; bypass soft penalty; genreless-endpoint fallback. (interior step + final-pier connection)
- `src/playlist/pier_bridge_builder.py` — pass `X_genre_dense` to the beam; genre-floor relaxation in the segment backoff.
- `config.yaml` — `genre_steering_enabled: true` + per-mode `weight_genre_*`, `genre_edge_floor_*`, relaxation defaults.
- Tests: `tests/unit/test_genre_edge_steering.py` (new), extend `tests/unit/test_pmi_svd.py`-style config tests.

---

## Task 1: Config knobs — PierBridgeConfig + PierBridgeTuning + resolution

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (PierBridgeConfig dataclass, after line 68)
- Modify: `src/playlist/config.py` (PierBridgeTuning ~line 114; resolve_pier_bridge_tuning ~line 300-318)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (~line 95-105)
- Test: `tests/unit/test_genre_edge_steering.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_genre_edge_steering.py`:

```python
import numpy as np
import pytest

from src.playlist.config import resolve_pier_bridge_tuning


def test_tuning_genre_steering_defaults_off():
    """No overrides -> steering off, inert genre knobs, legacy weights intact."""
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=None)
    assert t.genre_steering_enabled is False
    assert t.weight_genre == 0.0
    assert t.genre_edge_floor == 0.0
    # Legacy narrow weights unchanged when steering off
    assert abs(t.weight_bridge - 0.7) < 1e-6
    assert abs(t.weight_transition - 0.3) < 1e-6


def test_tuning_genre_steering_renormalizes_weights():
    """When enabled with weight_genre, the three edge weights renormalize to sum 1."""
    overrides = {
        "pier_bridge": {
            "genre_steering_enabled": True,
            "weight_genre_narrow": 0.20,
            "genre_edge_floor_narrow": 0.40,
            # leave bridge/transition at narrow defaults 0.7/0.3
        }
    }
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert abs(t.genre_edge_floor - 0.40) < 1e-6
    total = t.weight_bridge + t.weight_transition + t.weight_genre
    assert abs(total - 1.0) < 1e-6
    # genre share is 0.20 / 1.20
    assert abs(t.weight_genre - (0.20 / 1.20)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -q`
Expected: FAIL — `AttributeError: 'PierBridgeTuning' object has no attribute 'genre_steering_enabled'`.

- [ ] **Step 3: Add fields to PierBridgeConfig**

In `src/playlist/pier_bridge/config.py`, immediately after line 68 (`genre_penalty_strength: float = 0.10`), add:

```python
    # Genre edge safeguards & steering (opt-in; code default OFF).
    # When enabled, the beam scores genre on the dense embedding, rejects edges
    # below genre_edge_floor, and adds weight_genre * genre_sim as a third term
    # (bridge/transition/genre weights are pre-renormalized to sum to 1).
    genre_steering_enabled: bool = False
    weight_genre: float = 0.0
    genre_edge_floor: float = 0.0
```

- [ ] **Step 4: Add fields to PierBridgeTuning**

In `src/playlist/config.py`, in the `PierBridgeTuning` dataclass (after `genre_penalty_strength: float` at line 114), add:

```python
    genre_steering_enabled: bool = False
    weight_genre: float = 0.0
    genre_edge_floor: float = 0.0
```

- [ ] **Step 5: Resolve + renormalize in resolve_pier_bridge_tuning**

In `src/playlist/config.py`, replace the `tuning = PierBridgeTuning(...)` construction block (currently lines ~309-317) with:

```python
    genre_steering_enabled = bool(pier_raw.get("genre_steering_enabled", False))
    weight_genre, src = _resolve_mode_number_with_source(
        pier_raw, "weight_genre", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["weight_genre"] = src
    genre_edge_floor, src = _resolve_mode_number_with_source(
        pier_raw, "genre_edge_floor", mode_s, 0.0, source_prefix="pier_bridge"
    )
    sources["genre_edge_floor"] = src

    # When steering is active, genre is a co-equal edge weight: renormalize the
    # (bridge, transition, genre) triple to sum to 1 so the score stays in range.
    if genre_steering_enabled and float(weight_genre) > 0.0:
        _wsum = float(weight_bridge) + float(weight_transition) + float(weight_genre)
        if _wsum > 0:
            weight_bridge = float(weight_bridge) / _wsum
            weight_transition = float(weight_transition) / _wsum
            weight_genre = float(weight_genre) / _wsum

    tuning = PierBridgeTuning(
        transition_floor=float(transition_floor),
        bridge_floor=float(bridge_floor),
        weight_bridge=float(weight_bridge),
        weight_transition=float(weight_transition),
        genre_tiebreak_weight=float(genre_tiebreak_weight),
        genre_penalty_threshold=float(genre_penalty_threshold),
        genre_penalty_strength=float(genre_penalty_strength),
        genre_steering_enabled=bool(genre_steering_enabled),
        weight_genre=float(weight_genre),
        genre_edge_floor=float(genre_edge_floor),
    )
    return tuning, sources
```

- [ ] **Step 6: Pass the fields into PierBridgeConfig**

In `src/playlist/pipeline/pier_bridge_overrides.py`, in the `PierBridgeConfig(...)` construction (around line 95-105, where `genre_tiebreak_weight=float(tuning.genre_tiebreak_weight)` appears), add three kwargs alongside it:

```python
        genre_steering_enabled=bool(tuning.genre_steering_enabled),
        weight_genre=float(tuning.weight_genre),
        genre_edge_floor=float(tuning.genre_edge_floor),
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist/config.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-steering): add genre edge config knobs + weight renormalization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: InfeasibleHandlingConfig — genre-floor relaxation fields

**Files:**
- Modify: `src/playlist/run_audit.py` (`InfeasibleHandlingConfig` dataclass ~line 35; `parse_infeasible_handling_config` ~line 96)
- Test: `tests/unit/test_genre_edge_steering.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_genre_edge_steering.py`:

```python
def test_infeasible_handling_genre_floor_fields_default():
    from src.playlist.run_audit import InfeasibleHandlingConfig, parse_infeasible_handling_config
    cfg = InfeasibleHandlingConfig()
    assert cfg.genre_floor_relaxation_enabled is True
    assert cfg.min_genre_edge_floor == 0.0
    parsed = parse_infeasible_handling_config({
        "enabled": True, "min_genre_edge_floor": 0.15, "genre_floor_relaxation_enabled": False,
    })
    assert parsed.genre_floor_relaxation_enabled is False
    assert abs(parsed.min_genre_edge_floor - 0.15) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py::test_infeasible_handling_genre_floor_fields_default -q`
Expected: FAIL — `AttributeError: ... 'genre_floor_relaxation_enabled'`.

- [ ] **Step 3: Add the fields**

In `src/playlist/run_audit.py`, in `InfeasibleHandlingConfig` (immediately after the existing `min_transition_floor: float = 0.20` line), add:

```python
    genre_floor_relaxation_enabled: bool = True
    min_genre_edge_floor: float = 0.0
```

In `parse_infeasible_handling_config`, alongside the existing `min_transition_floor=...` line, add:

```python
        genre_floor_relaxation_enabled=bool(raw.get("genre_floor_relaxation_enabled", True)),
        min_genre_edge_floor=float(raw.get("min_genre_edge_floor", 0.0)),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py::test_infeasible_handling_genre_floor_fields_default -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/run_audit.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-steering): add genre-floor relaxation fields to InfeasibleHandlingConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Beam — dense signal, floor gate, steering term, soft-penalty bypass

This is the core. Add a `X_genre_dense` param, repoint the genre signal, gate on the floor, add the steering term, bypass the soft penalty, and skip both for genreless endpoints. Apply at the interior step AND the final-pier connection.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (signature ~line 195; `X_genre_for_sim` block ~668-698; interior scoring ~984-1032; final connection ~1412-1432)
- Test: `tests/unit/test_genre_edge_steering.py` (extend — uses the `test_beam_pace_gate.py` harness pattern)

- [ ] **Step 1: Write the failing test (floor rejects the off-genre candidate)**

Append to `tests/unit/test_genre_edge_steering.py`:

```python
from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _diag4():
    # 4 tracks: 0=pierA, 1=genre-OK candidate, 2=genre-misfit candidate, 3=pierB.
    # Sonic: all identical so sonic never decides (forces genre to break the tie).
    X = np.ones((4, 3), dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Dense genre (L2-normalized rows): cand 1 aligns with piers; cand 2 is orthogonal.
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [1.0, 0.0, 0.0],   # cand 1: same genre as piers (sim 1.0)
        [0.0, 1.0, 0.0],   # cand 2: orthogonal genre (sim 0.0)
        [1.0, 0.0, 0.0],   # pierB
    ], dtype=float)
    return Xn, dense


def test_beam_floor_rejects_off_genre_candidate():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.2, genre_edge_floor=0.5,
        weight_bridge=0.5, weight_transition=0.3,
    )
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert path == [1], f"expected genre-OK cand 1, got {path}"


def test_beam_steering_prefers_higher_genre_when_sonic_tied():
    Xn, dense = _diag4()
    # No floor (0.0) so cand 2 is allowed; steering weight should still rank cand 1 first.
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.3, genre_edge_floor=0.0,
        weight_bridge=0.4, weight_transition=0.3,
    )
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert path == [1]


def test_beam_legacy_unchanged_when_steering_off():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=False,
    )
    # Steering off: floor must NOT reject; both candidates valid, no crash.
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert len(path) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -k beam -q`
Expected: FAIL — `TypeError: _beam_search_segment() got an unexpected keyword argument 'X_genre_dense'`.

- [ ] **Step 3: Add the `X_genre_dense` parameter**

In `src/playlist/pier_bridge/beam.py`, in the `_beam_search_segment` signature, add a keyword parameter alongside the other `X_genre_*` kwargs (after `X_genre_smoothed: Optional[np.ndarray] = None,`):

```python
    X_genre_dense: Optional[np.ndarray] = None,
```

- [ ] **Step 4: Repoint the genre signal to dense + compute genre_present**

In `src/playlist/pier_bridge/beam.py`, immediately BEFORE the `vector_source = ...` line (currently ~line 672), insert the dense override and a presence mask. Then the existing block stays as the fallback:

```python
    # Genre-steering: prefer the dense PMI-SVD embedding when enabled + available.
    genre_present = None
    _steering = bool(getattr(cfg, "genre_steering_enabled", False))
    if _steering and X_genre_dense is not None:
        X_genre_for_sim = X_genre_dense  # rows already L2-normalized
        genre_present = np.linalg.norm(X_genre_dense, axis=1) > 1e-9
    else:
        vector_source = str(cfg.dj_genre_vector_source or "smoothed").strip().lower()
        # ... EXISTING block that assigns X_genre_for_sim (raw/smoothed) stays here ...
```

Note: indent the existing `vector_source`/`if vector_source == "raw"...else` block one level deeper so it sits under the new `else:`. `X_genre_for_sim` must be defined in both branches.

- [ ] **Step 5: Add floor gate + steering term in the interior scoring loop**

In `src/playlist/pier_bridge/beam.py`, replace the genre block in the candidate loop (currently ~lines 984-989, the `genre_sim = None ... combined_score += cfg.genre_tiebreak_weight * genre_sim` block) with:

```python
                genre_sim = None
                if X_genre_for_sim is not None:
                    genre_sim = _get_genre_sim(int(current), int(cand))
                _both_present = (
                    genre_present is None
                    or (genre_sim is not None
                        and bool(genre_present[int(current)])
                        and bool(genre_present[int(cand)]))
                )
                if _steering:
                    if genre_sim is not None and math.isfinite(genre_sim) and _both_present:
                        # Hard floor (safeguard): reject egregiously off-genre edges.
                        if genre_sim < float(getattr(cfg, "genre_edge_floor", 0.0)):
                            continue
                        # Steering: genre as a first-class (renormalized) edge weight.
                        if float(getattr(cfg, "weight_genre", 0.0)) > 0.0:
                            combined_score += float(cfg.weight_genre) * genre_sim
                else:
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if cfg.genre_tiebreak_weight:
                            combined_score += cfg.genre_tiebreak_weight * genre_sim
```

- [ ] **Step 6: Bypass the soft penalty when steering**

In `src/playlist/pier_bridge/beam.py`, in the non-tiebreak path (currently ~lines 1029-1032), guard the soft penalty with `not _steering`:

```python
                    if (not _steering) and genre_sim is not None and math.isfinite(genre_sim):
                        if penalty_strength > 0 and genre_sim < penalty_threshold:
                            combined_score *= (1.0 - penalty_strength)
                            genre_penalty_hits += 1
```

- [ ] **Step 7: Apply floor + steering at the final-pier connection**

In `src/playlist/pier_bridge/beam.py`, at the final-pier connection (the `genre_sim = _get_genre_sim(int(last), int(pier_b))` at ~line 1415 and its tiebreak/penalty), wrap with steering logic. Replace the existing genre block there with:

```python
            genre_sim = _get_genre_sim(int(last), int(pier_b))
            _both_present_final = (
                genre_present is None
                or (genre_sim is not None
                    and bool(genre_present[int(last)])
                    and bool(genre_present[int(pier_b)]))
            )
            if _steering:
                if genre_sim is not None and math.isfinite(genre_sim) and _both_present_final:
                    if genre_sim < float(getattr(cfg, "genre_edge_floor", 0.0)):
                        continue  # this beam state cannot legally connect to pier_b
                    if float(getattr(cfg, "weight_genre", 0.0)) > 0.0:
                        final_edge_score += float(cfg.weight_genre) * genre_sim
            else:
                if genre_sim is not None and math.isfinite(genre_sim):
                    if cfg.genre_tiebreak_weight:
                        final_edge_score += cfg.genre_tiebreak_weight * genre_sim
```

Note: confirm the surrounding loop variable is `last` and that `continue` skips the current beam state. If the final connection is not inside a loop that allows `continue`, instead set a `valid = False` flag and skip appending the state. Read ~1405-1490 before editing.

- [ ] **Step 8: Run beam tests to verify they pass**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py -k beam -q`
Expected: PASS (3 passed).

- [ ] **Step 9: Run the existing beam test to confirm no regression**

Run: `python -m pytest tests/unit/test_beam_pace_gate.py tests/unit/test_progress_arc.py -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/playlist/pier_bridge/beam.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-steering): dense genre signal + floor gate + steering term in beam

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Builder — pass dense to the beam + genre-floor relaxation

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` (beam call sites pass `X_genre_dense`; segment backoff genre-floor relaxation)
- Test: `tests/unit/test_genre_edge_steering.py` (extend — relaxation via `build_pier_bridge_playlist` is covered by integration in Task 6; here we unit-test the floor-attempts helper)

- [ ] **Step 1: Pass `X_genre_dense` to every beam call**

In `src/playlist/pier_bridge_builder.py`, near where `X_genre_norm` is derived (~line 421-426), add:

```python
    X_genre_dense = getattr(bundle, "X_genre_dense", None)
```

Then at EACH `_beam_search_segment(...)` call (there are calls around lines 1099, 1318; find all via `grep -n "_beam_search_segment(" src/playlist/pier_bridge_builder.py`), add the kwarg:

```python
                        X_genre_dense=X_genre_dense,
```

- [ ] **Step 2: Write the failing test for the genre-floor attempts helper**

Append to `tests/unit/test_genre_edge_steering.py`:

```python
def test_genre_floor_attempts_steps_down():
    # The relaxation helper should produce a descending sequence from the initial
    # floor toward min_genre_edge_floor when relaxation is enabled.
    from src.playlist.pier_bridge_builder import _genre_floor_attempts_for_test
    attempts = _genre_floor_attempts_for_test(initial=0.40, minimum=0.10, enabled=True)
    assert attempts[0] == 0.40
    assert attempts[-1] <= 0.10 + 1e-9
    assert all(attempts[i] >= attempts[i + 1] for i in range(len(attempts) - 1))
    # Disabled -> only the initial floor.
    assert _genre_floor_attempts_for_test(initial=0.40, minimum=0.10, enabled=False) == [0.40]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py::test_genre_floor_attempts_steps_down -q`
Expected: FAIL — `ImportError: cannot import name '_genre_floor_attempts_for_test'`.

- [ ] **Step 4: Add the relaxation helper + apply it in the segment backoff**

In `src/playlist/pier_bridge_builder.py`, add a module-level helper (near the other module-level helpers, not inside a closure):

```python
def _genre_floor_attempts_for_test(initial: float, minimum: float, enabled: bool) -> list[float]:
    """Descending genre-edge-floor sequence (initial -> minimum), step 0.10.
    Pure helper exposed for unit testing; mirrors the relaxation used in the
    segment backoff loop."""
    if not enabled or minimum >= initial - 1e-9:
        return [float(initial)]
    attempts = [float(initial)]
    cur = round(float(initial) - 0.10, 2)
    while cur > minimum + 1e-9 and len(attempts) < 5:
        attempts.append(cur)
        cur = round(cur - 0.10, 2)
    if not any(abs(a - minimum) < 1e-9 for a in attempts):
        attempts.append(float(minimum))
    return attempts
```

Then, in the segment-build loop, after the existing relaxation tiers have failed to produce a `segment_path` (the same place the `transition_floor` relaxation tier lives — search for `_transition_floor_attempts`), add a parallel genre-floor relaxation tier. Reuse `_run_segment_backoff_attempts` with a `genre_edge_floor` override via `dataclasses.replace`. Concretely, add a parameter to `_run_segment_backoff_attempts` (mirroring the existing `transition_floor_override`):

```python
        genre_edge_floor_override: Optional[float] = None,
```

and at the top of that function, right after `cfg = cfg_attempt_base`:

```python
        if genre_edge_floor_override is not None:
            cfg = replace(cfg, genre_edge_floor=float(genre_edge_floor_override))
```

Then after the transition-floor relaxation tier, add (only when steering + relaxation enabled):

```python
        if segment_path is None and bool(getattr(cfg_base, "genre_steering_enabled", False)) \
           and infeasible_handling and infeasible_handling.enabled \
           and infeasible_handling.genre_floor_relaxation_enabled:
            _gfloors = _genre_floor_attempts_for_test(
                float(cfg_base.genre_edge_floor),
                float(infeasible_handling.min_genre_edge_floor),
                True,
            )
            for _gf in _gfloors[1:]:  # first already attempted
                _g_result = _run_segment_backoff_attempts(
                    cfg_attempt_base=cfg_base,
                    segment_allow_detours=segment_allow_detours_base,
                    segment_g_targets=segment_g_targets,
                    pier_a=pier_a, pier_b=pier_b, interior_len=interior_len,
                    pier_a_id=pier_a_id, pier_b_id=pier_b_id, seg_idx=seg_idx,
                    recent_boundary_artists=_recent_artists_for_segment(seg_idx),
                    genre_edge_floor_override=float(_gf),
                )
                if _g_result["segment_path"] is not None:
                    segment_path = _g_result["segment_path"]
                    # (copy the same result fields the transition-floor tier copies)
                    break
```

Note: copy the exact set of result fields the existing `transition_floor` relaxation tier copies from `_t_result` (chosen_bridge_floor, backoff_used_count, etc.) — read that block and mirror it precisely to avoid stale locals.

- [ ] **Step 5: Run the helper test + import check**

Run: `python -m pytest tests/unit/test_genre_edge_steering.py::test_genre_floor_attempts_steps_down -q && python -c "import src.playlist.pier_bridge_builder"`
Expected: PASS, then `OK` (clean import).

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge_builder.py tests/unit/test_genre_edge_steering.py
git commit -m "feat(genre-steering): wire dense matrix to beam + genre-floor relaxation tier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: config.yaml — enable steering with per-mode defaults

**Files:**
- Modify: `config.yaml` (under `playlists.ds_pipeline.pier_bridge` and `...pier_bridge.infeasible_handling`)

- [ ] **Step 1: Inspect current pier_bridge config block**

Run: `grep -n "pier_bridge:\|infeasible_handling:\|weight_bridge\|genre_tiebreak" config.yaml`
Read the surrounding block to match indentation and placement.

- [ ] **Step 2: Add the steering knobs**

Under `playlists.ds_pipeline.pier_bridge:` add (initial calibration values — refined in Task 6):

```yaml
        genre_steering_enabled: true
        weight_genre_strict: 0.30
        weight_genre_narrow: 0.20
        weight_genre_dynamic: 0.12
        weight_genre_discover: 0.06
        genre_edge_floor_strict: 0.50
        genre_edge_floor_narrow: 0.40
        genre_edge_floor_dynamic: 0.25
        genre_edge_floor_discover: 0.10
```

Under `...pier_bridge.infeasible_handling:` add:

```yaml
          genre_floor_relaxation_enabled: true
          min_genre_edge_floor: 0.0
```

- [ ] **Step 3: Verify config loads and resolves**

Run:
```bash
python -c "from src.playlist.config import resolve_pier_bridge_tuning; import yaml; c=yaml.safe_load(open('config.yaml')); ov={'pier_bridge': c['playlists']['ds_pipeline']['pier_bridge']}; t,_=resolve_pier_bridge_tuning(mode='narrow', similarity_floor=0.35, overrides=ov); print('steering', t.genre_steering_enabled, 'wgenre', round(t.weight_genre,3), 'floor', t.genre_edge_floor, 'sum', round(t.weight_bridge+t.weight_transition+t.weight_genre,3))"
```
Expected: `steering True wgenre ~0.143 floor 0.4 sum 1.0`.

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "feat(genre-steering): enable genre edge safeguards in config.yaml (initial calibration)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Calibration & integration validation (live artifact)

Requires the live artifact + dense sidecar. Marked `@pytest.mark.slow` / `@pytest.mark.integration`.

**Files:**
- Test: `tests/integration/test_genre_steering_integration.py` (new)

- [ ] **Step 1: Write the integration test (Smiths coherence improves; Charli stays feasible)**

Create `tests/integration/test_genre_steering_integration.py`:

```python
import numpy as np
import pytest
from pathlib import Path

from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_requires = pytest.mark.skipif(not ART.exists(), reason="live artifact required")
SMITHS = "de11fcb727aae7853a1b6c1e0d89ab25"      # This Charming Man
CHARLI = "5dda14ae880acbcc911e32710c50d5a5"      # a Charli XCX track


def _mean_edge_genre(bundle, track_ids):
    D = bundle.X_genre_dense
    ti = bundle.track_id_to_index
    sims = []
    for a, b in zip(track_ids, track_ids[1:]):
        ia, ib = ti.get(str(a)), ti.get(str(b))
        if ia is None or ib is None:
            continue
        na, nb = np.linalg.norm(D[ia]), np.linalg.norm(D[ib])
        if na < 1e-9 or nb < 1e-9:
            continue
        sims.append(float(D[ia] @ D[ib]))
    return float(np.mean(sims)) if sims else 0.0


@pytest.mark.integration
@pytest.mark.slow
@_requires
def test_smiths_edge_genre_coherence_improves_with_steering():
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))

    base = generate_playlist_ds(artifact_path=str(ART), seed_track_id=SMITHS,
                                mode="narrow", length=30, random_seed=42,
                                overrides={"pier_bridge": {"genre_steering_enabled": False}})
    steered = generate_playlist_ds(artifact_path=str(ART), seed_track_id=SMITHS,
                                   mode="narrow", length=30, random_seed=42,
                                   overrides={"pier_bridge": {
                                       "genre_steering_enabled": True,
                                       "weight_genre_narrow": 0.20,
                                       "genre_edge_floor_narrow": 0.40,
                                   }})
    g_base = _mean_edge_genre(bundle, base.track_ids)
    g_steer = _mean_edge_genre(bundle, steered.track_ids)
    assert g_steer > g_base, f"steering should raise mean edge genre sim: base={g_base:.3f} steered={g_steer:.3f}"


@pytest.mark.integration
@pytest.mark.slow
@_requires
def test_charli_narrow_still_feasible_with_steering_and_relaxation():
    load_artifact_bundle.cache_clear()
    res = generate_playlist_ds(artifact_path=str(ART), seed_track_id=CHARLI,
                               mode="narrow", length=40, random_seed=42,
                               overrides={"pier_bridge": {
                                   "genre_steering_enabled": True,
                                   "weight_genre_narrow": 0.20,
                                   "genre_edge_floor_narrow": 0.40,
                                   "infeasible_handling": {
                                       "enabled": True,
                                       "genre_floor_relaxation_enabled": True,
                                       "min_genre_edge_floor": 0.0,
                                   }}})
    assert res is not None and len(res.track_ids) >= 30
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/integration/test_genre_steering_integration.py -q -m "slow"`
Expected: PASS. If `test_smiths...` fails (no improvement), the floor/weight need tuning — see Step 3.

- [ ] **Step 3: Calibrate floor/weights against the reference seeds**

Run the research diagnostic to pick floor/weights on the dense scale, then adjust `config.yaml` (Task 5) values:
```bash
python scripts/research_genre_similarity.py 2>&1 | sed -n '/^E\./,$p'
```
Goal: `genre_edge_floor` sits in the gap between same-scene (~0.85) and adjacent-but-off pairs. Confirm by re-running the Smiths case in the GUI/CLI and checking Ramones/Halo Benders no longer appear in the first interior. Iterate config values (no code change) until coherent.

- [ ] **Step 4: Full regression**

Run: `python -m pytest -m "not slow and not gui" -q`
Expected: all pass (no regression in the existing suite).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_genre_steering_integration.py config.yaml
git commit -m "test(genre-steering): integration coherence + feasibility; calibrate floors

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Out of scope (do NOT implement here)
- General sonic-scoring re-examination (separate initiative).
- Pier-dedup spacing bug (5 vs 6 piers; `dict.fromkeys` at `pier_bridge_builder.py:324`).
- Mode admission-floor recalibration (broader genre tuning).
