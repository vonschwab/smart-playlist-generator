# SP3 Mini-Piers v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert high-character waypoints as extra piers inside long bridges — chosen up front so the beam can't drift past them — to structurally bound within-bridge sag.

**Architecture:** A pure, unit-tested module (`mini_pier_select.py`) provides `select_waypoint` (relative smoothness floor → anti-center pick) and `plan_pier_sequence` (greedy split of the longest segment until all ≤ K). `build_pier_bridge_playlist` calls `plan_pier_sequence` right after seed-ordering, behind a config flag, so the augmented pier list flows through the existing segment machinery unchanged. Off ⇒ byte-identical.

**Tech Stack:** Python 3.11, numpy, pytest. MuQ sonic space (`X_full_norm`, unit rows).

## Global Constraints

- Off by default (`mini_pier_enabled=False`) ⇒ byte-identical to today. Pier-bridge config goldens + `test_audit_matches_beam` must stay green.
- Pure module has NO engine imports (numpy only), like `collapse_metric.py` / `seed_character.py`.
- All checks RELATIVE to the piers (smoothness floor = relative to best available min-sim; centrality = relative to the local between-region), never a global absolute center.
- Selection excludes seed/pier-artist tracks and already-used indices (waypoints are real playlist members subject to diversity).
- TDD: failing test first, minimal code, frequent commits. Run pytest directly, never piped (`python -m pytest -q -x path::name`), bounded by timeout.
- Starting knob values (spec): `mini_pier_max_interior K=5`, `mini_pier_smoothness_margin=0.12`, `k_broad=150`.

---

### Task 1: `select_waypoint` — the pick (relative smoothness floor → anti-center)

**Files:**
- Create: `src/playlist/pier_bridge/mini_pier_select.py`
- Test: `tests/unit/test_mini_pier_select.py`

**Interfaces:**
- Produces: `select_waypoint(pier_a: int, pier_b: int, candidate_indices: Sequence[int], X_full_norm: np.ndarray, *, margin: float = 0.12, k_broad: int = 150, exclude: frozenset[int] = frozenset()) -> Optional[int]` — returns the chosen waypoint's global index, or None if no feasible candidate.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mini_pier_select.py
import numpy as np
from src.playlist.pier_bridge.mini_pier_select import select_waypoint

def _unit(M):
    M = np.asarray(M, float)
    return M / np.linalg.norm(M, axis=1, keepdims=True)

def test_picks_between_and_off_center():
    # A and B piers; c1 is central-to-both (blur), c2 is between but more distinctive,
    # c3 is a distant outlier (must be excluded by the smoothness floor).
    X = _unit([
        [1, 0, 0, 0],    # 0 pier A
        [0, 1, 0, 0],    # 1 pier B
        [1, 1, 0, 0],    # 2 c1: max between (the blur center)
        [1, 1, 0.6, 0],  # 3 c2: between + a distinctive component
        [0, 0, 1, 1],    # 4 c3: distant outlier
    ])
    got = select_waypoint(0, 1, [2, 3, 4], X, margin=0.12, k_broad=3)
    assert got in (2, 3)          # never the distant outlier c3
    assert got == 3               # among the smooth pair, the less-central one wins

def test_excludes_pier_and_excluded_indices():
    X = _unit(np.eye(6))
    assert select_waypoint(0, 1, [0, 1], X) is None          # only piers -> nothing
    assert select_waypoint(0, 1, [2, 3], X, exclude=frozenset({2, 3})) is None

def test_returns_none_on_empty_pool():
    X = _unit(np.eye(4))
    assert select_waypoint(0, 1, [], X) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_select.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.playlist.pier_bridge.mini_pier_select'`.

- [ ] **Step 3: Write the module**

```python
# src/playlist/pier_bridge/mini_pier_select.py
"""SP3 mini-pier v2 selection (pure functions, unit-testable).

Pick a waypoint to pin inside a long bridge: relative smoothness floor (candidates
within `margin` of the best available min-sim to BOTH piers, so the pick is genuinely
between them and adapts to close vs cross-niche) -> anti-center within (the least
central relative to the local between-region, so it's on-character, not the wallpaper).
See docs/superpowers/specs/2026-06-30-sp3-mini-piers-design.md.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def select_waypoint(
    pier_a: int,
    pier_b: int,
    candidate_indices: Sequence[int],
    X_full_norm: np.ndarray,
    *,
    margin: float = 0.12,
    k_broad: int = 150,
    exclude: frozenset[int] = frozenset(),
) -> Optional[int]:
    piers = {int(pier_a), int(pier_b)}
    cand = np.array(
        [int(c) for c in candidate_indices if int(c) not in piers and int(c) not in exclude],
        dtype=int,
    )
    if cand.size == 0:
        return None
    simA = X_full_norm[cand] @ X_full_norm[int(pier_a)]
    simB = X_full_norm[cand] @ X_full_norm[int(pier_b)]
    minsim = np.minimum(simA, simB)
    # between-region = the k_broad most-between candidates (stable local center + floor)
    k = int(min(max(1, k_broad), cand.size))
    broad_local = np.argpartition(-minsim, k - 1)[:k]
    broad = cand[broad_local]
    best = float(minsim[broad_local].max())
    smooth_mask = minsim[broad_local] >= best - float(margin)
    smooth = broad[smooth_mask]
    if smooth.size == 0:
        return int(broad[int(np.argmax(minsim[broad_local]))])
    center = X_full_norm[broad].mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm < 1e-12:
        return int(smooth[0])
    center = center / norm
    cent = X_full_norm[smooth] @ center
    return int(smooth[int(np.argmin(cent))])
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_select.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/mini_pier_select.py tests/unit/test_mini_pier_select.py
git commit -m "feat(sp3): select_waypoint — smoothness-floor + anti-center pick"
```

---

### Task 2: `plan_pier_sequence` — greedy split of the longest segment

**Files:**
- Modify: `src/playlist/pier_bridge/mini_pier_select.py`
- Test: `tests/unit/test_mini_pier_select.py`

**Interfaces:**
- Consumes: `select_waypoint` (Task 1).
- Produces: `plan_pier_sequence(ordered_seeds: Sequence[int], total_tracks: int, candidate_indices: Sequence[int], X_full_norm: np.ndarray, *, max_interior: int, margin: float, k_broad: int, exclude_base: frozenset[int], max_waypoints: int) -> list[int]` — the augmented pier list (seeds + inserted waypoints, in order).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_mini_pier_select.py
from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence

def test_no_split_when_segments_short():
    X = _unit(np.eye(10))
    # 3 seeds, 30 tracks -> interior 27 / 2 segs ~13; but max_interior huge -> no split
    seq = plan_pier_sequence([0, 1, 2], 30, list(range(3, 10)), X,
                             max_interior=99, margin=0.12, k_broad=5,
                             exclude_base=frozenset(), max_waypoints=3)
    assert seq == [0, 1, 2]

def test_splits_longest_until_under_K():
    # 2 seeds, long interior -> must insert waypoints so each segment interior <= K.
    X = _unit(np.random.default_rng(0).normal(size=(60, 8)))
    seq = plan_pier_sequence([0, 1], 20, list(range(2, 60)), X,
                             max_interior=5, margin=0.20, k_broad=30,
                             exclude_base=frozenset(), max_waypoints=5)
    assert seq[0] == 0 and seq[-1] == 1     # original piers stay at the ends
    assert len(seq) > 2                      # at least one waypoint inserted
    # every segment's even-split interior is <= K
    n_seg = len(seq) - 1
    interior = 20 - len(seq)
    base = interior // n_seg
    assert base <= 5

def test_respects_max_waypoints_cap():
    X = _unit(np.random.default_rng(1).normal(size=(60, 8)))
    seq = plan_pier_sequence([0, 1], 40, list(range(2, 60)), X,
                             max_interior=3, margin=0.30, k_broad=30,
                             exclude_base=frozenset(), max_waypoints=2)
    assert len(seq) - 2 <= 2                  # never more than max_waypoints inserted
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_select.py::test_splits_longest_until_under_K`
Expected: FAIL — `cannot import name 'plan_pier_sequence'`.

- [ ] **Step 3: Add `plan_pier_sequence` to the module**

```python
# add to src/playlist/pier_bridge/mini_pier_select.py

def _even_split_lengths(total_interior: int, num_segments: int) -> list[int]:
    base, rem = divmod(total_interior, num_segments)
    return [base + (1 if i < rem else 0) for i in range(num_segments)]


def plan_pier_sequence(
    ordered_seeds,
    total_tracks: int,
    candidate_indices,
    X_full_norm: np.ndarray,
    *,
    max_interior: int,
    margin: float,
    k_broad: int,
    exclude_base: frozenset[int] = frozenset(),
    max_waypoints: int = 8,
) -> list[int]:
    """Greedily split the longest segment (by even-split interior) by inserting a
    waypoint between its two piers, until every segment's interior <= max_interior,
    no feasible waypoint remains, or max_waypoints is reached. Returns the augmented
    pier list; identical to ordered_seeds when nothing needs splitting."""
    piers = [int(s) for s in ordered_seeds]
    used = set(piers) | {int(e) for e in exclude_base}
    for _ in range(int(max_waypoints)):
        num_seg = len(piers) - 1
        interior = int(total_tracks) - len(piers)
        if num_seg < 1 or interior < 1:
            break
        lengths = _even_split_lengths(interior, num_seg)
        seg = int(np.argmax(lengths))
        if lengths[seg] <= int(max_interior):
            break
        wp = select_waypoint(
            piers[seg], piers[seg + 1], candidate_indices, X_full_norm,
            margin=margin, k_broad=k_broad, exclude=frozenset(used),
        )
        if wp is None:
            break
        piers.insert(seg + 1, wp)
        used.add(wp)
    return piers
```

- [ ] **Step 4: Run to verify all pass**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_select.py`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/mini_pier_select.py tests/unit/test_mini_pier_select.py
git commit -m "feat(sp3): plan_pier_sequence — greedy split of the longest segment"
```

---

### Task 3: Config knobs + override wiring

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (PierBridgeConfig fields)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (read from pb_overrides)
- Modify: `tests/unit/goldens/pipeline/*.json` (add the new inert defaults)
- Test: `tests/unit/test_mini_pier_overrides.py`

**Interfaces:**
- Produces: `PierBridgeConfig.mini_pier_enabled: bool`, `.mini_pier_max_interior: int`, `.mini_pier_smoothness_margin: float`.

- [ ] **Step 1: Write the failing test** (mirror `test_roam_overrides.py`'s `_apply` helper)

```python
# tests/unit/test_mini_pier_overrides.py
from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

def _apply(overrides):
    cfg, _, _, _ = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides=overrides, pb_overrides=overrides.get("pier_bridge", {}),
        artist_playlist=False, dry_run=True, audit_cfg=None, resolved_variant="raw")
    return cfg

def test_mini_pier_defaults_off():
    c = _apply({"pier_bridge": {}})
    assert c.mini_pier_enabled is False
    assert c.mini_pier_max_interior == 5
    assert c.mini_pier_smoothness_margin == 0.12

def test_mini_pier_overrides_propagate():
    c = _apply({"pier_bridge": {"mini_pier_enabled": True,
                                "mini_pier_max_interior": 4,
                                "mini_pier_smoothness_margin": 0.1}})
    assert c.mini_pier_enabled is True
    assert c.mini_pier_max_interior == 4
    assert c.mini_pier_smoothness_margin == 0.1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_overrides.py`
Expected: FAIL — `AttributeError: 'PierBridgeConfig' object has no attribute 'mini_pier_enabled'`.

- [ ] **Step 3: Add the fields** to `PierBridgeConfig` in `src/playlist/pier_bridge/config.py` (next to `seed_character_*`, ~line 105):

```python
    # SP3 mini-piers (off by default -> byte-identical). Insert high-character
    # waypoints as extra piers in long bridges so the beam can't sag past them.
    mini_pier_enabled: bool = False
    mini_pier_max_interior: int = 5        # split any segment whose interior exceeds K
    mini_pier_smoothness_margin: float = 0.12
```

- [ ] **Step 4: Wire the overrides** in `src/playlist/pipeline/pier_bridge_overrides.py` — add the bool next to `seed_character_mode`, and the numeric two to the numeric-cast loop:

```python
    if isinstance(pb_overrides.get("mini_pier_enabled"), bool):
        pb_cfg = replace(pb_cfg, mini_pier_enabled=bool(pb_overrides.get("mini_pier_enabled")))
    # ...and add to the existing numeric-cast tuple:
    #   ("mini_pier_max_interior", int), ("mini_pier_smoothness_margin", float)
```

- [ ] **Step 5: Run the override test**

Run: `python -m pytest -q -x tests/unit/test_mini_pier_overrides.py`
Expected: PASS (2 passed).

- [ ] **Step 6: Update the config goldens** (the serialized PierBridgeConfig gained 3 keys)

```bash
python - <<'PY'
import json, pathlib
gd = pathlib.Path("tests/unit/goldens/pipeline")
add = {"mini_pier_enabled": False, "mini_pier_max_interior": 5, "mini_pier_smoothness_margin": 0.12}
for p in sorted(gd.glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    for k, v in add.items(): d.setdefault(k, v)
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
PY
```

- [ ] **Step 7: Run the config goldens**

Run: `python -m pytest -q -x tests/unit/test_pipeline_smoke_golden.py`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/test_mini_pier_overrides.py tests/unit/goldens/pipeline/
git commit -m "feat(sp3): mini_pier config knobs + override wiring (off by default)"
```

---

### Task 4: Integrate into the beam (behind the flag) + validate

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py:913-915` (splice waypoints, derive segment math from `len(ordered_seeds)`)
- Test: `tests/unit/test_var_bridge_integration.py` (add an off-by-default byte-identical + on-feature multi-pier case) OR a new `tests/integration/test_mini_pier_integration.py`

**Interfaces:**
- Consumes: `plan_pier_sequence` (Task 2), the config knobs (Task 3).

- [ ] **Step 1: Write the failing integration test** (through `generate_like_gui`, per the playlist-testing skill — multi-pier, real config chain). Mark `@pytest.mark.integration @pytest.mark.slow` and skip if the artifact is absent. Use the William Tyler / Hayden Pedigo / Steve Hiett / Songs:Ohia / Bill Callahan seed set from `test_var_bridge_integration.py`.

```python
# tests/integration/test_mini_pier_integration.py
from pathlib import Path
import pytest
from src.features.artifacts import load_artifact_bundle
from src.playlist.ds_pipeline_runner import generate_playlist_ds
from tests.support.gui_fidelity import gui_ui_state, resolve_gui_overrides, resolve_gui_genre_params

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_req = pytest.mark.skipif(not ART.exists(), reason="live artifact required")
SEEDS = ["f28fd5cebac845cf64fee59d5ac3b3aa", "b8f8aa0e86f977f9fcb26f615e130ac9",
         "42473b911cef5674e56b8e2ce87df7cb", "49f8bba75408d4e0e0e000d1dc708add"]

def _gen(mini_pier: bool):
    ui = gui_ui_state(cohesion_mode="dynamic", genre_mode="dynamic",
                      sonic_mode="dynamic", pace_mode="dynamic")
    ov = resolve_gui_overrides(ui)
    if mini_pier:
        pb = dict(ov.get("pier_bridge", {}) or {})
        pb["mini_pier_enabled"] = True
        pb["mini_pier_max_interior"] = 4
        ov = {**ov, "pier_bridge": pb}
    gp = resolve_gui_genre_params(ui)
    b = load_artifact_bundle(str(ART))
    seeds = [s for s in SEEDS if s in b.track_id_to_index]
    if len(seeds) < 3:
        pytest.skip("seeds absent from this artifact build")
    return generate_playlist_ds(artifact_path=str(ART), seed_track_id=seeds[0],
        anchor_seed_ids=seeds, mode="dynamic", pace_mode="dynamic", length=30,
        random_seed=0, overrides=ov, artist_style_enabled=False, artist_playlist=False, **gp)

@pytest.mark.integration
@pytest.mark.slow
@_req
def test_mini_pier_off_matches_baseline_length():
    load_artifact_bundle.cache_clear()
    assert len(_gen(mini_pier=False).track_ids) == 30

@pytest.mark.integration
@pytest.mark.slow
@_req
def test_mini_pier_on_generates_and_changes_ordering():
    load_artifact_bundle.cache_clear()
    off = _gen(mini_pier=False).track_ids
    load_artifact_bundle.cache_clear()
    on = _gen(mini_pier=True).track_ids
    assert len(on) == 30                 # length preserved (waypoints take interior slots)
    assert list(on) != list(off)         # the flag actually changed the playlist
```

- [ ] **Step 2: Run to verify the on-feature test fails**

Run: `python -m pytest -q -x tests/integration/test_mini_pier_integration.py::test_mini_pier_on_generates_and_changes_ordering`
Expected: FAIL — `on == off` (flag not wired into the beam yet).

- [ ] **Step 3: Splice waypoints into the beam.** In `src/playlist/pier_bridge_builder.py`, replace the multi-seed `else` block (currently lines 913-915):

```python
    else:
        if bool(getattr(cfg, "mini_pier_enabled", False)):
            from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence
            # exclude seed/pier-artist tracks (waypoints are real piers; keep them
            # off the seed artists) using normalized track_artists.
            import numpy as _np
            _artists = _np.array([" ".join(str(x).split()).lower()
                                  for x in bundle.track_artists])
            _pier_artists = {_artists[i] for i in ordered_seeds}
            _exclude = frozenset(int(i) for i in _np.where(
                _np.isin(_artists, list(_pier_artists)))[0])
            ordered_seeds = plan_pier_sequence(
                ordered_seeds, total_tracks, candidate_pool_indices, X_full_norm,
                max_interior=int(cfg.mini_pier_max_interior),
                margin=float(cfg.mini_pier_smoothness_margin),
                k_broad=150, exclude_base=_exclude,
                max_waypoints=max(0, total_tracks // 4),
            )
            logger.info("Mini-piers: %d waypoint(s) inserted (piers now %d)",
                        len(ordered_seeds) - num_seeds, len(ordered_seeds))
        num_segments = len(ordered_seeds) - 1
        total_interior = total_tracks - len(ordered_seeds)
```

Note: `num_seeds` at line 914-915 becomes `len(ordered_seeds)` — byte-identical when the flag is off (no waypoints ⇒ `len(ordered_seeds) == num_seeds`). Leave the single-seed-arc branch (lines 906-912) untouched.

- [ ] **Step 4: Run the on/off integration tests**

Run: `python -m pytest -q -x -m "not slow" tests/unit/test_mini_pier_select.py tests/unit/test_mini_pier_overrides.py` then (with artifact) `python -m pytest -q tests/integration/test_mini_pier_integration.py`
Expected: unit PASS; integration on-feature test now passes (`on != off`, length 30).

- [ ] **Step 5: Off-by-default regression sweep**

Run: `python -m pytest -q -m "not slow" tests/unit/test_pier_bridge_smoke_golden.py tests/unit/test_audit_matches_beam.py tests/unit/test_pipeline_smoke_golden.py`
Expected: PASS — off is byte-identical.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge_builder.py tests/integration/test_mini_pier_integration.py
git commit -m "feat(sp3): splice mini-pier waypoints into the beam (behind flag)"
```

---

### Task 5: Collapse-harness validation + tuning

**Files:** none (research runs). Uses `scripts/research/collapse_eval.py`, `collapse_rescore.py`, `collapse_sweep_compare.py`.

- [ ] **Step 1:** Run the collapse corpus with mini-piers on, at K=5 (override generation_budget_s:60 for speed, seed_character_mode:anti_center strength 2.0 to stack on SP2-B):

```bash
python scripts/research/collapse_eval.py --repeats 1 --tag mp_K5 \
  --override '{"playlists":{"ds_pipeline":{"pier_bridge":{"generation_budget_s":60,"seed_character_mode":"anti_center","seed_character_strength":2.0,"mini_pier_enabled":true,"mini_pier_max_interior":5}}}}'
```

- [ ] **Step 2:** Rescore both faces vs the SP2-B baseline and check the quality floor held:

```bash
python scripts/research/collapse_sweep_compare.py muq_B20 mp_K5
```

Expected: within-bridge sag (esp. dreampop) drops below the SP2-B baseline while seed_sim / worst_edge hold. If not, sweep K (3, 4) and margin (0.08, 0.15).

- [ ] **Step 3:** Audition — generate real playlists (diverse + few-seed) with mini-piers on and confirm by ear that the long-bridge middles hold character. **Real playlists are the verdict.**

---

## Self-Review

- **Spec coverage:** Trigger (Task 2 `plan_pier_sequence`, length K) ✓; selection smoothness-floor + anti-center (Task 1) ✓; global up-front insertion (Task 4, before segment math) ✓; relative checks (Task 1 uses local between-region center + relative floor) ✓; config knobs K/margin (Task 3) ✓; var-bridge composition (untouched — waypoints set structure, var-bridge flexes lengths downstream) ✓; validation harness + audition (Task 5) ✓. Phase-2 distance trigger + hazy-pull + v1 deletion are spec-deferred, correctly absent.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `select_waypoint` returns `Optional[int]`; `plan_pier_sequence` consumes it and returns `list[int]`; integration assigns to `ordered_seeds` (list[int]); config names match across Tasks 3-4.
