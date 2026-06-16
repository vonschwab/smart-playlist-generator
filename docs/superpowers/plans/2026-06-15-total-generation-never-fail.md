# Total (never-fail) Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Mandatory reading before any generation test:** the `playlist-testing` skill — always reproduce through `tests/support/gui_fidelity.generate_like_gui`, never a hand-built single-seed.

**Goal:** Make pier-bridge generation *total* — it always returns a playlist (relaxing guideline gates per-segment, with a terminal guaranteed placement, and a "relaxed to fit" report) instead of raising `ValueError("Segment N infeasible")`.

**Architecture:** The relaxation ladder already exists in `pier_bridge_builder.py` (bridge_floor → transition-floor tier → genre-arc tier → micro-piers → failure). We (1) fix two min-floor defaults that make the transition tier inert, (2) parameterize the beam to disable the remaining gates for a single "terminal" attempt, (3) add a terminal tier (terminal beam attempt + greedy last resort) before the failure return, (4) report what bent. Spec: `docs/superpowers/specs/2026-06-15-total-generation-never-fail-design.md`.

**Tech Stack:** Python (dataclasses, numpy), the `gui_fidelity` test harness; React/FastAPI for the GUI notice.

**Operating rules (this repo):** work in this worktree off master; pytest **redirected to a file, never piped** (`python -m pytest … -q > out.txt 2>&1` then read it); never stage `config.yaml`; no `git push`; ff master after each task or two.

**Baseline:** `master` @ `ccd2070`. The failing repro (must turn green):
```python
generate_like_gui(seeds=["29a6637c9ba785f6270b114b37e59594","afd9ee94229bde6f31c853bfbe754730",
  "a7cf50c432f58d0df81fcbb22c4bd674","8539afd5d87ff30c3180863dced469c8","631f693758a8c5de622d750a08cbf6ee"],
  cohesion_mode="dynamic", genre_mode="narrow", sonic_mode="narrow", pace_mode="dynamic",
  artist_spacing="strong", length=30, random_seed=0)
```

---

### Task 1: Fix the inert min-floor defaults

**Files:** Modify `src/playlist/run_audit.py` (`InfeasibleHandlingConfig`, ~line 21); Modify `config.example.yaml` (`infeasible_handling` block ~line 579); Test: `tests/unit/test_infeasible_min_floors.py` (create).

- [ ] **Step 1: Failing test** — `tests/unit/test_infeasible_min_floors.py`:
```python
from src.playlist.run_audit import InfeasibleHandlingConfig

def test_min_floors_default_to_zero():
    cfg = InfeasibleHandlingConfig()
    # Defaults must let the transition tier relax BELOW a 0.20 transition_floor,
    # and the genre-arc tier reach percentile 0. (Old defaults: 0.20 / 0.5 — inert.)
    assert cfg.min_transition_floor == 0.0
    assert cfg.min_genre_arc_percentile == 0.0
```
- [ ] **Step 2: Run, expect FAIL** — `python -m pytest tests/unit/test_infeasible_min_floors.py -q > /tmp/t1.txt 2>&1` then read `/tmp/t1.txt`. Expected: assertion error (defaults are 0.20 / 0.5).
- [ ] **Step 3: Implement** — in `run_audit.py` change the two dataclass defaults:
```python
    min_transition_floor: float = 0.0
    ...
    min_genre_arc_percentile: float = 0.0
```
Also set them explicitly in `config.example.yaml`'s `infeasible_handling` block:
```yaml
        min_transition_floor: 0.0
        min_genre_arc_percentile: 0.0
```
- [ ] **Step 4: Run, expect PASS.** Same command.
- [ ] **Step 5: Commit** — `git add src/playlist/run_audit.py config.example.yaml tests/unit/test_infeasible_min_floors.py && git commit -m "fix(pier-bridge): default infeasible min-floors to 0 so the relax tiers actually relax"`

---

### Task 2: Add the `guarantee_feasible` knob

**Files:** Modify `src/playlist/run_audit.py` (`InfeasibleHandlingConfig` + `parse_infeasible_handling_config`, ~line 78); Test: append to `tests/unit/test_infeasible_min_floors.py`.

- [ ] **Step 1: Failing test** — append:
```python
from src.playlist.run_audit import parse_infeasible_handling_config

def test_guarantee_feasible_defaults_true_and_parses():
    assert InfeasibleHandlingConfig().guarantee_feasible is True
    assert parse_infeasible_handling_config({"guarantee_feasible": False}).guarantee_feasible is False
```
- [ ] **Step 2: Run, expect FAIL** (`AttributeError: guarantee_feasible`). Redirect to file.
- [ ] **Step 3: Implement** — add to the dataclass: `guarantee_feasible: bool = True`; and in `parse_infeasible_handling_config` add `guarantee_feasible=bool(raw.get("guarantee_feasible", True)),`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git add src/playlist/run_audit.py tests/unit/test_infeasible_min_floors.py && git commit -m "feat(pier-bridge): add infeasible_handling.guarantee_feasible (default true)"`

---

### Task 3: Beam "terminal mode" — disable the remaining gates for one attempt

**Files:** Modify `src/playlist/pier_bridge/beam.py` (the beam entry signature + the gate sites: genre-arc `~1194-1198`, local-sonic `~1219`, the progress gate, the pace gate); Test: `tests/unit/test_beam_terminal_mode.py` (create).

The beam already takes per-attempt overrides (`transition_floor_override`, `genre_arc_floor_percentile_override`). Add **one** new param `terminal_mode: bool = False`. When true, the beam skips ALL guideline `continue`s: the genre-arc floor drop (`beam.py:1194-1198`), the local-sonic edge policy (`beam.py:1219`, treat `None` as "keep"), the sonic-progress/monotonic gate, and the pace gate — and uses `transition_floor=0`/`bridge_floor=0`. Scoring still runs (to rank), only the hard rejects are bypassed.

- [ ] **Step 1: Failing test** — build a tiny synthetic bundle where the gated beam returns `None` (no candidate clears the genre-arc + progress gates) but `terminal_mode=True` returns a full path. (Use the existing beam test fixtures in `tests/` as the pattern — search `tests/` for an existing `beam` unit test to mirror its bundle construction; do **not** hand-roll config — reuse the fixture helpers.)
```python
# Skeleton — fill the bundle from the existing beam-test fixture helper:
def test_terminal_mode_returns_path_when_gated_beam_cannot():
    ctx = make_beam_ctx_with_no_gated_continuation()   # reuse fixture
    assert run_beam(ctx, terminal_mode=False) is None
    path = run_beam(ctx, terminal_mode=True)
    assert path is not None and len(path) == ctx.interior_len
```
- [ ] **Step 2: Run, expect FAIL** (`terminal_mode` unknown / no path). Redirect to file.
- [ ] **Step 3: Implement** — thread `terminal_mode` from the beam entry to each gate `continue`; guard every guideline reject with `if not terminal_mode and <reject condition>: continue`. Leave invariant rejects (already-used, artist/min-gap) intact. Keep scoring unchanged.
- [ ] **Step 4: Run, expect PASS.** Then run the existing beam tests to confirm no regression: `python -m pytest tests/ -k beam -q > /tmp/t3.txt 2>&1` and read it.
- [ ] **Step 5: Commit** — `git add src/playlist/pier_bridge/beam.py tests/unit/test_beam_terminal_mode.py && git commit -m "feat(pier-bridge): beam terminal_mode bypasses guideline gates for the never-fail terminal attempt"`

---

### Task 4: Terminal tier in `pier_bridge_builder.py` (the guarantee)

**Files:** Modify `src/playlist/pier_bridge_builder.py` (insert before the failure return at `:1908`, and thread `terminal_mode` through `_run_segment_backoff_attempts`); Test: `tests/integration/test_gui_fidelity_regressions.py` (append; mark `@pytest.mark.integration @pytest.mark.slow`).

- [ ] **Step 1: Failing test** — append the repro:
```python
import pytest
from support.gui_fidelity import generate_like_gui

@pytest.mark.integration
@pytest.mark.slow
def test_hard_seed_pair_never_fails():
    res = generate_like_gui(
        seeds=["29a6637c9ba785f6270b114b37e59594","afd9ee94229bde6f31c853bfbe754730",
               "a7cf50c432f58d0df81fcbb22c4bd674","8539afd5d87ff30c3180863dced469c8",
               "631f693758a8c5de622d750a08cbf6ee"],
        cohesion_mode="dynamic", genre_mode="narrow", sonic_mode="narrow",
        pace_mode="dynamic", artist_spacing="strong", length=30, random_seed=0)
    tids = getattr(res, "track_ids", res)
    assert len(tids) == 30  # currently raises ValueError("Segment 1 infeasible …")
```
- [ ] **Step 2: Run, expect FAIL** (`ValueError: Segment 1 infeasible`). `python -m pytest tests/integration/test_gui_fidelity_regressions.py::test_hard_seed_pair_never_fails -q -m "integration and slow" > /tmp/t4.txt 2>&1` then read.
- [ ] **Step 3: Implement** — in `pier_bridge_builder.py`, just before line 1908 (`if infeasible_handling and infeasible_handling.enabled:`), insert:
```python
        if segment_path is None and infeasible_handling and infeasible_handling.guarantee_feasible:
            # Terminal guarantee: one all-gates-off beam attempt, then a greedy last resort.
            _term = _run_segment_backoff_attempts(
                cfg_attempt_base=cfg_base, segment_allow_detours=True,
                segment_g_targets=segment_g_targets, segment_g_targets_dense=segment_g_targets_dense,
                pier_a=pier_a, pier_b=pier_b, interior_len=interior_len,
                pier_a_id=pier_a_id, pier_b_id=pier_b_id, seg_idx=seg_idx,
                recent_boundary_artists=_recent_artists_for_segment(seg_idx),
                transition_floor_override=0.0, genre_arc_floor_percentile_override=0.0,
                terminal_mode=True,
            )
            if _term["segment_path"] is not None:
                segment_path = _term["segment_path"]
                _relaxed = ["transition_floor→0", "genre steering", "sonic progress (terminal)"]
            else:
                segment_path = _greedy_terminal_path(
                    pier_a=pier_a, pier_b=pier_b, interior_len=interior_len,
                    candidates=last_segment_candidates, X_full_norm=X_full_norm,
                    X_end=X_end_tr_norm, used=used_track_indices,
                )
                _relaxed = ["all guideline gates", "diversity (terminal greedy)"]
            if segment_path is not None:
                warnings.append({
                    "type": "relaxation", "scope": "segment", "segment_index": int(seg_idx),
                    "bridge": f"{pier_a_id} -> {pier_b_id}", "relaxed": _relaxed,
                    "severity": "invariant" if "greedy" in _relaxed[-1] else "guideline",
                })
```
Thread a `terminal_mode: bool = False` kwarg through `_run_segment_backoff_attempts` (→ the beam call). Implement `_greedy_terminal_path` as a module helper: take the unused candidates, score `0.5*sonic_cos_to_prev + 0.5*sonic_cos_to_pier_b`, greedily pick `interior_len` ordered by increasing distance to `pier_b`. (Match the bundle field names used elsewhere in this file — `X_full_norm`, `X_end_tr_norm`.)
- [ ] **Step 4: Run, expect PASS** (30 tracks, no ValueError). Same command.
- [ ] **Step 5: Commit** — `git add src/playlist/pier_bridge_builder.py tests/integration/test_gui_fidelity_regressions.py && git commit -m "feat(pier-bridge): terminal guarantee — generation is now total"`

---

### Task 5: Surface relaxations to the GUI

**Files:** `src/playlist/pipeline/core.py` (carry the `relaxation` warnings onto the result; do not raise when a segment was terminally filled), `src/playlist_gui/worker.py` (include in the result payload), web schemas + `web/src` (render a dismissible notice). Test: extend the Task 4 integration test + a web schema unit test.

- [ ] **Step 1: Failing test** — extend `test_hard_seed_pair_never_fails` to assert the report is present:
```python
    relax = [w for w in getattr(res, "warnings", []) if w.get("type") == "relaxation"]
    assert any(w["segment_index"] == 1 for w in relax)
```
(If `generate_like_gui` does not surface `warnings`, assert via the builder result it returns — check `tests/support/gui_fidelity.py` for what the harness exposes and assert on that; wire `relaxation` warnings through if the harness drops them — that itself is a fidelity fix worth a Trap-Catalog row.)
- [ ] **Step 2: Run, expect FAIL.** Redirect to file.
- [ ] **Step 3: Implement** — carry `warnings` (filtered to `type=="relaxation"`) from `PierBridgeResult` through `pipeline/core.py` onto the generation result and out through the worker NDJSON `result` payload (mirror how `metrics` flows). Add a `relaxations: list[...]` field to the web `PlaylistOut` schema and a `RelaxationNotice` component in `web/src` rendered above the playlist (mirror an existing notice/banner). Guideline severity = info; invariant = warning.
- [ ] **Step 4: Run, expect PASS.** Backend test green; `npm --prefix web run build` green.
- [ ] **Step 5: Commit** — explicit paths; `git commit -m "feat(web): surface pier-bridge relaxations as a GUI notice"`

---

### Task 6: No-regression + full verification

- [ ] **Step 1: No-regression test** — append to `test_gui_fidelity_regressions.py`: a *known-feasible* multi-pier case asserts **no** `relaxation` warning and a 30-track result. Run it (redirect to file), expect PASS.
- [ ] **Step 2: Full suites** — `python -m pytest -q -m "not slow" > /tmp/suite.txt 2>&1` then read it (expect the ~1674 baseline still green); `python -m pytest -q -m "integration and slow" -k gui_fidelity > /tmp/int.txt 2>&1`; `npm --prefix web run build`. Quote real pass counts.
- [ ] **Step 3: Manual** — restart `serve_web.py`, regenerate the failing seeds in the GUI, confirm a 30-track playlist + the relaxation notice.
- [ ] **Step 4: Commit** any remaining test/doc; update the `playlist-testing` skill Trap Catalog if a fidelity gap was found (Task 5).

---

## Self-Review

**Spec coverage:** §1 min-floors → Task 1; §5 flag → Task 2; §2 beam terminal mode → Task 3; §3 terminal tier (beam attempt + greedy) → Task 4; §4 reporting → Task 5; testing → Tasks 4/6. ✓ All spec sections mapped.

**Placeholder scan:** Config values, the dataclass fields, the insertion point (`:1908`), and the repro seeds are exact. **Two tasks (3 and the greedy helper / GUI plumbing in 4–5) intentionally specify by file:line anchor + "mirror the existing pattern" rather than pasting code I have not fully read** (the beam gate `continue`s and the worker→GUI `metrics` flow). The implementer must read those exact sites — flagged inline. This is the honest boundary of what the planning read covered; everything load-bearing (terminal insertion, config, repro test) is concrete.

**Type consistency:** `terminal_mode: bool` is threaded identically through beam entry → `_run_segment_backoff_attempts` → the beam call. `guarantee_feasible` is the single gate on the terminal tier. The `relaxation` warning dict shape is identical in Task 4 (emit) and Task 5 (assert/render).

**Known risk to verify during execution:** that disabling gates is strictly scoped to `terminal_mode=True` (Task 3 Step 4 runs the existing beam tests to catch a leak), and that the harness actually surfaces `warnings` (Task 5 Step 1 — may need a fidelity fix).
