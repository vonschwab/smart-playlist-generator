# Variable Bridge Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each pier-bridge choose its own length (within bounds) so it lands smoothly on the next pier — lifting the worst transition across the playlist — without becoming a crutch.

**Architecture:** Reuse the existing fixed-length beam as a black box. For each segment, score the bridge it produces at each candidate interior length by its **bottleneck** (worst edge incl. the return), and **greedily** pick the length that maximizes that bottleneck — but only flex off the nominal even-split length when the nominal edge is genuinely weak and a flex beats it by ε (the anti-crutch). The soft total band `[N−m, N+m]` lets segments flex independently (no global zero-sum coordination needed). Default-off → byte-identical to today.

**Tech Stack:** Python 3.11, numpy, existing `pier_bridge` package + `transition_metrics.score_transition_edge`, pytest. No new deps.

> **Plan note (deliberate simplification of the spec):** the spec describes a global max-min DP over per-length tables via an integrated beam capture. This plan realizes the same *principle* (per-segment minimax landing, bounded, prefer-N + ε, soft band) with **multi-call + greedy-prefer-nominal**, because (a) the beam is a 5.3k-LOC hotspot and multi-call needs zero surgery and gets the arc/genre targets right per length automatically, and (b) the cross-segment artist carry makes the segments sequentially dependent, which greedy honors and a global DP would fight. The full global DP / integrated capture is a documented v2 refinement.

## Global Constraints

- Python 3.11+. numpy only; no new deps.
- A configured knob that can't act is a **startup error / loud warning**, never a silent no-op.
- 90 s generation ceiling is **hard** — the multi-call flex MUST be budget-guarded (skip flex when time is short).
- Faithful generation/validation go through the policy layer (artist mode) — `scripts/research/slider_differentiation_eval.py`.
- New default once validated, not opt-in-off (discipline #22) — ships **disabled** until the worst-edge gate passes.
- Run pytest directly with `-q -p no:cacheprovider`, bounded by the tool timeout — never piped through `tail`/`head`.
- Default-OFF means **byte-identical to today** — guard every change behind `variable_bridge_length is True`.
- **Bottleneck** = min edge `T` over the complete bridge `pier_a → interior → pier_b`, including the return. **Nominal** = the current even-split length. **Total band** = track count in `[N−m, N+m]`.
- Out of scope: outlier piers (pier-selection problem), duration-targeting, the abandoned beam-redundancy lever.

---

## File Structure

- **Create** `src/playlist/pier_bridge/var_bridge.py` — `segment_bottleneck`, `choose_segment_length` (the pure, testable core).
- **Modify** `src/playlist/pier_bridge/config.py` — 5 knobs on `PierBridgeConfig` (after `generation_budget_s`, ~line 318).
- **Modify** `src/playlist/pier_bridge_builder.py` — wire greedy length selection into the segment loop (~line 1722); replace even-split consumption (~line 898) when enabled.
- **Modify** `src/playlist_generator.py` (artist sites 1973, 2861) + `src/playlist/pipeline/pier_bridge_overrides.py` (after the `pace_bridge_floor` block ~line 125) — knob passthrough.
- **Create** `tests/unit/test_var_bridge.py`.

> **Pre-build cleanup (do once, before Task 1):** branch fresh off master (`git checkout -b worktree-variable-bridge master`). Master has the two crash fixes and **none** of the abandoned beam-redundancy code, so "stripping Lever 1" = simply starting from master. Bring the two design docs (`docs/superpowers/specs/2026-06-27-variable-bridge-length-design.md`, this plan) over from `worktree-collapse-levers` via `git checkout worktree-collapse-levers -- <path>`.

---

### Task 1: Bottleneck + greedy length-selection core

**Files:**
- Create: `src/playlist/pier_bridge/var_bridge.py`
- Test: `tests/unit/test_var_bridge.py`

**Interfaces:**
- Produces:
  - `segment_bottleneck(nodes, edge_score) -> tuple[float, int]` — `nodes` = `[pier_a, *interior, pier_b]`; `edge_score(a, b) -> float`; returns `(min_edge_T, weakest_edge_index)`.
  - `choose_segment_length(nominal, lo, hi, build_and_score, *, good_enough, eps) -> tuple[int, object]` — `build_and_score(l) -> (path, bottleneck)`; tries `nominal` first, flexes within `[lo, hi]` only if the nominal bottleneck `< good_enough`, returns the length with the best bottleneck, preferring the length closest to `nominal` among those within `eps` of the best. Returns `(chosen_length, chosen_path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_var_bridge.py
from src.playlist.pier_bridge.var_bridge import segment_bottleneck, choose_segment_length


def test_bottleneck_is_min_edge_and_its_index():
    # nodes 0..3; edge score = fixed table
    edges = {(0, 1): 0.8, (1, 2): 0.2, (2, 3): 0.9}
    b, idx = segment_bottleneck([0, 1, 2, 3], lambda a, c: edges[(a, c)])
    assert b == 0.2 and idx == 1            # weakest edge is 1->2 at position 1


def test_choose_keeps_nominal_when_already_good():
    calls = []
    def build(l):
        calls.append(l)
        return ([10] * l, 0.7)               # every length scores 0.7
    chosen_l, path = choose_segment_length(6, 4, 8, build, good_enough=0.5, eps=0.02)
    assert chosen_l == 6 and path == [10] * 6
    assert calls == [6]                       # nominal good enough -> no flex, ONE build


def test_choose_flexes_to_best_bottleneck_when_nominal_weak():
    scores = {4: 0.10, 5: 0.45, 6: 0.12, 7: 0.40, 8: 0.20}   # nominal 6 is weak
    def build(l):
        return ([l], scores[l])
    chosen_l, path = choose_segment_length(6, 4, 8, build, good_enough=0.5, eps=0.02)
    assert chosen_l == 5 and path == [5]      # 5 has the best bottleneck (0.45)


def test_choose_prefers_nominal_within_epsilon():
    scores = {5: 0.46, 6: 0.45, 7: 0.30}      # 5 best but 6 within eps -> keep 6
    def build(l):
        return ([l], scores[l])
    chosen_l, _ = choose_segment_length(6, 5, 7, build, good_enough=0.9, eps=0.02)
    assert chosen_l == 6                       # nominal preferred when within eps of best


def test_choose_respects_band_clamp():
    # lo/hi narrower than nominal +/- flex: only 6,7 allowed
    scores = {6: 0.1, 7: 0.9}
    chosen_l, _ = choose_segment_length(6, 6, 7, lambda l: ([l], scores[l]), good_enough=0.5, eps=0.02)
    assert chosen_l == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_var_bridge.py -q -p no:cacheprovider`
Expected: FAIL (`var_bridge` module missing).

- [ ] **Step 3: Write the implementation**

```python
# src/playlist/pier_bridge/var_bridge.py
"""Variable bridge length — pick each segment's interior length to maximize its
worst edge (bottleneck), flexing off the nominal only when it earns it.

The bottleneck of a bridge is the weakest edge over pier_a -> interior -> pier_b,
INCLUDING the return edge, so a shorter bridge can never hide a bad landing.
"""
from __future__ import annotations

from typing import Callable


def segment_bottleneck(nodes, edge_score: Callable[[int, int], float]) -> tuple[float, int]:
    """Min edge score over the complete bridge nodes=[pier_a, *interior, pier_b],
    and the index of the weakest edge (0 = pier_a->first)."""
    best = float("inf")
    best_i = 0
    for i in range(len(nodes) - 1):
        s = float(edge_score(int(nodes[i]), int(nodes[i + 1])))
        if s < best:
            best, best_i = s, i
    return best, best_i


def choose_segment_length(nominal: int, lo: int, hi: int,
                          build_and_score: Callable[[int], tuple], *,
                          good_enough: float, eps: float) -> tuple[int, object]:
    """Choose interior length in [lo, hi] maximizing the segment bottleneck.

    Tries the nominal first; if its bottleneck >= good_enough, keeps it (no flex,
    one build). Otherwise builds the other allowed lengths and picks the best
    bottleneck, preferring the length CLOSEST to nominal among those within eps of
    the best (the prefer-N + eps anti-crutch). Returns (chosen_length, chosen_path)."""
    nom = max(lo, min(hi, int(nominal)))
    nom_path, nom_b = build_and_score(nom)
    if nom_b >= good_enough:
        return nom, nom_path
    results = {nom: (nom_b, nom_path)}
    for l in range(lo, hi + 1):
        if l not in results:
            path, b = build_and_score(l)
            results[l] = (b, path)
    best_b = max(b for b, _ in results.values())
    near = [l for l, (b, _) in results.items() if b >= best_b - eps]
    chosen = min(near, key=lambda l: (abs(l - nom), l))    # closest to nominal, then smaller
    return chosen, results[chosen][1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_var_bridge.py -q -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/var_bridge.py tests/unit/test_var_bridge.py
git commit -m "feat(var-bridge): bottleneck + greedy prefer-nominal length selection (pure core)"
```

---

### Task 2: Config knobs + wiring

**Files:**
- Modify: `src/playlist/pier_bridge/config.py:318`
- Modify: `src/playlist_generator.py:1973`, `:2861`
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (after pace_bridge_floor override, ~line 125)
- Test: `tests/unit/test_pipeline_smoke_golden.py` (golden regen — additive fields)

**Interfaces:**
- Produces: `PierBridgeConfig` gains `variable_bridge_length: bool = False`, `variable_bridge_flex: int = 2`, `variable_bridge_band: int = 5`, `variable_bridge_min_edge: float = 0.30`, `variable_bridge_epsilon: float = 0.02`. All reach the artist path (read from `ds_cfg["pier_bridge"]`) and the seeds path (`pb_overrides`).

- [ ] **Step 1: Add the dataclass fields**

In `src/playlist/pier_bridge/config.py`, after `generation_budget_s` (~line 318):

```python
    # --- Variable bridge length (default OFF until the worst-edge gate passes) ---
    # Each segment may flex its interior length to land more smoothly on the next
    # pier; lengths reallocate within a soft total band. OFF => even split (byte-identical).
    variable_bridge_length: bool = False
    variable_bridge_flex: int = 2          # k: +/- interior tracks a segment may flex
    variable_bridge_band: int = 5          # m: total track count may land in [N-m, N+m]
    variable_bridge_min_edge: float = 0.30  # only flex a segment whose nominal worst edge is below this
    variable_bridge_epsilon: float = 0.02   # prefer nominal length unless a flex beats it by > eps
```

- [ ] **Step 2: Wire the artist-style sites**

In `src/playlist_generator.py`, in BOTH `pier_cfg = PierBridgeConfig(...)` blocks (1973 and 2861), after the last argument (`genre_admission_percentile=...`), add (match each block's indentation — 20 spaces at 1973, 24 at 2861):

```python
                    variable_bridge_length=bool((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_length", False)),
                    variable_bridge_flex=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_flex", 2)),
                    variable_bridge_band=int((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_band", 5)),
                    variable_bridge_min_edge=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_min_edge", 0.30)),
                    variable_bridge_epsilon=float((ds_cfg.get("pier_bridge") or {}).get("variable_bridge_epsilon", 0.02)),
```

- [ ] **Step 3: Wire the seeds/DS path**

In `src/playlist/pipeline/pier_bridge_overrides.py`, after the `pace_bridge_floor` override block (~line 125):

```python
    if isinstance(pb_overrides.get("variable_bridge_length"), bool):
        pb_cfg = replace(pb_cfg, variable_bridge_length=bool(pb_overrides.get("variable_bridge_length")))
    for _k, _cast in (("variable_bridge_flex", int), ("variable_bridge_band", int),
                      ("variable_bridge_min_edge", float), ("variable_bridge_epsilon", float)):
        _v = pb_overrides.get(_k)
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            pb_cfg = replace(pb_cfg, **{_k: _cast(_v)})
```

- [ ] **Step 4: Regenerate the config-shape goldens (additive fields)**

Run: `python -m pytest tests/unit/test_pipeline_smoke_golden.py -q -p no:cacheprovider -k test_pier_bridge_config_matches_golden`
Expected: FAIL on the 4 scenarios, each diff showing ONLY the 5 new keys added.

Confirm purely additive, then regenerate:
```bash
rm tests/unit/goldens/pipeline/discover_with_dj_bridging.json tests/unit/goldens/pipeline/dynamic_default.json tests/unit/goldens/pipeline/narrow_progress_arc_dry_run.json tests/unit/goldens/pipeline/narrow_with_pier_bridge_overrides.json
python -m pytest tests/unit/test_pipeline_smoke_golden.py -q -p no:cacheprovider -k test_pier_bridge_config_matches_golden
python -m pytest tests/unit/test_pipeline_smoke_golden.py -q -p no:cacheprovider -k test_pier_bridge_config_matches_golden
```
Then `git diff --stat tests/unit/goldens/pipeline/` — expect 4 files, +5 lines each, 0 deletions.

- [ ] **Step 5: Verify imports + defaults**

Run: `python -c "import src.playlist_generator, src.playlist.pipeline.pier_bridge_overrides; from src.playlist.pier_bridge.config import PierBridgeConfig as C; c=C(); print('OK', c.variable_bridge_length, c.variable_bridge_flex, c.variable_bridge_band, c.variable_bridge_min_edge, c.variable_bridge_epsilon)"`
Expected: `OK False 2 5 0.3 0.02`.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist_generator.py src/playlist/pipeline/pier_bridge_overrides.py tests/unit/goldens/pipeline/
git commit -m "feat(var-bridge): config knobs + artist/seeds wiring (default off)"
```

---

### Task 3: Builder integration — greedy variable length in the segment loop

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` (even-split ~line 898; segment loop ~line 1722; beam call ~line 1486)
- Test: `tests/unit/test_pier_bridge_smoke_golden.py` (engine-off byte-identical) + a targeted variable-length generation assertion in `tests/unit/test_var_bridge_integration.py`

**Interfaces:**
- Consumes: `segment_bottleneck`, `choose_segment_length` (Task 1); the 5 config knobs (Task 2); `score_transition_edge(context, a, b)` from `src.playlist.transition_metrics` (returns a dict with key `"T"`).
- Produces: when `cfg.variable_bridge_length`, each segment's interior length is chosen by `choose_segment_length`; total stays in `[N−m, N+m]`; OFF → the even-split path unchanged.

- [ ] **Step 1: Establish the even-split nominal (keep it; don't delete)**

The even-split block (~line 898) computes `segment_lengths` as today — KEEP it. It is now the **nominal** per segment. No change here when the feature is off.

- [ ] **Step 2: Add the greedy selection inside the segment loop (guarded)**

Inside `for seg_idx in range(num_segments):` (~line 1722), the body currently calls `_beam_search_segment(pier_a, pier_b, interior_length, ...)` once with `interior_length = segment_lengths[seg_idx]`. Wrap that single call so that, when the feature is on, it is driven by `choose_segment_length`. Add near the top of the builder (before the loop), the time origin and a transition context handle:

```python
    # Variable bridge length: greedy per-segment length, budget-guarded. (Lever 2)
    _vbl = bool(getattr(cfg, "variable_bridge_length", False))
    _vbl_k = int(getattr(cfg, "variable_bridge_flex", 2))
    _vbl_band = int(getattr(cfg, "variable_bridge_band", 5))
    _vbl_good = float(getattr(cfg, "variable_bridge_min_edge", 0.30))
    _vbl_eps = float(getattr(cfg, "variable_bridge_epsilon", 0.02))
    _vbl_total_dev = 0   # running sum of (chosen_len - nominal_len), clamped to +/- band
    if _vbl:
        from src.playlist.pier_bridge.var_bridge import segment_bottleneck, choose_segment_length
        from src.playlist.transition_metrics import score_transition_edge
```

Then, in the loop body, replace the single fixed-length beam call with a builder closure + selection. Where the body has `interior_length = segment_lengths[seg_idx]` and the `_beam_search_segment(... interior_length ...)` call producing `segment_path`, do:

```python
        nominal_len = int(segment_lengths[seg_idx])
        if not _vbl or num_segments <= 0:
            interior_length = nominal_len
            segment_path, soft_hits, soft_edges, beam_fail = _run_one_beam(interior_length)  # existing call, factored
        else:
            # band clamp: keep the running total within [N-m, N+m]
            lo = max(1, nominal_len - _vbl_k, nominal_len - (_vbl_band + _vbl_total_dev))
            hi = nominal_len + min(_vbl_k, _vbl_band - _vbl_total_dev)
            hi = max(lo, hi)
            # budget guard: if we've eaten the budget, do not flex this segment
            _elapsed = _seconds_since_start()
            if _elapsed > float(getattr(cfg, "generation_budget_s", 60.0)) * 0.55:
                lo = hi = nominal_len
            def _build(l):
                path, _sh, _se, _bf = _run_one_beam(l)
                nodes = [int(pier_a), *[int(x) for x in path], int(pier_b)]
                b, _ = segment_bottleneck(nodes, lambda a, c: float(score_transition_edge(transition_metric_context, a, c).get("T", 0.0)))
                return (path, _sh, _se, _bf, b)
            def _build_and_score(l):
                r = _build(l)
                return ((r[0], r[1], r[2], r[3]), r[4])
            chosen_len, chosen = choose_segment_length(nominal_len, lo, hi, _build_and_score,
                                                       good_enough=_vbl_good, eps=_vbl_eps)
            segment_path, soft_hits, soft_edges, beam_fail = chosen
            _vbl_total_dev += (chosen_len - nominal_len)
            interior_length = chosen_len
            logger.info("Var-bridge seg %d: nominal=%d chosen=%d (total_dev=%+d)", seg_idx, nominal_len, chosen_len, _vbl_total_dev)
```

NOTE for the implementer: factor the existing `_beam_search_segment(...)` call at ~line 1486 into a local `_run_one_beam(interior_length)` closure that returns `(segment_path, soft_genre_penalty_hits_segment, soft_genre_penalty_edges_scored_segment, beam_failure_reason)` — it already has all other args (`pier_a, pier_b, candidates, …, transition_metric_context, …`) in scope; only `interior_length` varies. Add a `_seconds_since_start()` helper (or reuse the existing deadline/`time.monotonic()` origin the builder already tracks for `generation_budget_s`). Confirm `transition_metric_context` is the in-scope context variable passed to `_beam_search_segment` (it is — grep the call site).

- [ ] **Step 3: Engine-off byte-identical check**

Run: `python -m pytest tests/unit/test_pier_bridge_smoke_golden.py -q -p no:cacheprovider`
Expected: PASS (default `variable_bridge_length=False` → only the even-split path runs → unchanged).
Run: `python -m pytest tests/unit/ -q -p no:cacheprovider -k "pier or beam or roam"`
Expected: PASS.

- [ ] **Step 4: Targeted on-feature generation test**

```python
# tests/unit/test_var_bridge_integration.py
import pytest
from tests.helpers.gui_fidelity import generate_artist_playlist  # multi-pier policy-layer harness


@pytest.mark.integration
def test_variable_bridge_holds_total_in_band_and_helps_or_holds_worst_edge():
    base = generate_artist_playlist("Real Estate", tracks=30, overrides={})
    flex = generate_artist_playlist("Real Estate", tracks=30,
                                    overrides={"pier_bridge": {"variable_bridge_length": True}})
    assert 25 <= len(flex.tracks) <= 35                      # total in band [N-5, N+5]
    assert flex.min_transition >= base.min_transition - 1e-6  # worst edge not regressed
```

(Implementer: use the project's real multi-pier policy-layer harness — see the `playlist-testing` skill — not a hand-built single-seed config. If `generate_artist_playlist` isn't the exact helper name, use the established `gui_fidelity` entry point the other generation tests use.)

Run: `python -m pytest tests/unit/test_var_bridge_integration.py -q -p no:cacheprovider -m integration`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

Run: `ruff check src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/var_bridge.py && mypy src/playlist/pier_bridge/var_bridge.py`
```bash
git add src/playlist/pier_bridge_builder.py tests/unit/test_var_bridge_integration.py
git commit -m "feat(var-bridge): greedy variable length in builder segment loop (default off, budget-guarded)"
```

---

### Task 4: Validate "always good" + activate

**Files:**
- Modify: `config.yaml` + `src/playlist/pier_bridge/config.py` (`variable_bridge_length` default) — ONLY after the gate passes.
- Create: `docs/VARIABLE_BRIDGE_FINDINGS_2026-06-27.md`

- [ ] **Step 1: Worst-edge sweep on the corpus**

Using `scripts/research/slider_differentiation_eval.py` (artist policy layer), run the weak-edge + control seeds **off vs on**: Vegyn, Modest Mouse, Horse Jumper of Love, Carly Rae Jepsen, Real Estate, Deerhunter, Aphex Twin, The Smiths, Cocteau Twins. For each: min `T`, distinct artists, track count, wall seconds. Two runs per cell for determinism. Override `playlists.ds_pipeline.pier_bridge.variable_bridge_length: true`.

- [ ] **Step 2: Gate (all must hold vs OFF)**

- **Worst edge lifts or holds** on every seed — min `T` improves on the weak-edge seeds (Vegyn 0.023, Modest Mouse 0.024, Horse Jumper 0.079, Carly Rae 0.092) and no seed regresses beyond noise.
- **Total in band** — every cell's track count ∈ `[N−5, N+5]`, sitting at N unless a flex earned it.
- **Diversity not reduced**; **no adjacent same-artist**; **arc** loss not worse where energy/progress is active.
- **wall < 90 s** every cell (the multi-call + budget guard must hold this); deterministic (two runs identical).

Record results in `docs/VARIABLE_BRIDGE_FINDINGS_2026-06-27.md` with N and pool sizes stated.

- [ ] **Step 3: Flip the default ON — only if Step 2 passes**

Set `PierBridgeConfig.variable_bridge_length = True` and add it to `config.yaml`. Re-run the full fast suite:
Run: `python -m pytest -q -m "not slow" -p no:cacheprovider`
Expected: PASS (regenerate any transition-affected goldens; inspect each diff before accepting).

- [ ] **Step 4: Commit**

```bash
git add config.yaml src/playlist/pier_bridge/config.py docs/VARIABLE_BRIDGE_FINDINGS_2026-06-27.md tests/
git commit -m "feat(var-bridge): activate variable bridge length default-on after worst-edge gate"
```

- [ ] **Step 5: Perceptual audition (manual gate, Dylan):** generate a few weak-edge seeds (Vegyn, Carly Rae, Horse Jumper) and confirm the previously-broken transitions are gone and the arcs still feel intentional. Only after this passes is the feature done.

---

## Self-Review

**Spec coverage:** minimax-over-length landing → greedy `choose_segment_length` maximizing the bottleneck (T1, T3); bottleneck incl. return edge → `segment_bottleneck` over `[pier_a,*interior,pier_b]` (T1); soft band `[N−m,N+m]` → running `_vbl_total_dev` clamp (T3); prefer-N + ε → `choose_segment_length` eps tiebreak (T1); per-segment bound k → `lo/hi` clamp (T3); budget guard → elapsed>0.55·budget skips flex (T3); arc preserved → multi-call runs each length with its own `interior_length` so per-step targets are correct automatically (architecture); default-off byte-identical → `_vbl` guard + kept even-split (T3); validation worst-edge/in-band/diversity/budget (T4); out-of-scope outlier piers/duration honored (no code touches them). Deviation from spec (greedy vs global DP, multi-call vs integrated capture) flagged at top.

**Placeholder scan:** no TBD/TODO. T3 carries explicit implementer notes (factor `_run_one_beam`, confirm `transition_metric_context` name, time origin) rather than guessing exact line content in the 5.3k-LOC file; T4 lists exact gate thresholds and seeds.

**Type consistency:** `segment_bottleneck(nodes, edge_score)->(float,int)` and `choose_segment_length(nominal,lo,hi,build_and_score,*,good_enough,eps)->(int,object)` used identically in T1 tests and the T3 builder; `build_and_score(l)->(path_tuple, bottleneck)` matches the `_build_and_score` closure; config field names (`variable_bridge_length/flex/band/min_edge/epsilon`) consistent across T2/T3/T4; `score_transition_edge(context,a,b)["T"]` matches `transition_metrics.py:153`.
