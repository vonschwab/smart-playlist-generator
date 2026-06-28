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
- Modify: `src/playlist/pier_bridge_builder.py` (even-split nominal ~line 898; segment loop ~line 1722; per-segment build = genre targets ~1756/1810 + `_run_segment_backoff_attempts` ~1926 + fallbacks ~1981–2150)
- Test: `tests/unit/test_pier_bridge_smoke_golden.py` (engine-off byte-identical) + a targeted variable-length generation assertion in `tests/unit/test_var_bridge_integration.py`

**Interfaces:**
- Consumes: `segment_bottleneck`, `choose_segment_length` (Task 1); the 5 config knobs (Task 2); `score_transition_edge(context, a, b)` from `src.playlist.transition_metrics` (returns a dict with key `"T"`).
- Produces: when `cfg.variable_bridge_length`, each segment's interior length is chosen by `choose_segment_length`; total stays in `[N−m, N+m]`; OFF → the even-split path unchanged.

- [ ] **Step 1: Establish the even-split nominal (keep it; don't delete)**

The even-split block (~line 898) computes `segment_lengths` as today — KEEP it. It is now the **nominal** per segment. No change here when the feature is off.

- [ ] **Step 2: Factor the per-segment build into a length-parameterized closure, and drive it with `choose_segment_length` when enabled**

This is a careful refactor of an intricate, **length-coupled** loop body in a 5.3k-LOC hotspot. READ the segment loop (builder ~1722–2150) before editing. (Controller verified the structure below.)

**The real structure — the loop body does NOT call `_beam_search_segment` directly.** For each segment it:
1. sets `interior_len = segment_lengths[seg_idx]` (~1740);
2. builds the genre-arc targets `segment_g_targets` (`_build_genre_targets(..., interior_length=interior_len, ...)`, ~1756, under `if cfg.dj_bridging_enabled and X_genre_norm is not None`) and `segment_g_targets_dense` (`build_taxonomy_genre_targets(..., interior_length=interior_len, ...)`, ~1810, under the genre-steering condition) — **both length-coupled**;
3. computes length-INDEPENDENT `segment_far_stats` (~1771, pier-pair only — reuse, do NOT rebuild per length);
4. calls `attempt_result = _run_segment_backoff_attempts(*, cfg_attempt_base=…, segment_allow_detours=…, segment_g_targets=segment_g_targets, segment_g_targets_dense=segment_g_targets_dense, pier_a=pier_a, pier_b=pier_b, interior_len=interior_len, pier_a_id=…, pier_b_id=…, seg_idx=seg_idx, recent_boundary_artists=…)` (~1926; def is keyword-only ~1026) → `segment_path = attempt_result["segment_path"]` (~1939);
5. runs infeasibility **fallbacks** if `segment_path is None` (transition/genre-floor relaxation retries, micro-pier; ~1981–2150).

**The seam:** extract steps 2 + 4 (the length-coupled genre-target build + the primary `_run_segment_backoff_attempts` call) into a local closure
```
def _build_segment_at(interior_len: int) -> dict:   # returns the attempt_result dict ("segment_path", soft-penalty counts, …)
```
keyed only on `interior_len` (capture `pier_a/pier_b/seg_idx/cfg/far_stats/recent_boundary_artists/…`). The length-INDEPENDENT far_stats (step 3) and the fallbacks (step 5) stay in the loop body, **outside** the closure.

**When the feature is OFF** (`variable_bridge_length=False`): `attempt_result = _build_segment_at(segment_lengths[seg_idx])` exactly once → **byte-identical to today** (same genre targets, same backoff, same fallbacks). This is the single most important property to preserve — prove it with Step 3.

**When ON:** add the config reads before the loop (`_vbl`, `_vbl_k`=flex, `_vbl_band`=band, `_vbl_good`=min_edge, `_vbl_eps`=epsilon, `_vbl_total_dev=0`), importing `segment_bottleneck`/`choose_segment_length` from `src.playlist.pier_bridge.var_bridge` and `score_transition_edge` from `src.playlist.transition_metrics` under `if _vbl:`. Then in the loop body drive the closure:
- `nominal = int(segment_lengths[seg_idx])`. Band-clamp: `lo = max(1, nominal − _vbl_k, nominal − (_vbl_band + _vbl_total_dev))`, `hi = max(lo, nominal + min(_vbl_k, _vbl_band − _vbl_total_dev))` (keeps the running total in `[N−m, N+m]`).
- **Budget guard (the hard 90 s ceiling):** before flexing, if `(deadline is not None and time.monotonic() > deadline − 5.0)` OR `(time.monotonic() − _pb_build_start) > 0.55 * float(cfg.generation_budget_s)`, force `lo = hi = nominal` (no flex). `deadline` is the builder param (~457); `_pb_build_start = time.monotonic()` is set ~1721.
- `edge_T = lambda a, c: float(score_transition_edge(transition_metric_context, a, c).get("T", 0.0))`.
- `def _build_and_score(l): r = _build_segment_at(l); p = r["segment_path"]; b = segment_bottleneck([int(pier_a), *[int(x) for x in p], int(pier_b)], edge_T)[0] if p else float("-inf"); return (r, b)`.
- `chosen_len, attempt_result = choose_segment_length(nominal, lo, hi, _build_and_score, good_enough=_vbl_good, eps=_vbl_eps)`; then `interior_len = chosen_len`, `_vbl_total_dev += chosen_len − nominal`, `segment_path = attempt_result["segment_path"]`, and continue the loop body so the existing fallbacks run on `segment_path` exactly as today when it is `None`. `logger.info("Var-bridge seg %d: nominal=%d chosen=%d total_dev=%+d", seg_idx, nominal, chosen_len, _vbl_total_dev)`.

Verified facts to rely on (do not re-derive): `_run_segment_backoff_attempts` keyword-only signature at ~1026; `transition_metric_context` built ~731, passed to the beam ~1525; `deadline` param ~457; `_pb_build_start` ~1721; `score_transition_edge(context, a, b)["T"]` at `transition_metrics.py:153`. If anything doesn't match when you read the code, STOP and report BLOCKED rather than guessing.

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
