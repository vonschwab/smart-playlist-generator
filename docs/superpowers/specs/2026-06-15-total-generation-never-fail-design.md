# Total (never-fail) generation — design (rev. 2)

- **Date:** 2026-06-15
- **Status:** Approved (revised after reading the merged code; ready for implementation plan)
- **Origin:** Recommendation #1 of `audit/beam_pier_bridge_audit_2026-06-15.md`.
- **Baseline:** `master` @ `ccd2070`.

## Principle

Generation is **total**: it always returns a playlist as long as the library has enough tracks
for the requested length. Genre/sonic/pace are **guidelines** that relax per-segment as far as
needed; diversity/recency/dedup are **invariants** that hold through every guideline rung and are
breached only by a terminal fallback when physically unavoidable. Every bend is recorded and
surfaced. This honors principle 25 ("infeasible bridge → progressively relax … don't crash").

## What's actually wrong (revised — the ladder already exists)

Reading the merged `pier_bridge_builder.py`, the relaxation ladder is **already present and
enabled** (the original draft of this spec was wrong to assume a from-scratch build):

- bridge_floor backoff (`_run_segment_backoff_attempts`, gated on `infeasible_handling.enabled`,
  which **is** true in the live config — `[0.02,0.01,0.0]` is the *enabled* backoff),
- a transition-floor tier (`pier_bridge_builder.py:1693-1736`),
- a genre-arc-floor tier that jointly sweeps transition floor (`1738-1800`),
- a micro-pier tier (`1813-1894`),
- then it returns a **failure** `PierBridgeResult` (`1912-1921`) → `pipeline/core.py:703` raises the
  `ValueError` the user sees.

**Three concrete defects make it fail anyway:**

1. **The transition-floor tier is inert.** `InfeasibleHandlingConfig.min_transition_floor`
   defaults to **0.20** and the live `transition_floor` is also **0.20**, so
   `_transition_floor_attempts` short-circuits (`pier_bridge_builder.py:839`) and returns a single
   value — the tier never lowers the one gate the audit found binding (0.20 > MERT pairwise median
   ~0.144). `min_genre_arc_percentile` (0.40) likewise never reaches 0.
2. **Even fully relaxed, the existing tiers don't guarantee feasibility.** Empirically confirmed:
   a faithful `generate_like_gui` re-baseline with `min_transition_floor=0.0` **and**
   `min_genre_arc_percentile=0.0` **still fails** at "Segment 1 … no valid continuations at
   step=0." So the binding gate at step 0 is one the tiers do **not** relax — the sonic-progress
   requirement, the local-sonic edge policy (`beam.py:1219`), or the pace gate.
3. **There is no terminal guarantee** — after every tier the builder returns failure.

## Design

### 1. Fix the inert min-floor defaults

`src/playlist/run_audit.py` `InfeasibleHandlingConfig`: `min_transition_floor: 0.20 → 0.0` and
`min_genre_arc_percentile: 0.5 → 0.0`. Mirror in `config.example.yaml` (and document for
`config.yaml`). This lets the *existing* tiers relax their floors to zero — cheap relaxations the
ladder should exhaust before the terminal placement. (Necessary but, per defect #2, not
sufficient.)

### 2. Parameterize the beam for an "all-gates-off" terminal attempt

`pier_bridge/beam.py` already accepts `transition_floor_override` and
`genre_arc_floor_percentile_override`. Add overrides to also disable, for a single attempt, the
gates the tiers don't currently relax:
- sonic-progress / monotonic-progress requirement,
- the local-sonic edge policy (`_apply_local_sonic_edge_policy`),
- the pace gate (`pace_bridge_floor`).
With all of these off and `transition_floor=0` / `bridge_floor=0` / genre-arc off, any unused track
can follow any track → the beam can always produce a path while a candidate pool is non-empty.

### 3. Terminal tier in `pier_bridge_builder.py`

Immediately before the failure return (`pier_bridge_builder.py:1908`), when `segment_path is None`
and `cfg.guarantee_feasible`:
1. **Terminal beam attempt** — one `_run_segment_backoff_attempts` call in "terminal mode" (all
   guideline gates off, per §2). This succeeds in essentially all real cases.
2. **Greedy last resort** — if the terminal beam still returns nothing (degenerate pool), pick the
   `interior_len` best-scoring **unused** candidates by raw sonic+genre score and order them by
   progress toward `pier_b`. Breaches invariants (min-gap/recency) only if no alternative exists.
   Cannot fail while ≥ `interior_len` unused tracks exist.
Set `segment_path` and record a relaxation event (see §4). The existing failure return stays as the
hard floor only when `guarantee_feasible` is `false`.

### 4. Reporting

The builder already appends structured `warnings` (`pier_bridge_builder.py:1804,1852`). Add a
typed `relaxation` warning per segment that records the **deepest tier used** and the gates
dropped:
```
{ "type": "relaxation", "scope": "segment", "segment_index": 1,
  "bridge": "I'm on Fire → I Am So Happy With My Little Dog",
  "relaxed": ["transition_floor→0", "genre steering", "sonic progress (terminal)"],
  "severity": "guideline" | "invariant" }
```
Surface the list on the generation result → worker → API → GUI as a **dismissible notice** above
the playlist. Guideline bends render as info; invariant breaches as a warning. Existing per-segment
INFO logs stay.

### 5. Backward-compat flag

`pier_bridge.guarantee_feasible`, **default `true`**. `false` restores the legacy failure return
(useful for calibration/debugging where you *want* to see infeasibility).

## Non-goals
- The other audit recs (tower-vs-MERT dead path, genre-layer consolidation, floor recalibration)
  are separate.
- No change to the seed-order optimizer.
- *(Optional, deferred):* pinning + adding a dedicated ladder rung for the exact step-0 binding gate
  (so we relax it specifically before the terminal). The terminal guarantee makes generation
  total without it; this is a later quality refinement.

## Testing (through the `gui_fidelity` multi-pier harness)
- **Repro → fixed (integration):** the exact 5-seed case (`genre_mode=narrow sonic_mode=narrow
  pace_mode=dynamic cohesion_mode=dynamic`, length 30) now returns a **full 30-track playlist**
  with a populated `relaxation` report naming the *I'm on Fire → I Am So Happy…* bridge (currently
  raises `ValueError`). Add to `tests/integration/test_gui_fidelity_regressions.py`.
- **No-regression (integration):** a normally-feasible run produces **no `relaxation` warning** and
  **byte-identical** output (terminal tier never fires).
- **Unit:** `min_transition_floor` default 0.0 lets `_transition_floor_attempts` step below 0.20;
  the terminal beam attempt with all gates off returns a path on a pool where the gated beam
  returns none; the greedy last resort returns `interior_len` unused tracks; `guarantee_feasible:
  false` restores the failure return.

## Files touched
| File | Change |
|---|---|
| `src/playlist/run_audit.py` | `InfeasibleHandlingConfig` min-floor defaults → 0.0 |
| `src/playlist/pier_bridge/beam.py` | per-attempt overrides to disable progress / local-sonic / pace gates |
| `src/playlist/pier_bridge_builder.py` | terminal tier before the failure return; emit `relaxation` warning |
| `src/playlist/pier_bridge/config.py` (+ `pier_bridge_overrides.py`) | `guarantee_feasible` knob |
| `src/playlist/pipeline/core.py` | carry `relaxation` warnings; don't raise when feasible-guaranteed |
| `src/playlist_gui/worker.py` + web schemas + `web/src` | surface relaxations as a GUI notice |
| `config.example.yaml` | min-floor defaults + `guarantee_feasible: true` |
| `tests/…` | per above |

## Risks
- **Terminal quality:** the terminal placement can pick a jarring track on a truly impossible
  bridge — acceptable per the principle (connected playlist + warning beats a crash), and the
  report makes it visible so the user can re-seed.
- **Disabling gates in the beam** must be strictly scoped to the terminal attempt — a leak would
  silently drop quality on normal runs. The unit test (gated-beam-none vs terminal-beam-path)
  guards this.
