# Total (never-fail) generation — design

- **Date:** 2026-06-15
- **Status:** Approved (brainstorming complete; ready for implementation plan)
- **Origin:** Recommendation #1 of `audit/beam_pier_bridge_audit_2026-06-15.md`.
- **Baseline:** `master` @ `ccd2070` (post the cleanup/v6 ↔ web-search merge).

## Principle

Generation is **total**: it always returns a playlist as long as the library has enough tracks
for the requested length. The genre / sonic / pace sliders are **guidelines** that relax
per-segment as far as needed; diversity / recency / dedup are **invariants** that hold through
every guideline rung and are breached only by a terminal fallback when physically unavoidable.
Every bend is recorded and surfaced.

This replaces today's behavior, where a hard bridge raises
`ValueError("Segment N infeasible …")` (`pipeline/core.py:703`) and the whole run 500s — a
violation of the project's own principle 25 ("infeasible bridge → progressively relax … don't
crash").

## Confirmed problem (current master)

A faithful `generate_like_gui` re-baseline (5 piers; `genre_mode=narrow sonic_mode=narrow
pace_mode=dynamic cohesion_mode=dynamic`) **still fails** on `ccd2070`:
`Segment 1 infeasible under bridge_floor backoff (attempted=[0.02, 0.01, 0.0];
last_reason=no valid continuations at step=0)`. The segment pool had ~2289 candidates at
`bridge_floor=0` — so it is the *per-edge* gates rejecting every one, not pool starvation.

## Decisions (from brainstorming)

1. **Per-segment, surgical relaxation.** Only the failing bridge loosens its own gates; the rest
   of the playlist keeps the user's exact settings.
2. **Tiered.** Guidelines (genre/sonic/pace) relax freely; invariants (diversity/recency/dedup)
   hold through all guideline rungs and are breached only by the terminal fallback.

## Design

### 1. Where it hooks in

`pier_bridge_builder.py`'s per-segment backoff today relaxes only `bridge_floor` (0.02 → 0) and
widens the BPM/onset caps, then raises (the `infeasible under bridge_floor backoff` message at
`pier_bridge_builder.py:1888`). We replace the *raise* with a continued, ordered relaxation
**ladder** that re-runs the segment beam at progressively looser gates, ending in a terminal
placement that cannot fail. The beam (`pier_bridge/beam.py`) gets its gate thresholds
**parameterized per attempt** so the same search runs at the relaxed floors:
- the genre-arc steering floors `arc_step_floor` / `cfg.genre_arc_floor` (`beam.py:1194-1198`),
- the sonic-progress requirement + `transition_floor`,
- the pace gate (`pace_bridge_floor`).

### 2. The per-segment relaxation ladder

Re-attempt the segment after each rung; record what bent. Stop at the first rung that yields a
full segment.

1. `bridge_floor → 0`, widen BPM/onset *(existing behavior)*
2. **genre-arc steering floors → 0** — drop genre steering for this one bridge
3. **sonic-progress relaxed + `transition_floor` → 0** — allow a non-progressing / low-cosine step
4. **pace gate → off** for this bridge
   — *guideline tiers exhausted; invariants still enforced above this line* —
5. **Terminal guarantee:** place the best-scoring **unused** candidate (by the beam's combined
   score) ignoring the guideline gates. Cannot fail while an unused track exists. Only here may an
   invariant (min-gap / recency) be breached, and only when there is no alternative — recorded as
   an invariant breach.

**Ladder order rationale:** genre steering is the softest guideline to drop on a hard bridge
(it's a cultural-tag target, not a sonic adjacency), so it relaxes first; sonic-progress/transition
next; pace last. Order is a module-level constant, tunable.

### 3. Reporting

`pier_bridge_builder` emits a structured per-segment list:
```
{ "bridge": "I'm on Fire → I Am So Happy With My Little Dog",
  "relaxed": ["genre steering", "sonic progress"],
  "severity": "guideline" | "invariant" }
```
carried on the generation result (a new `relaxations` field, alongside the existing metrics) →
worker → API → GUI. The GUI shows a **dismissible non-blocking notice** above the playlist
("Relaxed to fit: the *I'm on Fire → I Am So Happy With My Little Dog* bridge dropped genre
steering + sonic progress to stay connected"). Guideline bends render as info; invariant breaches
as a stronger warning. The existing per-segment INFO logs stay.

### 4. Backward-compat

Behind `pier_bridge.guarantee_feasible`, **defaulting to `true`** (the current crash is a bug,
not a feature). `false` restores the legacy raise — useful for calibration/debugging where you
*want* to see infeasibility.

### 5. Non-goals

- Does **not** touch the audit's other recommendations (tower-vs-MERT dead path, genre-layer
  consolidation, floor recalibration). Those are separate efforts; this one only guarantees
  feasibility on top of whatever gates exist.
- Does **not** change the seed-order optimizer (the order that produced the hard adjacency is
  left as-is; the bridge is made to always succeed instead).
- No new sonic/genre features; no topology change.

## Testing

Through the `gui_fidelity` multi-pier harness (per the `playlist-testing` skill — never a
hand-built single-seed):

- **Repro → fixed (integration):** the exact 5-seed case above must now return a **full 30-track
  playlist** with a populated `relaxations` report naming the *I'm on Fire → I Am So Happy…* bridge
  (currently raises `ValueError`). Reference this spec + the fixing commit; add to
  `tests/integration/test_gui_fidelity_regressions.py`.
- **No-regression (integration):** a normally-feasible run produces an **empty** `relaxations`
  list and **byte-identical** track output (the ladder never fires).
- **Unit (`pier_bridge_builder` / `beam`):** ladder ordering (genre → sonic → pace → terminal);
  terminal fallback places an unused track when all guideline gates are off; `guarantee_feasible:
  false` restores the raise.

## Files touched

| File | Change |
|---|---|
| `src/playlist/pier_bridge_builder.py` | backoff → relaxation ladder + terminal; emit `relaxations` |
| `src/playlist/pier_bridge/beam.py` | accept per-attempt relaxed thresholds (genre-arc / progress / transition / pace) |
| `src/playlist/pier_bridge_diagnostics.py` | capture relaxation events |
| `src/playlist/pipeline/core.py` | carry `relaxations`; do not raise when `guarantee_feasible` |
| `src/playlist_gui/worker.py` + web schemas + `web/src` | surface `relaxations` as a GUI notice |
| `config.example.yaml` (+ local `config.yaml`) | `pier_bridge.guarantee_feasible: true` |
| `tests/integration/test_gui_fidelity_regressions.py`, unit tests | per above |

## Risks / open items

- **Terminal fallback quality:** placing "best unused candidate ignoring gates" can pick a jarring
  track on a truly impossible bridge. Acceptable per the principle (a connected playlist + a
  warning beats a crash), but the report must make it visible so the user can re-seed.
- **`transition_floor` post-merge** is still 0.20 vs MERT pairwise median ~0.144 — the ladder makes
  hard bridges *feasible* but does not *recalibrate*; that's audit rec #4, separate.
- The relaxation order is a heuristic; expose it as a constant so it can be tuned without a code
  change.
