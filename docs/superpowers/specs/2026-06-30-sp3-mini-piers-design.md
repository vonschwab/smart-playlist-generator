# SP3 — Mini-Piers v2 (structural anti-sag) — Design

**Date:** 2026-06-30
**Status:** Design approved (brainstorming). Next: implementation plan.
**Depends on:** SP2-B (`seed_character_mode: anti_center`, live on master) and the
collapse harness + within-bridge sag metric ([[project_collapse_attack_design]]).

## Problem

Within-bridge sag ([[project_collapse_attack_design]] face 2): the interior of a
bridge between two character-X piers loses that character and settles into the local
generic average. SP2-B (soft anti-center penalty) reduces this but is a **partial**
dent — it *plateaus* (e.g. dreampop sag 117%→101% no matter the strength), because a
soft re-rank can't fix a bridge if the on-character tracks aren't in the pool: it
demotes the blur, but if the alternative is also blur, the bridge still sags.

The structural remedy: **pin a high-character waypoint inside a long bridge** so the
beam *cannot* drift past it. A shorter span anchored on-character at both ends can't
sag as far — this bounds the sag structurally rather than re-ranking softly.

## Why not v1

v1 = the `dj_micro_piers` feature (`src/playlist/pier_bridge/micro_pier.py`,
`_attempt_micro_pier_split`). It failed for two reasons this design fixes:
1. **Reactive + local.** It fires only *when a segment fails* and splits it in place,
   rebuilding that segment and **depleting the downstream pools** so later segments
   regress ([[project_mini_pier_v1_failed_validation]]).
2. **Blur-pinning selection.** Its metric is `max_min_sim` — the track closest to
   *both* piers, which is the central blur (verified in the SP3 probe: `max_min_sim`
   pins AprilBlue / Wild Nothing–Live-In-Dreams / The Umbrellas, cent 0.87–0.92).
3. It lives in the **`dj_bridging` path**, which the live config disables
   (`genre_steering_enabled supersedes dj_bridging`), so it is dead in the path we run.

v2 is a fresh mechanism in the **live segment_scored/steering path**; it does not touch
v1's code (whether to delete the dead `dj_micro_piers` path is a separate cleanup).

## Design

### Trigger — length, up front, recursive (phase 1)

After seed-ordering (`ordered_seeds`) and **before** any segment is built: for any
segment whose interior would exceed **K** tracks (start **K = 5**, config knob),
select a waypoint and recursively split until every segment's interior ≤ K.

Length-only first because it is simple, deterministic, and computed before any build
(so it is global-safe). **Phase-2 refinement (documented, not built now):** a
pier-distance-aware trigger — sag risk ≈ `length × (1 − sim(pier_a, pier_b))` — since
distant piers are the ones whose shortest smooth path crosses the generic center.

### Selection — smoothness floor → anti-center (validated by eye)

For a segment `(pier_a, pier_b)`, from the **global admitted candidate pool** (the
deduped universe already gated for the run), excluding seed/pier artists and
already-used tracks:

1. **Relative smoothness floor.** Keep candidates whose `min(sim(c, A), sim(c, B))` is
   within `margin` (start **0.12**) of the best available min-sim. Relative, so it
   adapts to close piers (tight floor) vs cross-niche piers (looser). This is the fix
   for outlier-chasing — it excludes distant off-character tracks (e.g. the
   ambient-guitar pick that surfaced with a naive top-K).
2. **Anti-center within.** Among those genuinely-smooth candidates, pick the
   **least-central** relative to the local between-region centroid (SP2-B's winning
   signal). This escapes the wallpaper the smoothness floor alone would keep.

Validated in the SP3 probe (`scripts/research/mini_pier_probe.py`): between iconic
piers it picks distinctive on-niche waypoints (Beach House Myth~Space Song →
Alvvays / Bilinda Butchers / For Tracy Hyde; Slowdive Kisses~Alison → Cate le Bon /
The Embassy / Peel Dream Magazine) and never the Cocteau Twins / Candy Claws blur,
while `max_min_sim` pins the wallpaper every time. **Behavior is relative:** the
waypoint matches the piers' own distinctiveness level — iconic piers → distinctive
waypoints; genuinely hazy piers → the *least*-hazy option (their baseline is haze).

### Global-awareness — the v1 fix

Waypoints are chosen **up front and spliced into the pier sequence** (they become real
piers). `num_segments`, `total_interior`, and `segment_lengths` then recompute over the
augmented pier set, so each shorter segment draws its **own** pool between its **own**
closer piers. No reactive rebuild, no downstream depletion.

### Composition

- **var-bridge** flexes interior *lengths* within the now-shorter segments; mini-piers
  set the pier *structure*. Orthogonal.
- The waypoint is a **real track** the listener hears, occupying an interior slot —
  total playlist length unchanged.
- It is a normal playlist member for diversity: it respects `min_gap`, artist caps, and
  seed/pier-artist exclusion.

## Components

- **New pure module** `src/playlist/pier_bridge/mini_pier_select.py` (unit-testable,
  no engine — like `collapse_metric.py` / `seed_character.py`):
  - `select_waypoint(pier_a, pier_b, candidate_indices, X_full_norm, margin, exclude) -> Optional[int]`
    — the smoothness-floor + anti-center pick; returns None if no feasible candidate.
  - `plan_waypoints(ordered_seeds, total_interior, K, ...) -> list[int]`
    — recursively decides split points and returns the augmented pier sequence.
- **Integration** in `build_pier_bridge_playlist` (`pier_bridge_builder.py`), right
  after `ordered_seeds` is finalized (~line 902) and before `segment_lengths` (~920),
  behind a config flag. Flag off ⇒ `ordered_seeds` unchanged ⇒ byte-identical.
- **Config** (`PierBridgeConfig` + `apply_pier_bridge_overrides`, off by default):
  `mini_pier_enabled: bool = False`, `mini_pier_max_interior: int = 5` (K),
  `mini_pier_smoothness_margin: float = 0.12`.

## Validation

- Off-by-default ⇒ pier-bridge config goldens + audit-matches-beam stay green.
- Collapse harness (`collapse_eval.py` + `collapse_rescore.py`): CI **and** within-bridge
  sag vs the SP2-B baseline, under the quality floor (seed-sim + worst-edge must not
  crater). Target the dreampop plateau specifically.
- **Real playlists are the verdict** — audition on diverse and few-seed runs (the long
  bridges are where sag lives).

## Knobs to sweep in implementation

- `K` (trigger length) and `margin` (smoothness floor) — swept like SP2-B's strength.

## Deferred / out of scope

- Pier-distance-aware trigger (phase 2, once the machinery is proven).
- "Pull hazy-seeded playlists *up* toward character" vs match-the-piers (default) — a
  by-ear taste knob to settle later.
- Deleting the dead v1 `dj_micro_piers` path — separate cleanup, not required for v2.
