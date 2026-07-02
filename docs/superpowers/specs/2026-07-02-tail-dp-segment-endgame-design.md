# Segment tail re-optimization (tail-DP) — design

**Date:** 2026-07-02
**Status:** approved (design), pending implementation plan
**Scope:** per-segment "endgame" fix for the beam's landing blindness, plus two
wiring-hygiene fixes found by the Phase-2 investigation. Full bidirectional
build (c-full) explicitly deferred with a reconsider trigger.

## Evidence (Phase-2 beam investigation, 2026-07-02)

Instrumented pure-beam Alvvays runs (edge repair disabled via `t_floor=0`):

- The beam's landing edges are chosen effectively blind: the last interior slot
  is scored with only `dest_pull = 0.1·cos` awareness of the destination pier;
  per-step minimax prunes on interior-edges-so-far. **6/9 segments left >0.2 of
  in-pool landing quality unused; mean 0.292.**
- Attribution at the worst segment (M83 "Outro", landing T=0.059): anti_center
  penalized nothing (0.000 both candidates) — the collapse-preventer is NOT
  hiding good candidates. The damage was TWO slots deep: given the committed
  second-to-last track, no candidate could both connect and land.
- Tail-window probe (re-open last 2 slots, exact DP over the segment pool):

  | seg | chosen | tail1 (repair ceiling) | tail2 (this design) | fullOpt (segment ceiling) |
  |----|-------|-------|-------|--------|
  | M83 Outro | 0.059 | 0.393 | 0.545 | 0.583 |
  | Magdalena Bay | 0.300 | 0.515 | 0.643 | 0.788 |
  | Forget About Life | 0.472 | 0.678 | 0.799 | 0.800 |
  | Bored In Bristol | 0.546 | 0.859 | 0.917 | 0.931 |
  | Many Mirrors | 0.692 | 0.882 | 0.891 | 0.902 |

  `fullOpt` = exact-length bottleneck-path DP over the pool (upper bound,
  constraints ignored) ≈ what a full bidirectional rebuild could reach.
  **tail2 captures ~85–95% of fullOpt at ~10× less risk**; on the hardest
  segment even fullOpt is pool-capped (0.583), so bidirectional adds little.
- Probe scripts: `C:\Users\Dylan\.claude\jobs\5368076c\tmp\probe_beam_landing.py`,
  `probe_tail_dp.py` (session scratch; the DP math below is validated by them).

## Decisions (Dylan, 2026-07-02)

- Build **c-tail** (this design). **Skip (a)** (last-step landing term) — subsumed.
- **Defer c-full (bidirectional beams + join).** Reconsider trigger: post-c-tail
  audits showing segments with a large achieved-vs-ceiling gap (the bottleneck-DP
  ceiling probe machinery exists to measure this).
- Break-glass edge repair stays as the downstream global safety net; c-tail is a
  per-segment quality pass, not a replacement.
- Future (not this spec): seed ordering could consume the per-pier-pair
  bottleneck ceiling to avoid low-ceiling pier pairings (M83-class segments).

## Design

### 1. Tail-DP pass (new module `src/playlist/pier_bridge/tail_dp.py`)

Pure, unit-testable function applied to each **finalized** segment (after the
beam and after var-bridge picks the segment length), before the segment is
appended to the playlist:

- **Window:** the last `min(2, interior)` interior slots of the segment
  (positions adjacent to the landing pier). Interior 1 degenerates to a single
  slot; interior 0 → no-op.
- **Objective:** maximize the **window min-edge** —
  `min( T(prefix_end→x), T(x→y), T(y→pier_b) )` for the 2-slot case — where T is
  the exact shared transition metric the beam uses (`_score_shared_transition`
  context). Pure T; path-shaping modifiers get no vote (matches the repair
  precedent and the attribution evidence that modifiers were not the culprit).
- **Candidates:** the segment's own candidate pool. Hard constraints enforced:
  tracks already used anywhere in the playlist so far (including this segment's
  kept prefix); dedup by identity track-key; pier/seed-artist interior bans
  (`disallowed_artist_keys` as in repair); positional **min_gap** (distance
  ≤ min_gap violates, matching `_enforce_min_gap_global`), checked against the
  kept prefix AND across the upcoming pier; per-artist caps where applicable.
  Implementation shape: vectorized prefilter (used/banned artists) → DP on the
  filtered T matrices → validate top-K pairs against the exact positional
  checks until one passes.
- **Acceptance (never-worse):** replace the window only if the new window
  min-edge exceeds the current window min-edge by ≥ `tail_dp_epsilon` (0.02).
  Otherwise leave the segment untouched. Never raises; on any internal error,
  log and keep the original segment.
- **Weak-landing trigger (`tail_dp_floor`, default 0.30):** re-optimize a segment
  only when its current landing-window min-edge is below the floor — matching the
  break-glass repair's under-threshold philosophy (Dylan, 2026-07-02: only override
  the beam's pace/genre balancing when the endgame is actually weak, not for a
  marginal pure-T gain on an already-good landing; superseding the first draft's
  always-on choice, which a pair of pace/genre-gate tests flagged as over-reaching).
  `tail_dp_floor: 0` = always-on (no gate). Never-worse (≥ `tail_dp_epsilon`) still
  guards each accepted swap. Cost ≈ 3 vectorized matrix ops per *triggered* segment
  — negligible against the 90s budget.
- **Config (Layer 4):** `pier_bridge.tail_dp_enabled: true` — live default;
  rollback `false` restores today's beam output byte-identically.
  `tail_dp_epsilon: 0.02`. Window size fixed at 2 (not a knob; YAGNI).
- **Diagnostics:** one INFO line per applied re-optimization (segment, old→new
  window min-edge, swapped tracks) + a per-generation summary count; a
  `tail_dp` entry in the segment diagnostics dict.

### 2. Composition order (unchanged pieces stay unchanged)

beam → var-bridge length choice → **tail-DP (this)** → cross-segment assembly →
global break-glass edge repair (t_floor 0.30) → reporter. Repair still catches
cross-segment boundary edges and anything tail-DP's constraints refused.

### 3. Wiring hygiene (same feature, separate commits)

- **F2 — lying tuning log in artist mode:** `apply_pier_bridge_overrides`
  (`pier_bridge_overrides.py:52-77`) logs "Pier-bridge tuning resolved:
  weight_bridge=… weight_transition=…" from `tuning`, then discards tuning when
  a pre-built `pier_bridge_config` is supplied (artist mode). Fix: log the
  values from the pb_cfg actually in effect, and when the pre-built config wins,
  say so explicitly (e.g. "(artist-style pier config supplied; resolved tuning
  weights not applied)"). No behavior change — logging only.
- **F3 — stale hard-floor comments in `beam.py`:** "Hard floors: transition +
  bridge-local" (~1248) and "Hard floor on final transition" (~1800) describe a
  T-gate that `is_broken_transition` no longer applies. Reword to
  "anti-alignment safety only (no T-floor since the roam promotion)".

## Testing

- **Unit (pure DP helper):** synthetic 2-D vectors (repair-test pattern,
  `center_transitions=False` so T = plain cosine): picks the max-min pair;
  never-worse guard keeps original when no pair clears epsilon; 1-slot
  degenerate case; used/dedup/artist/min_gap refusals; deterministic tie-break.
- **Builder integration:** segment path through the builder with tail_dp on vs
  off (`tail_dp_enabled=false` byte-identical to today).
- **Live validation:** pure-beam Alvvays run with tail-DP on and repair off —
  weak-segment window min-edges should land near the probe's tail2 column
  (≈0.54/0.64/0.80/0.92); then a normal run (repair on) to confirm composition;
  suite + goldens (config-snapshot goldens gain the new knobs — deliberate
  regen, diff-audited).

## Deferred

- **c-full (bidirectional beams + meet-in-the-middle join):** reconsider only if
  post-c-tail audits show a persistent achieved-vs-ceiling gap. Costs direction-
  aware rewrites of progress monotonicity, genre-arc targets, roam corridors,
  var-bridge in the hottest file.
- **(a) last-step landing term:** subsumed by c-tail.
- **Ceiling-aware seed ordering:** use the bottleneck-DP per-pier-pair ceiling in
  seed ordering to avoid low-ceiling adjacencies. Separate future design.
