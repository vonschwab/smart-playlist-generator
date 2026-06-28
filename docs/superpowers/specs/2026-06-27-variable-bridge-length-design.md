# Variable Bridge Length — Design

**Date:** 2026-06-27
**Status:** Design approved (brainstorming) → implementation plan next.

## Goal

Make playlists **always good** (no broken transition) and **sometimes great**, by
letting each pier-bridge **choose its own length** so it lands smoothly on the
next pier — instead of the rigid even split that forces a segment to pad with a
bad edge just to "make length." This directly targets north-star #5: *the worst
edge defines the experience.*

Success = on the validation corpus, the **worst edge lifts** across seeds
(measured min transition `T`), **without** reducing diversity, breaking the
energy/progress arc, exceeding the 90 s budget, or pushing the track count
outside the requested band.

## Background (what the research established)

- The worst edges across every sweep this session were **not collapse** — they
  were weak **pier-return / outlier edges**, mostly on *unflagged* seeds (Vegyn
  minT 0.023, Modest Mouse 0.024, Horse Jumper 0.079, Carly Rae 0.092). The
  beam-redundancy lever (Lever 1) was abandoned as a 1-in-12 niche
  ([[project_beam_redundancy_negative_result]]); this is the higher-value lever.
- Today `segment_lengths` is a rigid even split
  (`pier_bridge_builder.py` ~line 898: `base_length = total_interior //
  num_segments`). Every segment must place an exact interior count regardless of
  how bridgeable its two piers are — so a segment whose piers don't connect
  smoothly in N tracks is still forced to place N, padding with a bad edge.
- The beam **already keeps the next pier in mind**: every candidate's score
  includes `bridge_score = 2·sim_a·sim_b/(sim_a+sim_b)` (harmonic mean of
  similarity to *both* piers) plus a `dest_pull` toward pier_b. What's missing is
  letting the bridge **choose when to land**.
- The **min-bottleneck / minimax worst-edge** principle is already used by the
  roam-corridors engine ([[project_roam_corridors_engine]]) — this is the same
  idea applied to *length*, not new machinery.

## Core mechanism — minimax-over-length adaptive landing

Per segment, instead of building paths of a single fixed `interior_length`, the
beam produces the **best complete bridge at each length** in a bounded range, and
we pick the length whose *whole traversal* is smoothest.

1. **Nominal length** `L_nom` per segment = the current even-split value. The
   per-segment range is `[L_nom − k, L_nom + k]`, clamped to a hard floor
   (≥ 1 interior track).
2. **Variable landing in the beam.** As the beam extends partial paths, at every
   step `l ≥ L_nom − k` it evaluates each surviving length-`l` partial as a
   *complete* bridge — append pier_b, and compute that bridge's **bottleneck**
   (the worst edge over pier_a→…→pier_b, *including the return* last→pier_b). The
   beam keeps extending up to `L_nom + k`. Output per segment: a table
   `{ l : (path_l, bottleneck_l) }` for `l ∈ [L_nom−k, L_nom+k]`, where `path_l`
   is the best complete bridge of length `l` (the surviving length-`l` partial
   whose completed bottleneck is highest).
3. This realizes the intuition exactly:
   - *keep finding good candidates → let it roll*: if extending keeps every edge
     strong, the longer bridge's bottleneck stays high → it wins;
   - *next best doesn't reach the pier → cut short*: if the only way to extend is
     through a track that bridges poorly (to the next track **or** the pier), that
     path's bottleneck drops → a shorter bridge that already landed cleanly wins.

The return edge is **always** part of the bottleneck, so nothing can be hidden by
trimming — the trimmed bridge still has to land.

## Global reallocation — soft total band, prefer-N + ε (the anti-crutch)

Each segment's per-length bottleneck table feeds a single cheap global step that
chooses **one length per segment**.

- **Total is a soft band, not a rule.** "30 tracks" means `[N−m, N+m]` (default
  m=5 → 25–35). Track count is an arbitrary proxy (a 30-track punk playlist ≈
  45 min, a 30-track afrobeat playlist ≈ hours), so exact-N is false precision.
- **Objective = max-min bottleneck.** Choose the length vector
  `(l_1,…,l_S)`, each `l_s ∈ [L_nom_s − k, L_nom_s + k]`, with total in
  `[N−m, N+m]`, that **maximizes the playlist's worst edge** =
  `min_s bottleneck_s(l_s)`. Computed by a small DP over
  `(segment, cumulative_total) → best achievable min-bottleneck`
  (O(segments × total_range × lengths) — trivial).
- **Tie-break toward N (the earns-its-keep guard).** Let `B*` be the best
  achievable min-bottleneck over the band. Among all length vectors achieving
  `≥ B* − ε`, pick the one whose **total is closest to N** (further tie → closest
  to the even split). So the playlist **only moves off N when doing so strictly
  improves the worst edge by more than ε** — it cannot drift to the bottom of the
  band to quietly trim hard edges. This single rule is what makes it a feature,
  not a crutch.

## Anti-crutch guardrails (Dylan's central concern)

1. **Principled trigger** — flex is driven by traversal need (bottleneck), not
   "this edge scored low, delete it."
2. **Bounded** — `[L_nom−k, L_nom+k]` per segment, `[N−m, N+m]` overall.
3. **Earns its keep** — the prefer-N + ε tiebreak: no flex unless it beats the
   nominal's worst edge by a margin.
4. **Nothing hidden** — the return edge is always in the bottleneck.
5. **Shape preserved** — energy/progress arc, min-gap diversity, pier spacing all
   survive the reflow (see Interactions).

## Architecture

- **`src/playlist/pier_bridge/beam.py`** — `_beam_search_segment` gains a
  variable-length mode: accept `(l_min, l_max)` (or `interior_length` + flex `k`)
  and return the per-length table `{ l : (interior_path_l, bottleneck_l) }`
  instead of a single fixed-length path. The fixed path stays the default when
  the feature is off (return only `l = interior_length`). The bottleneck uses the
  same edge metric (`_score_shared_transition`) already computed per step.
- **`src/playlist/pier_bridge_builder.py`** — replace the even-split consumption:
  when enabled, run each segment's beam in variable mode, collect the per-length
  tables, run the **max-min reallocation DP** (new helper, e.g.
  `src/playlist/pier_bridge/allocate.py::reallocate_bridge_lengths(tables, N, m, k, eps)
  -> list[int]`), then assemble the chosen `path_l` per segment. When disabled,
  the current even-split path is byte-identical.
- **Config** (`PierBridgeConfig`): `variable_bridge_length: bool = False`,
  `variable_bridge_flex: int = 2` (k), `variable_bridge_band: int = 5` (m),
  `variable_bridge_epsilon: float` (ε, on the calibrated-T scale). Default OFF
  until the validation gate passes (discipline #22), wired through the artist +
  seeds paths like the other pier-bridge knobs.

## Interactions (must be preserved)

- **Energy / progress arc** — per-step targets are position-based; variable
  lengths shift positions. The arc targets must be computed **relative to the
  chosen segment length** (fraction-of-segment), not an absolute step index, so a
  shorter/longer bridge still traces the intended contour. Verify the arc loss is
  unchanged for the nominal length.
- **Diversity / min-gap** — per-segment one-per-artist and cross-segment carry
  are adjacency rules, independent of length; they continue to apply within each
  candidate path. Confirm no adjacent same-artist after reflow.
- **Relaxation cascade** — variable length *helps* feasibility (a shorter segment
  is easier to fill); the existing `_run_segment_backoff_attempts` backoff still
  governs a segment that can't fill even `L_nom − k`.
- **Determinism & budget** — the per-length table adds bounded work (≤ 2k+1
  completions per segment); must stay deterministic and within 90 s.

## Out of scope (this spec)

- **Outlier piers** — a pier nothing bridges to at *any* length (e.g. Vegyn's
  0.023 may be this) is a pier-**selection** problem, not a length problem.
  Keeping it out is what stops this feature from becoming the crutch that pretends
  to fix it.
- **Duration-based length** — targeting minutes instead of track count (the real
  fix for punk-vs-afrobeat) is a separate, larger change to how the system
  defines "length." Parked ([[project_variable_bridge_length_idea]]).
- **The abandoned beam-redundancy lever** — superseded; its code is stripped
  before this is built.

## Validation gate (the "always good" test)

On the corpus (the ~16 seeds swept this session, through the artist policy
layer), with the feature on vs off:
- **Worst edge lifts** — min `T` improves on the weak-edge seeds (Vegyn 0.023,
  Modest Mouse 0.024, Horse Jumper 0.079, Carly Rae 0.092), and no seed's worst
  edge regresses beyond noise.
- **Total stays in band** — every playlist's track count ∈ `[N−m, N+m]`, and sits
  at N unless a flex earned it.
- **Diversity not reduced**, **arc preserved** (energy/progress loss not worse),
  **min-gap intact** (no adjacent same-artist), **wall < 90 s**, deterministic.
- Default-OFF until the gate passes, then flip to the live default (keep the knob
  for rollback). Perceptual audition (Dylan) is the final gate.

## Self-review

- **Placeholders:** none. The two deferred specifics (exact DP formulation, ε
  scale) are bounded by an explicit contract (`reallocate_bridge_lengths` signature,
  max-min objective, ε on the calibrated-T scale) for the plan to implement test-first.
- **Consistency:** the bottleneck always includes the return edge (matches
  "nothing hidden"); prefer-N+ε matches "earns its keep"; bounds match "not too
  many"; the beam-already-knows-the-pier point matches the existing `bridge_score`.
- **Scope:** one mechanism (variable landing) + one global step (reallocation),
  default-off, single plan. Outlier piers and duration explicitly excluded.
- **Ambiguity:** "length" = interior track count per segment; "bottleneck" =
  min edge over the complete bridge incl. return; "total band" = track count in
  `[N−m, N+m]`; nominal = even split.
