# Phase 1 expected retirements — pre-registered before the corridor re-sweep

**Date:** 2026-07-17 · **Author:** Task 7 (prep) · **Purpose:** per
`docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md` and the design spec's
"automated completeness net," this list is written and committed **before**
the controller runs the full corridor-flag-ON knob sweep, so the sweep's
changed→inert transitions can be checked against a falsifiable prediction
rather than rationalized after the fact. Every entry is grounded in the
actual `pooling_mode == "corridor"` branches in
`src/playlist/pier_bridge_builder.py` (grep-complete: all 7 occurrences on
`c551f77` read and traced), not the spec's aspirational end-state table —
Task 7 runs mid-Phase-1 (dev flag only, legacy path still present), so only
what Tasks 1-6 actually wired is in scope. Each field's Phase-0a baseline
status (`docs/corridor_baseline/knob_sweep.json`) is quoted so the
prediction is checkable field-by-field once the real sweep lands.

Method: for each `if pooling_mode == "corridor"` branch, identify what
config family it swaps out entirely, then check whether that family's
fields were already `inert`/`unmapped`/`skipped_type`/absent at Phase 0a
(baseline `legacy` mode). Only fields that were **`changed` at Phase 0a**
and are **code-confirmed unreachable under `pooling=corridor`** are
registered as expected transitions — everything already inert at baseline
is excluded per the contract ("fields recorded inert at baseline are not
preservation targets").

## Legitimate retirement (corridor supersedes it correctly)

| Field | Baseline status (both cells) | Rationale |
|---|---|---|
| `playlist.pier_config.collapse_segment_pool_by_artist` | `changed` | Read only inside `_build_segment_candidate_pool_scored` (the `dj_union`/scored segment-pool builder, `pier_bridge_builder.py:1761`). The segment-pool builder selection at `pier_bridge_builder.py:1668-1717` is `if pooling_mode == "corridor": _build_corridor_segment_pool(...) / elif pool_strategy == "legacy": ... / else: _build_segment_candidate_pool_scored(...)` — corridor mode always takes the first branch, so the `else` branch (and everything it reads) never executes. No other read site. **Expected: `changed` → `inert`.** This is also CLAUDE.md's known project gotcha ("Segment-pool one-per-artist collapse is OFF by default... the beam enforces per-segment artist diversity on its own") — the corridor path structurally can't reach the knob that CLAUDE.md already says shouldn't be re-enabled, so its retirement is aligned with existing project intent, not a regression. |

No other field met both bars (Phase-0a `changed` + code-confirmed corridor-unreachable). The `dj_pooling_k_*`/`dj_pooling_cache_enabled`/`dj_pooling_debug_compare_baseline`/`initial_neighbors_m`/`max_neighbors_m`/`initial_bridge_helpers`/`max_bridge_helpers` fields are ALSO unreachable under corridor (same `else`-branch/`elif pool_strategy=="legacy"` argument), but were **already `inert` at Phase 0a** (verified against `knob_sweep.json`) — not a new transition, excluded per the contract's baseline-inert rule. `segment_pool_strategy` (`dj_pooling_strategy`) itself is a string field, perturbed to `skipped_type` at Phase 0a (never tested by the mechanical sweep either way) — its VALUE is moot under corridor (the `if pooling_mode == "corridor"` check short-circuits before `pool_strategy` is ever read), worth naming in Task 8's retired-warning list even though the sweep can't observe a transition for it.

## NOT legitimate retirements — contract gaps (BLOCKING, not pre-registered as expected)

These two fields are expected to show `changed` → `inert` in the real sweep,
but this is **not** a legitimate retirement — Contract Category C1
("Duration soft penalty ... duration-outlier probe still demoted by ≥
baseline margin") requires them GREEN, not retired. Flagging here so the
controller does not silently accept them as "corridor legitimately
retires this."

| Field | Baseline status (Bill_Evans_Trio/open) | Why it goes inert | Contract impact |
|---|---|---|---|
| `candidate_pool.duration_penalty_weight` | `changed` (jaccard=0.590, n_position_diffs=10/30, delta_min_T=+0.017) | `_build_corridor_segment_pool`'s single call to `build_eligible_universe` (`pier_bridge_builder.py:624-636`) passes `duration_reference_ms=None, duration_cutoff_multiplier=1.0, duration_penalty_weight=0.0` **hardcoded**, regardless of `cfg`. Documented in-code as a deliberate Phase-1 no-op deferred to "Task 4/5" (`pier_bridge_builder.py:572-583`, restated at `:1286-1298`) — Tasks 4/5 (relevance mask + widening ladder; bangers/tag/tail-DP/repair reseats) did not revisit this. **Empirically confirmed** this task (Task 7 smoke sweep + a dedicated C1 probe, both real generations under `pooling=corridor`, Bill Evans Trio/open): jaccard=1.0, n_position_diffs=0, delta_min_T=0.0, delta_mean_T=0.0 — fully inert, margin **0 < baseline margin**. | **Category C1 RED.** Duration soft-penalty is not rehomed onto the corridor path despite Task 2's title ("Eligible universe + C1/C10 rehome"); C10 (instrumental) WAS rehomed for real (see the probe below) but C1 was not. Blocks the merge gate ("every Category C term explicitly rehomed... a term with no rehoming target is a spec defect") until fixed. |
| `candidate_pool.duration_cutoff_multiplier` | `changed` (jaccard=0.561, n_position_diffs=20/30, delta_min_T=-0.0034) | Same root cause — feeds the same hardcoded-off `duration_rank_penalty` path. Empirically confirmed inert in this task's smoke sweep (jaccard=1.0, n_position_diffs=0, delta_min_T=0.0). | Same C1 gap. |

**Recommendation for the controller:** do not fold these two into Task 8's
"expected retired, add to warning list" bucket. They need a real C1 rehome
(wire a real `duration_reference_ms` — seed-track duration median, same
source `candidate_pool.py` already computes — and the real
`duration_penalty_weight`/`duration_cutoff_multiplier` into the
`build_eligible_universe` call at `pier_bridge_builder.py:629-631`) before
Phase 1 can be contract-GREEN. This is new implementation work, out of
Task 7's prep scope.

## Not sweep-observable, but code-confirmed bypassed (manual coverage note)

The `infeasible_handling.*` config family (`backoff_steps`,
`min_bridge_floor`, `max_attempts_per_segment`, `widen_search_on_backoff`,
`extra_neighbors_m`, `extra_bridge_helpers`, `extra_beam_width`,
`extra_expansion_attempts`, `transition_floor_relaxation_enabled`,
`min_transition_floor`, `genre_arc_relaxation_enabled`,
`min_genre_arc_percentile`) is explicitly bypassed under
`pooling_mode == "corridor"`:

- `_bridge_floor_attempts` forces single-attempt "regardless of
  `infeasible_handling.enabled`" when `pooling_mode == "corridor"`
  (`pier_bridge_builder.py:1412-1423`, explicit "anti-double-ladder gate"
  comment).
- The transition-floor relaxation tier is gated
  `pooling_mode != "corridor"` (`pier_bridge_builder.py:2879-2884`).
- The genre-arc-floor relaxation tier is gated `pooling_mode != "corridor"`
  (`pier_bridge_builder.py:2938-2943`).
- All three comments cite the same rationale: `_run_corridor_widening_ladder`
  (Task 4) is the corridor path's sole segment-level recovery mechanism: a
  second relaxation axis firing on top of it would be the "double-ladder"
  design spec §2 warns against.

**This family never appears in the flattened effective-config blob the
mechanical sweep reads** (verified: zero records for any `infeasible_handling.*`
key at Phase 0a, in either cell) — `infeasible_handling` is a separate
function parameter to `build_pier_bridge_playlist`
(`pier_bridge_builder.py:399`), constructed by the caller from its own
config path, not a field of the `PierBridgeConfig` instance the effective-
config logger serializes. The mechanical "no knob goes inert" sweep
structurally cannot see this transition — it is a coverage gap in the
automated net, not a contract violation, and is exactly the kind of thing
Contract Category B/D's "hand-authored assertions... cover what isn't a
single knob" clause exists for. Recorded here so Task 8's retired-warning
list includes it even though no sweep record will ever flag it, and so a
future reader doesn't mistake "the sweep shows nothing changed" for
"nothing changed."

## Already inert at Phase 0a (stays inert, not a new transition)

For completeness — these were already `inert`/`unmapped`/`skipped_type` in
`legacy` mode at Phase 0a (verified against `knob_sweep.json`), independent
of corridor, so their continued inertness under corridor is a non-event:
`candidate_pool.max_pool_size` (Phase-0 subtraction already killed the
artist-cap/backstop machinery that consumed it — `candidate_pool.py:1395-1409`,
"Vestigial (Phase 0 corridor rework, 2026-07-16)"), `candidate_pool.target_artists`,
`candidate_pool.candidates_per_artist`, `candidate_pool.seed_artist_bonus`
(all `unmapped` — genuinely hardcoded/computed, no yaml path ever reached
them, per `perturb.py`'s `_CANDIDATE_POOL_FIELD_MAP`), `playlist.pier_config.dj_pooling_k_local/k_toward/k_genre/k_union_max/step_stride/cache_enabled/debug_compare_baseline`,
`playlist.pier_config.initial_neighbors_m/max_neighbors_m/initial_bridge_helpers/max_bridge_helpers`,
`playlist.pier_config.dj_relaxation_*` (dj_bridging family, off by default),
`playlist.one_each_candidate_relaxation` (string, `skipped_type`).

## Explicitly NOT a retirement (confirmed still wired)

- `playlist.pier_config.segment_pool_genre_weight` — the brief's own
  worked example. `changed` at Phase 0a (jaccard-affecting). Read directly
  by `_build_corridor_segment_pool` (`pier_bridge_builder.py:1236`,
  `genre_blend_weight = ... getattr(cfg, "segment_pool_genre_weight", 0.0)`)
  and passed into `build_corridor`'s own genre blend
  (`genre_blend_weight=genre_blend_weight`) — genuinely rehomed onto the
  corridor path, not retired.
- C10 instrumental lean (`playlist.pier_config.instrumental_enabled`,
  `instrumental_penalty_weight`) — passed for real into
  `build_eligible_universe` (`pier_bridge_builder.py:633-634`,
  `instrumental_enabled=bool(cfg.instrumental_enabled),
  instrumental_penalty_weight=float(cfg.instrumental_penalty_weight)`).
  Confirmed live by this task's C10 probe (see the Task 7 report):
  jaccard=0.622, n_position_diffs=7/30, delta_min_T=+0.0077 — the term
  fires under corridor. Flag-off at Phase 0a baseline (contract note: "record
  its corridor-path effect as a NEW baseline datum, not a comparison") —
  this probe result IS that new datum.
- Popularity / Oops-All-Bangers gate (`playlist.popularity_*`,
  `playlists.bangers.*`) — reseated onto `corridor_universe` directly
  (`pier_bridge_builder.py:637-704`, "Task 5 reseat... Applied ONCE here,
  at universe build").
- Edge repair (`edge_repair_*`) — reseated onto the union of final segment
  corridors (`pier_bridge_builder.py:3675-3689`, "Task 5 reseat").
- Tag-steering pool guarantee — reseated into both `corridor_universe`
  (bangers-gate exemption, `:655-667`) and `_build_corridor_segment_pool`
  (force-include, `:1248-1271`).

## Cross-reference

`docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md` Category C1/C10,
`docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md` §2/§3,
`docs/corridor_baseline/knob_sweep.json` (Phase-0a baseline),
`.superpowers/sdd/p1-task-7-prep-report.md` (this task's full report,
smoke evidence, C1/C10 probe raw numbers).
