# Phase 1 contract knob-sweep verdict — corridor flag ON

**Date:** 2026-07-17 · **Author:** Task 7 (controller-directed triage) · Compares
`docs/corridor_baseline/phase1_knob_sweep_corridor.json` (fresh, `pooling=corridor`)
against `docs/corridor_baseline/knob_sweep.json` (Phase 0a, legacy) per
`docs/CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md` §merge-gate #3. Pre-registered
expectations: `docs/corridor_baseline/phase1_expected_retirements.md`.
Comparison script (throwaway): scratchpad `compare_sweeps.py`, re-run on demand.

## Harness completeness

- `n_checkpoints` (413) == checkpoint-file count in
  `logs/corridor_baseline/sweep_phase1_corridor/` (`Bill_Evans_Trio_open` 206 +
  `Swirlies_home` 207, both minus `_reference.json`). Sweep is complete, not a
  partial/truncated run.
- All 413 checkpoint files have mtimes between 2026-07-17 20:08:19 and
  21:31:17 — i.e. the *entire* sweep ran in one continuous pass **after**
  both e34a5ad (19:13) and f142677 (19:35) landed. Ruled out: stale
  checkpoints left over from a pre-fix partial run contaminating this
  result (checked file mtimes directly; also confirmed `changed` and
  `inert` statuses are interleaved throughout the run's timeline, not
  clustered by wall-clock position — rules out a "sweep degrades near the
  end" artifact).

## Verdict: **FAIL — 13 unexplained RED fields**

## GREEN (changed → changed): 45

Full agreement between baseline and corridor — knob still moves the
playlist under corridor pooling, same as under legacy. Full list in the
JSON; representative entries: `candidate_pool.min_sonic_similarity`,
`candidate_pool.similarity_floor`, `playlist.pier_config.center_transitions`,
`playlist.pier_config.dj_ladder_smooth_min_sim`,
`playlist.pier_config.disallow_pier_artists_in_interiors` (Bill_Evans_Trio_open
cell only — see RED below, this field splits by cell).

## RED (changed → inert/did_not_resolve, NOT pre-registered): 15 records / 13 unique fields

Code-confirmed **not** structurally bypassed — grepped current
`pier_bridge_builder.py` and found live corridor-path reads for every one of
these (`disallow_pier_artists_in_interiors`: lines 1271/1803/1870/1944/3302/3740;
`progress_arc_enabled`: 1819/1886/1960/4062; `edge_repair_enabled`:
3736/3956/3967; `segment_pool_genre_weight`: 1308/1835/1902/1976). So this is
not the same "corridor branch never reached" story as the
`collapse_segment_pool_by_artist` retirement — the code path IS exercised;
the perturbation just produced a **bit-identical** playlist (jaccard=1.0,
0 position diffs, delta_min_T=delta_mean_T=0.0 in every case, not "close to
inert" — exactly inert).

| Field | Cell | Baseline status/jaccard | Corridor status | Note |
|---|---|---|---|---|
| `candidate_pool.duration_cutoff_multiplier` | Bill_Evans_Trio_open | changed, jaccard=0.561 | inert | Pre-registration flagged this BLOCKING; f142677 fixed it in the *other* cell (Swirlies_home: now changed, jaccard=0.941) but not this one — see "Duration knobs" below. |
| `candidate_pool.duration_penalty_weight` | Bill_Evans_Trio_open | changed, jaccard=0.590 | inert | Same split; Swirlies_home now changed (jaccard=0.711). |
| `playlist.pier_config.disallow_pier_artists_in_interiors` | Swirlies_home | changed, jaccard=0.784 | inert | Bill_Evans_Trio_open cell is GREEN for this same field — cell-specific, not structural. |
| `playlist.pier_config.edge_repair_enabled` | Swirlies_home | changed, jaccard=0.784 | inert | |
| `playlist.pier_config.edge_repair_margin` | Swirlies_home | changed, jaccard=0.886 | inert | |
| `playlist.pier_config.progress_arc_enabled` | Bill_Evans_Trio_open | changed, jaccard=0.600 | inert | RED in **both** cells — no surviving GREEN cell for this field. |
| `playlist.pier_config.progress_arc_enabled` | Swirlies_home | changed, jaccard=0.711 | inert | |
| `playlist.pier_config.segment_pool_genre_weight` | Bill_Evans_Trio_open | changed, jaccard=0.600 | inert | RED in **both** cells. Pre-registration's own worked example ("Explicitly NOT a retirement... genuinely rehomed") — contradicted empirically. |
| `playlist.pier_config.segment_pool_genre_weight` | Swirlies_home | changed, jaccard=0.886 | inert | |
| `playlist.pier_config.segment_pool_max` | Bill_Evans_Trio_open | changed, jaccard=0.600 | inert | |
| `playlist.pier_config.tail_dp_epsilon` | Swirlies_home | changed, jaccard=0.692 | inert | |
| `playlist.pier_config.variable_bridge_epsilon` | Swirlies_home | changed, jaccard=0.615 | inert | |
| `playlist.pier_config.variable_bridge_flex` | Bill_Evans_Trio_open | changed, jaccard=0.455 | inert | Largest baseline effect in the RED list (jaccard 0.455) — biggest died effect. |
| `playlist.pier_config.variable_bridge_length` | Bill_Evans_Trio_open | changed, jaccard=0.694 | inert | Swirlies_home cell for this field is GREEN (changed, unaffected). |
| `playlist.pier_config.variable_bridge_min_edge` | Swirlies_home | changed, jaccard=0.684 | inert | Bill_Evans_Trio_open cell for this field is **WOKEN** (was inert at baseline, now changed) — same field, opposite transition in the other cell. |

**Mechanism hypothesis (unverified, reported per rule 3 — not rationalized
away):** Every RED is bit-exact-identical (jaccard=1.0, 0 diffs, delta=0.0),
not "muted" — that pattern plus the live code-read confirmation suggests
either (a) these two probe corpora's corridor beam decisions are more
converged/less sensitive to these specific perturbation magnitudes than
legacy's pool-ranking was (a real but narrow-evidence-base finding — only 2
cells sampled), or (b) a downstream clobber analogous to the already-documented
`pace_bridge_floor` pattern (`perturb.py:77-94`: a config value is written by
one layer, then unconditionally overwritten by another before the corridor
call site ever reads it) exists for one or more of these fields and simply
hasn't been traced yet. Distinguishing (a) from (b) needs either more probe
cells or a source trace per field — out of this triage's scope. **Flagging
un-rationalized per rule 3; controller decides next steps.**

## AMBER-expected (changed → inert/did_not_resolve, pre-registered): 2

| Field | Cell | Baseline jaccard | Corridor status |
|---|---|---|---|
| `playlist.pier_config.collapse_segment_pool_by_artist` | Bill_Evans_Trio_open | 0.196 | inert |
| `playlist.pier_config.collapse_segment_pool_by_artist` | Swirlies_home | 0.585 | inert |

Matches `phase1_expected_retirements.md`'s only pre-registered legitimate
retirement, exactly as predicted (`_build_corridor_segment_pool` always wins
the branch, the `else` branch that reads this field never executes). Task 8
deletion+warning-list plan: delete the dead `collapse_segment_pool_by_artist`
read site in the legacy `else` branch when legacy is removed; until then, a
startup warning if the corridor flag is on and this key is set non-default
(per CLAUDE.md's "configured knob that can't act" rule).

## Duration knobs (C1) — explicit verification requested by the brief

**NOT cleanly verified. Split result, not a clean fix.**

The pre-registration (written before e34a5ad/f142677) predicted
`candidate_pool.duration_penalty_weight`/`duration_cutoff_multiplier` would
stay `changed → inert` and treated that as BLOCKING pending a real C1 rehome.
f142677's commit message claims the rehome is done (`duration_penalty_values`
threaded into `_beam_search_segment`). Empirically:

- **Swirlies_home: fixed.** Both fields now `changed` (jaccard 0.941 / 0.711,
  nonzero delta_mean_T) — the beam rehome is live and moves the playlist.
- **Bill_Evans_Trio_open: still inert** (jaccard=1.0, delta=0.0 for both
  fields) — same bit-exact-inert pattern as the 13 RED fields above.

Additionally, e34a5ad added new `PierBridgeConfig` fields
(`duration_penalty_enabled/_weight/_cutoff_multiplier`) that surface in the
effective-config blob under **new** leaf names
(`playlist.pier_config.duration_cutoff_multiplier`,
`.duration_penalty_enabled`, `.duration_penalty_weight` — distinct keys from
the pre-existing `candidate_pool.duration_*` leaves). Per the commit message
these are populated by `core.py`'s `generate_playlist_ds` **from
`cfg.candidate`** (i.e. mirror the `candidate_pool.*` values), not from their
own `playlists.ds_pipeline.pier_bridge.duration_*` yaml path. The sweep's
naive prefix-map perturbation writes to that dead yaml path and gets
`did_not_resolve` in both cells for all three — the harness needs a
`_PIER_CONFIG_FIELD_MAP` redirect entry for these three leaves (pointing back
at the `candidate_pool.*` source), the same pattern already used for
`pace_bridge_floor`/`bpm_stability_min`/`center_transitions`/`transition_floor`.
This is a **harness gap**, not evidence the fields themselves are dead — the
real knob is `candidate_pool.duration_*`, already covered above.

**Verdict on the brief's specific ask:** the fix made the knob live in one of
two sampled cells, not both. Not contract-GREEN yet. Recommend the controller
treat this the same as the other 13 REDs (same bit-exact-inert signature in
the same cell) rather than as a separately-resolved item.

## WOKEN (inert → changed, corridor woke a knob): 11

| Field | Cell | Corridor jaccard | Mechanism guess |
|---|---|---|---|
| `playlist.pier_config.initial_beam_width` | Bill_Evans_Trio_open | 0.304 | Corridor's beam-width pinning (Task 6) makes beam width a live lever where legacy's pool-then-beam pipeline diluted its effect before selection. |
| `playlist.pier_config.initial_beam_width` | Swirlies_home | 0.784 | Same. |
| `playlist.pier_config.roam_mutual_proximity` | Swirlies_home | 0.561 | Corridor's own roam/widening-ladder geometry (Task 4) reads this directly where legacy routed it through the now-superseded pool-collapse path. |
| `playlist.pier_config.roam_width_sonic` | Swirlies_home | 0.561 | Same family as above. |
| `playlist.pier_config.tail_dp_floor` | Swirlies_home | 0.784 | Tail-DP reseat (Task 5) gives this floor a live read it didn't reliably have under legacy pooling. |
| `playlist.pier_config.transition_floor` | Swirlies_home | 0.757 | Corridor's tighter, beam-scored candidate set makes the transition floor a binding constraint more often. |
| `playlist.pier_config.variable_bridge_min_edge` | Bill_Evans_Trio_open | 0.348 | Opposite of this same field's RED in Swirlies_home — cell-specific sensitivity, consistent with the RED section's "converged decisions" hypothesis rather than a uniform code change. |
| `playlist.pier_config.weight_bridge` | Bill_Evans_Trio_open | 0.935 | Corridor beam scoring (per-edge score is the corridor's core decision surface) gives bridge weight direct leverage legacy's pool-then-beam split diluted. |
| `playlist.pier_config.weight_genre` | Bill_Evans_Trio_open | 0.935 | Same — beam-native scoring wakes weight_* terms generally. |
| `playlist.pier_config.weight_genre` | Swirlies_home | 0.833 | Same. |
| `playlist.pier_config.weight_transition` | Swirlies_home | 0.806 | Same. |

Pattern: mostly the `weight_*`/`initial_beam_width`/`roam_*` family — plausibly
explained by corridor routing scoring through the beam directly rather than
through a separate pool-ranking stage. Worth a one-line note in Task 8's
docs but not blocking.

## Config drift (Phase 0 → now)

**No fields disappeared** (no `only in baseline` set) — nothing was silently
retired without a corresponding sweep record.

**9 new field leaves**, all sane against the actual Phase 1 commits:

| New field | Sanity check |
|---|---|
| `playlist.pier_config.pooling` | The corridor/legacy switch itself (Task 3) — expected new knob. |
| `playlist.pier_config.corridor_widen_attempts` | Task 4 widening-ladder param — expected. |
| `playlist.pier_config.corridor_widen_improvement_epsilon` | Task 4 — expected. |
| `playlist.pier_config.corridor_widen_step` | Task 4 — expected. |
| `playlist.pier_config.corridor_width_percentile` | Task 1/4 corridor-builder param — expected. |
| `playlist.pier_config.title_hard_exclude_flags` | e34a5ad title-hygiene wiring — expected (this is the tuple-typed field the commit message says it fixed a `json.dumps` crash for). |
| `playlist.pier_config.duration_cutoff_multiplier` | e34a5ad's new `PierBridgeConfig` mirror field — see "Duration knobs" section; needs a harness redirect, not itself a red flag. |
| `playlist.pier_config.duration_penalty_enabled` | Same. |
| `playlist.pier_config.duration_penalty_weight` | Same. |

## Harness health

- `override_failed`: **0** in the corridor sweep (confirmed: status set is
  exactly `{changed, inert, did_not_resolve, unmapped, skipped_type}` — no
  `override_failed`/`error` value anywhere in 413 records). Contract
  requirement met.
- `error`: **0**. Met.
- `did_not_resolve`: **12** (baseline had 6). The baseline's 6
  (`experiment_bridge_balance_weight`/`experiment_bridge_min_weight`/
  `experiment_bridge_scoring_enabled` × 2 cells) survive unchanged into the
  corridor sweep — pre-existing, already-documented non-issue. The **6 new**
  ones are exactly the 3 new `playlist.pier_config.duration_*` leaves × 2
  cells, root-caused above (harness redirect gap, not a contract violation).
  All 12 are accounted for — no undocumented `did_not_resolve`.
- Status-count totals: corridor
  `{changed: 60, did_not_resolve: 12, inert: 222, skipped_type: 69, unmapped: 50}`
  sums to 413 == `n_checkpoints`. Consistent.

## Overall

**CONTRACT FAIL.** 13 unique fields regress `changed → inert` under the
corridor flag without a pre-registered retirement, spanning
`segment_pool_genre_weight` (explicitly claimed "confirmed still wired" by
the pre-registration), `edge_repair_*`, `progress_arc_enabled`,
`variable_bridge_*`, `tail_dp_epsilon`, `disallow_pier_artists_in_interiors`,
`segment_pool_max`, plus the two C1 duration fields (which the brief hoped
were resolved — only half-resolved, one of two cells). All 13 are
code-confirmed *not* structurally bypassed (live reads found in
`pier_bridge_builder.py` at HEAD), and the sweep itself is a clean, complete,
single-pass, post-fix run with zero `override_failed`/`error`. This is
reported un-rationalized per rule 3 — root cause (real insensitivity in a
2-cell sample vs. an undiscovered downstream clobber) is not established and
is the controller's call.
