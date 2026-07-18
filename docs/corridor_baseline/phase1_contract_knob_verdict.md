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

**RESOLVED — see "RED root-cause resolutions" below (2026-07-17 follow-up).
Verdict amended to CONTRACT PASS-AS-AMENDED: all 13 fields root-cause to
category (a), CONDITIONALLY LIVE.**

## RED root-cause resolutions

Follow-up task, same branch. Every RED field was probed with hand-built
`scripts.corridor_baseline.runner.run_cell(artist, detent, set_paths={...})`
pairs — real generations, `pooling=corridor` forced via `set_paths`, diffed
by `track_ids` jaccard/order (throwaway probe scripts, not committed; exact
`set_paths` given per row below for reproducibility). All 13 land in
**category (a): CONDITIONALLY LIVE** — the knob fires when its firing
condition occurs; the two `SWEEP_CELLS` at their default corridor settings
just never trigger that condition. **Zero (b), zero (c).**

### Mechanism families

**1. `segment_pool_max` / `segment_pool_genre_weight` — share one root cause.**
`build_corridor` (`src/playlist/pier_bridge/corridor.py:188-193`) only
truncates to `segment_pool_max` when `size_before_cap > cap`; at the sweep's
default corridor width (`corridor_width_percentile=0.85`), corridor sizes run
162-358 per segment (see the harness-completeness note at the top of this
doc) — well under the default cap of 800 — so the cap **never binds** on
either sweep cell. `segment_pool_genre_weight` only feeds `rank_scores`
(`corridor.py:124-137`), which matters for two things: (a) which candidates
survive the `segment_pool_max` truncation, and (b) `final_candidates`'
*list order* — but `_beam_search_segment` (`src/playlist/pier_bridge/beam.py`)
re-scores and re-sorts every candidate itself and truncates to `beam_width`
by its own score (`beam.py:1765`, `next_beam[:beam_width]`), so input-list
order carries zero signal downstream. With the cap never binding, genre-weight
perturbations only ever reorder a set that (a) never changes and whose (b)
never matters — pure no-op, exactly matching both fields' bit-exact-inert
signature.
  - PROBE (`segment_pool_max`, default width 0.85/widen=2, `pooling=corridor`
    only): `segment_pool_max=100` vs the `=800` default — Bill Evans
    Trio/open jaccard=0.378 vs reference. `=50` vs `=800`: BET/open
    jaccard=0.326, Swirlies/home jaccard=0.212.
  - PROBE (`segment_pool_genre_weight`, same base + `segment_pool_max=50`
    forced so the cap now binds): weight 0.0 vs 0.25 (default) vs 1.0 —
    BET/open jaccard 0.0-vs-default=0.167, 0.0-vs-1.0=0.235; Swirlies/home
    0.0-vs-default=0.224, 0.0-vs-1.0=0.145. Membership changes once capping
    is real.
  - **Disposition:** no code fix needed (both fields ARE wired correctly);
    the sweep's fixed `×1.5` perturbation (800→1200) is structurally
    incapable of ever exercising the cap on these 2 cells' natural corridor
    sizes — this is the sweep's own blind spot, not the knob's. Worth a
    one-line note for Task 8/9: the *effective* default cap value relative to
    typical corridor sizes should be revisited if `segment_pool_genre_weight`
    is meant to matter in production, since at today's defaults it's a no-op
    on typical-sized corridors, not just on these 2 sampled cells.

**2. `edge_repair_enabled` / `edge_repair_margin` (Swirlies_home).**
`repair_playlist_edges` (`src/playlist/repair/edge_repair.py`) only touches
edges with `T < transition_floor`; corridor's widening ladder
(`_run_corridor_widening_ladder`) already drives every edge above floor on
the sweep's default-width cells (confirmed: `below_floor=0` in both cells'
reference runs), so there is nothing for repair to fix regardless of
`enabled`/`margin`.
  - PROBE: forced sub-floor edges via
    `corridor_width_percentile=0.985, corridor_widen_attempts=0`
    (Swirlies/home) — CorridorWiden logs show seg 2 EXHAUSTED at
    `min_edge_T=0.120` and seg 4 at `0.145`, both `< floor=0.200`.
    `edge_repair.enabled=False` vs the True/default reference: jaccard=0.818,
    `min_transition` drops 0.545→0.340 (repair demonstrably fixing the weak
    edges). `edge_repair.margin=0.5` (vs default 0.05) reproduces the
    **exact same** degraded output as `enabled=False` (margin so strict no
    swap ever clears it) — same jaccard, same `min_transition`.
  - **Disposition:** no action. CONDITIONALLY LIVE, cleanly demonstrated.

**3. `tail_dp_epsilon` (Swirlies_home).**
`optimize_segment_tail` (`src/playlist/pier_bridge/tail_dp.py:93-94,109-110,134-135`)
gates on `floor` (`tail_dp_floor`, default 0.30) **first** — if the existing
landing window already clears the floor it returns `None` without ever
consulting `epsilon`. Corridor's default-width cells mostly already clear
0.30.
  - PROBE: same sub-floor base as #2 (seg 2/4 land at 0.120/0.145, well under
    `tail_dp_floor=0.30`) — `tail_dp.epsilon=0.5` (vs default 0.02) now
    differs from reference: jaccard=0.875.
  - **Disposition:** no action. CONDITIONALLY LIVE.

**4. `variable_bridge_length` / `_flex` / `_min_edge` / `_epsilon` (multiple
cells).** `choose_segment_length` (`src/playlist/pier_bridge/var_bridge.py:37-40`)
tries the nominal length first; if its bottleneck already clears
`good_enough` (`variable_bridge_min_edge`, default 0.30) it returns
immediately with `flexed=False` — `_flex`, `_epsilon`, and `_min_edge`'s own
magnitude are then never consulted. Under corridor's default-width cells
(widening ladder already active), nominal segments mostly clear 0.30.
  - PROBE 1 (sub-floor via tight width + 0 widen attempts, default
    `segment_pool_max=800`, Swirlies/home): `_vbl` genuinely **searches**
    (log: `Var-bridge seg 2: ... flexed=True`, `seg 4: ... flexed=True`) but
    the search never finds a length beating nominal (`chosen == nominal`
    both times) — so `variable_bridge_length` True/False, `_flex` magnitude,
    `_epsilon`, and `_min_edge` all still produce bit-identical output at
    this setting alone (matches the original RED finding exactly).
  - PROBE 2 (added a scarce pool, `segment_pool_max=15`, on top of PROBE 1's
    base, so path length itself — not corridor width — is the real
    bottleneck): `variable_bridge_length=True` (reference) vs `False` now
    diverges sharply — the reference run actually **lengthens the playlist**
    (34 tracks vs the requested 30 when flexing is off), jaccard=0.422,
    `min_transition` 0.506 vs 0.292. Reproduced on Bill Evans Trio/open too,
    though that cell stayed jaccard=1.0 (no sub-floor segments arose there
    even at this setting — cell-specific, consistent with #6 below).
  - PROBE 3 (same scarce-pool base where flexing is confirmed to win):
    `variable_bridge_flex=1` (vs default 2) jaccard=0.571;
    `variable_bridge_min_edge=0.95` (vs default 0.30) jaccard=0.327;
    `variable_bridge_epsilon=0.5` (vs default 0.02) jaccard=0.422 — each
    independently moves the output substantially once flexing actually wins.
  - **Disposition:** no action. CONDITIONALLY LIVE for the whole family.

**5. `disallow_pier_artists_in_interiors` (Swirlies_home RED;
Bill_Evans_Trio_open already GREEN at Phase 1).** PROBE at default settings
reproduces the split exactly: BET/open True vs False jaccard=0.622 (live,
matches the original sweep); Swirlies/home True vs False jaccard=1.000
(bit-identical, matches the original RED). Cell-specific: for these
particular Swirlies piers, no Swirlies-authored track ranks inside the
corridor's threshold-passing set to begin with, so the exclusion mask has
nothing to remove — a genuine no-op for this cell only, not a structural
corridor defect (the code path IS exercised and DOES change output on the
other cell, and the mask logic itself — `pier_bridge_builder.py:1270-1271,
1283-1284` — is unconditional and correct for both cells).
  - **Disposition:** no action. CONDITIONALLY LIVE, cell-specific — confirms
    the original doc's own hypothesis verbatim.

**6. `progress_arc_enabled` (RED in both cells).** Wiring confirmed correct
via the nested `progress_arc.enabled`/`progress_arc.weight` path (matches
`perturb.py`'s own field map; a first probe attempt using the WRONG flat
`progress_arc_enabled` path silently no-opped, confirming by contrast that
the *real* sweep's field-mapped path was never the bug). At the sweep's
actual perturbation magnitude (bool flip only, `progress_arc_weight` stays
at its default 0.25), both cells reproduce bit-exact identical
(jaccard=1.000) — the original RED finding is real, not a probe artifact.
Raising `progress_arc.weight` to 3.0 (12x default) while `enabled=True`
flips the output substantially: BET/open jaccard=0.875, Swirlies/home
jaccard=0.432.
  - **Disposition:** no code fix. CONDITIONALLY LIVE, but flagging a tuning
    concern for Task 8/9 (not blocking): the *default* `progress_arc_weight`
    (0.25) is empirically too small to ever matter under corridor's tighter,
    more-converged candidate pools on these 2 cells specifically — worth
    revisiting the default magnitude if progress-arc steering is meant to be
    load-bearing under corridor, since a mechanically-correct term that never
    fires at its shipped default is functionally the same as dead code in
    production even though this sweep correctly reports it as "wired."

**7. `candidate_pool.duration_cutoff_multiplier` / `duration_penalty_weight`
(Bill_Evans_Trio_open cell only — Swirlies_home already resolved by
f142677).** Per `.superpowers/sdd/p1-task-7-c1beam-report.md`: the beam-level
rehome (commit f142677) is confirmed genuinely active by a dedicated
magnitude probe — a "Corridor beam duration penalty: active" log line plus
904/1984 (45%) of the eligible universe penalized (mean=0.11, max=0.48),
tens of thousands of candidate evaluations demoted per segment on
Bill_Evans_Trio_open itself. The null result there is a **candidate-geometry
finding** (Bill Evans Trio recordings cluster tightly in length — the
winning candidates in that artist's corridors simply never happen to be
duration outliers), not a wiring defect — the identical architecture to
C10's already-accepted null pattern (instrumental lean, `jaccard=0.622` on
the one cell where it fires).
  - **Disposition:** no action beyond what f142677 already shipped.
    CONDITIONALLY LIVE / candidate-geometry null. Fold into this resolution
    rather than treating as separately unresolved, per the brief.

### Verdict table

| Field | Cell(s) | Category | Disposition |
|---|---|---|---|
| `segment_pool_max` | both | (a) | No action; cap-vs-corridor-size blind spot noted for Task 8/9 |
| `segment_pool_genre_weight` | both | (a) | No action; same root cause as above |
| `edge_repair_enabled` | Swirlies_home | (a) | No action |
| `edge_repair_margin` | Swirlies_home | (a) | No action |
| `tail_dp_epsilon` | Swirlies_home | (a) | No action |
| `variable_bridge_length` | Bill_Evans_Trio_open | (a) | No action |
| `variable_bridge_flex` | Bill_Evans_Trio_open | (a) | No action |
| `variable_bridge_min_edge` | Swirlies_home | (a) | No action |
| `variable_bridge_epsilon` | Swirlies_home | (a) | No action |
| `disallow_pier_artists_in_interiors` | Swirlies_home | (a) | No action, cell-specific |
| `progress_arc_enabled` | both | (a) | No action; default-weight tuning concern noted |
| `candidate_pool.duration_cutoff_multiplier` | Bill_Evans_Trio_open | (a) | No action; candidate-geometry null (f142677) |
| `candidate_pool.duration_penalty_weight` | Bill_Evans_Trio_open | (a) | No action; candidate-geometry null (f142677) |

**Harness gap fixed separately** (not a RED field, but blocked 6 of the 12
`did_not_resolve` records): `scripts/corridor_baseline/perturb.py`'s
`_PIER_CONFIG_FIELD_MAP` gained a redirect for the 3
`playlist.pier_config.duration_*` mirror leaves
(`duration_penalty_enabled`/`_weight`, `duration_cutoff_multiplier`) to their
real yaml source (`playlists.ds_pipeline.candidate_pool.duration_*`) — same
pattern as the pre-existing `pace_bridge_floor`/`bpm_stability_min`/
`center_transitions`/`transition_floor` redirects. Unlike `pace_bridge_floor`
this is **not** a dead outlet: `pipeline/core.py:873-875` mirrors
`cfg.candidate.duration_*` into `pb_cfg` unconditionally, with nothing
discarding it first — so the redirect resolves to a real, live path.
Covered by 3 new assertions in `tests/unit/test_corridor_baseline_perturb.py`
(`test_config_path_duration_mirror_fields_redirect_to_candidate_pool`).

### Overall (amended)

**CONTRACT PASS-AS-AMENDED.** All 13 RED fields are category (a),
CONDITIONALLY LIVE — each fires under a real, reachable firing condition;
the 2 `SWEEP_CELLS` at their default corridor settings (generous
`segment_pool_max`, wide-enough `corridor_width_percentile`, an
already-effective widening ladder) simply never trigger that condition
naturally. Zero (b) genuinely-dead knobs, zero (c) regressions. One
non-blocking concern carried forward for Task 8/9: `progress_arc_weight`'s
shipped default (0.25) is empirically inert on both sampled cells even when
`progress_arc_enabled=True` — mechanically correct but tuned too weak to
matter under corridor at its current default, worth a deliberate look before
corridor becomes the only path.
