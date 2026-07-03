# Pier-Bridge / Beam-Search Deep Dive

This is the mechanism-level companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) §"Runtime: the
pier-bridge engine" — that doc gives the map, this one gives the code. It answers *how* the beam
scores a candidate, *how* the weak-edge cascade is sequenced, and *where* each mode axis touches
the search. For **why** a mechanism is shaped this way (rejected alternatives, measured effect
sizes), see `DESIGN_RATIONALE.md`. For **how to tune** a knob, see
[`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md) (the "Knob N" cross-references below
point there). This doc does not re-list every config default — it explains the mechanism the
default drives.

> **Superseded framing, for anyone who read the old version of this file.** This document used to
> describe "Phase 2 (opt-in) / Phase 3 / Legacy (default)" DJ-bridging genre routing as the
> headline feature. That framing is gone. The **vector-mode + IDF + coverage-bonus genre router**
> (`dj_bridging_enabled`) still exists in the codebase, but it is an **opt-in, off-by-default**
> subsystem now — the default genre-arc mechanism is the simpler **taxonomy-graph steering**
> described in §7. Sonic scoring is a single **MuQ** embedding (512-d contrastive); there is no
> rhythm/timbre/harmony tower split left to reweight (`ARCHITECTURE.md` §"Sonic feature space").

---

## 1. Topology, in one pass

1. **Seeds → piers.** For an artist run with `artist_style.enabled` (live default `true`; shipped
   `config.example.yaml` default `false` — see the gap note in `ARCHITECTURE.md` §"Configuration
   model"), the artist's catalog is medoid-clustered in `src/playlist/artist_style.py`
   (`cluster_artist_tracks`) into per-cluster piers *before* this module is ever called — clustering
   is upstream, driven from `src/playlist_generator.py:1710-1747`. Otherwise a legacy per-seed pier
   list is used directly. Either way, `build_pier_bridge_playlist`
   (`src/playlist/pier_bridge_builder.py:438`) only ever orders and bridges whatever
   `seed_track_ids` it receives — it has no clustering logic of its own.
2. **Pier ordering.** `_order_seeds_by_bridgeability` (`src/playlist/pier_bridge/seeds.py:77-170`)
   permutes the piers to maximize total (or, for roam corridors, minimum) pairwise bridgeability —
   exhaustive for ≤6 seeds (`math.factorial(n)` permutations scored), greedy nearest-neighbor above
   that. The per-pair score blends a bridgeability heuristic, sonic cosine, and genre cosine
   (`weight_bridge`/`weight_sonic`/`weight_genre`, default all-bridge). Single-seed artist runs skip
   this and duplicate the one seed as both start and end pier — an arc structure
   (`pier_bridge_builder.py:849-855`).
3. **Mini-pier splicing (structural anti-sag, §8.2).** If `mini_pier_enabled`, extra waypoint
   "piers" are spliced into the ordered sequence *before* segment lengths are computed
   (`pier_bridge_builder.py:857-882`) — so a spliced run has more, shorter segments than
   `len(seeds) - 1`.
4. **Segment length allocation.** `total_interior = total_tracks - len(ordered_seeds)` is split
   evenly across `num_segments = len(ordered_seeds) - 1`, remainder to the earliest segments
   (`pier_bridge_builder.py:886-892`). Variable bridge length (§9, pass 1) can later flex an
   individual segment's length within a small band.
5. **Per-segment candidate pool.** Default `segment_pool_strategy: segment_scored`
   (`src/playlist/segment_pool_builder.py::SegmentCandidatePoolBuilder`) scores every remaining
   candidate jointly against *both* piers and takes the top `segment_pool_max` (400, cap 1200) —
   not a union of each pier's neighbor list. The legacy union-of-neighbors pool
   (`_build_segment_candidate_pool_legacy`, `src/playlist/pier_bridge/pool.py:58`) and the
   DJ-bridging S1(local)/S2(toward)/S3(genre-waypoint) union pool
   (`dj_bridging.pooling.strategy: dj_union`) still exist but are debug/opt-in paths.
6. **Per-segment beam search** fills the interior (§2-§6).
7. **Assembly** concatenates segments, dropping duplicate pier boundaries
   (`pier_bridge_builder.py:2851-2874`).
8. **The weak-edge recovery cascade** (§9) runs once, per-segment then globally.
9. **Export** to M3U / Plex (outside this module).

---

## 2. The beam search — entry point and control flow

`_beam_search_segment` (`src/playlist/pier_bridge/beam.py:230-272`) is the unit of work: one call
per segment (or per half-segment, for a micro-pier fallback — §8.3). Signature: `(pier_a, pier_b,
interior_length, candidates, X_full, X_full_norm, X_start, X_mid, X_end, X_genre_norm, cfg,
beam_width, **~25 kwargs)` → `(interior_indices | None, genre_penalty_hits, edges_scored,
failure_reason | None)`. The returned indices exclude both piers.

Control flow:

1. **Setup** (`beam.py:289-946`) — resolve every penalty knob off `cfg`, build the local-sonic-edge
   policy closure, DJ-waypoint targets (if `dj_bridging_enabled`), genre-arc targets
   (`g_targets_override`, injected by the caller — §7), the progress-projection coordinate per
   candidate, roam-corridor penalty arrays, the SP2 anti-center centroid (§8.1), and the pairwise
   genre-edge floor closure (§7).
2. **`interior_length == 0`** is a fast-path special case: score the direct pier_a→pier_b edge and
   return immediately, bypassing the loop (`beam.py:681-699`).
3. **Beam loop**, `for step in range(interior_length)` (`beam.py:1044-1768`): for every surviving
   `BeamState` × every candidate, apply hard gates, compute the additive/multiplicative score
   (§3), extend the state, then sort and truncate `next_beam` to `beam_width` (`beam.py:1650-1656`).
4. **Final connection** (`beam.py:1770-1913`): score each surviving state's last track against
   `pier_b` with the same machinery.
5. **Selection** (§6) picks the winning `BeamState` from `final_candidates`.

---

## 3. Per-step scoring formula

The core additive term (`beam.py:1256-1267`) is:

```python
bridge_score = harmonic_mean(sim(cand, pier_a), sim(cand, pier_b))
combined_score = cfg.weight_bridge * bridge_score + cfg.weight_transition * trans_score
```

`trans_score` comes from `score_transition_edge` — the same calibrated-cosine function the final
reporter uses (§4). The harmonic-mean bridge-score formula (`2*sim_a*sim_b/(sim_a+sim_b)`) is
computed inline in `beam.py:1256-1267` and, identically, as `_compute_bridge_score`
(`pool.py:27-55`) for the upstream candidate-pool scoring — same formula, two call sites (the
latter also supports an opt-in experimental min/balance blend, off by default). A separate
`dest_pull = eta_destination_pull * cos(cand, pier_b)` term is added at path-extension time, not
folded into `combined_score` itself (`beam.py:1428`, `1571`).

From there, terms are applied to `combined_score` **in this order** (`beam.py:1270-1416`):

| # | Term | Direction | Gate or penalty | Config |
|---|------|-----------|------------------|--------|
| 1 | SP2 anti-center (§8.1) | subtract | penalty | `seed_character_mode`, `seed_character_strength` |
| 2 | Pace (BPM+onset+energy) | subtract | soft penalty (or hard `continue` — §5) | `bpm_bridge_soft_penalty_strength`, `onset_bridge_soft_penalty_strength`, energy step/arc |
| 3 | Progress-target deviation | subtract | soft penalty | `progress_penalty_weight` |
| 4 | Progress-arc loss (opt-in) | subtract | soft penalty | `progress_arc_*` |
| 5 | Progress-arc max-step (opt-in) | subtract | penalty or hard gate | `progress_arc_max_step_mode` |
| 6 | Genre arc vote (§7) | add | **hard** floor + soft bonus | `weight_genre`, `genre_arc_floor`, arc_step_floor |
| 7 | Layered-transition delta (dormant) | add | soft | `layered_transition_scoring_enabled` |
| 8 | Roam-corridor penalties (opt-in) | subtract | soft | `roam_width_sonic/genre/energy` |
| 9 | Local-sonic-edge policy | subtract, or reject | configurable hard/soft | `local_sonic_edge_penalty_*` |
| 10 | Title-artifact penalty (opt-in) | subtract | soft | `title_artifact_penalty_*` |
| 11 | Pairwise genre-edge floor (§7) | subtract | soft | `genre_pair_floor`, `genre_pair_penalty` |
| 12 | Legacy genre multiplicative demotion | `*= (1 - strength)` | soft | `genre_penalty_threshold/strength` |
| 13 | Popularity demotion (Bangers) | `*= (1 - strength·(1-p))` | soft | `popularity_penalty_strength` |
| 14 | DJ waypoint delta (opt-in) | add | soft | `dj_waypoint_*` |
| 15 | DJ coverage bonus (opt-in) | add | soft | `dj_genre_use_coverage` |

`_compute_duration_penalty` is defined in `beam.py:150-214` but **never called** — the module
docstring confirms it is not yet wired into the live score (duration penalty is applied elsewhere,
in candidate-pool construction). Don't assume it fires inside the beam.

**Hard gates** (candidate rejected outright, never scored) fire *before* this table, inside the
same loop: `bridge_floor`, artist-diversity (§5), progress-monotonicity regression (§5), the
transition anti-alignment check, and — when a soft-penalty strength is `0.0` — the BPM/onset bands
fall back to their legacy hard `continue` (item 2 above; see `PLAYLIST_ORDERING_TUNING.md` Knob 5).

---

## 4. Transition calibration — single source of truth

Every transition cosine (beam scorer *and* the post-hoc reporter) is remapped through one
calibrated logistic, `_calibrate_transition_cos` (`src/playlist/pier_bridge/vec.py:35-61`):

```
sigma(gain * (x - center) / scale)
```

— a Platt-style sigmoid that replaced a legacy linear `(x + 1) / 2` rescale (which compressed the
realistic edge-cosine band into a narrow, low-contrast slice of `[0,1]`). `center`/`scale` are
**variant-keyed** (`src/playlist/transition_metrics.py:21-26`, `TRANSITION_CALIB_BY_VARIANT`):
MuQ is `(0.594, 0.092)` — the sole registered variant since sonic towers/MERT were deleted. An
unrecognized variant **raises** rather than silently saturating every edge
(`transition_metrics.py:47-56`) — this codebase's standing rule that a configured-but-unusable
knob is a startup error.

Two call paths share the one implementation, so the beam and the reporter can never diverge:

- **Scalar**, per-edge: `score_transition_edge` (`transition_metrics.py:183-235`) — used by the
  beam's `trans_score`, edge repair, edge delete, and the final per-edge audit report.
- **Vectorized**, batch: `tail_dp.batch_T` (`src/playlist/pier_bridge/tail_dp.py:27-69`) —
  numpy-re-expresses the *identical* formula elementwise for tail-DP's window search (it can't call
  the scalar function per-element without being too slow). Guarded by a dedicated parametrized test
  (`test_batch_T_matches_score_transition_edge_both_branches`) that must stay green if the
  calibration formula ever changes — a manual-mirror module is a divergence risk by construction.

`is_broken_transition` (`transition_metrics.py:238-262`) is intentionally narrow: the old
`transition_floor` **hard gate on `T` is removed** (the calibrated `T` already discriminates well
enough that the beam's own optimization — plus roam's worst-edge minimax — out-performs an
eliminate-on-floor rule, which only adds cascade/timeout risk). What remains is a safety-only check
on the **raw**, uncalibrated end→start cosine (`T_centered_cos < centered_cos_floor`, default
`-0.5`) — catching an edge that is actually *anti-aligned*, which "essentially never fires" in
practice. This function backs the edge-repair trigger (§9) and diagnostics; it does not gate the
beam's own candidate selection.

---

## 5. Hard constraints inside the beam

**Progress monotonicity.** A 1-D coordinate `t ∈ [0,1]` is precomputed per candidate by projecting
onto the pier_a→pier_b chord in sonic space (`beam.py:754-778`). A candidate whose `t` regresses
below `state.last_progress - progress_monotonic_epsilon` is **rejected outright**
(`beam.py:1234-1237`) — this is what prevents "teleporting"/bouncing mid-bridge. A softer
`progress_penalty_weight * |cand_t - target_t|` term (table item 3) additionally penalizes
deviation from the expected per-step progress even among candidates that pass the hard gate. An
optional, more elaborate "progress-arc" layer (non-linear target curve, Huber/squared/abs loss,
max-step gate-or-penalty, autoscaling) sits on top when `progress_arc_enabled` — off by default.

**Artist diversity.** `BeamState.used_artists` (`beam.py:145`) is a running set of artist keys
already placed in *this segment's* path. It is seeded from three sources
(`beam.py:951-994`): the pier artists (if `disallow_pier_artists_in_interiors`), the seed artist
(if `disallow_seed_artist_in_interiors`), and — the cross-segment interaction point —
`recent_global_artists`, an artist-key list the **caller** (`pier_bridge_builder.py`) resolves from
the last `min_gap` positions of the playlist prefix built by *prior* segments. `beam.py` does not
compute the `min_gap` window itself; it treats the carried-in list identically to in-segment usage,
so within-segment diversity and cross-segment `min_gap` are enforced by the same one mechanism,
just fed from two different sources. Identity resolution (ensemble suffixes, "feat." collaborators)
runs through `resolve_artist_identity_keys` when artist-identity mode is enabled, so `The Beatles`
and `Paul McCartney feat. The Beatles` collapse to one key for this check (Layer-1 principle 10).

**Beam width and infeasibility widening.** `beam_width` is a plain scalar parameter to
`_beam_search_segment` — the function has **no widening logic of its own**
(confirmed by grep: the only two references are the parameter and the final truncation line). All
widening happens in the caller:

| Trigger | Mechanism | Site |
|---|---|---|
| First attempt | `beam_width = cfg.initial_beam_width` | `pier_bridge_builder.py:1102` |
| `infeasible_handling` retry | `beam_width += extra_beam_width`, capped | `pier_bridge_builder.py:1111-1114` |
| Expansion-attempt loop retry | `beam_width = min(beam_width * 2, max_beam_width)` (doubling, paired with `segment_pool_max` doubling) | `pier_bridge_builder.py:1603` |

`initial_beam_width` / `max_beam_width` are **single top-level fields on `PierBridgeConfig`**
(`config.py:73-74`, shipped 20/100) — not per-`cohesion_mode`, unlike almost every other pier-bridge
knob (§10).

---

## 6. Selection objective: total-score vs. worst-edge

`_select_best_beam_state` (`beam.py:124-136`) picks the winning path from `final_candidates`:

- **`total_score`** (default): highest cumulative `score`.
- **`min_edge`**: lexicographic — first maximize the path's **weakest single-edge** transition
  score (`_state_min_edge`, `beam.py:108-121`), then break ties by total score. This is the direct
  implementation of Layer-1 principle 5 ("the worst edge defines the experience").

```python
objective = "min_edge" if cfg.worst_edge_minimax_enabled else cfg.min_edge_objective
```

`worst_edge_minimax_enabled` **forces** `min_edge` regardless of `min_edge_objective`'s own value —
and it doesn't only affect the final pick. It also changes the **per-step pruning sort order**
(`beam.py:1652-1655`): with it on, `next_beam` is sorted by `(min_edge, score)` at *every* step, so
weak-edge protection compounds across the whole segment rather than being a final-selection-only
tiebreak. See `PLAYLIST_ORDERING_TUNING.md` Knob 3.

---

## 7. Genre in the beam — two independent mechanisms, plus a demoted third

**7.1 Arc vote (default, `genre_steering_enabled: true`).** The beam scores each candidate's
closeness to a per-step genre **target vector**, injected via `g_targets_override`
(`beam.py:1290-1306`). Two hard floors gate it — a per-segment percentile-derived `arc_step_floor`
and an absolute `genre_arc_floor` fallback — both **reject** (`continue`) a candidate below them;
only candidates that clear both get the additive `weight_genre * arc_sim` bonus. This is a hard
gate wrapped around a soft bonus, not a pure soft mechanism — worth remembering when a segment goes
infeasible on a broad-genre pair.

The target vectors themselves come from one of two sources, chosen by `genre_steering_source`
(`pier_bridge_builder.py:1846-1919`), and — critically — `_require_usable_genre_steering`
(`pier_bridge_builder.py:415-435`, called at construction time) **raises** if the configured source
can't act, rather than silently producing dead targets:

- **`taxonomy`** (canonical default, rebuild-robust): `build_taxonomy_genre_targets`
  (`src/playlist/pier_bridge/taxonomy_steering.py:263-354`) canonicalizes each pier's top tags,
  walks a hub-damped shortest path between them over the SP3a taxonomy graph
  (`TaxonomySteering.arc_adjacency`, built from `src/genre/graph_similarity.py`), converts each
  waypoint label to a taxonomy-smoothed vector over the *artifact* genre vocabulary, and
  interpolates per interior step. Needs no per-track taxonomy assignments — it steers directly on
  the in-artifact `X_genre_raw`. Returns `None` (inert for that segment, not an error) only when
  *neither* pier has a canonicalizable genre.
- **`dense`** (legacy, opt-in): `build_dense_genre_targets` interpolates the 64-dim PMI-SVD
  embedding, optionally routed through a genre-graph ladder. Requires the dense sidecar
  (`X_genre_dense`); if the configured source is `dense` and that sidecar is unavailable,
  generation raises at setup rather than silently steering on nothing.

**7.2 Pairwise genre-edge floor (default, soft).** A second, independent mechanism penalizes the
*adjacent-track* genre similarity directly, deliberately as a soft demotion rather than a gate — a
hard reject here was found to "detonate the infeasibility/expansion machinery on broad-genre
segments" (inline design note, `beam.py:847-853`). `GenrePairSimProvider.sim(a, b)`
(`taxonomy_steering.py:166-195`) scores a track pair as the **max** hub-damped taxonomy similarity
over each track's top canonicalized tags — this `max` metric was chosen over a soft-cosine
alternative that was built and evaluated and lost (`DESIGN_RATIONALE.md` §"Genre metric"). Below
`genre_pair_floor`, the edge score is demoted by a flat `genre_pair_penalty` (`beam.py:1356-1367`
mid-segment, `1806-1818` for the final pier-adjacent edge — explicitly called out as "the
highest-stakes edge in the segment").

**7.3 The DJ-bridging ladder system (`dj_bridging_enabled`, off by default) — what this doc used to
be about.** A third, considerably deeper genre-routing system still exists in the codebase, gated
entirely behind `dj_bridging_enabled: false` (both the dataclass default and the shipped
`config.example.yaml`): vector-mode multi-genre interpolation with IDF weighting and a coverage
bonus (`_build_genre_targets`, `src/playlist/pier_bridge/genre_targets.py:62-195`), S1(local) /
S2(toward) / S3(genre-waypoint) union candidate pooling (`dj_bridging.pooling.strategy: dj_union`),
seed-ordering weights, connector-bias detour logic, and a proactive relaxation-attempt ladder
(`micro_pier.py::_build_dj_relaxation_attempts` — §8.3). This is the entire subject of the old
version of this document. It is not the live default genre-arc mechanism today — §7.1/§7.2 are —
but it remains a fully-wired, tested, opt-in path for anyone who wants direct genre-label ladder
routing instead of the taxonomy arc. Its targets (`_g_targets`) are computed and passed to the
beam completely separately from the arc-steering targets (`_g_targets_dense`) of §7.1
(`pier_bridge_builder.py:1827-1919`) — the two systems can coexist but do not share state.

---

## 8. Anti-sag scoring (collapse prevention)

Long bridges tend to **sag**: interior tracks drift toward the dense, genre-blurred average of the
local candidate pool instead of representing the seeds' actual character. Two shipped-on levers
attack this from different angles, plus an older reactive mechanism that solves a different problem
entirely (segment infeasibility, not sag) and is easy to confuse with the newer one by name.

### 8.1 SP2 — anti-center (scoring fix)

`anti_center_penalty(cand_center_sim, bridge_score, strength)` = `strength * max(0,
cand_center_sim - bridge_score)` (`src/playlist/pier_bridge/seed_character.py:18-22`) — zero
whenever the candidate is at least as pier-like as it is central; otherwise it subtracts the excess.
The centroid (`cand_center_sim`) is precomputed once per segment (`beam.py:823-836`): the mean of
the segment's **L2-normalized candidate vectors**, itself re-normalized before the dot product.
Piers are excluded from this centroid *transitively* — not by any filtering inside `beam.py`, but
because piers are already members of `used_track_ids` by the time the upstream pool builder forms
`candidates`, so they never reach `beam.py` as candidates in the first place. The penalty is applied
first in the per-step score (table item 1, §3): `combined_score -=
anti_center_penalty(...)`. Mode `off` (strength `0`) computes nothing and leaves the score
byte-identical to pre-SP2 behavior. An earlier second selector, `hubness` (kNN in-degree
deflation), was evaluated alongside `anti_center`, lost, and has been **deleted from the
codebase** — `seed_character_mode` accepts only `off | anti_center` now.

### 8.2 SP3 — mini-piers (structural fix)

Where anti-center nudges scoring, mini-piers change the *topology* before the beam ever runs.
`plan_pier_sequence` (`src/playlist/pier_bridge/mini_pier_select.py:60-117`) greedily splits the
longest segment (by even-split interior length) by inserting a waypoint pier between its two
endpoints, repeating until every segment's interior is ≤ `mini_pier_max_interior` (5), no feasible
waypoint remains, or a safety cap (`total_tracks // 4`) is hit. `select_waypoint`
(`mini_pier_select.py:17-52`) picks that waypoint with a two-stage rule: first a **smoothness
floor** — restrict to candidates within `mini_pier_smoothness_margin` (0.12) of the best
available `min(sim_to_pier_a, sim_to_pier_b)`, so the pick is genuinely *between* the two piers, not
just near one of them — then, among those, the **least central** relative to the local
between-region (the same anti-center idea as §8.1, applied to pick a high-character point rather
than a generic one). This runs once, unconditionally, in `pier_bridge_builder.py:857-882` — **before**
segment lengths are even computed (§1, step 3-4) — so a spliced pier is a real pier: it gets its own
segments on both sides, each independently beam-searched, and the beam structurally cannot drift
past it.

**Waypoint artist exclusion.** Waypoints are excluded from all seed/pier artists' other tracks
(`pier_bridge_builder.py:861-870`, normalized `track_artists` match) so a mini-pier can't
accidentally land on the same artist as an existing pier.

### 8.3 Micro-piers — a different, older mechanism (don't confuse with 8.2)

`src/playlist/pier_bridge/micro_pier.py::_attempt_micro_pier_split` predates SP3 and solves a
different problem: **reactive** recovery when a segment's beam search fails outright (returns
`None` after all backoff/expansion attempts), not proactive anti-sag. It is only invoked *inside*
the per-segment loop, only on beam failure (`pier_bridge_builder.py:2247`), gated by
`dj_micro_piers_enabled` (and, for its "detour" variant, `dj_bridging_enabled` +
`dj_allow_detours_when_far`). On trigger, it picks an intermediate connector track and runs two
half-length beam searches around it — a rescue for an infeasible segment, not a quality lever for a
feasible one. `_build_dj_relaxation_attempts` (`micro_pier.py:24-106`) is the broader ladder this
belongs to: baseline → relax waypoint weight → relax pool/beam effort → relax connector bias →
(optionally) relax the transition floor — a fixed sequence of increasingly permissive retries, each
only reached if the previous one still failed. **The naming collision is real**: "mini-pier" (SP3,
§8.2, proactive, structural, shipped on) and "micro-pier" (this section, reactive, DJ-bridging-only,
off unless `dj_micro_piers_enabled`) are unrelated features that happen to sound alike.

---

## 9. The weak-edge recovery cascade

After the beam (and any mini-pier/micro-pier structural changes) produces a playlist, a **fixed
four-pass cascade** lifts remaining weak or broken edges, escalating from least- to
most-destructive. It runs **once, top to bottom — not a retry loop**: each pass hands its
(possibly mutated) playlist to the next; nothing re-runs. Two passes are per-segment (inside the
segment loop, before assembly); two are global (after all segments are assembled):

| # | Pass | Scope | Call site | Mechanism | Trigger |
|---|------|-------|-----------|-----------|---------|
| 1 | **Variable bridge length** (add-only) | per-segment | `pier_bridge_builder.py:2004` → `var_bridge.choose_segment_length` | Tries every interior length in `[nominal-flex, nominal+flex]`, keeps the best bottleneck (includes the pier-return edge), prefers nominal within `epsilon` | nominal worst edge `< variable_bridge_min_edge` (0.30) |
| 2 | **Tail-DP** | per-segment (after var-bridge finalizes the path, before segment assembly) | `pier_bridge_builder.py:2501` → `tail_dp.optimize_segment_tail` | Re-opens the last `min(2, interior)` slots and **exactly** maximizes the landing-window min-edge over the segment's own candidate pool (never-worse) | window min-edge `< tail_dp_floor` (0.30; `0` = always) |
| 3 | **Edge repair** (break-glass) | global, after assembly | `pier_bridge_builder.py:2889` → `repair.edge_repair.repair_playlist_edges` | Swaps **one** interior track per triggered edge for a pool candidate that lifts `min(T_in, T_out)` by ≥ `margin`; length-preserving | `T < edge_repair_t_floor` (0.30) **or** catastrophic `T_centered_cos < centered_cos_floor` (−0.5) |
| 4 | **Edge delete** (remove-only, last resort) | global, immediately after repair | `pier_bridge_builder.py:2930` → `repair.edge_delete.delete_broken_edges` | Deletes **one** interior endpoint of the worst edge only if the merged edge strictly beats it (never-worse) and doing so won't breach a bystander artist's `min_gap`; up to `max_deletions` (4) | worst `T < edge_delete_floor` (0.30) |

Design intent is swap → add → remove, each strictly more destructive than the last, and the
2026-07-02 reorder rationale (why edge-delete moved to *after* edge-repair, not before) is recorded
in `docs/superpowers/plans/2026-07-02-weak-edge-cascade-reorder.md` /
`DESIGN_RATIONALE.md`. Every pass shares the **same 0.30 trigger floor**, which produces a known
"deadzone": an ugly-but-legal edge at, say, `T ≈ 0.46` gets no fixer attention from anything.
Edge-repair has also been seen flagging edges the final reporter later scores as healthy
(`T ≈ 0.66-0.79` against the 0.30 floor) — an open T-mismatch discrepancy, not yet root-caused.
Full knob tables, per-pass config defaults, and the `config.example.yaml` gap (edge-repair has no
block in the shipped template, so a fresh clone runs it off) live in
`PLAYLIST_ORDERING_TUNING.md` Knob 4.

> **Naming collision to watch for.** `playlists.ds_pipeline.repair` (nested `enabled` /
> `max_iters` / `max_edges` / `objective: gap_penalty`, resolved into `src/playlist/config.py`'s
> `RepairConfig`) is a **different, unrelated** config block — it belongs to the legacy greedy
> `constructor.py` path, which is **dead code** (`pipeline/core.py` unconditionally takes the
> pier-bridge branch). Don't confuse it with `edge_repair_enabled` (pass 3 above), which is the
> live pier-bridge cascade's break-glass swap.

---

## 10. The four mode axes, the beam, and tag-steering

`cohesion_mode` is the axis that actually reaches the beam and cascade; `genre_mode` / `sonic_mode`
/ `pace_mode` shape the *candidate pool* upstream (`ARCHITECTURE.md` §"The four mode axes"). Inside
this module, every per-mode pier-bridge knob — `bridge_floor_<mode>`, `weight_bridge_<mode>` /
`weight_transition_<mode>`, `soft_genre_penalty_threshold/strength_<mode>`, `genre_pair_floor_<mode>`,
`genre_arc_floor_<mode>`, `sonic_admission_percentile_<mode>`, and more — resolves through one
shared priority chain, `_resolve_mode_number_with_source` (`src/playlist/config.py:149-185`), not
through any logic in `pier_bridge_builder.py` itself (which only ever reads the already-resolved
scalar, e.g. `cfg.bridge_floor`):

1. A mode-suffixed override key (`{key}_{cohesion_mode}`) if present and numeric.
2. A bare scalar key (`{key}`) if present and numeric.
3. A nested dict (`{key}: {mode: value}`), falling back to its own `default` entry.
4. A hardcoded Python default.

`cohesion_mode: strict` therefore activates every `_strict`-suffixed knob across the beam and
cascade at once — there is no per-knob mode override needed. **Beam width is the deliberate
exception**: `initial_beam_width` / `max_beam_width` have no mode-suffixed variants at all (§5) —
raising them is a global search-cost knob, independent of `cohesion_mode`.

**Tag-steering** (artist-mode GUI chips — a per-request soft lean, not a mode) reaches two points,
neither of which is inside `beam.py` or `pier_bridge_builder.py` directly:

- **Pool lever** (`tag_steering_pool_blend`, default `0.5`): blends the resolved tag target into
  the dense genre-admission centroid used for candidate-pool admission
  (`src/playlist/candidate_pool.py:824-835`) — `(1-blend) * seed_centroid + blend * tag_target`,
  re-normalized. Selecting tags forces `genre_admission_aggregate=centroid` even when the config
  requests `per_seed` (`candidate_pool.py:784-786`), since the blend only makes sense against one
  centroid.
- **Pier lever** (`tag_steering_pier_weight`, default `0.3`): adds an on-tag affinity bonus to each
  candidate's medoid score during artist-style pier clustering (`src/playlist/artist_style.py`,
  `medoid_tag_weight`, wired from config at `playlist_generator.py:1744-1745`). **Dormant unless
  `artist_style.enabled`** — the medoid-clustering path it lives in doesn't run otherwise.

Neither lever ever gates a candidate; with no tags selected, `resolve_tag_steering_target` returns
`None` and both levers are inert — byte-identical to a run with no steering at all. A designed
third lever (a beam-stage, per-edge tag bonus) does not exist yet. Full mechanism and degenerate
cases: `PLAYLIST_ORDERING_TUNING.md` Knob 6b.

---

## 11. Time budget

A single shared `deadline` (`time.monotonic()`-based, computed once by the caller in `core.py`, or
`None` when `generation_budget_s <= 0`) is threaded into `build_pier_bridge_playlist` as a
parameter and checked at four points: segment start, the expansion-attempt loop, the bridge-floor
backoff loop, and micro-pier eligibility. A second, independent, **hardcoded** cap,
`_SEGMENT_RELAXATION_BUDGET_S = 40.0` (`pier_bridge_builder.py:166`, marked `TODO: promote to a
config knob`), bounds cumulative relaxation-tier time across all segments — but it is itself
disabled (set to infinity) whenever `deadline is None`, i.e. whenever `generation_budget_s` is
`0`. Shipped `config.example.yaml` sets `generation_budget_s: 0` — **quality-first**: the beam,
variable-bridge flexing, and mini-piers all run to completion regardless of wall-clock cost. See
`PLAYLIST_ORDERING_TUNING.md` Knob 10 for the trade-off and the (currently unenforced) 90s design
ceiling.

---

## 12. File map

| File | Role |
|------|------|
| `src/playlist/pier_bridge_builder.py` | Top-level orchestrator: pier ordering, mini-pier splicing, segment loop, assembly, cascade dispatch (~3.2k LOC — one large function with nested closures) |
| `src/playlist/pier_bridge/beam.py` | The beam search itself: per-step scoring, hard gates, selection objective (~2k LOC) |
| `src/playlist/pier_bridge/vec.py` | Single-source transition-cosine calibration (`_calibrate_transition_cos`) |
| `src/playlist/transition_metrics.py` | Variant-keyed calibration bands, `score_transition_edge`, `is_broken_transition` |
| `src/playlist/pier_bridge/seed_character.py` | SP2 anti-center penalty (pure function) |
| `src/playlist/pier_bridge/mini_pier_select.py` | SP3 waypoint selection + pier-sequence splicing |
| `src/playlist/pier_bridge/micro_pier.py` | Reactive beam-failure fallback + DJ relaxation-attempt ladder |
| `src/playlist/pier_bridge/var_bridge.py` | Variable bridge length (cascade pass 1) |
| `src/playlist/pier_bridge/tail_dp.py` | Tail-DP landing optimization (cascade pass 2) |
| `src/playlist/repair/edge_repair.py` | Break-glass edge repair (cascade pass 3) |
| `src/playlist/repair/edge_delete.py` | Remove-only edge delete (cascade pass 4) |
| `src/playlist/pier_bridge/taxonomy_steering.py` | Taxonomy-graph arc targets + pairwise genre-pair provider (default genre mechanism) |
| `src/playlist/pier_bridge/genre_targets.py` | Vector-mode/IDF/coverage genre targets (`dj_bridging`, opt-in) |
| `src/playlist/pier_bridge/genre.py` | Genre-vector helpers shared by both genre systems |
| `src/playlist/pier_bridge/seeds.py` | Pier ordering, connector selection, pool dedupe |
| `src/playlist/pier_bridge/pool.py` | Bridge-score kernel + legacy union candidate pool |
| `src/playlist/segment_pool_builder.py` | Default `segment_scored` candidate pool builder |
| `src/playlist/pier_bridge/config.py` | `PierBridgeConfig` dataclass (~70 fields) + `PierBridgeResult` |
| `src/playlist/tag_steering.py` | Tag-to-target resolver shared by the pool and pier levers |
| `src/playlist/artist_style.py` | Medoid pier clustering (upstream of this module) |
| `src/playlist/mode_presets.py` | `genre_mode` / `sonic_mode` / `pace_mode` presets |
| `src/playlist/config.py` | `_resolve_mode_number_with_source` — the per-`cohesion_mode` knob resolver |

See also: [`ARCHITECTURE.md`](ARCHITECTURE.md) (map), [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md)
(knob-by-knob tuning), [`CONFIG.md`](CONFIG.md) (full key reference), `DESIGN_RATIONALE.md` (why),
[`CLEANUP_LIST.md`](CLEANUP_LIST.md) (tracked gaps, incl. the `edge_repair` config.example gap).
