# Playlist Ordering Tuning Recipe

A practical, knob-by-knob guide for playlist ordering and track-quality problems in the
pier-bridge engine. For orientation on *what the pieces are*, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for *why* a default landed where it did, see
[`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) — this doc states the knob and the trade-off, not
the experiment behind it.

> **Reading the defaults below.** Same three layers as `ARCHITECTURE.md`: (1) **dataclass
> defaults** (`src/playlist/pier_bridge/config.py`) are the rollback baseline — every
> experimental lever is off/legacy here; (2) **`config.example.yaml`** is the shipped default —
> what a clean install actually runs; (3) **`config.yaml`** is the live, gitignored config on one
> machine. Each knob below names its config key and states which of the three layers it ships
> at, so you can always see the rollback path.

**Scope note:** sonic similarity is a single MuQ contrastive embedding — there is no
rhythm/timbre/harmony tower to reweight and no variant switch to document. See
`ARCHITECTURE.md` §"Sonic feature space" for the embedding itself; the knobs below are all about
pool composition, beam scoring, and post-beam repair, not sonic-space choice.

---

## Symptoms that suggest these knobs may help

- High-`T` transitions still feel jarring (texture/density mismatch despite 0.9+ scores)
- Demo, live, or medley tracks appearing unprompted in playlists
- A few catastrophically bad edges (`T < 0.20`) per playlist despite a high mean `T`
- `min_transition` in stats is much lower than `mean_transition`
- A long bridge's interior tracks feel like generic filler, not like either pier ("sag")
- Ambient/beatless seeds strand a segment, or a playlist takes far longer than others to build

---

## Step 1: Enable the diagnostic audit (always do this first)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true
```

Dataclass default: `False`. Not set in `config.example.yaml` (diagnostic-only, zero behavior
change, opt in per debugging session).

Generate a playlist. The log will include a **"Selected-edge audit"** section
(`src/playlist/reporter.py::emit_selected_edge_audit`) with per-edge:

- `T`, `T_centered_cos`, `S` (sonic cosine), `G` (genre similarity)
- `bridge_score` (harmonic mean sim to both piers)
- `trans_beam` (beam's internal transition score — comparable to `T`)
- `progress_t`, `progress_jump`
- `local_sonic_cos` (raw cosine of the edge, uncentered)
- `local_pen`, `genre_pen` (penalties applied)
- `title_flags` (artifact flags for the destination track)
- `bpm=A->B dist=D` (BPM values + log-distance, when both tracks have trusted BPM)
- `⚠` prefix on edges below the transition floor

Also watch for `WARNING: T-mismatch edge` lines
(`src/playlist/reporter.py::diagnose_t_mismatch`). These are regression signals: the beam and
final reporter share the same transition metric, so a mismatch usually means stale audit data or
a missing-data fallback, not normal tuning drift.

---

## Knob 0 (removed): sonic is a single MuQ embedding

**There is nothing to align here anymore.** SP-B (merged 2026-07-01/02) deleted the hand-built
rhythm/timbre/harmony towers and the MERT/tower-variant switch entirely — sonic similarity now
runs on one contrastive **MuQ** embedding (`OpenMuQ/MuQ-MuLan-large`, 512-d, `center_l2`
post-processed). `tower_weights` and `transition_weights` are gone from `config.example.yaml`,
and the runtime call site hardcodes the diagnostic field to `None`
(`src/playlist/pipeline/core.py:1242`, kept only for downstream consumers that still read the
key). There's no rhythm/timbre/harmony split left to reweight, so the beam and the reporter score
plain cosine in the same MuQ space by construction — not because two config blocks happen to
match. If an old `config.yaml` still sets `tower_weights` / `transition_weights`, they're inert
and safe to delete; nothing reads them.

This heading stays numbered "Knob 0" so older cross-references in this doc ("do Knob 0 first")
still resolve, but there's nothing to do here anymore — go straight to Knob 1.

See `ARCHITECTURE.md` §"Sonic feature space" for the current single-embedding picture and
`DESIGN_RATIONALE.md` for the towers → MERT → MuQ removal arc.

---

## Pool-level artist cap + never-starve backstop (retired 2026-07, corridor Phase 0)

The `candidates_per_artist`, `target_artists`, `max_pool_size`, `min_pool_size`, and `seed_artist_bonus` knobs (under `playlists.ds_pipeline.candidate_pool` and `playlists.ds_pipeline.pier_bridge.*_pool_size`) are retired. The beam now enforces diversity natively; pre-beam pool artist caps and min-pool backstops are no longer needed. See `docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md` for the rationale.

---

## Corridor pooling (Phase 1, 2026-07): the segment admission mechanism

`docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md`, flipped to the sole
pier-bridge pooling path (2026-07-17); see
`docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` §5.0 for the full universe → corridor → widening →
reseat walkthrough. This section is the knob-by-knob tuning reference.

**Per-mode widths + escape hatch.** `corridor_width_percentile` (`Optional[float]`, default
`None`) is the plain global override — set it to force one width regardless of `sonic_mode`, the
same "tuning escape hatch" pattern as every other per-mode knob in this codebase. Left `None` (the
default), the active `sonic_mode` resolves a width via the per-mode fields below
(`src/playlist/pier_bridge/config.py:415-419`, `resolve_corridor_width_percentile` in
`corridor.py:293`):

| Config key | Default | `sonic_mode` | Effect of raising it |
|---|---|---|---|
| `corridor_width_percentile_strict` | 0.985 | `strict` | Tighter corridor (fewer, closer candidates) — home/strict cells |
| `corridor_width_percentile_narrow` | 0.9675 | `narrow` | Midway; **provisional** — interpolated, not directly corpus-probed |
| `corridor_width_percentile_dynamic` | 0.95 | `dynamic`, unset, or unrecognized | The historical flat pin; open/dynamic cells |
| `corridor_width_percentile_discover` | 0.93 | `discover` | Widest of the four named modes — **provisional**, not directly corpus-probed |
| *(hardcoded, no field)* | 0.0 | `off` | Whole eligible universe qualifies — no sonic narrowing at all |

**Tuning:** raising a mode's percentile tightens that mode's corridor (fewer members, higher
min-sim floor) — use when a mode feels too loose/generic. Lowering it widens (more members, lower
floor) — use when a mode craters (a segment can't find a path, or the widening ladder fires on
nearly every segment). `narrow`/`discover` are calibrated by interpolation only; if you tune them,
re-run the 12-cell corpus (`scripts/corridor_baseline/capture_corpus.py`) to confirm the change
before trusting it — see the known-issues note in `docs/corridor_baseline/phase1_contract_report.md`.

**The widening ladder.** Three knobs govern the sole segment-level recovery mechanism (replaces
every legacy relaxation tier for corridor segments — see the flow doc §5.0 point 4 for the full
state machine):

| Config key | Default | Meaning |
|---|---|---|
| `corridor_widen_step` | 0.05 | Percentile subtracted from `width_percentile` per widen attempt (lower percentile = wider corridor) |
| `corridor_widen_attempts` | 2 | Max widen attempts after the initial (un-widened) build, once the quality trigger (`min_edge_T < transition_floor`) fires |
| `corridor_widen_improvement_epsilon` | 0.02 | Attempt ≥2 only widens further if the *previous* attempt improved the best-seen `min_edge_T` by more than this; otherwise the ladder stops early and hands the best-seen path to the repair stack |

**Tuning:** a corpus with frequent `CorridorWiden[seg N] EXHAUSTED` warnings (see `LOGGING.md`)
either needs a wider starting `corridor_width_percentile_<mode>` (fewer widen cycles needed at
all) or a larger `corridor_widen_attempts`/smaller `corridor_widen_step` (finer-grained widening,
more attempts to find a working width) — the latter costs wall-clock (each attempt re-builds the
corridor + re-runs the beam). `corridor_widen_improvement_epsilon` trades quality-chasing against
wall-clock: lower it to keep widening on marginal gains (slower, occasionally better); raise it to
stop sooner (faster, accepts a merely-adequate path sooner). The Task 6 remediation report
(`.superpowers/sdd/p1-task6-remediation-report.md`) has the empirical basis for 0.02 — an earlier
*predictive* gate (widen only if anchor support looked promising) was tried and **falsified** by a
real cell (Alex G/home) that needed a widen the predictor said wouldn't help; the current gate is
purely empirical (did the last attempt actually help) for that reason.

**`segment_pool_max` / `max_segment_pool_max`.** Unchanged in *meaning* from the pre-corridor
world (still the cap on a segment's ranked candidate count after `build_corridor`'s harmonic-mean
ranking, with `force_include` ids exempt from the cap), but now capping a **corridor**'s ranked
output rather than a KNN-union pool's. `segment_pool_max` defaults to 400, escalating (doubling,
capped at `max_segment_pool_max`=1200) via the separate, pre-existing expansion-attempts loop
inside `_run_segment_backoff_attempts` — this is an orthogonal axis from the widening ladder's
`width_percentile`: widening controls *which* candidates qualify as corridor members; the pool-max
expansion controls *how many* of the ranked, qualifying members the beam gets to see. Both can fire
in the same segment build.

**Small-fixture note (Task 8 finding, keep in mind for new unit tests).** Corridor's
percentile-based membership degenerates on very small (2-4 candidate) synthetic fixtures — a fixed
floor and a percentile behave identically only in the large-N limit. Several existing unit tests
needed an explicit `corridor_width_percentile=0.0` override to restore the admission behavior
their fixed-floor-era fixtures assumed. If you write a new tiny-fixture unit test against the
corridor path, expect the same: either build a fixture with enough rows for percentile math to be
meaningful, or override `corridor_width_percentile=0.0` explicitly rather than relying on a
per-mode default.

**Retired keys.** Corridor's Task 8 flip retired 24 warning conditions (`pooling`,
`collapse_segment_pool_by_artist`, the 14-key legacy `infeasible_handling.*` relaxation ladder, the
6-key `dj_bridging.pooling.*` KNN-union family + its `strategy: dj_union` value, and Artist mode's
`per_cluster_candidate_pool_size`/`pool_balance_mode`) — every one now fires a loud startup warning
via `_warn_retired_keys` rather than silently no-op-ing. Full list + rationale per key:
`docs/CONFIG.md` (struck-through rows) and `.superpowers/sdd/p1-task-8-report.md` §4; don't
re-derive it from scratch, it's already enumerated there.

---

## Knob 1: Title-artifact penalty

**Use when:** the audit shows `title_flags` like `demo`, `live`, `medley` on bad-feeling tracks.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      title_artifact_penalty:
        enabled: true
        weights:
          demo: 0.10
          live: 0.05
          medley: 0.20
          remix: 0.10
          instrumental: 0.08
          version: 0.05
          take: 0.10
          outtake: 0.15
          alternate: 0.10
```

Dataclass default: `title_artifact_penalty_enabled: False`, no weights. Not set in
`config.example.yaml` — opt-in, add it yourself once the audit shows a real problem.

**Tuning:** higher weights demote more strongly. `0.10` demotes roughly on the order of a
moderate bridge-score difference; `0.20` (medley) strongly discourages medleys unless there's no
alternative. Recognized flags are detected by `src/playlist/title_quality.py::detect_title_artifacts`
(word-boundary, case-insensitive); a flag missing from the `weights` dict contributes nothing.

**Warning:** weights above `0.30` can strand long narrow-style segments (artist mode). If
generation starts failing, dial weights down by half.

---

## Knob 2: Scaled local-sonic-edge penalty

**Use when:** the audit shows many `local_sonic_cos` values below 0.10 and `local_pen` is tiny
(< 0.03) — the penalty is decorative in `legacy` mode.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      local_sonic_edge_penalty_enabled: true
      local_sonic_edge_penalty_threshold: 0.15
      local_sonic_edge_penalty_mode: scaled
      local_sonic_edge_penalty_scale: 2.0
```

Dataclass default: `enabled: False`, `mode: legacy`. Not set in `config.example.yaml`.

**Tuning:**
- `threshold: 0.15` flags edges with raw sonic cosine below 0.15 (sonically anti-correlated or
  orthogonal).
- `scale: 2.0` produces penalties of 0.05–0.30 — comparable to bridge-score differences.
  `legacy` mode's `strength * (threshold - edge_cos)` math is preserved for anyone who explicitly
  wants it, but it under-penalizes; `scaled` is the recommended mode.
- Verify in the audit that `local_pen` values are now non-trivial (0.05+) on triggering edges.
- Start at `scale: 1.5`, watch `min_transition` in stats, raise to 2.0 if edges are still jarring.

---

## Knob 3: Worst-edge lexicographic beam objective

**Use when:** `min_transition` in stats is dramatically lower than `mean_transition`, and a few
single bad edges ruin an otherwise good playlist.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      min_edge_objective: min_edge   # default: total_score
```

Dataclass default: `"total_score"`. Not set in `config.example.yaml`.

**Effect:** the beam picks the path that maximizes the worst edge, breaking ties by total score
— the direct implementation of Layer-1 principle 5 ("the worst edge defines the experience").
Expected: `min_transition` rises noticeably; `mean_transition` may drop slightly.

**Warning:** if playlists start feeling "safe but flat," revert to `total_score`. Consider
**variable bridge length (Knob 7)** first — it targets the same symptom by flexing segment length
rather than changing what the beam optimizes for, and is the shipped default lever for this
symptom.

---

## Knob 4: The weak-edge recovery cascade (edge repair · tail-DP · edge-delete)

After the beam assembles a playlist, a fixed **four-pass cascade** lifts weak or broken transition
edges, escalating from least-destructive (swap) to most-destructive (delete). It runs **once, top
to bottom — it is not a retry loop:** each pass hands its (possibly mutated) playlist to the next
and no earlier pass re-runs, so a late deletion is never re-optimized by tail-DP.

| # | Pass | Scope | What it does | Trigger | Config (shipped default) |
|---|------|-------|--------------|---------|--------------------------|
| 1 | **variable bridge length** (add-only) | per-segment, pre-assembly | lengthens a weak segment's interior (never shortens) to land more smoothly — documented as **Knob 7** | worst edge `< variable_bridge_min_edge` (0.30) | `variable_bridge_length: true` |
| 2 | **tail-DP** | per-segment, pre-assembly | re-opens the last ≤2 interior slots of each just-finalized segment and exactly maximizes the landing-window min-edge over the segment pool (never-worse) | window min-edge `< tail_dp.floor` (0.30; 0 = always) | `tail_dp: {enabled: true, floor: 0.3, epsilon: 0.02}` |
| 3 | **edge repair** (break-glass) | global, post-assembly | swaps ONE interior track for a pool candidate that lifts the local `min(T_in, T_out)` by ≥ `margin` (never changes length) | `T < t_floor` (0.30) **or** catastrophic `T_centered_cos < centered_cos_floor` (−0.5) | `edge_repair: {enabled: true, t_floor: 0.3, centered_cos_floor: -0.5, margin: 0.05}` |
| 4 | **edge delete** (remove-only, last resort) | global, post-repair | deletes ONE interior endpoint of the worst edge — only if the merged edge strictly beats it (never-worse) and it won't breach a bystander artist's `min_gap` | worst `T < edge_delete.floor` (0.30), up to `max_deletions` (4) | `edge_delete: {enabled: true, floor: 0.3, max_deletions: 4}` |

Each pass is independently disableable via its `*_enabled` key. Dataclass rollback defaults are
**off** for variable-bridge and edge-repair, **on** for tail-DP and edge-delete; shipped
`config.example.yaml` turns all four on, including edge-repair — like tail-DP and edge-delete, it
ships live. Design intent (swap → add → remove) and the reorder reasoning live in
`DESIGN_RATIONALE.md`.

**When to reach for these:** the cascade is a post-beam safety net for weak edges the beam couldn't
avoid — prefer fixing the cause upstream first (Knobs 0–3, variable bridge length). The
`edge_repair.variety_guard` sub-knob (default off) rejects a repair swap whose new edge is *too*
similar to its neighbours — enable only for dynamic/discover playlists.

**Known limits (documented, not yet resolved — see `CLEANUP_LIST.md`):**
- **The fixer deadzone (0.30 – ~0.75).** Every trigger floor sits at 0.30, so an *ugly-but-legal*
  edge (e.g. a rough pier on-ramp at T ≈ 0.46) gets no fixer attention. If your worst edges cluster
  in this band, the deliberate levers are raising the floors or making them percentile-relative to
  the run — a policy choice, not a bug.
- **Edge-repair vs reporter T can disagree.** Edge-repair has been seen flagging edges the final
  reporter scores as healthy (T ≈ 0.66–0.79 against a 0.30 floor). Until that's root-caused, don't
  retune the floors against reporter numbers, and treat a `WARNING: T-mismatch edge` as a reason to
  distrust that run's quality metrics.
- **90 s ceiling can be breached** on large multi-seed runs (e.g. 10-seed / 50-track), since the
  shipped `generation_budget_s: 0` disables the time cap and the cascade adds work (see Knob 10).

---

## Knob 5: `pace_mode` (rhythm/tempo control)

**Use when:** a playlist has the right color/texture but drifts too far in pace or energy,
especially with slow/meditative seeds, deliberately high-energy seeds, or a beatless
(drone/ambient) pier.

```yaml
playlists:
  pace_mode: narrow   # strict | narrow | dynamic | off
```

Shipped default: `dynamic` (`config.example.yaml`, top-level `playlists.pace_mode`, commented
example; effective default even when absent). Presets live in
`src/playlist/mode_presets.py::PACE_MODE_PRESETS`.

**Effect:** `pace_mode` is **embedding-independent** — it gates on BPM and onset-rate log-distance
bands plus a soft rhythm penalty read from DB features, so it survives any sonic-embedding
change (this is why the old rhythm-cosine tower penalty is inert now — there's no rhythm tower
left to slice on the MuQ embedding, and why pace exists as its own axis rather than folding into
`sonic_mode`). It applies an admission floor when building the candidate pool, then a per-step
moving target inside the pier-bridge beam.

**Starting values (from the live presets):**

| Mode | BPM adm/bridge log-dist | Onset adm/bridge log-dist | Bridge soft-penalty strength | Use case |
|---|---|---|---|---|
| `strict` | 0.30 / 0.40 | 0.30 / 0.40 | 0.50 | Tight tempo fidelity — slow/meditative seeds |
| `narrow` | 0.50 / 0.60 | 0.50 / 0.60 | 0.40 | Moderate anchoring |
| `dynamic` | 0.75 / 0.85 | 0.75 / 0.85 | 0.30 | Gentle — catches double-time, allows drift (default) |
| `off` | ∞ / ∞ | ∞ / ∞ | 0.0 | No pace gate — rhythm still influences via the sonic embedding |

Bands are **soft penalties, not hard gates** (`bpm_bridge_soft_penalty_strength` /
`onset_bridge_soft_penalty_strength` > 0) — a hard gate on an outlier pier (e.g. a near-silent
ambient track) can strand a segment and detonate the relaxation cascade into minutes. The hard
gate still exists at `strength: 0` for anyone who wants the old behavior back.

**Beatless piers:** `bpm_trust_min_onset_rate` (preset 0.5 for strict/narrow/dynamic; dataclass
default 0.0/off) disables a segment's BPM band when its pier's onset rate is below this
threshold — BPM is meaningless on drone (a beatless track can read a confident, wrong BPM). The
onset band, which is reliable on beatless audio, stays active.

**Interaction with `sonic_mode`:** orthogonal. `sonic_mode` controls overall sonic similarity;
`pace_mode` constrains rhythm specifically. In multi-seed runs the beam's rhythm target
interpolates between adjacent piers, so a slow-to-fast route can still arc naturally when the
piers themselves differ.

**Energy admission-rescue** (`pace_rescue_k_energy`, part of the same preset table): re-admits
candidates rejected *only* by the BPM/onset bands, evenly spaced across sorted arousal — a relief
valve so pace gating doesn't starve a segment of arc range. Shipped `0` (off) in `dynamic`;
`strict`/`narrow` presets set 20/5. Off by default because the broader energy arc/contour feature
is still unvalidated — see `DESIGN_RATIONALE.md` "Pace / energy."

**Diagnostics:** watch for `Pace admission floor applied: floor=X rejected=N` in logs and the
`bpm=A->B dist=D` line in the selected-edge audit. If strict mode rejects too much of the pool or
makes segments infeasible, drop to `narrow` or `dynamic`.

---

## Knob 6: Local genre continuity (`soft_genre_penalty_*`)

**What it does.** Penalizes any beam edge whose candidate-to-previous-track genre similarity
drops below a per-mode threshold. The penalty multiplies the edge's combined beam score by
`(1 - strength)`, demoting (not gating) genre-jarring transitions — what suppresses a single-track
genre detour like a folk-punk track in the middle of a dream-pop run.

**Where it lives.** `src/playlist/pier_bridge/beam.py` (penalty application);
`src/playlist/config.py` (per-mode resolution). Genre similarity here is the taxonomy-graph
**`max`** metric (the max canonicalized-tag-pair similarity over the two tracks) — a soft-cosine
alternative was built and evaluated but rejected; see `DESIGN_RATIONALE.md` §"Genre metric."

**Note:** per-mode knobs (`soft_genre_penalty_threshold_narrow`, etc.) are resolved by
`playlists.cohesion_mode`. With the default `cohesion_mode: dynamic`, only `*_dynamic` keys
apply. Set `cohesion_mode` to `strict`/`narrow`/`discover` (or use `--cohesion-mode` on the CLI
for one run) to activate those per-mode values.

**Per-mode defaults, shipped in `config.example.yaml`:**

| Mode      | threshold | strength | Role                                    |
|-----------|-----------|----------|------------------------------------------|
| strict    | 0.82      | 0.40     | Hard enforcement of local continuity     |
| narrow    | 0.78      | 0.30     | Suppress single-track detours            |
| dynamic   | 0.73      | 0.15     | Light continuity nudge                   |
| discover  | 0.20      | 0.10     | Safety net only — allow variety          |
| off       | 0.20      | 0.10     | Safety net only — allow variety          |

**How to diagnose.** Run with `--log-level DEBUG` and look for per-segment
`Segment N: soft_genre_penalty_hits=H edges_scored=E threshold=T strength=S` lines
(`pier_bridge_builder.py`). The post-generation summary also reports total
`soft_genre_penalty_hits`.

- If `hits == 0` across all segments in a non-discover mode, the threshold is too low to be
  doing anything — raise it toward the observed `G genre` median in the summary.
- If `hits > 50%` of `edges_scored` in narrow or strict mode, the threshold is too high — you're
  penalizing the median edge, not just outliers. Lower toward the `G genre` p25–p33 range.
- If bridge relaxation warnings (`Segment N attempt 2: widened=True`) appear in narrow mode after
  recalibration, the penalty plus the gate is starving segments — lower `strength` first, then
  `threshold`.

**Caution.** A flat threshold of 0.80 was tried and reverted — it introduced worse worst-case
edges by shifting the beam onto paths with low `T` but high `G`, which the penalty can't see.
Raise only after confirming `min_T` and `below_floor` hold.

**Related, independent mechanism — raw-tag pool compatibility.** A second, separate soft
demotion (`genre_compatibility_enabled` / `genre_compatibility_penalty_strength`, live-only —
dataclass default off, not in `config.example.yaml`) applies during *candidate-pool
construction*, not at the beam edge: it demotes a candidate whose raw-tag overlap with the seed
is below `genre_compatibility_conflict_threshold` (0.15) — a pool-level filter, complementary to
this beam-level continuity penalty. Both are soft; a hard genre gate anywhere in the pipeline
detonates the relaxation cascade. **CLAUDE.md gotcha caveat:** the project doc's
`genre_conflict_min_confidence` / `genre_conflict_penalty_strength` key names are stale — the
real keys are `genre_compatibility_*` as above.

**Relationship to `genre_tiebreak_weight`.** The tiebreaker (default 0.05) nudges near-tied
edges; the penalty actively demotes below-threshold edges. They're independent — leave the
tiebreaker at 0.05 unless you have a specific reason to change it.

---

## Knob 6b: Tag steering (artist-mode soft genre lean)

**Use when:** an artist-mode playlist should lean toward one facet of the seed artist's own
catalog (e.g. their shoegaze-tagged tracks over their dreampop-tagged tracks) without hard
filtering out everything else.

**What it is.** A per-request, GUI-only feature — not a config default anyone tunes once and
forgets. The artist-mode generate screen shows up to 3 chips drawn from the seed artist's *own*
published genres (`GET /api/genres/for_artist` → `src/genre/authority.py::resolved_genres_for_artist`
— excludes `inferred_family` hub genres, top-12 by weight). Selected chips travel as
`steering_tags` on the request → `policy.derive_runtime_config` (`src/playlist_gui/policy.py:325-334`,
capped at 3, **web-only** like the other policy-owned knobs — there's no CLI flag) →
`playlists.ds_pipeline.pier_bridge.tag_steering_tags` override.

**It is always a soft lean, never a gate.** `src/playlist/tag_steering.py::resolve_tag_steering_target`
maps the chosen tags to a unit-norm mean of their dense-genre-embedding vocabulary rows. With no
tags selected it returns `None` silently — **byte-identical to legacy generation**, zero overhead.

Two independent, additive levers, both soft:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      tag_steering_pool_blend: 0.5    # 0 = ignore tags (legacy), 1 = pool centroid is pure target
      tag_steering_pier_weight: 0.3   # on-tag bonus in artist medoid (pier) scoring
```

- **Pool lever** (`tag_steering_pool_blend`, shipped `0.5`): blends the tag target into the
  dense genre-admission centroid used for candidate-pool admission
  (`candidate_pool.py:922-933`) — `(1-blend) * seed_centroid + blend * tag_target`, re-normalized.
  Selecting tags **forces `genre_admission_aggregate=centroid`** even if the config asked for
  `per_seed` (`candidate_pool.py:882-884`, logged), since the blend only makes sense against a
  single centroid.
- **Pier lever** (`tag_steering_pier_weight`, shipped `0.3`): adds an on-tag affinity bonus to
  each candidate's medoid score during artist-style pier clustering (`artist_style.py:923-939`,
  wired from config as `medoid_tag_weight` at `playlist_generator.py:1759-1761`). This lever is
  active in the shipped template — `artist_style.enabled` ships `true` in `config.example.yaml`,
  so the medoid-clustering code path it lives in runs by default for style-aware artist
  playlists.

**Degenerate cases (both loud, not silent):**
- No tags picked → resolver returns `None`, nothing logged beyond the normal path — this is the
  intentional legacy-identical no-op.
- Tags picked but the artifact has **no dense genre-embedding sidecar** (`X_genre_dense` /
  vocabulary missing) → `resolve_tag_steering_target` logs a **WARNING** ("dense genre sidecar
  has no vocabulary embedding — steering disabled for this run") and proceeds inert for that run.
  Same for tags that don't map to the vocabulary (per-tag WARNING) or that all map but net to a
  zero-norm target.

**Status:** pool + pier levers are shipped and live behind the GUI chips (`config.example.yaml`
lines ~292-295). A designed-but-not-built third lever — a beam-stage (per-edge) tag bonus — does
not exist yet; don't look for a `tag_steering_beam_weight` key. Tests:
`test_tag_steering*`, `test_authority_artist_genres`.

---

## The collapse-prevention stack (Knobs 7–9)

Long bridges tend to **sag**: interior tracks drift toward the dense, genre-blurred "average" of
the local pool rather than representing the seeds' actual character. This is now understood as
two related failure modes — cross-seed convergence and within-bridge sag — and three levers
target it, in the order you should reach for them. See `ARCHITECTURE.md` §"Anti-sag scoring
(collapse prevention)" for the map and `DESIGN_RATIONALE.md` for the evidence (including the
density-floor approach that was tried and abandoned).

### Knob 7: Variable bridge length

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      variable_bridge_length: true
      # variable_bridge_flex: 2
      # variable_bridge_band: 5
      # variable_bridge_min_edge: 0.30
      # variable_bridge_max_flex_segments: 3
```

Dataclass default: `False` (even split, rollback). **Shipped `true` in `config.example.yaml`** —
part of the collapse-core.

**What it does.** A rigid even split pads every segment to the same interior length regardless
of whether the pier-to-pier distance actually calls for it. Variable length instead builds the
nominal even split, and — only for a segment whose worst edge is below `variable_bridge_min_edge`
(0.30) — greedily tries every interior length in `[nominal-flex, nominal+flex]` (flex=2), keeps
the one with the best worst-edge (the "bottleneck," which includes the pier-return edge), and
prefers the nominal length unless a flexed length beats it by more than `variable_bridge_epsilon`
(0.02 — a prefer-nominal-first anti-crutch so it doesn't flex when it doesn't need to). The total
track count is allowed to land within `±variable_bridge_band` (5) of the requested playlist
length. `variable_bridge_max_flex_segments` (3) is a **deterministic cap** on how many segments
may actually flex per generation — it replaced an earlier wall-time guard so the cost is
predictable, not time-boxed.

**When to reach for it:** this is the first lever for "worst-edge is bad but average is fine" —
it's structurally cheaper than Knob 3's global objective change because it only perturbs the
specific segment that needs it.

**Trade-off:** flexing costs roughly 2.5–3x the generation time of the segments that actually
flex (every candidate length in the flex range gets a full trial build). It's inert on a seed set
where every segment nominal length is already the best fit (about half of real seed sets in
validation).

### Knob 8: Seed-character anti-collapse (SP2, `anti_center`)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      seed_character_mode: anti_center   # off | anti_center
      seed_character_strength: 2.0
```

Dataclass default: `seed_character_mode: "off"`, `strength: 0.0`. **Shipped
`anti_center` @ `2.0` in `config.example.yaml`.**

**What it does.** A *scoring* fix for within-bridge sag: it subtracts
`strength * max(0, cos(candidate, pool_centroid) - bridge_score)` from a candidate's combined
score — directly penalizing a candidate for sitting closer to the local segment's generic pool
center than to its own piers. The centroid is the mean of the segment's L2-normalized candidates,
**excluding the piers themselves**, so a candidate can't get credit just for being near the
seeds.

**Note — hubness was deleted.** An earlier second selector (`hubness`, a kNN in-degree
deflation) was evaluated alongside `anti_center` and lost (weaker effect, didn't scale) — it has
been **removed from the codebase**. `seed_character_mode` now accepts only `off` or
`anti_center`; don't look for a `hubness` option, and don't reintroduce it without new evidence.

**When to reach for it:** the default always-on lever against sag. `strength: 2.0` is the
validated sweet spot (measured sag reduction: electronic 60%→46%, dreampop 117%→101% on the
collapse harness, with playlist quality held). Turning it up further hasn't been validated —
treat this as a tuned constant, not a dial to crank.

**Trade-off / known limit:** it's a **partial** dent, not a fix. On some seed pairs (dreampop) it
plateaus around 101% — scoring alone can't fully prevent a long, genre-blurred segment from
sagging. That's what Knob 9 (mini-piers) exists to close.

### Knob 9: Mini-piers (SP3, structural anti-sag)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      mini_pier_enabled: true
      mini_pier_max_interior: 5
      mini_pier_smoothness_margin: 0.12
```

Dataclass default: `mini_pier_enabled: False`. **Shipped `true` / `max_interior: 5` /
`margin: 0.12` in `config.example.yaml`.**

**What it does.** Where Knob 8 nudges scoring, this is a *structural* fix: it splits any segment
whose interior would exceed `mini_pier_max_interior` (5) tracks by picking a high-character
waypoint — selected via a smoothness floor plus the same anti-center scoring as Knob 8 — and
pinning it as an **extra pier**. The beam then can't drift past that anchor; it's structurally
forced to route through a high-character point rather than the pool's generic average.
`mini_pier_smoothness_margin` trades some worst-edge quality for character (a smoother, more
generic waypoint is easier to bridge to but defeats the purpose; the margin bounds how much
smoothness you'll sacrifice character for).

**When to reach for it:** the residual-sag case Knob 8 plateaus on — long segments (interior >
5) with a diffuse local pool. Measured effect: dreampop within-bridge sag 103%→63%.

**Trade-off:** more piers means more segments, each independently beam-searched, so cost scales
with how many segments actually split. It composes with Knob 8 rather than replacing it — anti-
center still scores every candidate; mini-piers just adds structure the scoring alone can't
enforce.

### Roam corridors — live-only, advanced

Roam corridors (on-manifold kNN geodesic routing + minimax worst-edge ordering,
`playlists.ds_pipeline.pier_bridge.roam.*`) are a real, tested lever that is **live in the local
`config.yaml` but not in `config.example.yaml`** — genuinely advanced/opt-in, outside the
validated default bundle. Copy the relevant block from `config.yaml`'s comments if you want to try
it, and expect to tune `roam.width_sonic` / `genre_pair_floor` per library. (Edge repair, which an
earlier version of this note lumped in here, is part of the weak-edge recovery cascade in Knob 4
above — its absence from `config.example` is a template gap to fix, not an intentional exclusion.)

---

## Knob 10: `generation_budget_s` (time budget)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      generation_budget_s: 0   # 0 = disabled
```

Dataclass default: `60.0`. **Shipped `0` (disabled) in `config.example.yaml`.**

**What it does.** A single shared deadline computed once at generation start, threaded into
every segment loop and relaxation tier, so no combination of retries can collectively exceed the
budget. `0` **disables the time limit entirely** — every lever (beam, variable bridge length,
mini-piers) runs to completion regardless of wall-clock cost.

**Trade-off.** This is a deliberate quality-first choice while the collapse-prevention stack
(Knobs 7–9) is still being dialed in — none of those levers currently bail early to protect a
time budget, so a positive budget interacts with them by potentially truncating a flex/mini-pier
search mid-way rather than letting it finish. There's still a **90s hard ceiling as a design
target** (not separately enforced when the budget is 0) — a full generation should never
approach it in practice, but nothing currently kills a run that does. Set a positive value (the
dataclass default `60` is a reasonable starting point) once you want a hard speed guarantee back;
expect variable-bridge-length and mini-pier segments to be the first casualties of a tight
budget, since they're the ones that do extra trial work per segment.

---

## The four mode axes and per-cohesion-mode knobs

Cohesion-vs-discovery is exposed as four independent axes (`ARCHITECTURE.md` §"The four mode
axes"): `cohesion_mode` drives the beam; `genre_mode`, `sonic_mode`, `pace_mode` gate the
candidate pool. All default to `dynamic`.

The pier-bridge per-mode knobs — `bridge_floor_<mode>`, `weight_bridge_<mode>` /
`weight_transition_<mode>`, `soft_genre_penalty_threshold_<mode>` /
`soft_genre_penalty_strength_<mode>`, `genre_pair_floor_<mode>`, `genre_arc_floor_<mode>`,
`sonic_admission_percentile_<mode>`, etc. — are **keyed by `cohesion_mode`**, not by
`genre_mode`/`sonic_mode`. With `cohesion_mode: dynamic` (the default), only the `_dynamic` suffix
of each of these keys is read; setting `cohesion_mode: strict` activates the `_strict` values
across all of them at once. Use `--cohesion-mode` on the CLI to override for a single run without
editing config.

Two gotchas worth restating here (full detail in `ARCHITECTURE.md`):
- **Only the web GUI goes through the policy layer** (`src/playlist_gui/policy.py`). The CLI
  sets mode strings directly. A test/harness that bypasses policy will see modes as inert — the
  `playlist-testing` skill's harness routes through policy for exactly this reason.
- **Beam width is global, not per-mode**: `initial_beam_width` / `max_beam_width` (shipped
  default 20/100; `config.example.yaml` doesn't override the dataclass), doubling toward the cap
  on infeasibility, regardless of `cohesion_mode`. Raising both (e.g. to 40/200) widens the
  search at a roughly linear cost in generation time — a reasonable knob to reach for if
  infeasibility retries are common on your library, independent of any mode setting.

---

## Reading the audit table

Example bad edge entry:
```
⚠ Edge #15: Hideous Sun Demon - Gimmicks -> Stove - Nightwalk
  T=0.092 T_centered_cos=-0.817 S=0.306 G=1.000 | bridge=0.55 trans_beam=0.25 title_flags=-
  progress_t=0.850 progress_jump=0.100 local_sonic_cos=0.030 local_pen=0.021 genre_pen=0.000 below_floor=True
  bpm=118->131 dist=0.152
```

Interpretation:
- `⚠` + `below_floor=True` — this edge fell below the transition floor in the final emitted
  playlist.
- `T_centered_cos=-0.817` — the underlying centered cosine is strongly anti-correlated; the
  calibrated `T=0.092` reflects that correctly (this is the current sigmoid rescale, not the
  crushed linear rescale it replaced — see `DESIGN_RATIONALE.md` "Centered-transition sigmoid").
- `bridge=0.55` — candidate was moderately positioned between both piers.
- `trans_beam=0.25` vs `T=0.092` would be a metric regression in a healthy run — `trans_beam`
  should match `T` for the same edge unless the row is stale after repair or an input was
  missing.
- `local_sonic_cos=0.030` — very low raw cosine; scaled local-sonic penalty (Knob 2) would demote
  this significantly.
- `local_pen=0.021` — current legacy-mode penalty is tiny; confirms `scaled` mode would help.
- `bpm=118->131 dist=0.152` — within the `dynamic` pace band (0.75); pace is not the cause of
  this edge's badness.

Likely fix for this edge: check `T` vs `trans_beam` first. If they match and are both low, tune
upstream scoring (`local_sonic_edge_penalty_mode: scaled`, `min_edge_objective: min_edge`, or
variable bridge length if this segment's nominal length is a poor fit) before reaching for edge
repair as a fallback.

---

## Track replacement as post-generation refinement

Single-track replacement is a GUI refinement tool, not a pre-generation tuning knob. It uses the
most recent generation's in-memory artifact bundle and transition metric to search for one local
substitute between the previous and next playlist tracks. **Best Match** ranks by transition
quality; **Different Pace**, **Different Genre**, and **Different Sound** first keep
high-transition candidates, then re-rank the top 50 by BPM/rhythm, genre-vector, or
timbre+harmony divergence from the current track. If replacement suggestions are consistently
weak, tune candidate admission, transition weights, and pace/genre/sonic modes upstream before
relying on manual replacement.
