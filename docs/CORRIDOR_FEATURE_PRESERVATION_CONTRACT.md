# Corridor-First Pooling — Feature Preservation Contract

**Date:** 2026-07-13 · **Status:** draft for review · **Owner spec:**
`docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md`

## Purpose and the hard gate

The corridor rework must not degrade the playlist engine **in any way** — not just
transition quality (T), but every current behavior and feature. Aggregate metrics
(min_T, mean_T, distinct_artists) are necessary but **not sufficient**: a pure transform
(artist normalization) or a soft scoring term (duration penalty) can break while every
aggregate looks perfect. Precedent, this session: the pace gate once went inert in the live
path and no summary number showed it — only a per-knob differential test caught it.

**THE GATE:** corridor work merges to canonical **only when every feature in this contract is
GREEN** against the baseline captured on current `master`. This supersedes the spec's aggregate
acceptance bars — those become one row among many.

## How completeness is guaranteed (enumeration, not memory)

The feature list is derived from code surfaces, not recall:

1. **Effective-config spine** — every field in the resolved `pier_config` + `candidate_pool`
   config blob (~150 fields; the machine-readable source is any run's DS-success JSON line). Every
   knob is a feature, a gate, or a scoring term.
2. **The 44-gate inventory** (`docs/POOL_STARVATION_RESEARCH_2026-07-12.md`).
3. **The identity/normalization layer** (`artist_identity_resolver.py`, `identity_keys.py`,
   `normalization.py`, `src/genre/authority.py`).
4. **The reporter** (`src/playlist/reporter.py`) — every emitted metric/diagnostic.

**Automated completeness net — the "no knob goes inert" sweep.** On current `master`, record for
every config field whether perturbing it changes the output (the differential the dial audit uses,
generalized to the whole config). Post-corridor, **any field that changed output before but not
after is a caught regression.** This turns "did we remember every feature" into a mechanical check
over the entire config surface — the same logic that caught the dead pace gate, as a blanket
guarantee. Hand-authored assertions (below) cover what isn't a single knob: transforms, identity,
topology, reporting.

## Three test shapes

| Shape | Applies to | Assertion |
|---|---|---|
| **Byte-identical** | transforms, identity, and everything upstream of pool build | same input → same output (golden table / same anchors+exclusion sets) |
| **Reject-set / direction+strength** | hard gates | invariant gates: identical reject set; reseated gates: same direction, magnitude ≥ baseline |
| **Fires + magnitude** | soft scoring terms | term still applies; ranking influence (probe-candidate demote) ≥ baseline |

---

## Category A — Transforms & identity (assert BYTE-IDENTICAL)

Upstream of pooling; corridors must not touch them. Cheapest, strongest guarantee.

| # | Feature | Mechanism / source | Preservation assertion |
|---|---|---|---|
| A1 | **Artist normalization** — `The`-prefix strip, ensemble-suffix strip (Trio/Quartet/Band/Ensemble/Orchestra), lowercasing, punctuation | `normalization.py` → `norm_artist`/`artist_key` | golden raw→normalized table; every current row unchanged |
| A2 | **Diacritic / Unicode folding** in identity | both normalizers (fix `f01ac41`, `project_diacritic_artist_dedup`); beam re-derives identity from RAW artist | golden incl. the `Süss`≠`suss` regression case |
| A3 | **Collaboration inheritance** — `feat.`/`&`/`with` → multi-key identity | `artist_identity_resolver.resolve_artist_identity_keys`, `_cap_artist_keys_for_idx` returns a key SET | collab track yields the same identity-key set; min_gap/cap count it under each constituent |
| A4 | **Solo vs collaboration clustering scope** | `artist_style.cluster_artist_tracks` (log: `solo=N collab=M`) | same solo/collab partition for a given artist |
| A5 | **Alias / sibling linking** | `artist_aliases.py` + `data/artist_aliases.yaml` (`project_artist_alias_linking`): alias=merge, sibling=spaced ≥min_gap | `set_artist_link_map_for_testing` cases: alias merges, sibling spacing unchanged |
| A6 | **Version-dedup keys** | `(artist,title)` canonical collapse (`_dedupe_artist_indices`, `dedupe_pool_by_track_key`) | same collapse sets |
| A7 | **Genre authority resolution** | `src/genre/authority.py` → `release_effective_genres` (`genre-data-authority`) | same effective genres per album (contract for what steering/mask read) |
| A8 | **Computed identity underpins counting** (principle 10) | identity keys feed diversity/dedup/seed-exclusion | seed-artist and pier-artist interior exclusion resolve to the same keys |

## Category B — Hard gates

| # | Gate | Disposition under corridor | Assertion |
|---|---|---|---|
| B1 | **Recency pre-order exclusion** | INVARIANT (pool-construction, pre-order) — must NOT become post-order (v3.4 rule) | identical exclusion set; still pre-order |
| B2 | **Blacklist** | INVARIANT (exclusion set on universe) | identical reject set |
| B3 | **Duration HARD exclusion** (min_ms, `2.5×` seed median cutoff) | INVARIANT (exclusion set) | identical reject set for same seeds |
| B4 | **Title hygiene** (`title_hard_exclude_flags`, acapella/interlude/skit) | INVARIANT | identical reject set |
| B5 | **Sonic admission** (floor/percentile) | RESEATED → corridor membership | `sonic_mode` sweep still differentiates + directs (dial audit parity) |
| B6 | **Genre admission** (dense percentile + hard gate) | RESEATED → relevance mask | `genre_mode` sweep still admits/rejects same direction, strength ≥ baseline |
| B7 | **BPM admission gate** | RESEATED → beam band only | `pace_mode` strict still tightens BPM-spread ≥ baseline magnitude |
| B8 | **Onset admission band** | RESEATED → beam band only | `pace_mode` still constrains onset; energy-rescue behavior preserved or provably-subsumed |
| B9 | **Popularity / Oops-All-Bangers gate** | RESEATED → applied to corridor universe | with bangers on, every pooled track still in-cutoff |
| B10 | **BPM-trust on beatless piers** (`bpm_trust_min_onset_rate`) | preserved in beam bands | beatless-pier BPM disable still fires (`project_bpm_trust_beatless`) |

## Category C — Soft scoring terms (highest silent-drop risk — MUST be explicitly rehomed)

The pool ranks within itself today; corridor membership does NOT rank. **Every term below must be
explicitly rehomed onto corridor or beam scoring in Phase 1.** Each is tested by injecting a probe
candidate that the term should demote/promote and asserting the ranking influence is preserved.

| # | Term | Current knob(s) | Assertion |
|---|---|---|---|
| C1 | **Duration soft penalty** | `duration_penalty_weight=0.6`, `duration_reference_ms`=seed median, `duration_cutoff_multiplier=2.5` | duration-outlier probe still demoted by ≥ baseline margin |
| C2 | **Anti-center seed character** (collapse defense) | `seed_character_mode=anti_center`, `seed_character_strength=2.0` (`project_collapse_attack_design`) | center-collapse probe still repelled; collapse-attack metric ≥ baseline |
| C3 | **Popularity penalty** | `popularity_penalty_strength` | preserved (feature-flag off by default — assert still wireable) |
| C4 | **Local-sonic-edge penalty** | `local_sonic_edge_penalty_threshold=0.2`, `strength=0.5` | edge-jump probe still demoted |
| C5 | **Genre soft penalty** | `soft_genre_penalty_threshold=0.73`, `strength=0.15` | off-genre probe still demoted |
| C6 | **Genre tiebreak** | `genre_tiebreak_weight=0.05` | tie-broken toward genre-closer candidate |
| C7 | **Genre pair penalty** | `genre_pair_floor=0.1`, `genre_pair_penalty=0.5` | weak-genre-pair edge still demoted (stays SOFT — never a hard gate) |
| C8 | **Progress monotonicity / arc** | `progress_penalty_weight=0.15`, `progress_monotonic_epsilon=0.05` | non-monotone probe still penalized |
| C9 | **BPM / onset bridge soft penalties** | `bpm_bridge_soft_penalty_strength=0.3`, `onset_..=0.3` | rhythm-jump probe still demoted |
| C10 | **Instrumental lean** (pending branch) | `instrumental_penalty_weight` × P(voice) | composes with corridor; vocal-bridge demote preserved when enabled |
| C11 | **Cohesion bridge/transition weights** | `weight_bridge/weight_transition` per `cohesion_mode` | dial audit parity (already validated live) |

## Category D — Topology & structure invariants (assert same structure)

Upstream of / orthogonal to pooling — assert unchanged, since corridor sits between anchors and beam.

D1 pier-bridge topology · D2 mini-pier even spacing (`balance_gaps`) · D3 variable bridge length
(`variable_bridge_*`) · D4 popular-seeds pier override ("fire") · D5 pier-bridgeability veto
· D6 artist-pier scarcity + spacing · D7 tail-DP endgame · D8 edge repair · D9 edge delete
· D10 break-glass edge repair · D11 weak-edge cascade reorder · D12 roam corridors (geodesic
scoring role) · D13 beam worst-edge minimax · D14 mini-pier promotion (now corridor-armed —
must still promote, and the M1 desert must be gone).

## Category E — Steering (assert steering still moves the arc)

E1 genre taxonomy-graph steering (per-segment genre arc) · E2 tag steering Stage-1 (chips → pool +
pier lean) · E3 tag steering sonic-prototype (Stage-2b, if landed) · E4 genre arc floor
(`genre_arc_floor`, `_percentile`). Assertion: with the steer set, the realized genre/tag arc
still shifts vs. off, magnitude ≥ baseline.

## Category F — Reporting & diagnostics (assert still emitted)

F1 transition stats (min/mean/p10/p90) · F2 weakest-edge report · F3 distinct-artist count
· F4 per-playlist DEBUG log (`logs/playlists/`) · F5 run audits (`docs/run_audits/`) · F6 gate
tallies / pool waterfall lines · F7 the new corridor health line (size, width, widenings, per-anchor
support). Assertion: each still present in the output/log for a standard generation.

---

## Baseline capture — Phase 0a (precedes ALL deletion)

Run **on current `master`**, before Phase 0 touches code. Produces
`docs/corridor_baseline/feature_baseline.json` (committed) containing, per feature:

- **Transforms (A):** golden input→output tables (normalization/identity/collab/alias/dedup/genre).
- **Config sweep (B/C/E):** for the 6-artist × {home,open} corpus, every effective-config field's
  differential fingerprint (does perturbing it move the playlist; by how much) → the "no knob goes
  inert" reference.
- **Scoring probes (C):** per-term probe-candidate demote/promote magnitudes.
- **Topology (D):** structural fingerprints (pier positions, mini-pier spacing, segment count,
  repair/tail-DP fire counts, veto decisions).
- **Reporting (F):** presence checklist of every emitted line.

Harness: built on `tests/support/gui_fidelity.py::generate_like_gui` +
`scripts/research/slider_differentiation_eval.py` (reuse-first; both already faithful to the policy
layer). Corpus + `min_T`/coverage baselines already exist in
`scratchpad/starvation_validation.json` (2026-07-12) — extend, don't rebuild.

## The merge gate (added to the spec)

No corridor phase merges to canonical until:
1. Every Category A transform is byte-identical to baseline.
2. Every Category B/C/E feature is GREEN (reject-set / direction+strength / fires+magnitude).
3. The "no knob goes inert" sweep shows **zero** config fields that lost their effect.
4. Category D structural fingerprints unchanged (except D14, which must IMPROVE).
5. Category F: every diagnostic still emitted.
6. Aggregate bars (min_T flat-or-better, distinct ±2, wall ≤2×) — now necessary-but-subordinate.
7. Dylan's subjective listen-test on 2–3 playlists per phase.

## Open spec gap this contract closes

Spec §1 defines corridor **membership** only. The pool today also **scores** (Category C). Phase 1
MUST explicitly rehome every Category C term onto corridor/beam scoring; a term with no rehoming
target is a spec defect, not an implementation detail. The C-probes are the safety net.

## Cross-references

`docs/POOL_STARVATION_RESEARCH_2026-07-12.md`, spec `2026-07-12-corridor-first-pooling-design.md`,
memory: `project_gui_dial_knob_audit`, `project_diacritic_artist_dedup`,
`project_artist_alias_linking`, `project_publish_identity_fragmentation`,
`project_collapse_attack_design`, `project_tag_steering`, `project_bpm_trust_beatless`,
`project_instrumental_lean`, `feedback_never_fail_three_axes`, `feedback_golden_suite_config_fields`.
