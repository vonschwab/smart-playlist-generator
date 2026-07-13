# Pool starvation research — why pools starve, and whether it's us or the library

**Date:** 2026-07-12 · **Trigger:** SADE + Range=`home` shipped a T=0.028 edge (GUI dial audit).
**Question (Dylan):** does starvation reflect legitimate library scarcity, artificial self-restriction,
or an over-complicated pier/bridge/roam system?

**Answer up front:** overwhelmingly **manufactured, not library scarcity** — by three interacting
mechanisms, each individually reasonable, none aware of the others. The core topology (piers + beam)
is sound; the inelegance is that **six independently-evolved layers each run their own admission with
their own floor and no shared feedback**. Evidence below is from one fully-traced generation
(SADE + home, `scratchpad/home_debug.log`) plus artifact-level probes; validate on 2–3 more artists
before building (evaluation-methodology).

---

## The traced failure, end to end

Broken edge: position 26→27, `a.s.o. – Thinking → Portishead – All Mine`, **T=0.028, S=0.273**
(below_floor=1). Positions: seed piers at 1/11/21/30, mini-piers at 6/16/26 — **position 26 is a
mini-pier**; the broken edge is the first edge of segment 5 (`mini-pier → Frankie's First Affair`,
interior=3).

Waterfall (SADE + home):

| Stage | Count | Gate |
|---|---|---|
| Library | 42,582 | — |
| Pier-bridgeability genre mask | 5,457 eligible | `genre_floor=0.30` — 87% of library cut on genre before anything else |
| Pier-centric provisioning (`build_balanced_candidate_pool`) | allowed_ids=6,130 | top-400 sonic neighbors **per cluster of the seed piers only** |
| Blacklist −31, duration −82 | 6,099 → pool input 6,013 | |
| BPM −137, onset −417, sonic floor 0.61 (strict), genre hard gate −329 (+39 rescued) | 477 | sonic floor is adaptive percentile; strict lands at 0.61 |
| **Artist cap** (`target_artists=22 × candidates_per_artist=6`) | **251** | **−226 = 47% of survivors** |
| Pier-bridge dedup universe | 432 | +piers/helpers |
| Segment pools after gates | 224–427 | *numerically* healthy |
| Segment 5 best achievable min-edge (beam minimax) | **0.003** | qualitative desert |
| After tail-DP + edge repair (4/13) + refusals | **0.028 ships** | repairs pool-bound + pier-blocked |

## Verified mechanisms (each empirically tested)

### M1 — Pier-centric provisioning + late anchors = manufactured deserts (the root cause)
The candidate universe is provisioned as sonic neighbors **of the seed piers** (gate #8,
`artist_style.build_balanced_candidate_pool`), then `restrict_bundle` (#12) makes it irreversible —
segment pools, roam, and edge repair all draw from this ball. Later, **mini-pier promotion**
(balance_gaps) elevates an admitted-but-peripheral track (`a.s.o.`, admitted at ≥0.61 to a Sade pier)
into an **anchor** — whose own neighborhood was never provisioned.

Probes (MuQ space):
- `a.s.o. – Thinking` has **1,200 library neighbors ≥ 0.5** (Cornelius, Madeline Kenney, Men I Trust,
  TOPS — dream-pop) and sim **0.533 to the destination pier**. Not an outlier; not library scarcity.
- Its in-pool neighborhood ≈ only the dual-neighborhood tracks (Men I Trust, Tops) — see M2.
- Best library connector for the broken edge pair: bridge=0.618; **15 of top-20 connectors were
  admitted** — the failure was not admission of *this pair's* connectors but the anchor's desert.
- Beam minimax (`worst_edge_minimax_enabled=true`, verified lexicographic on real T) did its job:
  best available chain was 0.003. **The beam was handed an unsolvable segment.**

Prior art agreement: `project_mini_pier_v1_failed_validation` — "per-segment gate is LOCAL; promoting
depletes downstream pools" — predicted this class. The pier-bridgeability veto
(`project_pier_bridgeability`) fixed exactly this for **artist piers** but checks pier-pair sim, not
**in-pool neighbor support**, and mini-pier promotion doesn't get the equivalent check.
Roam corridors can't help: they roam the restricted bundle — the manifold was amputated at admission.

### M2 — Diversity triple-enforcement consumes connectors
Artist diversity is enforced 3× (pool artist-cap #23, segment collapse #33 [off by default — for this
exact reason], beam `used`/min_gap #34). The **pool-level cap deleted 47%** (226/477) of post-floor
survivors before the beam saw anything — duplicating what the beam enforces anyway (same rationale
that turned `collapse_segment_pool_by_artist` off). Then sequential segment construction **greedily
consumed the scarce dual-neighborhood connectors** (Men I Trust @24, Tops @25 — spent to *reach* the
mini-pier), and cross-segment artist blocking + min_gap denied them to segment 5. Segments have no
lookahead over shared connector resources.

### M3 — Every relaxation trigger is size- or feasibility-based; terminal quality has no feedback
The inventory found ~26/44 gates with fallbacks (bangers relax-to-fill, segment backoff ladder,
genre-gate ValueError retry, never-starve backstop…). **All fire on pool COUNT or hard infeasibility
(path=None). None fire on min_transition.** A pool can be numerically healthy (224) and qualitatively
bankrupt for a specific anchor pair, and nothing upstream ever learns. Post-hoc repairs (tail-DP,
edge repair, edge delete) are pool-bound, pier-blocked, and local — they cannot fix a desert.

### M4 — Redundant stacking and incoherences (inventory highlights)
Full 44-gate table: sub-agent inventory, this session. Highlights:
- **Sonic gated 4×** (pool floor/percentile → external-pool global floor → segment bridge_floor
  [min, or relaxed max for tag-steering] → beam bridge_floor [strict min again — **undoes the
  tag-steering relaxed admission**, #32 vs #36]).
- **Genre gated 4×** with four different references (artist profile mask 0.30; seed-track dense
  percentile; genre-neighbor pool; beam pair floor [soft]).
- **Rhythm 2×** (pool band vs seed; beam band vs interpolated target — only the beam one has a soft
  escape by default).
- **Dead code:** `pace_gate.filter_candidates_by_bpm_target`/`_onset_target` have no callers;
  `_enforce_min_gap_global` defined but not invoked (beam enforces live). Candidates for cleanup
  (principle 22).

## Verdict on the three hypotheses

1. **Legitimate scarcity?** Minor factor. Sade's genre-eligible neighborhood is thin-ish (5,457) and
   max pier-sim is 0.88–0.93, so a *tight* Sade pool is honestly small. But the shipped catastrophe
   required none of that: the library held 0.6+ bridges for the broken edge and 1,200 neighbors for
   the deserted anchor. Starvation as experienced = manufactured.
2. **Artificial restriction?** Yes — M1 (pier-centric universe frozen before anchors finalize) and
   M2 (pool-level artist cap; 87% genre pre-mask) are the quantified offenders.
3. **Too complicated / needs elegance?** The pier+beam topology is fine and validated. The elegance
   debt is **coordination**: six layers × private floors × no shared quality currency × no feedback.
   Complexity shows up as gates that re-check the same signal against different references and
   relaxation ladders that watch the wrong variable (count, not quality).

## Recommended direction (not yet designed — needs its own brainstorm + spec)

Ordered by leverage:
1. **Anchor-complete provisioning.** The universe must be provisioned around every anchor that will
   exist. Either (a) choose mini-piers *before* pool build and union their neighbor balls, or
   (b) re-provision on promotion (pull the promoted anchor's top-K library neighbors into the bundle,
   re-gated by the active mode floors). (b) is surgical; (a) is cleaner long-term.
2. **Single-enforcement for diversity.** Retire the pool-level artist cap (keep beam enforcement),
   exactly as was done for segment collapse. Frees ~47% of the strict-mode pool.
3. **Quality-triggered outer loop.** Generalize the home-cliff finding: when final min_T <
   transition_floor after repairs, re-provision around the failing anchor/segment (not just lower
   the global floor) and rebuild. Bounded (2–3 attempts).
4. **Coherence cleanups:** #32/#36 double bridge-floor contradiction; dead pace_gate filters;
   `_enforce_min_gap_global`; consider collapsing the four genre references.

**Validation gate before any build:** reproduce M1/M2 on ≥2 more artists (a spread-out artist and a
dense-neighborhood artist), seeds-mode included; run the golden suite + slider dial audit after each
change; success metric = worst-edge (min_T) distribution across the dial grid, not means.

## Multi-artist validation (same day — 5 artists × home/open, 10 real generations)

Corpus: Bill Evans Trio (dense jazz), The Strokes (garage), Swirlies (shoegaze), Aaliyah (R&B),
Alex G (lo-fi indie). Harness: `scratchpad/starvation_validation.py` (policy-layer faithful, DEBUG
logs `sv_*.log`, parsed JSON `starvation_validation.json`).

**M1 — CONFIRMED with dose-response.** Coverage = fraction of an anchor's top-100 library neighbors
clearing the run's sonic admission floor. In every `home` run the worst edge was **anchor-adjacent**,
and in 3/5 adjacent to that run's *lowest-coverage* anchor. Final min_T tracks minimum anchor
coverage monotonically:

| Artist (home) | floor | min anchor coverage | min_T | below_floor | top-20 library bridges admissible |
|---|---|---|---|---|---|
| Bill Evans Trio | 0.76 | 0.81 | 0.899 | 0 | 20/20 |
| Aaliyah | 0.71 | 0.41 (Yaya Bey) | 0.814 | 0 | 14/20 |
| The Strokes | 0.76 | 0.41 (Interpol) | 0.622 | 0 | 17/20 |
| Alex G | 0.62 | 0.30 (Waveform*) | 0.479 | 0 | 5/20 |
| Swirlies | 0.64 | **0.09** (Lovesliescrushing), 0.14 (Ride) | **0.018** | **3** | **2/20** |

The Swirlies crater is the SADE mechanism replicated: library held a 0.599 bridge for the broken
pair but the floor admitted only 2 of the top-20 connectors. Refinements: (a) **seed piers can also
be under-covered** (a Swirlies seed pier at 0.33 — the percentile floor centers on the artist's
joint ball, not each pier's own neighborhood); (b) low coverage is **necessary, not sufficient**
(Aaliyah's 0.41 anchor did no damage — needs unlucky geometry/consumption on top); (c) dense-artist
runs (Bill Evans) show tail-DP windows of 0.008 rescued to 0.899 — **the repair stack works
precisely when neighborhood support exists**, reinforcing that repairs can't fix deserts.

**M2 — CONFIRMED structural.** Artist-cap deletion share of the post-floor pool: home 14–51%,
open 32–60% (median ≈ 40%; e.g. Bill Evans open −825 of 1,373). Anti-correlates with floor
tightness — where floors bite hardest the cap bites least and vice versa, so the *combined*
restriction is large everywhere.

**M3 — CONFIRMED in sharper form.** Swirlies/home actually fired one feasibility relaxation
(`widened=1`) and **still shipped 3 below-floor edges** — the ladder exists but watches
infeasibility, not quality; a quality failure never re-triggers it. All healthy runs: zero firings.

Also noteworthy: Swirlies at the *default* GUI detent (`open`) landed min_T = 0.256 — marginal.
Spread-neighborhood artists sit near the cliff even without strict settings.

## Cross-references
- `scratchpad` probes: `connector_test.py`, `minipier_outlier_test.py`, `pier_proximity.py`,
  `repro_home.py` (session 2026-07-12)
- Memory: `project_gui_dial_knob_audit`, `project_pier_bridgeability`,
  `project_mini_pier_v1_failed_validation`, `project_foundation_similarity_research` (recall caveat:
  the sonic ranking itself under-recalls true soundalikes — compounds M1),
  `project_roam_corridors_engine`, `project_weak_edge_cascade_reorder`
- CLAUDE.md gotcha "Segment-pool one-per-artist collapse is OFF by default" — M2 is the same lesson
  one layer up.
