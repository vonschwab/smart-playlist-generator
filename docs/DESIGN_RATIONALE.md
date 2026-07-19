# Design Rationale

The lab notebook. The reference docs ([`ARCHITECTURE.md`](ARCHITECTURE.md),
[`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`](TECHNICAL_PLAYLIST_GENERATION_FLOW.md),
[`CONFIG.md`](CONFIG.md), …) say *what the system does and how*. This one says **why** — the
experiments that were run, what won, what was tried and rejected, and the evidence. It exists
because this project is run as a research program: the same measured question ("does X actually
beat Y on this library?") gets asked over and over, and the answers are load-bearing. When a
default looks arbitrary elsewhere, the reasoning is here.

Each entry is **Decision → Why → Evidence → Status**. Read the *methodology* section at the end
first if you're about to run an evaluation — most wrong turns below came from skipping it.

---

## Sonic embedding: towers → MERT → MuQ

The sonic similarity space has been replaced twice, each time because measurement — not intuition
— said the incumbent had a ceiling.

### The hand-built towers were perceptually coarse
- **Decision:** the original sonic space was a hand-built blend of three "towers" — rhythm (9–10
  d), timbre (57 d), harmony (2DFTM, 96 d) — weighted 0.20 / 0.50 / 0.30 (timbre-dominant).
- **Why it was abandoned:** the dominant timbre tower could not tell perceptually distinct music
  apart — it rated **Metallica ≈ Yeah Yeah Yeahs**. No reweighting fixed it; the ceiling was in
  the features, not the blend.
- **Evidence / hard-won detail:** the harmony tower was the subject of its own investigation. The
  original chroma-median harmony encoded *absolute key* (anti-correlated with human judgment in
  isolation, ρ −0.144); the fix was a key-invariant **2DFTM** representation (ρ → +0.210,
  pitch-shift cosine 0.99+). That lifted the blend but didn't lift the ceiling — timbre was still
  the limiter.
- **Status:** removed from the runtime path by SP-B (below). The tower code + the 2DFTM harmony
  sidecar are archived at `data/archive/mert_2026/`, not deleted.

### MERT (learned acoustic embedding) replaced the towers
- **Decision:** fold a learned **MERT-v1-95M** embedding (768-d, `whiten_l2` post-processed) into
  the artifact as the sonic space.
- **Why:** a learned self-supervised acoustic model captures timbre/idiom the hand-built towers
  couldn't. It keys on genre/idiom rather than recording medium (Black Flag → Dinosaur Jr., not
  → cassette-hiss).
- **Evidence:** cross-catalog, seed-artist-excluded neighbour QA — MERT beat the towers by
  **~45–93%**. A "collapse" incident that briefly claimed MERT had broken turned out to be a
  false alarm (see the methodology case study below).
- **Status:** superseded by MuQ; code removed by SP-B, data archived.

### MuQ (contrastive embedding) replaced MERT — SP-B
- **Decision (SP-B, 2026-07):** make **MuQ** (`OpenMuQ/MuQ-MuLan-large`, 512-d, contrastive
  audio-text) the **sole** sonic space. Delete the MERT and tower code paths entirely. Post-process
  with **`center_l2`** (mean-center + L2), *not* whitening.
- **Why:** MERT is a strong *acoustic* model but still misses *fine* similarity — soundalikes it
  should rank #1 landed rank ~1–3k. The fix for similarity is a **contrastive objective** (trained
  so cosine ≈ similarity), not a bigger acoustic model. Whitening was tested on MuQ and *hurt* it —
  MuQ is already well-conditioned, so `center_l2` is the right transform.
- **Evidence:** on Dylan's trusted-triplet gold set ("a known soundalike must beat a known
  near-negative"), **MuQ 84% vs MERT 73% vs CLAP 68%** — MuQ was specifically better, not
  contrastive-models-in-general. Validated end-to-end through the real config chain before the
  code deletion.
- **Status:** **live and sole.** There is no runtime variant choice; `sonic_variant_override`
  resolves to `muq`. MERT/towers are archived rollback material only.

### The centered-transition sigmoid
- **Decision:** score a transition edge as a **calibrated logistic** of the cosine, single-sourced
  so the beam scorer and the post-hoc reporter use identical math; calibration is per-variant
  (`muq` centered at 0.594).
- **Why:** the old linear `(x+1)/2` rescale wasted its output range on negative cosines that real
  edges never produce, compressing the realistic band into [~0.57, 0.75]. The good-vs-bad edge gap
  collapsed from 72% (raw) to 8% — the beam literally couldn't tell a good edge from a bad one.
- **Evidence:** the calibrated sigmoid restored the gap to ~88%; a real generation's worst edge-T
  went 0.13 → 0.78.
- **Status:** live. The old `transition_floor` hard gate was removed (roam-only). With SP-B the
  `transition_weights` / `tower_weights` knobs are gone — a single contrastive space has nothing
  to reweight.

---

## Genre: a published graph, not free-text tags

### The published authority
- **Decision:** the single source of truth for playlist-facing genres is
  `release_effective_genres` (in `metadata.db`), written **only** by the enrichment `publish`
  stage and read **only** through `src/genre/authority.py`. It's resolved against the SP3a layered
  taxonomy graph (`data/layered_genre_taxonomy.yaml`, a living GUI-grown artifact).
- **Why:** "playlists got worse as we added genre machinery" was traced to **four independently
  measured corruption mechanisms**, all from consumers/collectors trusting the wrong layer:
  (1) enrichment trusted a Bandcamp *label-storefront* page over the user's own file tags (and
  counted the same page twice); (1c) Last.fm tags fetched by artist-name string cross-contaminated
  identities (an LA ambient act got a Ukrainian hip-hop act's tags on all its albums, ~76 artists
  affected); (2) inferred hub-families ("rock", "indie") baked into the genre vector saturated
  similarity (random-pair cosine p50 ≈ 0.42 — almost no signal); and the QC ran *inside* the
  broken space and reported "healthy."
- **Evidence / fixes:** inferred-layer exclusion dropped genre-vector p50 0.42 → 0.12; source-split
  fusion (artist-page vs label-storefront weighting, never-drop local tags); and a **surgical delta
  migration** (267 releases, zero collateral) instead of wholesale re-derivation — two wholesale
  attempts had *un-decided* correct past calls.
- **Status:** live. The One Rule (consumers read the authority via `authority.py`; a richer-looking
  internal layer is a publish bug, not a reason to rewire) is enforced and documented in the
  `genre-data-authority` skill.

### `max` genre metric beats soft-cosine
- **Decision:** the runtime genre-edge metric is `max` (the maximum tag-pair similarity over two
  tracks' canonical tags), applied as a *soft* per-mode penalty — never a hard gate. A soft-cosine
  alternative was built, evaluated, and **rejected**.
- **Why:** once the sonic space dominates selection, the genre penalty has little room left to act.
  In that small room, `max`'s coarse "do these share a close tag?" catches the egregious disjoint
  edges as well as graded soft-cosine does.
- **Evidence:** a 7-cell sweep across 8 seeds — soft helped on only 1 of 4 wide trios and
  *cratered* the worst sonic edge on another; on cohesive seeds it was byte-identical to OFF while
  `max` lifted worst-genre-adjacency +0.08. **Key reframe:** the genre penalty can only *demote*
  an off-axis edge, never *promote* a genre-good neighbour — a rank-34 genre-good neighbour ("The
  Embassy") was absent from every playlist while rank-198/233 tracks were present. The lever for
  "include the great neighbour" is sonic admission + beam selection, **not** the genre metric.
- **Status:** `max` is live and the only metric shipped; the soft path was never merged.

---

## Selection & collapse prevention

### Pier-bridge with beam search
- **Decision:** every playlist is built by the pier-bridge topology — seeds are fixed **piers**,
  each adjacent pair is bridged by a **constrained beam search**, progress is monotonic in sonic
  space. The legacy greedy constructor is dead code.
- **Why:** seeds anchor structure (so the result reflects *this* listener's taste, not "popular and
  similar"); beam search optimizes the worst edge per segment ("the worst edge defines the
  experience" — a north-star commitment).
- **Status:** live; the current best topology.

### Variable bridge length shipped; density-floors abandoned
- **Decision:** a segment **flexes its interior length** (add-only) to land more smoothly on the
  next pier, instead of a rigid even split. The alternative — absolute-density floors / MMR to stop
  the beam collapsing into dense space — was **abandoned**.
- **Why (shipped):** a rigid even split padded a segment to a fixed count even when the piers didn't
  bridge smoothly in N tracks; letting the beam choose *when to land* lifts the worst edge cheaply.
- **Why (abandoned):** six density-floor/MMR formulations all failed — dense space is *on-character*
  for roughly half of seeds (so it can't be universally penalized), a pre-generation proxy couldn't
  predict actual drift, and even a gated version moved ~1 seed in 12.
- **Status:** variable bridge length is live; the density-floor lever is explicitly out of scope.

### Anti-sag: anti-center (SP2) + mini-piers (SP3)
- **Decision:** two levers stop long bridges from **sagging** into the generic local average — a
  *scoring* fix (`anti_center`: demote candidates closer to the local pool centroid than to their
  piers) and a *structural* fix (`mini_pier`: split an over-long segment by pinning a high-character
  waypoint as an extra pier).
- **Why:** the beam's own scoring rewards the blur — it's smooth *and* central to everything — so
  it needs an explicit counter-force. Scoring alone wasn't enough: anti-center at strength 2.0
  reduced measured sag (electronic 60→46%, dreampop 117→101%) but *plateaued* — dreampop stuck at
  101%. Structure closes the residual: mini-piers took dreampop sag 103% → 63%.
- **Evidence:** the collapse harness (`collapse_eval` / `collapse_rescore` / `collapse_sweep_compare`,
  against a gold-pairs set) measured both. A retired "hubness" variant of the scoring lever lost
  (weaker, didn't scale) and was deleted.
- **Status:** both live (shipped defaults on; dataclass rollbacks off). This is the "collapse
  prevention" system.

### The weak-edge recovery cascade
- **Decision:** after assembly, a fixed **four-pass** cascade lifts weak/broken edges, escalating
  least- to most-destructive: **variable-bridge (add) → tail-DP (re-optimize a segment tail) →
  edge-repair (swap one track) → edge-delete (remove one track)**. It runs once, not as a retry
  loop.
- **Why:** the beam can't always avoid a weak edge (an outlier pier, a starved segment). Ordering
  the fixers by destructiveness means the playlist is perturbed as little as possible — a swap is
  preferred to a deletion, and deletion is a genuine last resort guarded "never-worse" and against
  breaching a bystander artist's `min_gap`.
- **Status:** live. **Known open limits** (tracked in `CLEANUP_LIST.md`): a *deadzone* (every
  trigger floor is 0.30, so an ugly-but-legal edge at T ≈ 0.46 gets no attention — the corridor
  project's relative-trigger work below narrows but does not eliminate this), and an
  edge-repair-vs-reporter **T-mismatch** (repair has flagged edges the reporter scores as healthy)
  that must be root-caused before the floors are retuned.

---

## The corridor project (2026-07): pooling, repair triggers, pier quality

Two phases replacing how candidates are admitted per segment, when the repair cascade fires, and
which of a seed artist's own tracks become piers. Full detail: `TECHNICAL_PLAYLIST_GENERATION_FLOW.md`
§3.1/3.3/5.0/6.3, `PLAYLIST_ORDERING_TUNING.md`'s corridor + pier-quality sections.

### Pool-starvation research → corridor-first pooling
- **Decision:** replace the pre-generation seed-proximity-ball pool with a **per-segment corridor**
  — candidates whose similarity to *both* adjacent piers clears a self-calibrating percentile floor,
  drawn from a library-wide eligible universe computed once per generation. "The path defines the
  pool," not the other way around.
- **Why:** `docs/POOL_STARVATION_RESEARCH_2026-07-12.md` traced a shipped T=0.028 edge (SADE +
  `home`) to three validated mechanisms, dose-response confirmed on 6 artists: (M1) the seed-ball
  pool was frozen before mini-piers got promoted to anchors, so a promoted anchor's own real
  1,200-neighbor neighborhood was never provisioned; (M2) a pool-level per-artist cap deleted
  14–60% of post-floor survivors that the beam's own `min_gap` enforcement re-does anyway; (M3)
  every one of ~26 relaxation/fallback mechanisms fired on pool *size* or hard infeasibility, never
  on edge *quality* — a numerically healthy pool could still be qualitatively bankrupt for one
  specific anchor pair and nothing upstream would ever learn.
- **Evidence:** the failures were overwhelmingly *manufactured*, not library scarcity — the library
  held 0.6+-quality bridges for every broken edge examined. `docs/superpowers/specs/
  2026-07-12-corridor-first-pooling-design.md` was approved on this evidence, gated by
  `CORRIDOR_FEATURE_PRESERVATION_CONTRACT.md`'s non-degradation bar (every transform/gate/soft-term
  rehomed, zero features lost, an automated no-knob-goes-inert sweep).
- **Status:** live and sole (Phase 1 Task 8 flip, 2026-07-17) — the legacy `SegmentCandidatePoolBuilder`
  KNN-union and its 1,129-line implementation are deleted, not flagged off.

### The `restrict_bundle` discovery
- **Decision:** when the legacy per-cluster external pool (`build_balanced_candidate_pool`) was
  deleted at the Phase 1 flip, Artist mode's `allowed_track_ids` is kept for diagnostics/guard use
  only — `None` is passed to the actual eligible-universe build.
- **Why:** the flip's own post-change corpus caught a critical, unplanned regression:
  `allowed_track_ids` was still hard-clamping the bundle **upstream** of corridor's eligible-universe
  build via `restrict_bundle`, silently reintroducing the exact M1 seed-ball mechanism the whole
  project existed to remove — for Artist mode specifically, corridor had been scanning the ball, not
  the library.
- **Evidence:** root-caused (data/artifact drift and a `candidate_pool.py` code diff both ruled out
  byte-for-byte); after the fix, `below_floor` returned to 0/12 and the admitted universe grew
  2–15× for Artist mode.
- **Status:** live. Generalized into a standing review question for any future change to what feeds
  `build_eligible_universe`: is anything upstream still silently narrowing "the whole library"?

### Per-mode corridor widths
- **Decision:** `corridor_width_percentile` is resolved per `sonic_mode`
  (`corridor_width_percentile_{strict,narrow,dynamic,discover}`), not one global number, with a
  plain-override escape hatch for anyone who wants a fixed width regardless of mode.
- **Why:** narrow (`strict`/`home`-style) and wide (`dynamic`/`discover`-style) corridors need
  different floors to hold `below_floor=0` without over- or under-admitting; a single global
  percentile can't serve both.
- **Evidence:** Phase 1 pinned `strict` (0.985) and `dynamic` (0.95) from direct 4-cell probes;
  `narrow`/`discover` shipped as interpolations for lack of probe time. Phase 2 Task 4 ran 18 real
  generations (3 artists × both remaining detent families × 3 bracketing widths) and **superseded**
  both interpolations: `narrow` 0.9675 → **0.975**, `discover` 0.93 → **0.94**, each winning on mean
  AND worst-case min_T over every width tested, `below_floor=0` throughout. One confound disclosed,
  not resolved: `discover`'s win is partly `segment_pool_max=800` cap saturation on 2 of 3 probe
  artists.
- **Status:** live, all four modes now directly probed (none left as pure interpolation).

### The refuted ramp hypothesis
- **Decision:** do NOT build either of Task 2's originally scoped candidates (anchor-ramp
  force-include; support-triggered pre-beam widening).
- **Why:** Phase 1 left one known weakness — an apparent "outlier-anchor ramp exclusion" pattern,
  naturally hypothesized as corridor's sonic admission excluding the better connector. Task 1's
  mechanism probes (`docs/corridor_baseline/phase2_mechanism_probes.md`) reproduced both deep-dived
  cases directly against the production corridor functions and found the hypothesis **false** for
  both: Parquet Courts segment 4's known-good fix ("Theresa's Sound-World") was already a corridor
  member (41st of 43,241 by similarity), simply not ranked highly enough by the beam, and unused
  because `tail_dp_floor=0.3`/`transition_floor=0.2` both read "clears the floor" as "good enough"
  even with a 30–40-point-better connector sitting unused in the same pool. SADE/home showed a real
  partial match (a genre-relevance-mask exclusion of high-sonic-similarity candidates) but the
  actual best already-admitted candidate still went unused for the same reason, plus a structural
  gap: tail-DP only ever re-opens a segment's *last* slots, so a bad *first* edge is untouchable.
- **Evidence:** both original candidates patch pool admission; the pool was never the problem in
  either deep-dived case — building either would have cost real engineering time on an inert lever.
- **Status:** neither candidate built. Dylan chose the evidence-driven re-plan (relative repair
  triggers, below) over proceeding with the original A/B as scoped.

### Relative repair triggers: "fix broken" → "polish toward achievable"
- **Decision:** tail-DP and edge repair gate on `max(absolute_floor, reference_mean −
  relative_epsilon)` instead of a fixed absolute floor alone (`compute_relative_trigger_floor`,
  `src/playlist/pier_bridge/repair_triggers.py`) — a segment's own realized mean `T` for tail-DP, the
  whole playlist's mean `T` for edge repair.
- **Why:** the mechanism probes reframed the defect precisely — the floors were calibrated to catch
  *broken* edges, not edges merely *worse than what the run could achieve*. An edge that clears 0.30
  but sits well below its neighborhood's own achievable level previously got no fixer attention at
  all.
- **Evidence:** Parquet Courts segment 4 recovered 0.240 → 0.810 (almost exactly the legacy
  manual-swap reference of 0.802); SADE/home's structurally tail-DP-unreachable first edge recovered
  via edge repair, 0.452 → 0.730. 12-cell corpus: **12/12 min_T flat-or-better**, wall-clock *faster*
  (0.92×), `below_floor=0` throughout.
- **Status:** live, default `relative_epsilon=0.25` both mechanisms. `relative_epsilon ≤ 0` is the
  legacy-rollback escape hatch — **not symmetric** with the old `t_floor: 0` disable: once
  `edge_repair_relative_epsilon > 0`, `edge_repair_t_floor: 0` alone no longer fully disables the
  weak-`T` repair arm. The true full-legacy rollback is `edge_repair_relative_epsilon: 0.0`.

### Pier quality: within-artist support, not library-wide density
- **Decision:** demote (never exclude) medoid/pier candidates whose `compute_within_artist_support`
  — mean similarity to their own artist's other tracks, normalized by that artist's median — falls
  below typical; pair it with `reorder_avoiding_low_support_terminal`, a tie-break that prefers any
  same-pier-set re-walk keeping the lowest-support pier off a terminal (opening/closing) seat.
- **Why:** the historical Parquet Courts/Swirlies incidents both trace to an outlier EP cut winning
  a medoid slot and landing at the playlist's most exposed position, purely because clustering had
  no notion of "typical for this artist." The first estimator attempt — a whole-library
  top-100-neighbor density mirroring the existing pier-bridgeability veto — **failed** to separate
  the known outliers from normal candidates (`docs/corridor_baseline/phase2_task3_probe_findings.md`)
  and was not shipped.
- **Evidence:** `compute_within_artist_support` cleanly separated both known outliers (bottom
  ~10–20th percentile of their artist's candidates) and stayed stable across `k=5..20` — a
  fundamentally different reference frame (within-catalog typicality vs. library-wide density).
  12-cell corpus: 11/12 min_T flat-or-better (Bill Evans Trio +0.13/+0.20), one disclosed, bounded
  regression (Alex G/open −0.038, root-caused to forcing a different, imperfect pier into a
  now-vacated terminal seat — a structural trade-off with only two interior seats among four piers,
  not a systemic pattern).
- **Status:** live, default `pier_support_demotion_strength=1.0`. Deliberately **not** wired into
  mini-pier waypoint selection — a bridge waypoint isn't the seed artist's own catalog, so
  within-artist support has no defined meaning there, and the library-wide alternative tried in the
  same probe didn't discriminate the known cases either; an unvalidated mechanism was held out
  rather than guessed at.

---

## Pace / energy

### Pace rebuilt on BPM + onset bands
- **Decision:** pace gates on two embedding-independent **hard bands** (BPM log-distance,
  onset-rate log-distance) plus a soft rhythm penalty — reading DB features, not the sonic
  embedding. A beatless pier disables its own BPM band.
- **Why:** the old rhythm-cosine floor was near-noise and unsatisfiable for ambient artists. BPM is
  meaningless on beatless audio (a drone track read a confident, wrong 161.5 BPM identical to a
  wall-of-noise track); `tempo_stability` was a dead knob; and LUFS loudness is *not* energy
  (mastering-confounded — calm drone read louder than intense noise). Onset rate reliably separates
  beat-presence; arousal (Essentia emoMusic) is the validated intensity signal.
- **Status:** the BPM/onset bands are live. Because they read DB features, pace survived the
  MERT→MuQ migration unchanged. The **energy arc** extension is built but **parked** — measured
  redundant with MuQ for smoothness (MuQ-similarity predicts energy-closeness ~2.5× better than
  tempo-closeness), worth reviving only for *intentional directional* arcs, not anti-whiplash.
  **Corridor Phase 2 finding:** every per-mode pace band field (`bpm_bridge_max_log_distance`,
  `energy_arc_band`, `pace_rescue_k_energy`, …) was silently unreachable from `config.yaml` —
  `resolve_pace_mode` was called with no `overrides` argument at all. Fixed (`_resolve_pace_overrides`,
  `src/playlist/pipeline/core.py:246`) without changing any shipped default; see
  `PLAYLIST_ORDERING_TUNING.md` Knob 5 for the now-live override surface.

---

## Features

### Tag-steering (artist-mode soft lean)
- **Decision:** let a user pick ≤3 of the seed artist's **own published genres** as chips and lean
  the playlist toward them — a *soft* nudge, never a hard gate. Two additive levers: a **pool**
  lever (blend the tag target into the dense genre-admission centroid) and a **pier** lever (on-tag
  bonus in medoid scoring).
- **Why:** users want to say "give me the *dream-pop* side of this artist" without the brittleness
  of a hard genre filter. Softness is the whole point: no tags is byte-identical to legacy, and a
  degenerate target (unmappable tag, missing dense sidecar) degrades to a loud warning + inert, not
  a crash. The chips read the **authority** (`resolved_genres_for_artist`, observed-leaf+legacy,
  inferred hubs excluded) — a new authority reader, not a raw-tag read.
- **Status:** live (artist-mode GUI). A designed **stage-2 beam-level lever was deliberately not
  built** — the stage-1 gate ("build it only if logs show under-delivery") never tripped.

### Popular-seeds / "Oops All Bangers"
- **Decision:** two independent, off-by-default popularity features — `popular_seeds_mode`
  (bias pier *selection* toward an artist's hits) and `popularity_mode` (a pool *admission* gate so
  every track is a "banger").
- **Why:** a deliberate, opt-in "hits" mode, kept separate from seed selection so it never silently
  overrides the user's explicit seeds.
- **Status:** both live, both default off.

---

## Methodology case study: the MERT "collapse" false alarm

A 2026-06-24 session filed a **critical incident**: the live MERT space had "collapsed"
post-rebuild (a same-artist median cosine dropped 0.85 → 0.12; a known neighbour fell from rank 31
to 21,525). A recovery plan was drafted.

A second session **could not reproduce it** on any on-disk file — including the ones labelled
"broken," which gave the exact same healthy numbers as the incident's own stated baseline. Two
root causes: (1) a **bug in the incident's probe script** (it ranked the first track per artist by
substring match instead of the max over the artist's tracks, which is what the pool/beam actually
use); and (2) a **wrong health metric** — reading a raw same-artist cosine *drop* after `whiten_l2`
as "collapse," when that recentering is exactly what whitening is supposed to do to an anisotropic
embedding. **Rank** — what the beam uses — was fine the whole time.

The incident was retracted; no re-fold happened. It is preserved here because the *lesson* is the
point (below), and because "we almost re-ran 55 hours of CPU on one unverified number" is exactly
the kind of thing this notebook exists to prevent.

---

## Cross-cutting methodology (read before any evaluation)

Every decision above leaned on these, and every wrong turn came from skipping one:

1. **A metric computed inside the space you're changing can't validate that space.** Every QC pass
   needs an independent arm — ears, held-out labels, or a different modality. The genre corruption
   and the MERT false alarm both hid behind in-space "healthy" numbers.
2. **Report distributions and the worst case, never the mean.** Min / p10 / p50 / p90 and the
   weakest edge. The floor is the product ("the worst edge defines the experience"); a good mean
   routinely hides a broken edge.
3. **Inspect the actual diff before an irreversible write.** Summary counts lie — a "healthy" delta
   exploded to 5,525 removals on inspection, un-deciding correct past calls.
4. **A policy fix and a data migration are different jobs.** Don't re-derive the whole library to
   deploy a scoring change — use a surgical, additive/subtractive delta and grandfather the rest.
5. **Reconcile disagreeing probes against a baseline before you act.** The MERT incident shipped on
   a single number that a second probe immediately contradicted.

These are formalized in the `evaluation-methodology` skill and encoded in the engineering-discipline
layer of the project guide.
