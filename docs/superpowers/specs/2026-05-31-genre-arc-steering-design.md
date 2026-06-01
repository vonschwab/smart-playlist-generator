# Genre Arc Steering & Adaptive Calibration — Design

**Status:** Design approved 2026-05-31. Supersedes the prev-track model in `2026-05-30-genre-edge-safeguards-design.md` (which is partly implemented on branch `genre-edge-safeguards`).
**Author context:** During calibration brainstorming, the project owner reframed the goal: a bridge must *arc* between two piers in **both sonic and genre**, progressing track-by-track, **even when the piers are disparate** in both. Research then found the genre arc already exists as dormant infrastructure (`_build_genre_targets` waypoints), and that the Jan-2026 `dj_bridging_status_audit.md` independently diagnosed the same root cause found this session: genre is pooled but the beam selects on sonic, so genre never gets a vote (`chosen_from_genre_count ≈ 0`), and the waypoint score was suppressed to tie-break edges only.

## 1. Purpose

Make genre a **first-class, arc-shaped vote** in pier-bridge edge selection — the genre mirror of the existing sonic `bridge_score` — and make all genre floors **adaptive (percentile-relative)** so they survive embedding rebuilds and adapt to sparse-vs-dense seeds and disparate-vs-similar piers. This replaces the static track-to-previous-track floor from the earlier `genre-edge-safeguards` work, which fights the arc between disparate piers.

Non-goals: general sonic-scoring re-examination (separate initiative); LLM-prior sidecar rebuild.

## 2. Mechanism — genre arc as a first-class beam vote

The beam edge score becomes symmetric across sonic and genre, both "between-the-piers" arc terms:

```
score = w_bridge · sonic_bridge(cand)          # harmonic mean of sonic sim to pierA & pierB (EXISTS)
      + w_transition · sonic_transition(cand)  # step smoothness (EXISTS)
      + w_genre · genre_arc(cand, step)         # NEW first-class genre vote
```

- **`genre_arc(cand, step) = waypoint_sim(cand, g_target[step])`** — closeness of the candidate's dense genre vector to the per-step interpolated pier-A→pier-B genre target. First-class: added to `combined_score` unconditionally (NOT tie-break-gated — this fixes the audit's "genre never gets a vote" failure). Replaces Task 3's `genre_sim(cand, previous_track)`.
- **Dense arc targets — two route shapes, both in dense space:**
  - **`linear` (direct interpolation):** `g_target[step] = normalize((1 − frac)·dense[pierA] + frac·dense[pierB])`, `frac` = linear step fraction or arc curve. Built from `X_genre_dense`, so `waypoint_sim = dense·dense` is coherent. No IDF needed (the dense embedding already bakes in IDF + de-anisotropization). This is the automatic fallback.
  - **`ladder` (genre-hopping through niche rungs) — the primary mode for disparate piers:** rather than blend pier-A and pier-B genres directly (which muddies disparate pairs like Sonic Youth → The Replacements), walk a path of intermediate **niche** genres and interpolate along it.
    - **Rungs are niche genres pathed through a data-driven graph derived from `genre_emb`** (the 863×64 per-genre dense embedding): each genre's nearest genres above a cosine threshold form the adjacency graph. Because the embedding de-anisotropizes + IDF-weights, this graph reflects *niche* adjacency. **Broad hubs (rock/indie/alternative/pop, via a hub-exclusion list / IDF penalty) are excluded as path nodes**, so a path structurally *cannot* collapse into broad genres.
    - **Path = shortest route** from a pier-A niche genre (top IDF-weighted labels) to a pier-B niche genre, e.g. noise rock → post-punk → jangle/college rock → power pop.
    - **Each rung label → its `genre_emb` row (a dense vector);** `g_targets` interpolate *along the rung sequence* in dense space (reuses the existing ladder interpolation, but over dense rung vectors instead of sparse one-hot). When no niche path exists, fall back to `linear`.
  - This reworks `_build_genre_targets`: replace the sparse one-hot/smoothed rung vectors with dense `genre_emb` rungs, and feed it a `genre_emb`-derived niche graph (the small hand-curated `genre_similarity.yaml` is superseded as the graph source).
- **Genreless piers/candidates** (zero dense vector): if a pier has no dense vector, fall back to the existing neighbor-average (`_fallback_genre_vector`) or, failing that, skip the genre arc for that segment (sonic-only). A genreless *candidate* skips the arc floor + steering term (graceful, via the existing `genre_present` mask).
- **Weights** renormalize to sum to 1 when steering is on (`w_bridge + w_transition + w_genre = 1`), as already implemented in Task 1.

## 3. The two floors — adaptive and contextual

Both floors are **percentiles**, not absolute constants — so they survive embedding rebuilds — and both are **contextual** so the arc holds across sparse/dense seeds and disparate/similar piers.

- **Admission floor (candidate-pool genre gate; controls feasibility): per-seed.**
  `floor = dense-sim value at percentile P_admit of THIS seed's dense-sim-to-library distribution.`
  Computed at query time (one matrix-vector + percentile, negligible). A sparse seed (Charli) and a dense seed (Real Estate) get different *absolute* floors at the same `P_admit`, fixing sparse-vs-dense pool starvation.
- **On-arc floor (`genre_arc_floor`; the beam safeguard): per-segment.**
  `floor = waypoint_sim value at percentile P_arc of the segment pool's candidate→g_target[step] distribution.`
  A disparate-pier segment has a low *achievable* waypoint_sim ceiling; a per-segment percentile keeps the top `(1 − P_arc)` most on-arc candidates regardless of absolute achievability — so the arc holds even between far-apart piers, instead of the segment being declared infeasible.
- **`weight_genre`** stays a per-mode **scalar** vote weight (calibrated directly; renormalized with bridge/transition).
- **Relaxation** (reuses the existing tier): if a segment is infeasible, progressively raise the admitted fraction by lowering `P_arc` toward a floor before failing; per-seed admission relaxes `P_admit` likewise.

**Calibration knobs become:** `P_admit`, `P_arc`, and `weight_genre` **per cohesion mode**. Percentiles are stable across rebuilds; the per-seed / per-segment basis absorbs sparse-vs-dense and disparate-pier variation.

## 4. Calibration methodology

Metrics narrow to a shortlist; the owner auditions and picks (decided 2026-05-31).

- **Harness:** a reproducible, checked-in script (`scripts/calibrate_genre_arc.py`, in the `research_genre_*` family) that, given the *current* artifact + sidecar, sweeps a small grid over `(P_admit, P_arc, weight_genre)` per mode across the reference seeds and emits a markdown report + a shortlist. Ladder knobs (graph adjacency cosine threshold, max ladder steps, top-labels per pier, hub-exclusion list) get sensible defaults + a coarse check, not a full sweep. Idempotent and read-only w.r.t. data.
- **Reference seeds:** Charli XCX (sparse-niche), Real Estate (dense indie), Bill Evans (jazz), Beach House (dream pop), Minor Threat (punk). The disparate-pier test is a *measurement*, not a separate seed: the harness reports arc adherence + monotonicity for the most-disparate pier-pair within each seed's own playlist (Charli will exercise this most).
- **Metrics per (seed × mode × config):**
  - **Feasibility** — hard constraint: full playlist, no infeasible segment, across all reference seeds × modes. Any config that fails feasibility anywhere is disqualified.
  - **Arc adherence** — mean / min `waypoint_sim` along the interior (are interior tracks *on* the arc?).
  - **Arc monotonicity** — does interior genre similarity to pier-A fall and to pier-B rise across the bridge? The direct measure of "arcs in genre"; reported overall and for the most-disparate pier-pair. The old system never verified this.
  - **Ladder rung quality** (ladder mode) — are the chosen rungs niche (not broad hubs)? does the rung sequence progress monotonically from pier-A's genres to pier-B's in `genre_emb` space? Reported for the disparate cases (e.g. a Sonic Youth → Replacements-style pier pair).
  - **Selectivity** (admitted pool-size band), **worst-edge** (floor quality — Layer 1 #5), **distinct artists**, **mode separation** (strict < narrow < dynamic < discover, monotonic in pool size / arc tightness).
- **Output:** markdown report + **shortlist of 2-3 configs per mode** passing feasibility + bands, ranked by arc adherence + monotonicity. The owner generates playlists from the shortlist for a few seeds and picks by ear.
- **Cadence:** run once the data stabilizes (enrichment + human review + artifact/sidecar rebuild). Because the knobs are percentiles, future rebuilds need a feasibility re-verify, not a full re-sweep.

## 5. Scope & relationship to committed work

Revises branch `genre-edge-safeguards` (Tasks 1–6 committed); not a fresh build.

- **Reused as-is:** `genre_steering_enabled` flag + `weight_genre` (Task 1); relaxation fields + tier (Tasks 2/4); builder→beam wiring; config.yaml structure.
- **Reworked:** Task 3 beam scoring — prev-track `genre_sim` → waypoint **arc** (`waypoint_sim` vs dense `g_target`), made first-class (lift tie-break gating); `genre_edge_floor` → `genre_arc_floor`.
- **New:** dense `g_targets` path in `_build_genre_targets` (both `linear` and `ladder` route shapes); a `genre_emb`-derived niche genre-adjacency graph (supersedes the hand-curated `genre_similarity.yaml` as the ladder graph source); per-seed admission percentile (candidate_pool) + per-segment on-arc percentile (beam); arc-monotonicity + ladder-rung-quality metrics; calibration harness; percentile config knobs (`P_admit`, `P_arc`) per mode; ladder knobs (adjacency threshold, max steps, top-labels, hub-exclusion).
- **Out of scope:** sonic-scoring re-examination; LLM-prior sidecar rebuild.
- **Timing:** implement + calibrate **after** enrichment + human review + artifact/sidecar rebuild stabilize.

## 6. Testing

- **Unit:**
  - Dense arc target: `g_target` interpolates `dense[pierA]→dense[pierB]`; `frac=0`→pierA, `frac=1`→pierB; rows normalized.
  - First-class genre vote: with sonic tied, a higher-`waypoint_sim` candidate wins; the vote applies on every edge (not only tie-breaks).
  - On-arc floor: a candidate below the per-segment `P_arc` percentile is rejected; relaxation lowers it when infeasible.
  - Per-seed admission percentile: sparse vs dense synthetic seeds get different absolute floors at the same `P_admit`.
  - Genreless fallback: zero-dense candidate/pier skips arc floor + steering, no crash.
  - Arc monotonicity helper: returns increasing-toward-B / decreasing-from-A signal on a synthetic arc.
  - Ladder: `genre_emb`-derived graph excludes hub nodes; shortest path between two niche genres returns niche rungs; rung labels map to dense `genre_emb` vectors; `linear` fallback fires when no path exists.
- **Integration (live artifact, slow):** each reference seed feasible in every mode; interior arc is monotonic; disparate pier-pair still bridges.
- **Harness:** `calibrate_genre_arc.py` runs read-only, emits a report + non-empty shortlist for a stable artifact.
- **Regression:** full `pytest -m "not slow and not gui"` green; goldens updated for any renamed/added config fields.

## 7. Affected files (anticipated)

- `src/playlist/pier_bridge/genre_targets.py` — dense `g_targets` for both `linear` and `ladder` route shapes (dense `genre_emb` rungs over a niche genre graph).
- `src/playlist/pier_bridge/genre.py` (or a new `genre_graph.py`) — build the niche genre-adjacency graph from `genre_emb` cosine (kNN above threshold, hub-excluded); feed `_shortest_genre_path`.
- `src/playlist/pier_bridge/beam.py` — first-class genre-arc vote; per-segment on-arc floor; remove tie-break gating of the genre vote; supersede prev-track `genre_sim`.
- `src/playlist/candidate_pool.py` — per-seed admission percentile floor.
- `src/playlist/pier_bridge/config.py`, `src/playlist/config.py` — rename `genre_edge_floor`→`genre_arc_floor`; add `P_admit`/`P_arc` percentile knobs per mode; keep `weight_genre`.
- `src/playlist/run_audit.py` — arc-floor relaxation fields (rename from genre-floor).
- `scripts/calibrate_genre_arc.py` — new calibration harness.
- `config.yaml` / `config.example.yaml` — percentile knobs per mode.
- Tests: `tests/unit/test_genre_edge_steering.py` (extend/rename), `tests/integration/test_genre_steering_integration.py` (arc monotonicity + feasibility), golden updates.
