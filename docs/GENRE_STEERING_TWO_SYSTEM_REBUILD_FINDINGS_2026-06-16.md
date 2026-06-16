# Genre steering — two-system rebuild: findings

**Date:** 2026-06-16
**Status:** Parked research/findings. **Resume after the gold-genre-tagging work** (see `docs/GENRE_RESOURCES_AND_CLAUDE_ADJUDICATION_ROADMAP_2026-06-15.md`). This document is the "why + what," not an implementation plan — the rebuild depends on foundations that the gold work delivers, so it must not start first.

**Thesis (decided):** Genre steering should be **two systems, not one** — a *cohesion* system for **similar seeds** and a *bridge* system for **diverse seeds** — because holding a genre neighborhood and crossing a genre chasm are different problems that the current single mechanism conflates.

---

## 1. What the code does today (the conflation)

Per-segment genre guidance is built by **one function**, `_build_genre_targets` (`src/playlist/pier_bridge/genre_targets.py`), with sibling variants `build_dense_genre_targets` (PMI-SVD embedding) and `taxonomy_steering.build_taxonomy_genre_targets` (graph). All three are the **same paradigm**:

> Take pier A's genre vector and pier B's genre vector, and produce per-step target vectors by **interpolating from A to B**. The beam then scores each candidate by **cosine to the step's target**.

Modes are variations on that one idea:
- **vector / linear** (`cfg.dj_ladder_target_mode="vector"`): `g = (1−f)·vA + f·vB`, normalized, optional IDF weighting.
- **arc**: same, with a curved progress schedule (`_progress_target_curve`).
- **ladder**: walk `_shortest_genre_path` over a genre-similarity graph (edge cost `1 − sim`) from a top label of A to a top label of B, then interpolate along the path's waypoint vectors (one-hot or `_label_to_smoothed_vector`).

Two things are decisive:

1. **The same engine runs regardless of how far apart the piers are.** Nothing selects a different mechanism when A and B are genre-neighbors vs genre-strangers. Similar-seed segments just interpolate over a short distance.
2. **There is no dedicated "stay in the neighborhood" system.** For similar seeds, cohesion is an *accident* of a short interpolation plus the beam's transition floor and the `GenrePairSimProvider` genre-edge floor — not a positive, designed objective.

And the engine's core operation is **cosine to an interpolated genre vector**, which the codebase itself has documented as broken for discrimination:

> *(`taxonomy_steering.GenrePairSimProvider`)* "the graph-smoothed genre-vector **cosine** cannot separate bad edges from good ones (shared rock-family mass blurs everything to ~0.6–0.7); tag-level **max-sim** separates them cleanly."

The IDF weighting, the `coverage_A`/`coverage_B` decay bonus, and the broad-vs-specific label preference are all **patches** compensating for the fact that a single interpolation-by-cosine engine is the wrong primitive for at least one (really both) of the two jobs.

---

## 2. Why one engine cannot serve both jobs

**Similar seeds → a metric / neighborhood problem.**
The goal is to *hold* a coherent region: every step a small, perceptually-faithful genre move; the worst step still close (north star #5); a hub track must not be spuriously "close to everything." There is **no path to walk** — the target is roughly *constant* (the seeds' shared genre anchor). Interpolating A→B here is, at best, a no-op; at worst it manufactures drift the segment didn't ask for. The hard part is a **hubness-corrected local similarity** and a tight worst-edge bound — neither of which the current engine expresses.

**Diverse seeds → a routing / path problem.**
The goal is a *smooth geodesic* from A's genre region to B's: each hop small, the whole arc spanning a large genre distance, **every intermediate rung populated by real tracks in this library**, monotone progress, no jarring jump. The current "linear/vector" mode is actively wrong here: `(1−f)·vA + f·vB` produces **synthetic midpoint vectors that correspond to no real genre** (halfway between `shoegaze` and `gangsta rap` is not a genre), and cosine-scoring against that midpoint admits whatever blurs toward it. The "ladder" mode is the right *instinct* (walk real genres), but it's bolted onto the interpolation framework, uses the hand-calibrated similarity graph, and falls back to linear drift whenever a path isn't found.

These are not two tunings of one system. They are **two systems**: a neighborhood metric vs a geodesic router.

---

## 3. The two-system design

### System S — Cohesion (similar seeds)
- **Objective:** keep every candidate within a hubness-corrected neighborhood of the seeds' shared genre **anchor**; minimize per-step genre distance; bound the worst edge. *Diversity-within-cohesion* (a tight band around the anchor, not nearest-clones) so it doesn't collapse to one sound.
- **Mechanism:** compute the seeds' genre anchor (their common specific-genre core). Admit/score candidates by **mutual-proximity** similarity to the anchor (not raw cosine, not raw NN — mutual proximity is the literature's principled hubness fix). The per-step target is ~constant (the anchor), so there is no path machinery, no synthetic midpoints, no coverage-decay patch.
- **Worst-edge meaning:** the max genre step inside the neighborhood is bounded directly.

### System D — Bridge (diverse seeds)
- **Objective:** a geodesic path from A's genre region to B's, through **real, well-populated** intermediate genres, monotone, with the worst rung-to-rung jump bounded.
- **Mechanism:** shortest/smoothest path over a **faithful, hubness-corrected genre geometry** (typed edges: lineage / scene-adjacency / fusion), with **edge costs that are mutual-proximity-corrected** and **rungs filtered to genres that actually have enough tracks** in the library (so the arc never routes through an empty genre). Per-step targets interpolate **along the path of real genres** — never a straight line through vector space. Broad/umbrella genres may serve as connective tissue but cannot *be* the bridge (the hub guard, done by construction rather than a post-hoc cap).
- **Worst-edge meaning:** no single rung transition exceeds the bound; if it would, insert a rung (densify the ladder) rather than jump.

### Selection: per-segment, by pier genre-distance
The choice between S and D is **not global** — it's per **bridge segment**, decided by the genre distance between the two piers bounding that segment:
- Piers genre-close → **System S** (hold).
- Piers genre-far → **System D** (bridge).
- A single-genre/similar playlist naturally uses S on every segment; a wide multi-seed playlist uses D where piers are far and S where they're close.
- `cohesion_mode` (strict/narrow/dynamic/discover) tunes the **threshold and tightness** (how close counts as "hold," how big a bridge is allowed), rather than selecting a route shape.

This unifies the two systems under one dispatcher at the existing seam (`_build_genre_targets`), which becomes: *measure pier genre-distance → dispatch to S or D → return per-step guidance for the beam.*

---

## 4. Shared foundations — why this waits on the gold work

Both systems sit on **two foundations**, and both foundations are exactly what the gold/adjudication program is about:

1. **Accurate per-track genre *identities* (positions).** Tight, weighted, *specific*, facet-separated leaf sets — the gold-tagging + Claude-adjudicator deliverable. With today's bloated 12-genre authority, *both* systems are poisoned: System S's anchor is diluted by broad/redundant tags; System D's rungs and endpoints are mislocated. **Garbage positions → garbage neighborhoods and garbage paths.**
2. **A faithful, hubness-corrected genre *geometry* (distances + paths).** The graph-similarity matrix is a v1; "perfect" likely means a **mutual-proximity** geometry (and possibly a learned genre embedding grounded in audio+text, anchored to the curated taxonomy), replacing the hand-calibrated `edge_base`/`edge_span` constants in `src/genre/graph_similarity.py`.

**Mutual proximity is the single shared fix** — it corrects hubness for System S's neighborhood metric *and* System D's path edge costs. This is the cleanest reason to do the foundations first and the steering rebuild second: the same corrected geometry feeds both systems.

> Track-to-track genre similarity, in both systems, should be an **optimal-matching / best-match distance between two weighted genre sets** under the geometry's ground distance — not cosine over sparse vectors. (Both the literature and this repo's own "rock-family blur" scar point the same way.)

---

## 5. The seam, concretely (for the future implementer)

- **Today:** `_build_genre_targets(pier_a, pier_b, …)` returns `list[np.ndarray]` per-step targets; the beam scores `cosine(candidate_genre, target[step])` and adds the coverage bonus. Three parallel builders (raw-vector, dense PMI-SVD, taxonomy-graph) all follow this contract.
- **Rebuild:** keep the *contract* (the beam still consumes per-step genre guidance) but replace the *body* with the dispatcher:
  - `genre_distance(pier_a, pier_b)` over the corrected geometry → branch.
  - **S:** emit a (near-)constant anchor target + a neighborhood-radius admission, scored by mutual-proximity, not cosine-to-midpoint.
  - **D:** emit a populated, monotone geodesic ladder; targets interpolate along *real* rungs; worst-rung-jump bounded by rung insertion.
  - Retire the linear/vector synthetic-midpoint mode and the coverage-decay patch (they exist to paper over the missing neighborhood model).
- **Config:** collapse `dj_route_shape` / `dj_ladder_target_mode` / `dj_genre_vector_source` into the S/D dispatch + `cohesion_mode` thresholds. (A configured knob that can't act is a startup error — retire dead modes, don't leave them inert.)

---

## 6. Open questions / design considerations (resolve during the rebuild)

1. **Pier ordering for diverse seeds.** With ≥3 distant piers, the *order* in which they're visited is a TSP-in-genre-space problem — a good order minimizes total bridge distance and avoids backtracking. System D's quality depends on it; today pier order is set upstream (`seeds.py`) without a genre-distance objective. Decide whether ordering moves into System D.
2. **Facets as orthogonal modifiers.** Mood/texture/era/instrumentation should shape *transition smoothness* (and maybe the neighborhood radius), but they are not genres and must not enter the genre path. Where do they attach — a separate facet-smoothness term in the beam?
3. **Does System S need per-step targets at all,** or just a constant anchor + radius + a worst-edge cap? Simpler is likely better.
4. **Geometry: pure graph vs learned embedding.** Mutual-proximity over the curated graph may suffice; a learned genre embedding (audio+text, taxonomy-anchored) is the more ambitious option. Decide after the gold work exposes how good the positions can get.
5. **Validation.** Faithfulness on held-out **human triplets** ("is A closer to B or C?"), plus worst-edge audits per system — never self-scored in the space being built (the circularity rule).

---

## 7. Literature anchors

- **Hubness is the central documented failure of music similarity** (Aucouturier & Pachet 2004; Flexer et al.) — broad/hub items dominate nearest-neighbor lists; the principled fix is **mutual proximity / shared-neighbor** geometry (Schnitzer et al., *Mutual proximity graphs*). This is why both systems must be hubness-corrected by construction, not by a hand-tuned broad-pair cap.
- **Curated ontology ≈ or better than corpus co-occurrence** for genre similarity, and more explainable (Schreiber; ontology-guided multimodal, MDPI 2025) — supports the typed taxonomy graph as the geometry substrate.
- **Playlist sequencing = graph traversal + transition optimization** (Bittner & Gu) — supports System D as geodesic routing with a worst-edge/transition objective.
- **Social tags are noisy, long-tailed, and conflate genre/mood/instrument/junk** (Lamere) — supports separating genre from facet and denoising at the identity layer (the gold work).

---

## 8. Relationship to prior work

- Supersedes the single-arc **genre-edge-safeguards** redesign (memory: `project_genre_embedding_anisotropy`) — that kept the one-arc paradigm (waypoint vote + adaptive floors + niche ladder); the two-system cut is more fundamental and absorbs its good parts (niche ladder ≈ System D rungs; adaptive floors ≈ per-system worst-edge bounds).
- Consumes the genre-adjudication roadmap's output (accurate identities) and the taxonomy-graph geometry (SP4). **Does not block the gold work; the gold work unblocks it.**

---

## 9. One-line summary

Today's steering interpolates one genre vector toward another and scores by a cosine that is known to blur; it should be replaced by **two systems** — a hubness-corrected **neighborhood metric** for similar seeds and a hubness-corrected **geodesic router over real, populated genres** for diverse seeds — dispatched per segment by pier genre-distance, both standing on accurate per-track identities and a mutual-proximity geometry that the gold/adjudication work must deliver first.
