# Data Science–Centric Playlist Generation: What We Built

> **Status:** Candidate pooling, ordering, segment-aware transitions, batch evaluation, and post-pass repair are implemented and working as an end-to-end DS pipeline within `experiments/genre_similarity_lab/`.

---

## Plain-language explanation

### What we’re trying to do
We want playlists that feel **curated**: they should match the vibe of a seed track, flow smoothly from song to song, and avoid annoying repetition—especially repeating the same artist too much.

Traditional “similar tracks” approaches often fail in two ways:
1. They pick tracks that are “similar overall” but **transition badly** (the handoff feels jarring).
2. They clump artists or micro-scenes because they don’t enforce **artist diversity rules**.

This refactor builds a pipeline that treats playlist generation like an optimization problem:
- First, gather a **candidate pool** that’s relevant to the seed (sonic + genre similarity).
- Then, **order** those candidates into a playlist while enforcing constraints (artist caps, min-gap, no back-to-back repeats).
- Use **segment-aware sonic features** to judge transitions the way listeners hear them (end of current track → start of next).
- Finally, run a **repair pass** that can fix the weakest transitions without re-running the entire search.

### The process (high level)
1. **Build a feature artifact** from your DB: sonic vectors + genre vectors + artist identities.
2. **Learn genre similarity** from your library’s genre co-occurrence and use it to smooth sparse genre tags.
3. **Create hybrid embeddings** that combine sonic and (smoothed) genre information.
4. From a seed track, build an **artist-aware candidate pool** (relevant + diverse).
5. Build a playlist using a constrained search (beam search) that balances:
   - closeness to seed track
   - smooth transitions between neighboring tracks
   - artist novelty / controlled repetition
6. Optionally run a **repair pass** to improve the rough edges.
7. Run **batch evaluation** across many seeds to tune defaults and validate stability.

---

## Thorough explanation

## 1) Repo layout and “lab” strategy

This DS-centric approach lives in the experiments layer:

- `experiments/genre_similarity_lab/`
  - `data_matrix_lab.py`
  - `genre_cooccurrence_lab.py`
  - `embedding_knn_lab.py`
  - `playlist_candidate_pool_lab.py`
  - `playlist_construction_lab.py`
  - `playlist_eval_batch_lab.py`
  - `artist_key_audit_lab.py`
  - `artifacts/` (NPZ + CSV outputs)

The guiding idea is:
- Iterate fast in “labs”
- Produce deterministic artifacts + diagnostics
- Only later “promote” stable components into `src/` for production integration

---

## 2) Feature artifact: sonic + genres + artist identity

### `data_matrix_lab.py`
Builds the Step 1 artifact (NPZ) from your DB and metadata.

**Outputs include:**
- `X_sonic` (aggregate sonic features per track)
- `X_sonic_start`, `X_sonic_mid`, `X_sonic_end` *(optional; exported with `--export-sonic-segments`)*
- `X_genre_raw` (binary genre tags; cleaned MB/Discogs-only)
- `X_genre_smoothed` (optional smoothing using genre similarity)
- `track_ids`, plus metadata arrays (artist/title, etc.)
- `artist_keys` + provenance:
  - `artist_key_source`
  - `artist_key_missing`

### Artist identity robustness
A major early failure mode was empty/missing artist identities collapsing many tracks into one bucket (breaking diversity controls). Fixes implemented:
- Normalization function for artist keys
- Fallback logic:
  - prefer `tracks.norm_artist`
  - else normalize display artist name
  - else fallback to `unknown:<track_id>`
- Artifact now logs missing fraction, top keys, and saves missing flags.

---

## 3) Genre similarity and smoothing

### `genre_cooccurrence_lab.py` → `genre_similarity_matrix.npz`
Creates a canonical genre vocabulary and a similarity matrix using **co-occurrence** (Jaccard-like similarity).

### Why smoothing matters
Raw genre tags are sparse, inconsistent, and noisy. Smoothing helps by:
- letting related genres share signal
- improving robustness for lesser-tagged artists
- reducing “genre overfitting” on random single tags

The pipeline preserves both:
- **raw genres** for display/debug (`X_genre_raw`)
- **smoothed genres** for embedding construction (`X_genre_smoothed`)

---

## 4) Hybrid embeddings for similarity search

### `embedding_knn_lab.py`
Builds hybrid embeddings combining:
- PCA’d sonic features
- PCA’d smoothed genre features
- configurable weights

It also validates KNN sanity and ensures raw genres are displayed while smoothed genres influence similarity.

---

## 5) Artist-aware candidate pooling

### `playlist_candidate_pool_lab.py`
Given a seed track:
- computes similarity to all tracks using hybrid embeddings
- forms a **candidate pool** with mode-specific behavior
- groups by `artist_key` and caps per-artist contributions to prevent collapse

**Mode presets control:**
- similarity floors / thresholds
- max pool size
- per-artist caps (and optional seed-artist allowances)
- target distinct artists in the pool

**Diagnostics include:**
- seed summary (raw genres, effective artist key)
- config dump
- pool size & artist distribution
- top artists table
- preview of top similar tracks

**Optional output:**
- Save candidate pool as NPZ (`--save-pool-npz`) including indices, sims, artist keys, and params for reproducible downstream runs.

---

## 6) Playlist construction: constraints + scoring + beam search

### `playlist_construction_lab.py`
Builds the final ordered playlist from a candidate pool while enforcing constraints:

**Constraints implemented (mode-dependent):**
- max artist fraction / max per-artist count
- min-gap window (no artist repeats within last *k* tracks)
- no same-artist adjacency (unless extreme fallback)

### Multi-objective scoring
At each step, selection balances:
- **Seed cohesion:** similarity of candidate to seed
- **Local continuity:** transition similarity from previous to candidate
- **Diversity:** bonus for new artist / controlled repeats
- **Repeat logic:** repeat ramp and repeat bonuses when repeats are allowed and desirable

### Beam search
A beam search option improves ordering under stronger constraints and transition scoring.
- Segment mode can auto-increase beam width unless overridden.
- Beam source (CLI vs mode default) is logged and recorded.

---

## 7) Segment-aware transition scoring (end → start)

A major upgrade: transitions are judged using:
- `end(current)` → `start(next)` similarity using segment sonic vectors

### How it works
- Segment matrices (`X_sonic_start`, `X_sonic_end`) are scaled and PCA’d using the same pipeline as aggregate sonic features.
- Vectors are L2-normalized and compared with cosine similarity.
- The transition score is optionally blended with hybrid local similarity via `gamma`.

### Transition floors
Transition floors prevent “clunks.”

**Floor modes:**
- `off`: no floor
- `soft`: allow below-floor but apply penalty
- `hard`: filter out below-floor candidates; relax only if no feasible options

**Mode defaults (as tuned):**
- Dynamic defaults to **hard** floor (verified in batch)
- Narrow / Discover default to **soft** floors
- Floor values and penalty lambdas are now clearly “requested vs effective” in outputs

Diagnostics include:
- min/p10/median/mean/max of segment/hybrid/used local similarity
- count of below-floor options per step
- chosen-below-floor count
- worst 10 transitions table (with floor markers)

---

## 8) Adaptive novelty and repeat control

To avoid either:
- monotonous playlists (too few artists), or
- incoherent playlists (too many new artists too quickly),

we added:
- `avg_tracks_per_artist_target`
- novelty bonus that decays as distinct artists approach the target
- optional penalty for adding new artists beyond target
- repeat ramp bonus after target is reached

This makes “mode feel” tunable and stable.

---

## 9) Time-shaped seed cohesion (“alpha schedule”)

To control playlist arc (how much it “wanders” then returns), we implemented **time-varying alpha**:

- `alpha_schedule = constant | arc`
- `alpha_start`, `alpha_mid`, `alpha_end`
- `arc_midpoint`

Scoring uses `alpha_t` per step:
- Constant: stays anchored
- Arc: starts anchored → explores → returns

Batch A/B results showed:
- Arc improves transition robustness and diversity
- Constant preserves slightly higher seed cohesion

These tradeoffs are now measurable and tuneable.

---

## 10) Post-pass repair system

Ordering can still leave rough edges—especially in soft-floor modes—so we added a repair layer.

### Repair controls
- `--repair-pass {off,auto,on}`
- `--repair-trigger {auto,below_floor,always}`
- `--repair-target-floor` (repair can aim higher than construction floor for stress tests)
- `--repair-max-iters`, `--repair-window`, `--repair-max-edges`
- `--repair-ops` (swap/relocate/substitute_next/substitute_prev)
- deterministic `--random-seed` recorded in outputs

### Repair operators
- **swap / relocate**: reorder-only (cheap, but limited)
- **substitute_next**: replace the next track on a weak edge with an unused pool candidate that bridges better
- **substitute_prev**: optionally replace the previous track on a weak edge

### Repair objectives
- `lexicographic`: maximize min transition first (safe but can be too strict)
- `reduce_below_floor`: primarily reduce the number of below-floor transitions
- `gap_penalty`: uses continuous gap metrics:
  - `gap_sum`, `gap_mean`, `gap_p90` based on `max(0, floor - transition)`

### Safety / monotonicity
Repair now uses strict tuple-based acceptance:
- **never applies a move unless it improves the chosen objective**
- optional `--repair-assert-monotonic` prevents regressions (especially for below-floor metrics)

---

## 11) Batch evaluation and diagnostics

### `playlist_eval_batch_lab.py`
Runs playlist generation across many seeds and writes CSV results.

Key improvements:
- robust output path handling (explicit out path, verification after write)
- summaries computed from the same DataFrame written to disk
- “requested vs effective” fields captured for:
  - transition floor/mode/lambda
  - beam settings
  - alpha schedule values
- deterministic seed handling with `random_state` plumbing

Batch outputs include:
- per-mode summaries (means, percentiles)
- worst seeds by min transition
- counts of floor relaxations / below-floor choices
- drift metrics (seed drift via running mean embedding cosine vs seed)

---

## 12) Reproducibility

To reproduce “bad cases” and debug:
- candidate pools can be saved as NPZ
- playlist construction accepts `--random-seed`
- batch eval propagates fixed random state
- CSV outputs record `random_seed_used` plus effective params

This enables reliable A/B tests and tuning.

---

## 13) What’s next (the next phase)

### Integration phase (Plan Phase 4.5)
The next phase in the DS refactor plan is **integration into the production generator**, meaning:

1) Extract stable code into `src/` as reusable modules:
   - candidate pooling API
   - playlist construction API
   - repair API
   - evaluation helpers

2) Add a feature flag / config switch so production can run:
   - old path vs new DS path side-by-side

3) Add regression tests:
   - constraint invariants
   - deterministic outputs for fixed seeds
   - minimum transition guarantees in hard-floor modes
   - “no Last.fm genres” validation (MB/Discogs-only)

4) Add listening-oriented QA tooling:
   - playlist “heatmap” visualization (seed cohesion + transition quality + artist spacing)
   - curated seed sets (different genres/eras/recording styles)

---

## Appendix: Key files

- `experiments/genre_similarity_lab/data_matrix_lab.py`
- `experiments/genre_similarity_lab/genre_cooccurrence_lab.py`
- `experiments/genre_similarity_lab/embedding_knn_lab.py`
- `experiments/genre_similarity_lab/artist_key_audit_lab.py`
- `experiments/genre_similarity_lab/playlist_candidate_pool_lab.py`
- `experiments/genre_similarity_lab/playlist_construction_lab.py`
- `experiments/genre_similarity_lab/playlist_eval_batch_lab.py`
- `src/similarity_calculator.py` (segment vector support)
- `experiments/genre_similarity_lab/artifacts/` (NPZ/CSV outputs)

---
