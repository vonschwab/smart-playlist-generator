# Playlist Generator – DS Refactor Plan

**Scope:**
Redesign how the playlist generator measures similarity and builds playlists, using proper data-science layers:

1. better genre modeling,
2. hybrid sonic+genre embeddings,
3. playlist-aware candidate selection and ordering.

This document tracks:

* What we’re trying to achieve
* What’s already implemented
* What’s left to do

---

## 1. High-Level Goals

### 1.1. Desired behavior

We want playlists that:

* Feel **cohesive but not monotonous**

  * No tonal whiplash, no chaotic genre jumps
  * But not “50 tracks that are the same song in a trench coat”
* Have **good artist variety**

  * Multiple tracks from the seed artist are fine
  * But no “entire discography” dumps, no clumps of identical takes
* Have **good track-to-track flow**

  * Smooth transitions, with occasional mini-arcs / clusters
* Are **mode-aware**:

  * `--narrow`: tight, focused, “radio-like”
  * `--dynamic`: cluster-hopping but coherent
  * `--discover`: more adventurous, still anchored to the seed

### 1.2. Structural approach

We’re splitting the problem into three layers:

1. **Data layer / feature construction**

   * Sonic multi-segment features → a consistent vector
   * Genre features → cleaned, canonical, smoothed via similarity

2. **Embedding + similarity layer**

   * Sonic PCA + Genre PCA → hybrid embedding
   * Cosine similarities:

     * global (to seed),
     * local (track → track transitions)

3. **Playlist construction layer**

   * Artist-aware candidate pool (no more seed-artist flood)
   * Ordering with:

     * track-to-track smoothness,
     * seed anchoring,
     * artist/album constraints,
     * mode-specific personality.

---

## 2. Current Architecture & Files

**Key experimental directory:**

* `experiments/genre_similarity_lab/`

**Important scripts:**

* `data_matrix_lab.py`
  Build track-level matrices and artifacts (sonic + genre).
* `embedding_knn_lab.py`
  Build hybrid embeddings & run KNN neighbors.
* `album_genre_frequency_lab.py` *(mentioned, used earlier)*
  Inspect album-level genre frequency distribution.
* `genre_cooccurrence_lab.py`
  Build canonical genre list, co-occurrence & similarity matrix.
* `genre_similarity_matrix.npz`
  The genre similarity matrix built from co-occurrence.
* (Planned) `playlist_candidate_pool_lab.py`
  Build artist-aware candidate pools from seed track.
* (Planned) `playlist_construction_lab.py`
  Use candidate pool + transition scoring to build full playlists.

**Core artifact:**

* `experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz`

  Contains (currently):

  * `X_sonic`
    Sonic feature matrix (tracks × features, multi-segment collapsed).
  * `X_genre_raw`
    Binary genre matrix (tracks × genre_vocab) after normalization/filtering.
  * `X_genre_smoothed`
    Smoothed genre matrix using genre similarity.
  * `track_ids`
  * `track_artists` (display only)
  * `track_titles`
  * `artist_keys` (normalized artist identity, from `tracks.norm_artist`)
  * `genre_vocab`

Another artifact:

* `experiments/genre_similarity_lab/artifacts/genre_cooc_matrix.npz`
* `experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz`

Contain:

* Canonical genres
* Genre co-occurrence stats
* Genre similarity matrix `S` (genre × genre)

---

## 3. What We’ve Already Done

### 3.1. Sonic feature layer

**Goal:** Centralize sonic feature extraction and multi-segment handling.

**Done:**

* Added `SimilarityCalculator.build_sonic_feature_vector`:

  * Normalizes and flattens sonic features,
  * Handles multi-segment (beginning / middle / end) via stable aggregation,
  * Ensures consistent vector length (currently 27 dims in the artifact).
* Refactored existing sonic similarity code to use this helper (no behavior change in production, but experiments now share the same logic).
* `data_matrix_lab.py` now uses this helper to build `X_sonic`.

Result: Sonic features are consistent and ready for PCA, KNN, and future segment-aware transition logic.

---

### 3.2. Genre cleaning + source selection

**Goal:** Get rid of trash/meta/overly-broad tags, and prioritize high-quality sources.

**Done:**

* Identified that **Last.FM tags are mostly garbage** for our use case:

  * Tested coverage: Last.FM never uniquely contributed when MB/Discogs had nothing.
  * Decision: **Drop Last.FM entirely** from genre modeling in this DS pipeline.
* From DB + CSV analysis:

  * Built lists of:

    * **hard garbage tags**,
    * **meta/meta-broad tags** (years, “seen live”, artwork jokes, etc.),
    * **super-broad genres** (e.g. Discogs `folk, world, & country`).
  * Iterated with CSVs: `garbage_candidates`, `meta_candidates`, `meta_candidates_extra`, and a manually curated **real_trash** list.
* Introduced `data/genre_filters/meta_broad.csv` to centralize broad/meta genre filters:

  * Includes:

    * decades / year-like: `60s`, `1996`, etc.
    * formats: `soundtrack`, `theme`, `visual album`, etc.
    * marketing-ish tags: `urban`, `urban cowboy`, …
    * some location-only tags where appropriate.
* **Shared normalization & filtering helper**:

  * All labs (frequency, co-occ, step1 matrices) now:

    * Normalize genre strings (case, punctuation, etc.).
    * Apply config-driven `broad_filters` + filter sets from CSVs.

**Result:**

* Current DS pipeline uses **only MusicBrainz + Discogs** genres.
* Broad/meta/garbage tags are stripped before co-occurrence and matrix building.
* Genre vocab in `data_matrices_step1.npz` is much tighter and better behaved.

---

### 3.3. Canonical genre vocabulary + similarity

**Goal:** Build a principled genre similarity matrix so a tag like `jazz` understands `hard bop`, `cool jazz`, etc.

**Done:**

1. **Frequency analysis**

   * `album_genre_frequency_lab.py` builds `album_genre_frequency.csv`:

     * Counts how often each normalized genre appears at the album level.
   * After filtering out trash, a **canonical genre list** is derived.

2. **Co-occurrence**

   * `genre_cooccurrence_lab.py`:

     * Loads canonical genres.
     * Reads album-level genres from `metadata_lab.db`.
     * Builds a co-occurrence matrix (genre × genre) based on album-level co-tags.
     * Converts to Jaccard-like similarities.
     * Saves:

       * `genre_cooc_matrix.npz`
       * `genre_cooc_summary.csv` (diagnostics).
   * Example diagnostics:

     * `jazz` ↔ `hard bop`, `jazz-funk`, `contemporary jazz`, etc.
     * `rock` ↔ `indie rock`, `alternative rock`, etc.
     * `shoegaze` ↔ `dream pop`, etc.
     * `post-punk` ↔ `new wave`, `art punk`, etc.

3. **Genre similarity matrix**

   * Built `genre_similarity_matrix.npz` with:

     * canonical genre list,
     * similarity matrix `S`.
   * Updated to:

     * Use shared normalization/filtering.
     * Optionally include singletons when needed.
     * Provide **identity fallback** for genres not in `S` (so smoothing always works, even if `X_genre` includes new tags).

**Result:**

* We have a **data-driven genre similarity matrix** that lets:

  * `['jazz']` understand `['hard bop']`, `['cool jazz']`, etc.
  * `['post-punk']` understand `['new wave']`, `['art punk']`, etc.
* This is used to **smooth** genre vectors in Step 1.

---

### 3.4. Smoothed vs raw genre matrices

**Goal:** Use genre similarity for better coverage, but still keep raw tags for interpretability.

**Done:**

* `data_matrix_lab.py` now:

  * Builds `X_genre` (binary, normalized, filtered).
  * Copies `X_genre_raw = X_genre` before smoothing.
  * If `--use-genre-sim` is enabled:

    * Loads `genre_similarity_matrix.npz`.
    * Aligns `S` to current `genre_vocab` with identity fallback for missing genres.
    * Computes `X_genre_smoothed = X_genre_raw @ S_sub`.
    * Uses `X_genre_smoothed` for downstream PCA, but keeps `X_genre_raw` in the artifact.
  * Saves **both** to artifact:

    * `X_genre_raw`
    * `X_genre_smoothed`
* `embedding_knn_lab.py`:

  * Loads `X_genre_raw` and `X_genre_smoothed`.
  * Uses `X_genre_smoothed` for embedding.
  * Uses **`X_genre_raw` for display** in neighbor tables, so “genres” reflects actual MB/Discogs tags, not smoothed hub dimensions.

**Result:**

* Embeddings get the benefit of genre smoothing.
* Debug output now shows *actual* genres per track (e.g. `['jazz']`, `['indie rock', 'lo-fi']`).
* No more confusion from smoothed top-dimensions like `['abstract', 'acoustic', 'afrobeat']` showing up as if they were real tags.

---

### 3.5. Hybrid embedding + KNN sanity checks

**Goal:** Validate that the embedding space is musically sane.

**Done:**

* `embedding_knn_lab.py`:

  * Loads `X_sonic` and `X_genre_smoothed`.
  * Does StandardScaler + PCA on each:

    * configurable `--n-components` (capped at actual dims).
  * Builds **hybrid embedding**:

    [
    \text{hybrid} = [w_{\text{sonic}} \cdot \text{sonic_pca} ;||; w_{\text{genre}} \cdot \text{genre_pca}]
    ]

    with CLI flags:

    * `--mode sonic | genre | hybrid`
    * `--w-sonic`, `--w-genre`
  * Computes cosine similarity and prints top-K neighbors for a seed.

**Sanity checks done:**

* **Fela seed** → neighbors around dub, soul, experimental / groove stuff.
* **Ducks Ltd. seed** → Sleeping Bag, Built to Spill, Wombo, Cap’n Jazz, Pavement, Wet Leg, Wild Nothing, Interpol, etc.
* **Ahmad Jamal seed (with smoothing + raw display)**:

  * 24/25 neighbors are Ahmad Jamal / Vince Guaraldi jazz tracks, all tagged `['jazz']`.
  * One outlier: Zaiko Langa Langa `['congolese rumba']` at the fringe — a plausible exploratory neighbor.
* **Kurt Vile – “Cold Was The Wind”**:

  * Neighbors include Mount Eerie, Ada Lea, The Microphones, Stevie Dinner, Wombo, Pavement, Madeline Kenney, Truth Club, etc.
  * Tags like `['indie folk', 'indie rock', 'lo-fi', 'rock']` show a healthy indie/lo-fi cluster.

**Result:**

* The hybrid embedding + cleaned genres + genre smoothing is producing **musically coherent neighborhoods**.

---

### 3.6. Artist normalization in artifacts

**Goal:** Make sure “artist” in experiments means the same canonical artist as in production.

**Done:**

* `data_matrix_lab.py`:

  * Fetches `norm_artist` / normalized artist key from `tracks`.
  * Saves an `artist_keys` array aligned with `track_ids`.
  * Logs the number of unique artist keys.
* `embedding_knn_lab.py`:

  * Loads `artist_keys`.
  * For now, uses it only as metadata, but it’s ready for:

    * grouping,
    * artist caps,
    * candidate pool logic.

**Result:**

* We now have **canonical artist identity** in the artifacts, so future playlist logic doesn’t rely on raw `track_artists` strings.

---

## 4. What’s Still Left To Do

This is the roadmap from here.

### 4.1. Candidate pool construction (artist-aware)

**Status:** *Not implemented yet (planned design established).*

**Goal:** Given a seed track + mode + desired length, build a **candidate pool** that:

* Is strongly similar to the seed (hybrid similarity).
* Is **artist-diverse**, so we don’t start from “50 Jamal tracks”.
* Is mode-aware (`narrow` vs `dynamic` vs `discover`).

**Planned behavior:**

1. From hybrid embedding:

   * Compute `seed_sim[i] = cosine(hybrid[i], hybrid[seed])` for all tracks.
2. Apply a **mode-dependent similarity floor**:

   * narrow: ~0.35
   * dynamic: ~0.30
   * discover: ~0.25
3. Filter tracks passing the floor and not equal to the seed.
4. Group these by `artist_keys`.
5. For each artist:

   * compute `artist_seed_score = max(seed_sim for that artist)`.
6. Sort artists by `artist_seed_score`.
7. Choose:

   * a mode-dependent `max_artist_fraction_final` (already defined),
   * derive `max_per_artist_final = ceil(L * fraction)`,
   * define `candidate_per_artist` and `target_artists`, `max_pool_size` per mode.
8. Walk artists in rank order, taking up to `candidate_per_artist` tracks per artist (with a small bonus for the seed artist), until:

   * `pool_size >= max_pool_size` and
   * distinct artists ≥ `target_artists`.

**To implement:**

* New script: `playlist_candidate_pool_lab.py` (design prompt already outlined).
* Use `artist_keys` for all grouping, raw `track_artists` only for printing.
* Print pool diagnostics:

  * pool size,
  * distinct artists,
  * top artists by count and mean `seed_sim`.

---

### 4.2. Playlist ordering with constraints + modes

**Status:** *Not implemented yet.*

**Goal:** Given the candidate pool, construct an ordered playlist that:

* Respects **artist constraints**:

  * max share per artist:

    * narrow: 20%
    * dynamic: 12.5%
    * discover: 5%
  * **No adjacent repeats** by the same artist except in extreme fallback.
  * **Artist window (min_gap)**:

    * narrow: 3
    * dynamic: 6
    * discover: 9
      → artist appears at most once in any window of that size.
* Uses **multi-segment sonic analysis** for transitions:

  * Compare end-of-track A to start-of-track B (plus genre).
* Balances:

  * Global similarity to the seed (`seed_sim`),
  * Local similarity to the previous track (`local_sim`),
  * Diversity / novelty.

**Planned mechanics:**

1. **Scoring terms:**

   * `seed_sim(j)` – global similarity to seed (from hybrid embedding).
   * `local_sim(i,j)` – transition similarity (using appropriate segments and genres).
   * `diversity_bonus(j)` – reward rarely-used or new artists.
   * `penalties(j)` – for artist overuse, breaking min_gap, album repetition, etc.

2. **Score function:**

   For candidate `j` following track `prev`:

   [
   \text{score}(j) = \alpha \cdot \text{seed_sim}(j)
   + \beta  \cdot \text{local_sim}(\text{prev}, j)
   + \gamma \cdot \text{diversity_bonus}(j)
   + \text{penalties}(j)
   ]

   * `alpha, beta, gamma` are **mode-specific** and tunable.
   * Penalties include:

     * large negative if adding `j` would exceed `max_per_artist_final`,
     * negative if its `artist_key` appears in the last `min_gap` tracks,
     * negative if same album is overrepresented or repeated consecutively.

3. **Greedy construction loop:**

   * Start with the seed track.
   * At each step:

     * Consider a local neighborhood: top `M` candidates by `local_sim(prev, j)`.
     * Score them with the function above.
     * Choose the highest scoring track that doesn’t violate hard constraints.
     * Append it, mark it as used, update artist/album counts & windows.
   * Stop at desired length or when candidates exhausted.

**To implement:**

* New script: `playlist_construction_lab.py`.
* Reuse embedding construction logic (or import from a shared helper).
* Use `artist_keys` everywhere for artist logic.
* Implement `min_gap` and adjacency rules.
* Use segment-specific sonic vectors for `local_sim`.

---

### 4.3. Multi-segment transition modeling

**Status:** *Partially leveraged (aggregated for X_sonic), not yet exploited for ordering.*

**Goal:** Use beginning/middle/end sonic segments to improve transitions, not just overall similarity.

**To do:**

* Extend `SimilarityCalculator.build_sonic_feature_vector` or add a parallel helper to expose segment-specific embeddings:

  * e.g. `X_sonic_start`, `X_sonic_mid`, `X_sonic_end` or a function to pull the right slice.
* In `playlist_construction_lab.py`:

  * For `local_sim(prev, j)`:

    * Use end segment of `prev` vs start segment of `j` (plus genre component).
  * Optionally blend:

    * overall similarity,
    * segment transition similarity.

This can be a second phase after we’ve proven the simple overall hybrid embedding works for transitions.

---

### 4.4. Mode tuning + evaluation loop

**Status:** *Not started — depends on candidate + ordering prototype.*

**Goal:** Use your ears + domain knowledge to refine:

* `similarity_floor` per mode,
* candidate pool sizes,
* `alpha / beta / gamma` per mode,
* `max_artist_fraction_final`, `min_gap` (if needed),
* genre smoothing thresholds (top-k neighbors per genre, etc.).

**To do:**

* Build small CLI flows to:

  * export candidate pools + playlist runs to CSV/JSON.
  * log metrics (artist count, per-artist counts, average seed/local sim).
* Run manual listening tests with:

  * a few seed tracks across genres (jazz, indie rock, ambient, Afrobeat, etc.).
  * each mode (`narrow`, `dynamic`, `discover`).
* Iterate, commit tuned presets once the feel is right.

---

### 4.5. Integration into production playlist generator

**Status:** *Not started.*

**Goal:** Wire the DS lab logic into the main generator in a safe way.

**To do:**

* Decide integration point:

  * Does production call into the new embedding / candidate / ordering stack?
  * Or do we precompute embeddings and/or genre-smoothed features and reuse them?
* Create a config-driven switch:

  * e.g. `use_new_similarity_pipeline: true/false`.
* Ensure we respect existing constraints:

  * DB is still read-only.
  * No destructive changes to `data/metadata.db`.
* Add tests:

  * KNN neighbor sanity checks as regression tests.
  * Anti-regression tests for “no Last.FM genres leaking back in”.

---

## 5. How to Run the Current Experiments

### 5.1. Build full matrices with smoothing

```bash
python -m experiments.genre_similarity_lab.data_matrix_lab \
  --db-path experiments/genre_similarity_lab/metadata_lab.db \
  --config config.yaml \
  --max-tracks 0 \
  --use-genre-sim \
  --genre-sim-path experiments/genre_similarity_lab/artifacts/genre_similarity_matrix.npz \
  --save-artifacts
```

* Uses:

  * MB/Discogs genres only,
  * broad/meta filters from `config.yaml` + `meta_broad.csv`,
  * genre similarity smoothing with identity fallback.
* Produces:

  * `X_sonic`
  * `X_genre_raw`
  * `X_genre_smoothed`
  * `artist_keys`, etc.

### 5.2. Inspect neighbors for a seed track

```bash
python -m experiments.genre_similarity_lab.embedding_knn_lab \
  --artifact-path experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
  --seed-track-id <track_id> \
  --mode hybrid \
  --k 25
```

* Displays:

  * top-K neighbors,
  * raw genres (`X_genre_raw`),
  * artist names,
  * cosine similarities.

---

## 6. Quick Checklist

**Already done ✅**

* [x] Centralized sonic feature vector builder.
* [x] Multi-source genre cleaning; Last.FM removed.
* [x] Shared normalization + broad/meta/garbage filtering.
* [x] Canonical genre list from album-level frequency.
* [x] Genre co-occurrence + similarity matrix (`S`).
* [x] Genre smoothing in `data_matrix_lab` (`X_genre_smoothed`).
* [x] Raw vs smoothed genre matrices preserved in artifacts.
* [x] Hybrid PCA embedding + KNN lab.
* [x] Artist normalization (`artist_keys`) propagated into artifacts.
* [x] KNN sanity checks for multiple seeds look musically correct.

**Next steps 🚧**

* [ ] Implement `playlist_candidate_pool_lab.py` (artist-aware candidate pool).
* [ ] Implement `playlist_construction_lab.py` (ordering with constraints + modes).
* [ ] Add multi-segment-aware transition scoring for `local_sim`.
* [ ] Tune mode presets (`similarity_floor`, `alpha/beta/gamma`, artist caps).
* [ ] Wire new pipeline into production playlist generator under a config flag.
* [ ] Add basic regression tests for neighbors + playlist structure properties.

