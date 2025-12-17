# Playlist Generator v2 – Data-Science-Focused Redesign

## 1. Context

We have an existing playlist generator that:

- Uses **combined genre data** from MusicBrainz + Last.fm (+ file tags).
- Uses **multi-segment sonic analysis** per track (beginning / middle / end / average) with MFCCs, spectral features, tempo, key, etc.
- Uses a **curated genre similarity system** (multiple methods + ensemble).
- Uses a **hybrid similarity score** (genre + sonic) to pick tracks.
- Already has a reasonably clean, modular codebase and a new `/experiments` directory for prototyping.

The goal is to **re-imagine the playlist engine** from a data-science and systems perspective, not just tweak weights.

There is **no large labeled training set**; instead, we rely on **expert human judgment** (the user is an experienced music worker with DS background) to evaluate playlist quality. That means the initial focus is on **good representations, principled similarity, and controllable playlist behavior**, with optional learning/tuning later.

---

## 2. High-Level Goals

We want to design and prototype a new playlist engine with these properties:

1. **Cohesive but not monotonous**  
   - Playlists should feel like a coherent “world”: no genre whiplash, no abrupt mood clashes.
   - At the same time, tracks shouldn’t all feel like the same song slightly remixed.

2. **Good track-to-track flow**  
   - Local transitions between tracks should feel intentional and smooth (or deliberately high-energy, depending on mode), not random.
   - Multi-segment analysis (end of track A → beginning of track B) should matter.

3. **Support for “clusters” / “episodes” inside playlists**  
   - A playlist can be made of a few **mini-arcs** (clusters of related tracks) that transition sensibly between each other.
   - This allows a journey feel: subtle shifts in tone/genre over time.

4. **Parameterizable playlist “personality” via flags**  
   We want CLI flags / config options such as:

   - `--narrow` – Very cohesive, tight mood/genre; minimal exploration.
   - `--dynamic` – Cohesive overall but with more internal contrast and movement.
   - `--discover` – More exploration, but still avoids jarring transitions.

   Internally, these map to tunable parameters (candidate pool size, genre threshold, variety penalties, etc.).

5. **Experiments first, production second**  
   - All new ideas live in `/experiments` as self-contained modules.
   - Once something works and feels good in practice (human judgment), it can be integrated into the main generator.

---

## 3. Design Pillars

We will think of the system as three layers:

1. **Track Representations (Embeddings)**  
2. **Similarity & Transition Scoring**  
3. **Playlist Construction & Optimization**

Each layer should be swappable and configurable.

---

## 4. Layer 1 – Track Representations

### 4.1 Sonic Embeddings (Multi-Segment Aware)

Objective: build a compact **sonic embedding** per track that:

- Uses the existing multi-segment features:
  - Average MFCCs, chroma, spectral centroid, RMS, tempo, etc.
  - Dynamics features: differences between segments (e.g., middle vs beginning, end vs middle).
- Produces a fixed-length vector per track.
- Reduces dimensionality via PCA (or similar) to a smaller vector (e.g., 32D) and stores it in the DB as `sonic_embedding`.

Key requirements:

- Embedding should be **numerically stable** and easy to query.
- Existing code that uses raw features should still work (we can fall back when embeddings are missing).

### 4.2 Genre Embeddings

Objective: build a **genre embedding** per track that:

- Starts from the track’s combined genre list (MusicBrainz / Last.fm / file).
- Uses the existing **genre similarity system** to construct a high-dimensional “genre presence / influence” vector.
- Applies PCA (or similar) to yield a compact `genre_embedding` per track (e.g., 32D).

### 4.3 Combined Representation

Two options we want to explore:

1. Keep `sonic_embedding` and `genre_embedding` separate and combine similarities at scoring time.
2. Concatenate or otherwise combine them and derive a **joint `track_embedding`**.

For now, the plan is:

- Implement **separate sonic + genre embeddings**.
- Implement combined similarity as a **weighted sum of sonic and genre cosine similarities**.
- Keep the option open to build a joint embedding later.

---

## 5. Layer 2 – Similarity & Transition Scoring

### 5.1 Overall Track Similarity

We want a new main similarity function:

- Uses embeddings when available:
  - `sonic_sim = cosine(sonic_emb_1, sonic_emb_2)`
  - `genre_sim = cosine(genre_emb_1, genre_emb_2)`
- Combines them:
  - `similarity = w_sonic * sonic_sim + w_genre * genre_sim`
- Falls back to existing detailed sonic + genre similarity if embeddings are missing.

Configurable weights:

- `w_sonic` and `w_genre` should be configurable per playlist mode (`--narrow`, `--dynamic`, `--discover`).

### 5.2 Transition Score (End of A → Start of B)

We want an explicit **transition score** that focuses on:

- Sonic features from the **end segment** of track A.
- Sonic features from the **beginning segment** of track B.

This should:

- Use a sonic similarity function (same family as overall sonic similarity, but segment-specific).
- Optionally factor in:
  - Tempo differences.
  - Key compatibility.
  - Energy/brightness slopes.

The result: `transition_sim(A → B) ∈ [0, 1]`.

### 5.3 Neighbor Score (For Ordering)

We define a **neighbor score** between track A and B:

- `neighbor_score(A, B) = α * overall_similarity(A, B) + β * transition_similarity(A → B)`

Where:

- `α`, `β` are configurable.
- Different modes can change these (e.g., `--narrow` might weight overall similarity higher, `--dynamic` might put more emphasis on transitions).

---

## 6. Layer 3 – Playlist Construction & Optimization

### 6.1 Candidate Set Generation

Given seeds (track, artist, or existing small set), we want to:

1. Find a **candidate pool** of similar tracks using the embedding-based similarity:
   - For each seed, get K nearest neighbors in embedding space.
   - Merge and deduplicate into a candidate pool.

2. Apply filters and constraints:
   - Genre threshold / compatibility.
   - Duration / BPM ranges if desired.
   - Artist constraints (max tracks per artist, spacing).
   - Optional: minimum or maximum novelty (e.g., known vs seldom-played tracks, if that data exists later).

Parameters to tie to playlist modes:

- `--narrow`: smaller candidate pool, higher genre similarity threshold.
- `--dynamic`: moderate pool and threshold.
- `--discover`: larger pool, lower genre threshold, maybe more novelty.

### 6.2 Ordering as an Optimization Problem

Once we have a candidate pool and pick the subset for the playlist (N tracks):

- Build a graph where:
  - Nodes are track
