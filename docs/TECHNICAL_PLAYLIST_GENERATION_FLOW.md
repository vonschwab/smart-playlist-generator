# Technical Playlist Generation Flow
## Artist Mode Deep Dive: Bill Evans Trio Playlist Generation

**Last Updated:** 2026-01-03
**Purpose:** Comprehensive technical reference for understanding the complete playlist generation pipeline

---

## High-Level Summary

When you generate a playlist for "Bill Evans Trio", the system:

1. **Retrieves** all Bill Evans Trio tracks from your library database
2. **Selects** seed tracks (auto-clustered or explicit Seed List mode)
3. **Analyzes** your Last.fm listening history to exclude recently played tracks
4. **Clusters** the artist's catalog into 3-6 style groups using sonic similarity
5. **Builds** a candidate pool of ~1,200-2,000 external tracks with similar sonic/genre characteristics
6. **Constructs** a 30-track playlist using pier-bridge algorithm:
   - Places seeds as structural "piers" (anchor points)
   - Fills gaps between piers with smooth sonic transitions
   - Enforces artist diversity and recency constraints
7. **Validates** the result meets all quality constraints
8. **Exports** to M3U8 file and Plex

**Key Innovation:** The pier-bridge algorithm ensures the playlist flows smoothly while maintaining thematic coherence tied to the seed artist's musical identity.

---

## Detailed Technical Breakdown

### Phase 1: Initialization & Configuration
**Duration:** ~1-2 seconds
**Files:** `main_app.py`, `config_loader.py`, `src/playlist_generator.py`

#### 1.1 Configuration Loading
```python
# config_loader.py:18-25
config = Config(config_path="config.yaml")
config._apply_mode_presets()  # Apply genre/sonic mode presets
config._validate_config()      # Validate required fields
```

**Configuration Keys Applied:**
- `playlists.ds_pipeline.mode`: "dynamic" (balanced genre+sonic)
- `playlists.genre_mode` / `playlists.sonic_mode`: strict | narrow | dynamic | discover | off
- `playlists.genre_similarity.weight`: 0.50 (from mode preset)
- `playlists.sonic_weight`: 0.50
- `playlists.ds_pipeline.artifact_path`: Path to sonic embeddings
- `playlists.recently_played_filter.lookback_days`: 14
- `playlists.tracks_per_playlist`: 30

#### 1.2 Artifact Loading
```python
# src/features/artifacts.py:157-189
bundle = ArtifactBundle.from_file("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
```

**Loaded Arrays:**
- `X_sonic` (35881 × 137): Sonic feature embeddings from beat3tower analysis
  - Rhythm tower: 21 features (tempo, beat patterns)
  - Timbre tower: 83 features (spectral texture, tone color)
  - Harmony tower: 33 features (key, chord progressions)
- `X_genre_raw` (35881 × 732): Raw genre tag vectors (one-hot encoded)
- `X_genre_smoothed` (35881 × 732): Smoothed genre embeddings (similarity-based)
- `track_ids`, `track_artists`, `track_titles`: Metadata arrays

**Tower Configuration (beat3tower):**
```yaml
tower_weights:
  rhythm: 0.20   # BPM, beat strength, rhythmic complexity
  timbre: 0.50   # Spectral centroid, MFCCs, timbral texture
  harmony: 0.30  # Chroma features, key detection, harmonic content
```

---

### Phase 2: Artist Lookup & Track Retrieval
**Duration:** ~0.1 seconds
**Files:** `src/local_library_client.py`, `src/metadata_client.py`

#### 2.1 Library Query
```python
# local_library_client.py:89-147
def get_all_tracks(self) -> List[Dict]:
    cursor.execute("""
        SELECT t.track_id, t.title, t.artist, t.album,
               t.filepath, t.duration_ms
        FROM tracks t
        WHERE t.filepath IS NOT NULL
    """)
```

**Result:** 35,882 total tracks in library

#### 2.2 Artist Filtering
```python
# playlist_generator.py:1580-1589
def create_playlist_for_artist(self, artist: str, ...):
    all_tracks = self.library.get_all_tracks()
    artist_tracks = [
        t for t in all_tracks
        if self._fuzzy_artist_match(t['artist'], artist)
    ]
```

**Fuzzy Matching Logic:**
```python
# src/string_utils.py:138-173
def normalize_match_string(value: str, is_artist: bool = False):
    # 1. Lowercase
    # 2. Remove feat/ft/with/vs suffixes
    # 3. Remove parenthetical content
    # 4. Remove leading "the "
    # 5. Split on "and", ",", ";" → take first part
```

**Artist Normalization (NEW - with ensemble support):**
```python
# src/playlist/identity_keys.py:14-36
def normalize_primary_artist_key(value: str) -> str:
    # Use extract_primary_artist() which handles:
    # - "Bill Evans Trio" → "bill evans"
    # - "The Red Garland Trio" → "red garland"
    return extract_primary_artist(text, lowercase=True)
```

**Result for "Bill Evans Trio":**
- Matched tracks: 66 tracks
- Includes: "Bill Evans Trio", "Bill Evans", "Monica Zetterlund, Bill Evans"
- All normalized to: "bill evans"

---

### Phase 3: Seed Selection
**Duration:** ~0.2 seconds
**Files:** `src/playlist_generator.py`

#### 3.1 Seed Clustering
```python
# playlist_generator.py:1590-1605
def _select_seeds_for_artist(self, artist_tracks, seed_count=5):
    # K-means clustering on sonic embeddings
    # Select medoid (most central track) from each cluster
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=seed_count, random_state=0)
    labels = kmeans.fit_predict(X_sonic[artist_indices])

    seeds = []
    for cluster_id in range(seed_count):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(X_sonic[cluster_indices] - cluster_center, axis=1)
        medoid_idx = cluster_indices[np.argmin(distances)]
        seeds.append(artist_tracks[medoid_idx])
```

**Seed Selection Strategy:**
- **Goal:** Representative diversity across artist's sonic range
- **Method:** K-means on 32-dim PCA-reduced sonic embeddings
- **Count:** 4-5 seeds (minimum 4 required for pier-bridge)

**Example Seeds (Bill Evans Trio):**
1. "Some Day My Prince Will Come (Remastered Mono 2025 / Take 1)" (ballad, lyrical)
2. "Witchcraft (Remastered Stereo 2025)" (mid-tempo, swing)
3. "Come Rain Or Come Shine (Remastered Mono 2025 / Take 4)" (bebop, energetic)
4. "The Boy Next Door (Remastered Stereo 2025 / Take 1)" (intimate, introspective)

**Why Clustering?**
- Prevents all seeds from being similar ballads
- Ensures playlist spans artist's stylistic range
- Creates natural transition opportunities in pier-bridge

---

### Phase 4: Last.fm History Retrieval
**Duration:** ~0.5-2 seconds (cached) or ~5-10 seconds (fresh fetch)
**Files:** `src/lastfm_client.py`

#### 4.1 Cache Check
```python
# lastfm_client.py:89-124
cache_file = f"data/lastfm_cache_{username}_{days}d.json"
if os.path.exists(cache_file):
    cache_age = (datetime.now() - file_modified_time).total_seconds() / 86400
    if cache_age < 1.0:  # Cache valid for 1 day
        return load_from_cache()
```

#### 4.2 API Fetching (if needed)
```python
# lastfm_client.py:137-201
def get_recent_tracks(self, username: str, days: int = 90):
    # Calculate timestamp for lookback period
    since_ts = int((datetime.now() - timedelta(days=days)).timestamp())

    # Paginated API calls
    while page <= total_pages:
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "user.getRecentTracks",
                "user": username,
                "from": since_ts,
                "page": page,
                "limit": 200,
            }
        )
```

**Result:**
- Total scrobbles retrieved: 4,241 tracks
- Lookback period: 90 days
- Used for: Recency filtering (exclude recently played tracks)

#### 4.3 Recency Key Extraction
```python
# playlist_generator.py:1630-1650
def _extract_recency_keys(self, lastfm_tracks):
    keys = set()
    for track in lastfm_tracks:
        artist_key = extract_primary_artist(track['artist'], lowercase=True)
        title_key = normalize_title_for_dedupe(track['title'], mode='loose')
        keys.add((artist_key, title_key))
    return keys  # 938 unique (artist, title) pairs
```

---

### Phase 5: Artist Style Clustering
**Duration:** ~1-2 seconds
**Files:** `src/playlist/artist_style_clustering.py`

#### 5.1 Style Analysis
```python
# artist_style_clustering.py:156-248
def cluster_artist_styles(
    artist_tracks,
    bundle,
    k_min=3,
    k_max=6,
    target_k=5,
):
    # Step 1: Extract sonic embeddings for artist's tracks
    artist_indices = [bundle.track_ids.index(t['track_id']) for t in artist_tracks]
    X_artist = X_sonic[artist_indices]  # (66, 137)

    # Step 2: PCA reduction to 32 dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32)
    X_reduced = pca.fit_transform(X_artist)  # (66, 32)

    # Step 3: K-medoids clustering
    from sklearn_extra.cluster import KMedoids
    kmedoids = KMedoids(n_clusters=5, random_state=0)
    labels = kmedoids.fit_predict(X_reduced)

    # Step 4: Quality metrics
    intra_cluster_sim = compute_intra_cluster_similarity(X_reduced, labels)
    inter_cluster_sim = compute_inter_cluster_similarity(X_reduced, labels)
```

**Clustering Output (Bill Evans Trio):**
- **k = 5 clusters** (auto-selected based on silhouette score)
- **Cluster characteristics:**
  - Cluster 0: Ballads, slow tempo (17 tracks)
  - Cluster 1: Mid-tempo swing (14 tracks)
  - Cluster 2: Bebop, up-tempo (12 tracks)
  - Cluster 3: Modal jazz, atmospheric (11 tracks)
  - Cluster 4: Blues-tinged, bluesy (12 tracks)
- **Intra-cluster similarity:** 0.509 (median within-cluster cosine sim)
- **Inter-cluster similarity:** 0.359 (median between-cluster cosine sim)
- **Medoids:** 5 representative tracks (used as pier seeds)

#### 5.2 Per-Cluster Candidate Pools
```python
# artist_style_clustering.py:280-340
for cluster_id in range(n_clusters):
    # Find external tracks similar to this cluster's medoid
    medoid_idx = cluster_medoids[cluster_id]
    similarities = cosine_similarity(
        X_hybrid[medoid_idx].reshape(1, -1),
        X_hybrid[allowed_indices]
    )[0]

    # Top 400 candidates per cluster
    top_indices = np.argsort(similarities)[::-1][:400]
    cluster_pools[cluster_id] = top_indices
```

**Per-Cluster Pool Sizes:**
- Cluster 0: 400 candidates
- Cluster 1: 400 candidates
- Cluster 2: 400 candidates
- Cluster 3: 400 candidates
- Cluster 4: 400 candidates
- **Total unique candidates:** 1,620 tracks (some overlap between clusters)

**Internal Connectors:**
- Tracks from seed artist allowed ONLY at pier positions
- 61 Bill Evans tracks marked as "internal connectors"
- Used exclusively as pier anchors (not in bridge segments)

---

### Phase 6: Candidate Pool Generation
**Duration:** ~2-3 seconds
**Files:** `src/playlist/pipeline.py`, `src/playlist/candidate_generator.py`

#### 6.1 Hybrid Embedding Construction
```python
# pipeline.py:289-335
def build_hybrid_embedding(X_sonic, X_genre, sonic_weight=0.6, genre_weight=0.5):
    # Normalize weights to sum to 1.0
    total = sonic_weight + genre_weight  # 1.1
    w_sonic = sonic_weight / total       # 0.545
    w_genre = genre_weight / total       # 0.455

    # PCA reduction
    pca_sonic = PCA(n_components=32)
    X_sonic_reduced = pca_sonic.fit_transform(X_sonic)  # (35881, 32)

    pca_genre = PCA(n_components=32)
    X_genre_reduced = pca_genre.fit_transform(X_genre_smoothed)  # (35881, 32)

    # Weighted concatenation
    X_hybrid = np.hstack([
        X_sonic_reduced * w_sonic,
        X_genre_reduced * w_genre
    ])  # (35881, 64)

    return X_hybrid
```

**Hybrid Embedding Composition:**
- **Sonic component:** 32 PCA dims from 137 raw sonic features
  - Weight: 0.545 (normalized from 0.600)
  - Captures: rhythm, timbre, harmony similarity
- **Genre component:** 32 PCA dims from 732 genre tags
  - Weight: 0.455 (normalized from 0.500)
  - Captures: genre taxonomy, semantic similarity
- **Combined:** 64-dimensional hybrid space

#### 6.2 Candidate Filtering Pipeline
```python
# candidate_generator.py:410-520
def generate_candidate_pool(
    bundle,
    seed_track_ids,
    allowed_track_ids,  # From artist style clustering
    recency_keys,
    config,
):
    # Step 1: Restrict to allowed tracks (artist style pools)
    allowed_indices = [i for i, tid in enumerate(bundle.track_ids) if tid in allowed_track_ids]
    # Before: 35,881 tracks → After: 1,564 tracks

    # Step 2: Compute sonic similarity to seeds
    seed_indices = [bundle.track_ids.index(sid) for sid in seed_track_ids]
    X_seeds = X_sonic[seed_indices]  # (5, 137)

    # Max similarity across all seeds (multi-seed admission)
    similarities = np.max([
        cosine_similarity(X_sonic[allowed_indices], seed.reshape(1, -1))
        for seed in X_seeds
    ], axis=0)

    # Step 3: Sonic floor filter (mode-specific)
    sonic_floor = config.get('min_sonic_similarity_dynamic', 0.00)
    sonic_pass = similarities >= sonic_floor
    # Before: 1,559 → After: 901 (rejected: 658)

    # Step 3.5: Duration penalty (seed-relative)
    # Penalize candidates longer than the median seed duration before gating
    # Hard exclude candidates longer than duration_cutoff_multiplier * median seed duration

    # Step 4: Genre hard gate
    for idx in allowed_indices:
        genre_sim = max([
            compute_genre_similarity(bundle, idx, seed_idx)
            for seed_idx in seed_indices
        ])
        if genre_sim < config.get('min_genre_similarity', 0.30):
            genre_reject.add(idx)
    # Rejected: 162 tracks (below genre threshold)

    # Step 5: Artist diversity cap
    artist_counts = Counter()
    for idx in candidate_indices:
        artist_key = identity_keys_for_index(bundle, idx).artist_key
        artist_counts[artist_key] += 1

    max_per_artist = int(30 * 0.125)  # 3 tracks max per artist
    # Rejected: 323 tracks (artist cap exceeded)

    # Step 6: Recency filtering
    for idx in candidate_indices:
        keys = identity_keys_for_index(bundle, idx)
        if keys.track_key in recency_keys:
            recency_reject.add(idx)
    # Rejected: 0 tracks (none recently played from candidate pool)

    # FINAL POOL: 416 candidates from 131 unique artists
```

**Candidate Pool Statistics:**
- **Input:** 1,564 tracks (from artist style clustering)
- **Sonic filter:** 901 passed (658 rejected, floor=0.00)
- **Genre gate:** 739 passed (162 rejected, min_sim=0.30)
- **Artist cap:** 416 eligible (323 rejected, max 3 per artist)
- **Recency filter:** 416 final (0 rejected)
- **Distinct artists:** 131
- **Avg tracks per artist:** 3.2

---

### Phase 7: Pier-Bridge Construction
**Duration:** ~3-5 seconds
**Files:** `src/playlist/pier_bridge_builder.py`

This is the most complex and critical phase.

#### 7.1 Seed Ordering Optimization
```python
# pier_bridge_builder.py:1715-1761
def optimize_seed_order(seed_indices, X_transition):
    """Find optimal ordering of pier seeds to maximize flow."""
    best_score = -np.inf
    best_order = None

    # Try all permutations (5! = 120 permutations)
    for perm in itertools.permutations(seed_indices):
        score = 0.0
        for i in range(len(perm) - 1):
            # Transition quality between consecutive piers
            score += X_transition[perm[i], perm[i+1]]

        if score > best_score:
            best_score = score
            best_order = perm

    return best_order, best_score
```

**Optimal Seed Order (Bill Evans Trio):**
```
Original: [seed_0, seed_1, seed_2, seed_3, seed_4]
Optimized: [seed_4, seed_3, seed_2, seed_1, seed_0]
Score: 0.8890 (high inter-pier transition quality)

Ordered seeds:
1. Blue In Green (Remastered Stereo 2025)
2. Elsa (Remastered Stereo 2025 / Take 6)
3. Spring Is Here (Remastered Mono 2025 / Take 4)
4. All Of You (Take 2 / Live At The Village Vanguard / 1961)
5. Sweet And Lovely (Remastered Stereo 2025)
```

#### 7.2 Segment Length Distribution
```python
# pier_bridge_builder.py:1763-1789
def compute_segment_lengths(n_piers, target_length):
    """Distribute interior tracks across bridge segments."""
    n_segments = n_piers - 1  # 4 segments
    interior_tracks = target_length - n_piers  # 30 - 5 = 25

    # Base length per segment
    base_len = interior_tracks // n_segments  # 25 // 4 = 6
    remainder = interior_tracks % n_segments  # 25 % 4 = 1

    # Distribute remainder to first segments
    segment_lengths = [base_len + 1 if i < remainder else base_len
                      for i in range(n_segments)]

    return segment_lengths  # [7, 6, 6, 6]
```

**Segment Structure:**
```
Segment 0: Pier 1 → [7 bridge tracks] → Pier 2  (positions 1-9)
Segment 1: Pier 2 → [6 bridge tracks] → Pier 3  (positions 10-16)
Segment 2: Pier 3 → [6 bridge tracks] → Pier 4  (positions 17-23)
Segment 3: Pier 4 → [6 bridge tracks] → Pier 5  (positions 24-30)
```

#### 7.3 Per-Segment Bridge Construction
```python
# pier_bridge_builder.py:1791-1920
def build_bridge_segment(
    pier_a_idx,
    pier_b_idx,
    interior_length,
    candidate_pool,
    X_sonic,
    X_transition,
    config,
):
    """
    Build one bridge segment using beam search with progress constraint.

    Key insight: We're navigating from Pier A to Pier B in sonic space,
    progressively moving toward the destination while maintaining smooth transitions.
    """

    # STEP 1: Segment-scored candidate pool
    # Score each candidate by its bridge quality to BOTH piers
    pool_scores = []
    for cand_idx in candidate_pool:
        # Harmonic mean of similarity to both endpoints
        sim_a = X_sonic[cand_idx] @ X_sonic[pier_a_idx]
        sim_b = X_sonic[cand_idx] @ X_sonic[pier_b_idx]
        bridge_score = 2 * sim_a * sim_b / (sim_a + sim_b + 1e-9)
        pool_scores.append((cand_idx, bridge_score))

    # Sort by bridge score, take top 400
    pool_scores.sort(key=lambda x: x[1], reverse=True)
    segment_pool = [idx for idx, _ in pool_scores[:400]]
    # Segment 0 pool: 49 candidates (after bridge floor gate)

    # STEP 2: Bridge floor gating
    bridge_floor = config.get('bridge_floor_dynamic', 0.03)
    for cand_idx in segment_pool:
        sim_a = X_sonic[cand_idx] @ X_sonic[pier_a_idx]
        sim_b = X_sonic[cand_idx] @ X_sonic[pier_b_idx]
        if min(sim_a, sim_b) < bridge_floor:
            segment_pool.remove(cand_idx)
    # After gate: 49 candidates

    # STEP 3: Beam search with progress constraint
    beam_width = 20
    beam = [(pier_a_idx, [pier_a_idx], 0.0)]  # (current_idx, path, score)

    for step in range(interior_length):
        next_beam = []

        for current_idx, path, score in beam:
            # For each candidate in pool
            for cand_idx in segment_pool:
                # Skip if already used
                if cand_idx in path:
                    continue

                # Skip if violates artist constraints
                if violates_artist_gap(cand_idx, path, min_gap=6):
                    continue
                if violates_seed_artist_policy(cand_idx, seed_artist_key):
                    continue

                # Compute edge score
                transition_score = X_transition[current_idx, cand_idx]
                bridge_score = compute_bridge_score(cand_idx, pier_a_idx, pier_b_idx)
                edge_score = (
                    config.weight_transition * transition_score +
                    config.weight_bridge * bridge_score
                )

                # Progress constraint penalty
                progress_penalty = compute_progress_penalty(
                    cand_idx, pier_a_idx, pier_b_idx,
                    step, interior_length,
                    X_sonic,
                    epsilon=0.05,
                    weight=0.15
                )

                # Genre soft penalty (whiplash reduction)
                genre_penalty = 0.0
                genre_sim = compute_genre_similarity(cand_idx, current_idx)
                if genre_sim < config.genre_penalty_threshold:
                    genre_penalty = config.genre_penalty_strength

                # Final score
                final_score = score + edge_score - progress_penalty - genre_penalty

                next_beam.append((cand_idx, path + [cand_idx], final_score))

        # Keep top beam_width candidates
        next_beam.sort(key=lambda x: x[2], reverse=True)
        beam = next_beam[:beam_width]

        # Expand beam if needed
        if len(beam) < beam_width * 0.5:
            beam_width = min(beam_width * 2, 100)

    # STEP 4: Connect to Pier B
    best_path = None
    best_final_score = -np.inf

    for current_idx, path, score in beam:
        # Add Pier B as final track
        transition_score = X_transition[current_idx, pier_b_idx]
        final_score = score + transition_score

        if final_score > best_final_score:
            best_final_score = final_score
            best_path = path + [pier_b_idx]

    return best_path  # [pier_a, track1, track2, ..., track7, pier_b]
```

**Beam Search Visualization (Segment 0):**
```
Step 0: Start at Pier A (Blue In Green)
  Beam: [Blue In Green]

Step 1: Expand to 20 candidates
  Beam: [Blue In Green → Milt Jackson "Heartstrings"]  (score: 0.85)
        [Blue In Green → Freddie Hubbard "Lament"]     (score: 0.82)
        [Blue In Green → Wayne Shorter "Infant Eyes"]  (score: 0.80)
        ... (17 more)

Step 2: Expand each path
  Beam: [Blue In Green → Heartstrings → Lament For Booker]  (score: 1.67)
        [Blue In Green → Heartstrings → Infant Eyes]         (score: 1.64)
        ... (18 more)

... (steps 3-7)

Step 7: Connect to Pier B (Elsa)
  Best path: [Blue In Green → Heartstrings → Lament For Booker →
              Infant Eyes → Love Is Blindness → Stella By Starlight →
              The Christmas Song → Very Early → Elsa]
  Final score: 6.34
```

**Artist Diversity Enforcement:**
```python
# pier_bridge_builder.py:1157-1195
# Seed artist disallowed in bridge interiors
if disallow_seed_artist_in_interiors and seed_artist_key:
    seed_identity_keys = resolve_artist_identity_keys(seed_artist_key)
    # "bill evans" blocked from positions 2-8, 10-15, 17-22, 24-29

# Min gap constraint (6 positions)
used_artists = {}  # artist_key → last_position
for step, cand_idx in enumerate(path):
    artist_key = identity_keys_for_index(bundle, cand_idx).artist_key
    if artist_key in used_artists:
        gap = step - used_artists[artist_key]
        if gap < 6:
            reject_candidate()  # Too soon!
    used_artists[artist_key] = step
```

**Constraint Summary:**
- **Bridge floor:** 0.03 (min similarity to both piers)
- **Transition floor:** 0.20 (min local transition quality)
- **Min artist gap:** 6 positions
- **Seed artist policy:** Disallowed in bridge interiors
- **Progress constraint:** Monotonic movement toward destination
- **Duration penalty:** Applied in candidate pool (geometric curve vs max seed duration)
- **Genre soft penalty:** Reduce whiplash below 0.20 similarity

#### 7.4 Full Playlist Assembly
```python
# pier_bridge_builder.py:1920-1975
segments = []
for seg_id in range(n_segments):
    pier_a = ordered_seeds[seg_id]
    pier_b = ordered_seeds[seg_id + 1]
    interior_len = segment_lengths[seg_id]

    segment_path = build_bridge_segment(
        pier_a, pier_b, interior_len,
        candidate_pool, X_sonic, X_transition, config
    )

    segments.append(segment_path)

# Concatenate segments (remove duplicate piers at boundaries)
playlist = [segments[0]]  # Include first segment fully
for segment in segments[1:]:
    playlist.extend(segment[1:])  # Skip first pier (already added)

return playlist  # 30 tracks
```

**Final Playlist Structure:**
```
Position 01: [PIER 1] Bill Evans Trio - Blue In Green
Position 02: [BRIDGE] Milt Jackson - Heartstrings
Position 03: [BRIDGE] Freddie Hubbard - Lament For Booker
Position 04: [BRIDGE] Wayne Shorter - Infant Eyes
Position 05: [BRIDGE] Cassandra Wilson - Love Is Blindness
Position 06: [BRIDGE] Ryo Fukui - Stella By Starlight
Position 07: [BRIDGE] Vince Guaraldi Trio - The Christmas Song
Position 08: [BRIDGE] Bill Evans - Very Early  ← BLOCKED after fix!
Position 09: [PIER 2] Bill Evans Trio - Elsa
Position 10: [BRIDGE] Red Garland Trio - Summertime
Position 11: [BRIDGE] Red Garland - We Kiss in a Shadow  ← BLOCKED after fix!
...
Position 16: [PIER 3] Bill Evans Trio - Spring Is Here
...
Position 23: [PIER 4] Bill Evans Trio - All Of You
...
Position 30: [PIER 5] Bill Evans Trio - Sweet And Lovely
```

---

### Phase 8: Post-Processing & Validation
**Duration:** ~0.5 seconds
**Files:** `src/playlist/pipeline.py`, `src/playlist_generator.py`

#### 8.1 Recency Overlap Check
```python
# pipeline.py:580-610
def validate_recency_exclusions(playlist, recency_keys):
    """Ensure no recently played tracks slipped through."""
    overlap = 0
    for track in playlist:
        keys = identity_keys_for_index(bundle, track['bundle_idx'])
        if keys.track_key in recency_keys:
            overlap += 1
            logger.error(f"Recency violation: {track['artist']} - {track['title']}")

    assert overlap == 0, f"Recency filter leaked {overlap} tracks"
    # Expected: 0 overlaps (strict gate)
```

#### 8.2 Quality Metrics Computation
```python
# pier_bridge_builder.py:1980-2050
def compute_playlist_metrics(playlist_indices, X_transition, X_sonic):
    transitions = []
    for i in range(len(playlist_indices) - 1):
        score = X_transition[playlist_indices[i], playlist_indices[i+1]]
        transitions.append(score)

    return {
        "min_transition": min(transitions),      # 0.572
        "mean_transition": np.mean(transitions), # 0.754
        "p10_transition": np.percentile(transitions, 10),  # 0.679
        "p90_transition": np.percentile(transitions, 90),  # 0.990
        "below_floor": sum(t < 0.20 for t in transitions), # 0
    }
```

**Quality Report (Bill Evans Trio):**
```
Transition Quality (T):
  Min:  0.572  ✓ (above floor 0.20)
  Mean: 0.754  ✓ (strong overall flow)
  P10:  0.679  ✓ (no weak transitions)
  P90:  0.990  ✓ (many perfect transitions)

Sonic Similarity (S):
  Min:  -0.032 (one slightly negative edge)
  Mean:  0.540 (balanced diversity)
  P90:   0.716 (cohesive high end)

Genre Similarity (G):
  Min:   0.492 (solid minimum)
  Mean:  0.828 (strong genre coherence)
  P90:   0.958 (very tight at high end)

Artist Diversity:
  Unique artists: 19 / 30 tracks (63%)
  Max per artist: 5 tracks (Bill Evans Trio at piers)
  Avg gap: 6.2 positions
```

#### 8.3 Weakest Edge Reporting
```python
# pier_bridge_builder.py:2051-2090
weakest_edges = sorted(enumerate(transitions), key=lambda x: x[1])[:3]
for idx, score in weakest_edges:
    track_a = playlist[idx]
    track_b = playlist[idx + 1]
    logger.info(f"Weak edge #{idx+1}: T={score:.3f} S={sonic_sim:.3f} G={genre_sim:.3f}")
    logger.info(f"  {track_a['artist']} - {track_a['title']}")
    logger.info(f"  → {track_b['artist']} - {track_b['title']}")
```

**Weakest Transitions (Bill Evans Trio):**
```
#22 T=0.241  S=0.553  G=0.610
  Sonny Rollins - Softly, as in a Morning Sunrise
  → Bill Evans Trio - All Of You (Take 2 / Live At The Village Vanguard / 1961)
  [Jump back to seed artist at pier]

#14 T=0.286  S=0.406  G=0.712
  Cassandra Wilson - Skylark
  → Ryo Fukui - I Want To Talk About You
  [Genre shift: vocal jazz → instrumental]

#03 T=0.597  S=0.715  G=0.955
  Freddie Hubbard - Lament For Booker
  → Wayne Shorter - Infant Eyes
  [Strong overall, just lowest of high-quality edges]
```

---

### Phase 9: Export & Delivery
**Duration:** ~1-2 seconds
**Files:** `src/m3u_exporter.py`, `src/plex_exporter.py`

#### 9.1 M3U8 Export
```python
# m3u_exporter.py:45-98
def export_playlist(self, playlist_data, name):
    """Generate M3U8 file with EXTINF metadata."""
    filepath = os.path.join(self.output_dir, f"{name}.m3u8")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")

        for track in playlist_data['tracks']:
            duration_sec = int(track.get('duration_ms', 0) / 1000)
            f.write(f"#EXTINF:{duration_sec},{track['artist']} - {track['title']}\n")
            f.write(f"{track['filepath']}\n")

    return filepath
```

**M3U8 Output (E:\PLAYLISTS\Auto - Bill Evans Trio 2026-01-03.m3u8):**
```m3u8
#EXTM3U
#EXTINF:316,Bill Evans Trio - Blue In Green (Remastered Stereo 2025)
E:\Music\Bill Evans Trio\Portrait In Jazz (Remastered)\05 Blue In Green.flac
#EXTINF:284,Milt Jackson - Heartstrings
E:\Music\Milt Jackson\Bags & Trane\03 Heartstrings.flac
#EXTINF:442,Freddie Hubbard - Lament For Booker
E:\Music\Freddie Hubbard\Breaking Point!\05 Lament For Booker.flac
... (27 more tracks)
```

#### 9.2 Plex Export
```python
# plex_exporter.py:156-245
def create_or_update_playlist(self, name, track_filepaths):
    """Create playlist in Plex Media Server."""
    # Step 1: Build path index (map filepaths → Plex track objects)
    path_index = {}
    for track in plex_music.all():
        for location in track.locations:
            normalized_path = self._normalize_path(location)
            path_index[normalized_path] = track

    # Step 2: Match playlist tracks to Plex objects
    plex_tracks = []
    for filepath in track_filepaths:
        normalized = self._normalize_path(filepath)
        if normalized in path_index:
            plex_tracks.append(path_index[normalized])
        else:
            logger.warning(f"Track not found in Plex: {filepath}")

    # Step 3: Create or update playlist
    existing = plex.playlist(name) if name in plex.playlists() else None
    if existing:
        if config.get('plex.replace_existing', True):
            existing.delete()
            playlist = plex.createPlaylist(name, items=plex_tracks)
        else:
            existing.addItems(plex_tracks)
    else:
        playlist = plex.createPlaylist(name, items=plex_tracks)

    return playlist
```

**Plex Result:**
- Playlist name: "Auto - Bill Evans Trio 2026-01-03"
- Matched tracks: 30/30 (100%)
- Smart collection: No (static playlist)
- Visibility: Public

---

## Key Algorithms & Data Structures

### 1. Tower PCA (Sonic Embedding)
**Purpose:** Reduce high-dimensional sonic features while preserving structure

```python
# src/similarity/sonic_variant.py:234-298
def tower_pca_embedding(X_sonic, tower_dims, tower_pca_dims, tower_weights):
    """
    Multi-tower dimensionality reduction.

    Each tower (rhythm/timbre/harmony) is reduced independently,
    then weighted and concatenated.
    """
    rhythm_features = X_sonic[:, :21]     # BPM, beat strength, onset patterns
    timbre_features = X_sonic[:, 21:104]  # MFCCs, spectral features, texture
    harmony_features = X_sonic[:, 104:]   # Chroma, key, chord progressions

    # PCA per tower
    pca_rhythm = PCA(n_components=8)
    rhythm_reduced = pca_rhythm.fit_transform(rhythm_features)  # (N, 8)

    pca_timbre = PCA(n_components=16)
    timbre_reduced = pca_timbre.fit_transform(timbre_features)  # (N, 16)

    pca_harmony = PCA(n_components=8)
    harmony_reduced = pca_harmony.fit_transform(harmony_features)  # (N, 8)

    # Weighted concatenation
    X_tower = np.hstack([
        rhythm_reduced * tower_weights[0],   # 0.20
        timbre_reduced * tower_weights[1],   # 0.50
        harmony_reduced * tower_weights[2],  # 0.30
    ])  # (N, 32)

    return X_tower
```

**Why Towers?**
- Rhythm: Captures tempo/energy changes (important for flow)
- Timbre: Dominant tower for sonic texture (50% weight)
- Harmony: Captures tonal/modal similarity (jazz harmonic language)

### 2. Genre Smoothing
**Purpose:** Propagate genre similarity through the semantic graph

```python
# src/genre_similarity_v2.py:156-234
def smooth_genre_embeddings(X_genre_raw, genre_similarity_matrix, alpha=0.3):
    """
    Apply label propagation to genre vectors.

    Similar genres reinforce each other, creating a smooth genre space
    that captures semantic relationships beyond exact tag matching.
    """
    # X_genre_raw: (N, 732) one-hot encoded genre tags
    # genre_similarity_matrix: (732, 732) pairwise genre similarity

    # Normalize similarity matrix to stochastic matrix
    row_sums = genre_similarity_matrix.sum(axis=1, keepdims=True)
    P = genre_similarity_matrix / (row_sums + 1e-9)

    # Label propagation: X_smooth = (1-α)X + αXP
    X_propagated = X_genre_raw @ P  # (N, 732)
    X_smoothed = (1 - alpha) * X_genre_raw + alpha * X_propagated

    return X_smoothed
```

**Effect:**
- Track tagged "cool jazz" gets boost for "post-bop", "hard bop"
- Semantic relationships: bebop ↔ hard bop (0.85), jazz ↔ blues (0.72)
- Reduces false negatives from incomplete tagging

### 3. Ensemble Normalization
**Purpose:** Treat artist name variants as the same entity

```python
# src/artist_utils.py:13-54
def extract_primary_artist(artist: str, lowercase: bool = True) -> str:
    """
    Normalize artist identity with ensemble suffix handling.

    "Bill Evans Trio" → "bill evans"
    "The Red Garland Trio" → "red garland"
    "Vince Guaraldi Trio" → "vince guaraldi"
    """
    # Remove feat/with/vs
    base_artist = re.split(r"\s+(?:feat\.|ft\.|featuring|with|vs\.)\s+", artist)[0]

    # Check for ensemble suffix
    has_ensemble = re.search(r"\s+(Trio|Quartet|Quintet|Sextet|Septet|Octet)$",
                              base_artist, flags=re.IGNORECASE)

    if has_ensemble:
        # Remove leading "The"
        base_artist = re.sub(r"^The\s+", "", base_artist, flags=re.IGNORECASE)
        # Remove ensemble suffix
        base_artist = re.sub(r"\s+(Trio|Quartet|Quintet|Sextet|Septet|Octet)$",
                             "", base_artist, flags=re.IGNORECASE)

    return base_artist.lower() if lowercase else base_artist
```

**Critical for:**
- Seed artist exclusion policy (block all "Bill Evans" variants)
- Artist diversity constraints (gap enforcement across variants)
- Dedupe detection (prevent duplicate artists)

### 4. Progress Constraint
**Purpose:** Ensure bridge segments move monotonically toward destination

```python
# pier_bridge_builder.py:458-505
def compute_progress_penalty(
    candidate_idx,
    pier_a_idx,
    pier_b_idx,
    step,
    total_steps,
    X_sonic,
    epsilon=0.05,
    weight=0.15,
):
    """
    Penalize candidates that violate monotonic progress in sonic space.

    We project the candidate onto the A→B direction and ensure it's
    moving forward relative to the expected position.
    """
    # Direction vector from A to B in sonic space
    direction = X_sonic[pier_b_idx] - X_sonic[pier_a_idx]
    direction_norm = direction / (np.linalg.norm(direction) + 1e-9)

    # Project candidate onto A→B line
    candidate_vec = X_sonic[candidate_idx] - X_sonic[pier_a_idx]
    projection = np.dot(candidate_vec, direction_norm)

    # Expected progress at this step
    expected_progress = step / total_steps
    actual_progress = projection / np.linalg.norm(direction)

    # Penalty if moving backward
    if actual_progress < (expected_progress - epsilon):
        penalty = weight * (expected_progress - actual_progress)
    else:
        penalty = 0.0

    return penalty
```

**Effect:**
- Prevents "teleporting" (jumping back to Pier A area)
- Reduces ping-ponging between sonic regions
- Creates smooth sonic arc from A → B

### 5. Duration Penalty (Geometric Curve)
**Purpose:** Discourage overly long tracks in candidate pool selection
**Reference:** Median seed track duration (penalty applied before similarity floors)
**Hard cutoff:** Candidates > duration_cutoff_multiplier * median seed duration are excluded

```python
# candidate_pool.py:91-140
def compute_duration_penalty(candidate_duration_ms, reference_duration_ms, weight=0.60):
    """
    Four-phase geometric penalty based on percentage excess.

    0-20%:   Gentle (barely noticeable)
    20-50%:  Moderate (increasing)
    50-100%: Steep (strong discouragement)
    >100%:   Severe (track is 2x+ longer)
    """
    excess_ratio = (candidate_duration_ms - reference_duration_ms) / reference_duration_ms

    if excess_ratio <= 0:
        return 0.0  # Shorter tracks have no penalty

    if excess_ratio <= 0.20:
        # Phase 1: Gentle (power 1.5)
        penalty = weight * 0.05 * (excess_ratio / 0.20) ** 1.5
    elif excess_ratio <= 0.50:
        # Phase 2: Moderate (power 2.0)
        phase_ratio = (excess_ratio - 0.20) / 0.30
        penalty = weight * 0.05 + weight * 0.25 * (phase_ratio ** 2.0)
    elif excess_ratio <= 1.00:
        # Phase 3: Steep (power 2.5)
        phase_ratio = (excess_ratio - 0.50) / 0.50
        penalty = weight * 0.30 + weight * 0.45 * (phase_ratio ** 2.5)
    else:
        # Phase 4: Severe (power 3.0)
        phase_ratio = excess_ratio - 1.00
        penalty = weight * 0.75 + weight * 2.25 * (phase_ratio ** 3.0)

    return penalty
```

**Example Penalties (200s reference track):**
- 220s (+10%): penalty = 0.026 (negligible)
- 260s (+30%): penalty = 0.190 (moderate)
- 340s (+70%): penalty = 0.574 (steep)
- 440s (+120%): penalty = 1.524 (severe)
- 520s (+160%): excluded (hard cutoff at 2.5x)

---

## Performance Characteristics

### Computational Complexity

| Phase | Time | Complexity | Bottleneck |
|-------|------|------------|------------|
| Artifact Loading | 1-2s | O(N) | Disk I/O, numpy array loading |
| Artist Clustering | 1-2s | O(k·n·d²) | K-medoids on artist tracks |
| Candidate Pool | 2-3s | O(N·M·d) | Cosine similarity (N=35k, M=5 seeds) |
| Seed Ordering | 0.1s | O(k!) | Permutation search (5! = 120) |
| Pier-Bridge | 3-5s | O(s·b·p²) | Beam search per segment |
| **Total** | **7-13s** | | End-to-end generation |

**Variables:**
- N = Total tracks in library (35,881)
- n = Artist tracks (66 for Bill Evans)
- M = Number of seeds (5)
- k = Number of clusters (5)
- d = Embedding dimension (32-137)
- s = Segments (4)
- b = Beam width (20-100)
- p = Pool size per segment (50-400)

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Artifact Bundle | ~500 MB | X_sonic (35881×137), X_genre (35881×732) |
| Hybrid Embeddings | ~20 MB | Reduced to 64-dim (35881×64) |
| Distance Matrices | ~10 GB | Full pairwise (avoided via sparse computation) |
| Candidate Pool | ~50 MB | 416 tracks × metadata |
| Working Memory | ~1 GB | Peak during beam search |

**Optimization:**
- Sparse similarity computation (only compute N×M, not N×N)
- On-demand matrix access (no full distance matrix)
- Candidate pool pruning (416 instead of 35,881)

---

## Configuration Impact

### Mode Presets (Genre × Sonic)

| Mode | Genre Weight | Sonic Weight | Min Genre Sim | Pool Multiplier | Character |
|------|-------------|--------------|---------------|-----------------|-----------|
| **Strict + Strict** | 0.80 | 0.85 | 0.50 | 0.6 | Ultra-cohesive, tight genre match |
| **Narrow + Narrow** | 0.65 | 0.70 | 0.40 | 0.8 | Cohesive, same genre family |
| **Dynamic + Dynamic** | 0.50 | 0.50 | 0.30 | 1.0 | Balanced exploration ← DEFAULT |
| **Discover + Discover** | 0.35 | 0.35 | 0.20 | 1.2 | Maximum exploration |
| **Off + Dynamic** | 0.00 | 1.00 | N/A | 1.0 | Sonic-only (ignore genre tags) |
| **Dynamic + Off** | 1.00 | 0.00 | 0.30 | N/A | Genre-only (ignore audio features) |

**Effect on Candidate Pool:**
- Strict: ~200-300 candidates (very selective)
- Narrow: ~300-500 candidates
- **Dynamic: ~400-600 candidates** ← Bill Evans example
- Discover: ~800-1200 candidates (wide net)

### Artist Style Clustering (k parameter)

| k | Effect | Use Case |
|---|--------|----------|
| 3 | Broad styles | Artists with narrow range |
| 4-5 | Balanced | Most artists (DEFAULT) |
| 6-8 | Fine-grained | Artists with wide stylistic range |

**Auto-selection:** Uses silhouette score to pick optimal k in range [k_min, k_max]

---

## Common Edge Cases

### Edge Case 1: Artist with <4 Tracks
**Problem:** Pier-bridge requires minimum 4 seeds
**Solution:** Error message, suggest History mode
**Example:** "Haruomi Hosono has 2 tracks, need at least 4"

### Edge Case 2: All Candidates Recently Played
**Problem:** Recency filter removes entire candidate pool
**Solution:** Widen lookback window or disable recency filter
**Fallback:** Use artist's own tracks as connectors

### Edge Case 3: Infeasible Bridge Segment
**Problem:** No path exists that satisfies all constraints
**Solution:**
1. Reduce bridge_floor in steps (0.08 → 0.06 → 0.04 → 0.00)
2. Widen candidate pool (400 → 600 → 800)
3. Increase beam width (20 → 50 → 100)
4. As last resort, relax min_gap constraint

**Audit Report:** Generated to `docs/run_audits/` when enabled

### Edge Case 4: Ensemble Name Confusion
**Problem:** "Bill Evans Trio" vs "Bill Evans" treated as different artists
**Solution:** Ensemble normalization in `identity_keys.py`
**Fixed:** 2026-01-03 (commit 5fa9549)

---

## Debug & Introspection

### Enable Audit Reports
```yaml
# config.yaml
playlists:
  ds_pipeline:
    pier_bridge:
      audit_run:
        enabled: true
        out_dir: "docs/run_audits"
        include_top_k: 25
        max_bytes: 350000
```

**Output:** `docs/run_audits/pier_bridge_audit_<timestamp>.md`

**Contains:**
- Segment-by-segment construction log
- Top 25 candidates per segment with scores
- Bridge floor gates, penalties applied
- Edge quality breakdown
- Constraint violation tracking

### Verbose Logging
```bash
python main_app.py --artist "Bill Evans Trio" --verbose
```

**Enables:**
- DEBUG-level logs from `src.playlist_generator`
- Per-edge transition scores
- Candidate pool statistics
- Beam search state snapshots

### Quality Metrics
```python
# Logged automatically after generation
{
    "min_transition": 0.572,      # Weakest edge (should be >0.20)
    "mean_transition": 0.754,     # Average flow quality
    "below_floor": 0,             # Count of edges <0.20 (should be 0)
    "distinct_artists": 19,       # Unique artists (target: 15-25)
    "max_artist_count": 5,        # Most tracks per artist
}
```

---

## Future Optimizations

### Potential Improvements

1. **Cached Similarity Matrices**
   - Pre-compute X_hybrid for frequent artist/genre queries
   - Store in SQLite for O(1) lookup
   - Estimated speedup: 40-50%

2. **GPU Acceleration**
   - Move cosine similarity to GPU (CuPy/PyTorch)
   - Batch beam search expansions
   - Estimated speedup: 3-5x on CUDA hardware

3. **Incremental Candidate Pool**
   - Cache per-artist candidate pools
   - Only recompute when artifacts change
   - Estimated speedup: 60-70% on repeat generations

4. **Parallel Segment Construction**
   - Build bridge segments in parallel threads
   - Merge results with global conflict resolution
   - Estimated speedup: 2-3x on multi-core CPUs

5. **Smart Beam Pruning**
   - Use A* heuristic (distance to destination)
   - Prune beam more aggressively early in search
   - Estimated speedup: 20-30%

---

## Appendix: File Reference

### Core Pipeline Files
- `main_app.py`: Entry point, orchestration
- `src/playlist_generator.py`: High-level playlist logic
- `src/playlist/pipeline.py`: DS pipeline orchestrator
- `src/playlist/pier_bridge_builder.py`: Bridge construction (1,900+ lines)
- `src/playlist/candidate_generator.py`: Candidate pool filtering
- `src/playlist/artist_style_clustering.py`: Style analysis

### Supporting Modules
- `src/features/artifacts.py`: Artifact loading
- `src/playlist/identity_keys.py`: Artist/track normalization
- `src/similarity/sonic_variant.py`: Tower PCA embeddings
- `src/genre_similarity_v2.py`: Genre smoothing, taxonomy
- `src/lastfm_client.py`: Scrobble history retrieval
- `src/m3u_exporter.py`: M3U8 file generation
- `src/plex_exporter.py`: Plex integration

### Configuration Files
- `config.yaml`: Main configuration
- `data/genre_similarity.yaml`: Genre taxonomy overrides
- `data/artifacts/beat3tower_32k/data_matrices_step1.npz`: Sonic embeddings

### Database Files
- `data/metadata.db`: SQLite library database
- `data/lastfm_cache_<user>_90d.json`: Scrobble cache

---

## Glossary

**Artifact Bundle**: Pre-computed sonic/genre embeddings for entire library
**Beam Search**: Heuristic search maintaining top-k candidates at each step
**Bridge Floor**: Minimum similarity to both pier endpoints
**Ensemble Normalization**: Stripping "Trio", "Quartet" suffixes from artist names
**Hybrid Embedding**: Combined sonic + genre similarity space
**Medoid**: Most central point in a cluster (unlike centroid, always a real data point)
**Pier**: Structural anchor point in playlist (seed track)
**Progress Constraint**: Monotonic movement toward destination in sonic space
**Tower PCA**: Separate dimensionality reduction per sonic domain (rhythm/timbre/harmony)
**Transition Matrix**: Pairwise similarity of track endings to track beginnings

---

**End of Document**
