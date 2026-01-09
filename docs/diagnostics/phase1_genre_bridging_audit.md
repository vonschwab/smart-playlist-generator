# Phase 1 - DJ Genre Bridging Architecture Audit

**Date:** 2026-01-09
**Branch:** `dj-ordering`
**Goal:** Document current architecture before implementing mode=vector with IDF weighting

---

## Executive Summary

The current DJ genre bridging system uses a **shortest-path genre label selection** approach that collapses to hub genres ("indie rock") instead of respecting full multi-genre signatures. This audit documents the architecture to enable a targeted fix.

**Key Finding:** The system has partial infrastructure for genre vector interpolation but defaults to **onehot single-label waypoints** chosen via shortest path, causing the hub genre problem.

---

## 1. DJ Bridging Route Planning / Ladder Planning

### Location
**File:** `src/playlist/pier_bridge_builder.py`

### Key Functions

#### `_build_genre_targets()` (lines 1014-1250)
**Purpose:** Main entry point for genre waypoint planning
**Returns:** List of target genre vectors, one per interior step

**Flow:**
1. Extracts pier A and B genre vectors from `X_genre_norm`
2. Checks `route_shape` config: `"linear"` | `"arc"` | `"ladder"`
3. If `route_shape != "ladder"` → falls back to **linear interpolation**:
   ```python
   for i in range(interior_length):
       frac = _step_fraction(i, interior_length)
       g = (1.0 - frac) * g_a + frac * g_b  # Vector interpolation
       g_targets.append(_normalize_vec(g))
   ```
4. If `route_shape == "ladder"` → **onehot label mode** (current behavior)

---

#### `_select_top_genre_labels()` (lines 682-697)
**Purpose:** Extract top genre labels from a genre vector

**Parameters:**
- `g_vec`: Genre vector (N, G) from X_genre_smoothed
- `top_n`: How many labels to extract (default: 5)
- `min_weight`: Minimum weight threshold (default: 0.05)

**Logic:**
```python
weights = np.array(g_vec, dtype=float)
# Sort by weight, take top N above min_weight
indices = np.argsort(-weights)
labels = [genre_vocab[i] for i in indices[:top_n] if weights[i] >= min_weight]
```

**Problem:** Picks highest-weight labels, which are often **generic hub genres** when tracks have mixed signatures.

---

#### `_shortest_genre_path()` (lines 752-781)
**Purpose:** Find shortest path between two genre labels in the genre graph

**Algorithm:** Dijkstra's shortest path with edge costs = `1.0 - similarity`

**Parameters:**
- `graph`: Genre similarity graph (adjacency list)
- `start`: Starting genre label (from pier A)
- `goal`: Ending genre label (from pier B)
- `max_steps`: Maximum path length (default: 6)

**Returns:** List of genre labels along the path

**Problem:** Shortest path **favors common hub genres** that connect many other genres:
- Example: `shoegaze → indie rock → post-punk` (2 steps)
- Instead of: `shoegaze → dreampop → slowcore → post-punk` (3 steps but more representative)

---

#### Ladder Route Planning Flow (lines 1097-1235)

**Step 1:** Extract top labels from pier A and B
```python
labels_a = _select_top_genre_labels(g_a, genre_vocab, top_n=5, min_weight=0.05)
labels_b = _select_top_genre_labels(g_b, genre_vocab, top_n=5, min_weight=0.05)
```

**Step 2:** Find shortest path between any pair
```python
for la in labels_a:
    for lb in labels_b:
        path_labels = _shortest_genre_path(genre_graph, la, lb, max_steps=6)
        if path_labels:
            break  # Take first valid path
```

**Step 3:** Convert labels to vectors (onehot or smoothed)

**Mode: onehot (default)**
```python
# _label_to_genre_vector() - lines 784-795
vec = np.zeros(len(genre_vocab))
vec[genre_vocab_map[label]] = 1.0
```

**Mode: smoothed (opt-in via dj_ladder_use_smoothed_waypoint_vectors=true)**
```python
# _label_to_smoothed_vector() - lines 804-845
# For a label, find top-K similar genres and blend
for vocab_label in genre_vocab:
    sim = genre_similarity(label, vocab_label)
    if sim >= min_sim:
        scores.append((idx, sim))
scores.sort(reverse=True)
scores = scores[:top_k]
vec = weighted_blend(scores)  # Sum normalized by weights
```

**Step 4:** Interpolate between waypoint vectors
```python
# Lines 1237-1249
for i in range(interior_length):
    frac = i / (interior_length - 1)
    scaled = frac * (len(waypoint_vecs) - 1)
    idx = floor(scaled)
    local = scaled - idx
    g = (1 - local) * waypoint_vecs[idx] + local * waypoint_vecs[idx + 1]
    g_targets.append(_normalize_vec(g))
```

---

### Configuration Parameters

**File:** `src/playlist/pier_bridge_builder.py`, lines 135-167

```python
@dataclass
class PierBridgeConfig:
    dj_route_shape: str = "linear"  # linear | arc | ladder

    # Ladder planning
    dj_ladder_top_labels: int = 5          # How many labels to extract per pier
    dj_ladder_min_label_weight: float = 0.05  # Min weight threshold
    dj_ladder_min_similarity: float = 0.20    # Genre graph edge threshold
    dj_ladder_max_steps: int = 6              # Max path length

    # Waypoint vector mode
    dj_ladder_use_smoothed_waypoint_vectors: bool = False  # False = onehot
    dj_ladder_smooth_top_k: int = 10                       # Top-K for smoothed
    dj_ladder_smooth_min_sim: float = 0.20                 # Min sim for smoothed
```

**User's current config:**
```yaml
route_shape: ladder              # ✅ Ladder mode enabled
waypoints: 1                     # Result: only 1 waypoint ("indie rock")
labels: indie rock               # Problem: collapsed to hub genre
mode: onehot                     # Using onehot (single-label vectors)
```

---

## 2. Waypoint Scoring Integration

### Location
**File:** `src/playlist/pier_bridge_builder.py`

### Key Functions

#### `_waypoint_delta()` (lines 2048-2059)
**Purpose:** Compute score adjustment from genre waypoint alignment

**Formula:**
```python
delta = waypoint_weight * cosine_similarity(candidate_genre, target_genre)
delta = clamp(delta, -waypoint_cap, waypoint_cap)
if sim < waypoint_floor:
    delta -= waypoint_penalty * (waypoint_floor - sim)
return delta
```

**Parameters:**
- `waypoint_weight`: Multiplier (currently 0.25)
- `waypoint_cap`: Max absolute delta (currently 0.10)
- `waypoint_floor`: Penalty threshold (0.20)
- `waypoint_penalty`: Penalty strength (0.10)

---

#### Waypoint Scoring in Beam Search (lines 2386-2405)

**Step 1:** Compute genre similarity to waypoint target
```python
# Line 2388
waypoint_sim = float(np.dot(X_genre_norm[cand], g_target))
```
- `X_genre_norm[cand]`: Candidate track's genre vector (L2-normalized)
- `g_target`: Target genre vector for this step (from _build_genre_targets)

**Step 2:** Apply waypoint delta to combined score
```python
# Line 2404
waypoint_delta_val = _waypoint_delta(waypoint_sim)
combined_score += waypoint_delta_val  # Lines 2405
```

**Full scoring formula:**
```python
combined_score = (
    weight_bridge * bridge_score +          # 0.6 * harmonic_mean(sim_a, sim_b)
    weight_transition * trans_score +       # 0.4 * transition_sim
    genre_tiebreak_weight * genre_sim +     # 0.05 * local_genre_sim (optional)
    waypoint_delta_val                      # 0.25 * waypoint_sim (capped at 0.10)
)
```

**Key Insight:** Waypoint scoring is **additive**, not multiplicative, so it can be overridden by strong sonic signals unless waypoint_weight is high enough.

---

### Configuration Parameters

```python
dj_waypoint_weight: float = 0.15  # User increased to 0.25
dj_waypoint_cap: float = 0.05     # User increased to 0.10
dj_waypoint_floor: float = 0.20   # Below this, apply penalty
dj_waypoint_penalty: float = 0.10 # Penalty strength
```

---

## 3. Candidate Pool Construction (dj_union)

### Location
**File:** `src/playlist/segment_pool_builder.py`

### Key Function

#### `_build_dj_union_pool()` (lines 563-702)
**Purpose:** Build union of 3 candidate sources: S1 (local), S2 (toward), S3 (genre)

**Flow:**

**S1 - Local sonic neighbors (lines 598-605)**
```python
# Top-K most similar to pier A and B
k_a = k_local // 2
k_b = k_local - k_a
sim_a = np.dot(X_full_norm[candidates], X_full_norm[pier_a])
sim_b = np.dot(X_full_norm[candidates], X_full_norm[pier_b])
local_indices = topk(sim_a, k_a) + topk(sim_b, k_b)
```

**S2 - Toward-B progression (lines 607-635)**
```python
# For each step, interpolate target between A and B
for step in range(0, interior_length, stride):
    t = step / (interior_length - 1)
    target = (1 - t) * vec_a + t * vec_b  # Sonic space
    target = normalize(target)
    sims = np.dot(X_full_norm[candidates], target)
    toward_indices += topk(sims, k_toward)
```

**S3 - Genre waypoint neighbors (lines 637-663)**
```python
# For each step, find candidates similar to genre target
for step in range(0, interior_length, stride):
    if step >= len(genre_targets):
        break
    target = genre_targets[step]  # From _build_genre_targets()
    sims = np.dot(X_genre_norm[candidates], target)  # Cosine in genre space
    genre_indices += topk(sims, k_genre)
```

**Union & Deduplication (lines 669-687)**
```python
combined = dedupe(local_indices + toward_indices + genre_indices)
if len(combined) > union_max:
    # Score by harmonic mean, keep top union_max
    scores = [harmonic_mean(sim_a[i], sim_b[i]) for i in combined]
    combined = topk_by_score(combined, scores, union_max)
```

---

### Configuration Parameters

**User's current config:**
```yaml
pool_k_local: 200
pool_k_toward: 80
pool_k_genre: 80         # 80 genre candidates per step
pool_k_union_max: 900
pool_step_stride: 1      # Every step (no skipping)
```

**Diagnostics from user's log:**
```
dj_pool_source_genre=240       # Total genre candidates added (80 * 3 steps)
chosen_from_genre_count=0-1    # Only 0-1 selected from S3 per segment
```

**Problem:** Genre candidates are added to pool but **rarely selected** during beam search because:
1. Waypoint scoring (`+0.10` bonus) is weaker than sonic similarity differences
2. Genre targets are onehot hub labels ("indie rock") instead of specific signatures

---

## 4. Genre Matrix Loading

### Location
**File:** `src/features/artifacts.py`

### Data Structure

#### `ArtifactBundle` dataclass (lines 15-36)
```python
@dataclass(frozen=True)
class ArtifactBundle:
    X_genre_raw: np.ndarray       # (N, G) - raw genre vectors
    X_genre_smoothed: np.ndarray  # (N, G) - similarity-weighted
    genre_vocab: np.ndarray       # (G,) - genre label strings
```

**Dimensions:**
- `N`: Number of tracks (~32,000 in user's artifact)
- `G`: Number of unique genres in vocabulary

---

#### `load_artifact_bundle()` (lines 52-152)
**Purpose:** Load NPZ file and construct ArtifactBundle

**Required keys:**
```python
required_keys = {
    "track_ids",
    "artist_keys",
    "X_sonic",
    "X_genre_raw",        # Raw genre vectors
    "X_genre_smoothed",   # Smoothed genre vectors
    "genre_vocab",        # Genre label vocabulary
}
```

**Loading:**
```python
X_genre_raw = data["X_genre_raw"]          # (N, G)
X_genre_smoothed = data["X_genre_smoothed"] # (N, G)
genre_vocab = data["genre_vocab"]          # (G,)
```

---

### Genre Vector Semantics

**X_genre_raw:**
- Binary or count-based vectors
- Each dimension corresponds to a genre in `genre_vocab`
- Example: `[0, 0, 1, 0, 1, ...]` = track tagged with genres at indices 2 and 4

**X_genre_smoothed:**
- Similarity-weighted smoothing of raw vectors
- Incorporates genre-to-genre similarity
- Example: A "shoegaze" track gets non-zero weights for "dreampop", "slowcore", etc.

**Usage in DJ bridging:**
- **Route planning:** Uses `X_genre_smoothed[pier_a/b]` to extract top labels
- **Waypoint targets:** Can use either raw or smoothed (depends on mode)
- **Beam search scoring:** Uses `X_genre_norm` (normalized smoothed vectors)

---

## 5. Current "Onehot" Waypoint Logic

### Overview
**Default mode:** `ladder_waypoint_vector_mode: onehot`

**Implementation:** `_label_to_genre_vector()` (lines 784-795)

```python
def _label_to_genre_vector(
    label: str,
    genre_vocab: np.ndarray,
    genre_vocab_map: dict[str, int],
) -> Optional[np.ndarray]:
    idx = genre_vocab_map.get(str(label).strip().lower())
    if idx is None:
        return None
    vec = np.zeros((len(genre_vocab),), dtype=float)
    vec[int(idx)] = 1.0  # Single dimension = 1.0, all others = 0.0
    return vec
```

**Result:** Waypoint target is a **single-genre onehot vector**

**Example:** If shortest path chooses "indie rock" waypoint:
```python
g_target = [0, 0, 0, ..., 1.0, ..., 0]  # Only "indie rock" dimension = 1.0
             ^         ^
             |         └─ Position 87: "indie rock"
             └─ Positions 0-86, 88-G: all other genres
```

---

### Why This Causes Hub Genre Problem

**Problem 1: Label selection favors hubs**
```python
# Slowdive genre vector (smoothed):
g_slowdive = {
    "shoegaze": 0.45,
    "dreampop": 0.30,
    "indie rock": 0.15,  # Generic tag present but low weight
    "slowcore": 0.10
}

# _select_top_genre_labels extracts top 5:
labels_a = ["shoegaze", "dreampop", "indie rock", "slowcore", "ambient"]
          # ⬆ "indie rock" included because it's in top 5 by weight
```

**Problem 2: Shortest path picks the hub**
```python
# Deerhunter genre vector:
g_deerhunter = {
    "indie rock": 0.40,
    "post-punk": 0.25,
    "noise rock": 0.20,
    "shoegaze": 0.15
}

labels_b = ["indie rock", "post-punk", "noise rock", "shoegaze", "art rock"]

# Shortest path search:
for la in labels_a:  # ["shoegaze", "dreampop", "indie rock", ...]
    for lb in labels_b:  # ["indie rock", "post-punk", ...]
        path = _shortest_genre_path(la, lb)

# First match found:
# la="indie rock", lb="indie rock" → path=["indie rock"] (0 steps)
# ✅ Valid path, so algorithm returns immediately
```

**Result:** Waypoint collapses to "indie rock" instead of "shoegaze" → "dreampop" route

---

### Why Single-Label Onehot Is Lossy

**Original signature (Slowdive):**
```python
g_slowdive = [0.45, 0.30, 0.15, 0.10, ...]  # Multi-genre signature
             ^     ^     ^     ^
             |     |     |     └─ slowcore
             |     |     └─ indie rock
             |     └─ dreampop
             └─ shoegaze
```

**After onehot conversion:**
```python
g_target = [0, 0, 1.0, 0, ...]  # "indie rock" onehot
            ^     ^     ^
            |     |     └─ indie rock (100% of signal)
            |     └─ dreampop (LOST)
            └─ shoegaze (LOST)
```

**Impact on beam search:**
- Candidates scored by cosine similarity to `g_target`
- Only candidates with high "indie rock" weight get bonus
- "Shoegaze"-heavy candidates (Sweet Trip alternative: e.g., Cocteau Twins) penalized

---

## 6. Existing "Genre Vector" Mode

### Partial Support Exists

**Mode:** `dj_ladder_use_smoothed_waypoint_vectors: true`

**Implementation:** `_label_to_smoothed_vector()` (lines 804-845)

**What it does:**
1. For a waypoint label (e.g., "indie rock")
2. Compute similarity to all genres in vocab
3. Keep top-K similar genres above threshold
4. Build weighted blend vector

**Example:**
```python
# Input: label="shoegaze"
# Compute similarities:
sim("shoegaze", "shoegaze") = 1.00
sim("shoegaze", "dreampop") = 0.85
sim("shoegaze", "slowcore") = 0.70
sim("shoegaze", "indie rock") = 0.45
sim("shoegaze", "post-punk") = 0.40
# ... etc

# Keep top-10 above min_sim=0.20:
top_k = [
    (idx_shoegaze, 1.00),
    (idx_dreampop, 0.85),
    (idx_slowcore, 0.70),
    ...
]

# Build weighted vector:
vec = normalize([1.00, 0.85, 0.70, ...])  # Non-zero for multiple genres
```

**Result:** **Multi-genre waypoint vector** instead of onehot!

---

### Why This Doesn't Fully Solve the Problem

**Issue 1: Still based on shortest-path label selection**
- Smoothed mode uses the label chosen by shortest path
- If shortest path picks "indie rock", smoothed vector is based on "indie rock" similarities
- Still hub-biased, just less lossy

**Issue 2: No IDF weighting**
- All genres weighted equally by similarity
- Hub genres like "indie rock" still dominate if they appear in similarity neighborhood

**Issue 3: No coverage bonus near anchors**
- Waypoint scoring is constant throughout segment
- No boost for matching Slowdive's specific signature near step 0

---

### Config for Smoothed Mode

**User's current config:** `mode: onehot`

**To enable smoothed:**
```yaml
pier_bridge:
  dj_bridging:
    dj_ladder_use_smoothed_waypoint_vectors: true
    dj_ladder_smooth_top_k: 10
    dj_ladder_smooth_min_sim: 0.20
```

**Diagnostics field:**
```
ladder_waypoint_vector_mode: smoothed
ladder_waypoint_vector_stats: [
  {label: "indie rock", mode: "smoothed", nonzero: 8, top_labels: [...]}
]
```

---

## 7. Genre Frequency / IDF Tracking

### Current State: **No IDF/Rarity Weighting Exists**

**Searched patterns:**
- `genre.*frequency`
- `idf`
- `document.*frequency`
- `genre.*rarity`

**Results:** No existing IDF or genre frequency tracking in the codebase.

---

### What Would Be Needed

**IDF (Inverse Document Frequency) computation:**
```python
# Count tracks per genre
df = np.zeros(G)  # Document frequency per genre
for track_genres in X_genre_raw:
    for g_idx in np.nonzero(track_genres)[0]:
        df[g_idx] += 1

# Compute IDF
N = len(X_genre_raw)
idf = np.log(N / (df + 1))  # +1 smoothing to avoid division by zero

# Normalize
idf = idf / np.max(idf)  # Scale to [0, 1]
```

**Expected distribution:**
- Common genres (e.g., "rock", "indie rock"): `idf ≈ 0.1-0.3` (low weight)
- Rare genres (e.g., "shoegaze", "dreampop"): `idf ≈ 0.7-1.0` (high weight)

**Usage in genre vector weighting:**
```python
g_weighted = g_raw * idf  # Element-wise multiplication
g_weighted = normalize(g_weighted)
```

**Result:** Rare/expressive genres get higher influence in similarity computations.

---

### Caching Strategy

**Option 1: Compute once per run**
```python
# In pipeline.py or pier_bridge_builder.py
def _compute_genre_idf(X_genre_raw: np.ndarray) -> np.ndarray:
    df = (X_genre_raw > 0).sum(axis=0)  # Count tracks per genre
    N = X_genre_raw.shape[0]
    idf = np.log(N / (df + 1))
    return idf / np.max(idf)

# Cache in ArtifactBundle or pass to _build_genre_targets
```

**Option 2: Precompute and save to NPZ**
```python
# In artifact_builder.py (offline)
idf = _compute_genre_idf(X_genre_raw)
np.savez(
    artifact_path,
    ...,
    X_genre_raw=X_genre_raw,
    X_genre_idf=idf,  # New key
)

# In artifacts.py (loading)
genre_idf = data.get("X_genre_idf")  # Optional key
```

**Recommendation:** Start with Option 1 (compute per run) for rapid iteration, then move to Option 2 if computation is slow.

---

## 8. Summary: What Needs to Change

### Problems Identified

1. **Shortest-path label selection favors hub genres**
   - Picks "indie rock" over "shoegaze" because it's a shorter path

2. **Onehot waypoint vectors are lossy**
   - Discards multi-genre signature, keeps only one label

3. **No IDF/rarity weighting**
   - Common genres weighted equally to rare genres

4. **Constant waypoint scoring**
   - No coverage bonus near anchor tracks

---

### Existing Infrastructure to Reuse

✅ **Genre matrices:** `X_genre_raw`, `X_genre_smoothed` already loaded
✅ **Vector interpolation:** `_build_genre_targets()` supports linear/arc interpolation
✅ **Smoothed vectors:** `_label_to_smoothed_vector()` exists (partial solution)
✅ **S3 genre pooling:** `_build_dj_union_pool()` already uses genre targets
✅ **Waypoint scoring:** `_waypoint_delta()` integrates into beam search

---

### Proposed Changes (Phase 2 Design)

**Change 1: Add `mode=vector` to route planning**
- Skip shortest-path label selection
- Directly interpolate `X_genre_smoothed[pier_a]` → `X_genre_smoothed[pier_b]`
- Apply IDF weighting to emphasize rare genres

**Change 2: Compute and cache genre IDF**
- Add `_compute_genre_idf()` function
- Cache IDF vector in config or bundle

**Change 3: Add coverage bonus schedule**
- Increase `waypoint_weight` near anchors (steps 0-2, n-3 to n-1)
- Example: `weight = base_weight * (1.0 + coverage_bonus * proximity)`

**Change 4: Add detailed logging**
- Log top genres in target vector (with IDF-weighted scores)
- Log whether genre signal changed the winner
- Track `winner_changed` metric per step

---

## 9. File Reference Summary

| Component | File | Lines |
|-----------|------|-------|
| Route planning | `pier_bridge_builder.py` | 1014-1250 |
| Shortest path | `pier_bridge_builder.py` | 752-781 |
| Top label selection | `pier_bridge_builder.py` | 682-697 |
| Onehot conversion | `pier_bridge_builder.py` | 784-795 |
| Smoothed conversion | `pier_bridge_builder.py` | 804-845 |
| Waypoint scoring | `pier_bridge_builder.py` | 2048-2059, 2386-2405 |
| S3 genre pooling | `segment_pool_builder.py` | 637-663 |
| dj_union pool | `segment_pool_builder.py` | 563-702 |
| Artifact loading | `artifacts.py` | 52-152 |
| Config dataclass | `pier_bridge_builder.py` | 65-175 |

---

## 10. Next Steps (Phase 2)

With this architecture documented, Phase 2 will design:

1. **`mode=vector` implementation plan**
   - Config key: `dj_bridging.route_mode: vector` (vs `onehot`)
   - Skip label extraction, use direct interpolation

2. **IDF weighting integration**
   - Where to compute: `_build_genre_targets()` entry point
   - How to cache: Pass as parameter or add to config

3. **Coverage bonus schedule**
   - Formula: `bonus = coverage_weight * exp(-distance_to_anchor / decay)`
   - Where to apply: `_waypoint_delta()` function

4. **Logging enhancements**
   - Target genre breakdown (top 5 with IDF weights)
   - Per-step winner_changed tracking
   - Genre alignment distribution

**User approval needed before implementation.**
