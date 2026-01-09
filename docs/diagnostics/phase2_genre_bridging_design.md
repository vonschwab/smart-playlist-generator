# Phase 2 - DJ Genre Bridging Design (mode=vector + IDF + Coverage)

**Date:** 2026-01-09
**Branch:** `dj-ordering`
**Goal:** Fix hub-genre collapse by adding vector interpolation, IDF weighting, and coverage bonuses

---

## A) New Configuration Keys

### Location
**File:** `src/playlist/pier_bridge_builder.py`
**Dataclass:** `PierBridgeConfig` (lines 65-175)

### New Keys (defaults preserve existing behavior)

```python
@dataclass
class PierBridgeConfig:
    # ... existing keys ...

    # === GENRE VECTOR MODE ===
    dj_ladder_target_mode: str = "onehot"  # "onehot" | "vector"
    """
    Genre target generation mode:
    - "onehot": Extract labels via shortest path, convert to onehot vectors (legacy)
    - "vector": Direct multi-genre vector interpolation, bypass label selection
    """

    dj_genre_vector_source: str = "smoothed"  # "smoothed" | "raw"
    """
    Which genre matrix to use for vector targets:
    - "smoothed": X_genre_smoothed (similarity-weighted, recommended)
    - "raw": X_genre_raw (binary/count-based)
    """

    # === IDF WEIGHTING ===
    dj_genre_use_idf: bool = False
    """Enable IDF (inverse document frequency) weighting to down-weight common genres."""

    dj_genre_idf_power: float = 1.0
    """
    Exponent for IDF scaling: idf = idf ** power.
    - power=1.0: Standard IDF (linear)
    - power>1.0: Amplifies rare genre importance (e.g., 1.5-2.0)
    - power<1.0: Softens IDF effect
    """

    dj_genre_idf_norm: str = "max1"  # "max1" | "sum1" | "none"
    """
    IDF normalization strategy:
    - "max1": Scale so max(idf) = 1.0 (preserves relative differences)
    - "sum1": Scale so sum(idf) = 1.0 (probability-like)
    - "none": Raw log IDF values
    Recommendation: "max1" (interpretable scale, preserves ratios)
    """

    # === COVERAGE BONUS ===
    dj_genre_use_coverage: bool = False
    """Enable coverage bonus near anchors (rewards matching anchor's genre signature)."""

    dj_genre_coverage_top_k: int = 8
    """
    Number of top genres to consider for coverage computation.
    After IDF weighting, take top-K highest-weight genres from each anchor.
    """

    dj_genre_coverage_weight: float = 0.15
    """
    Weight multiplier for coverage bonus.
    Applied additively like waypoint_weight: score += coverage_weight * coverage_bonus
    Recommended: 0.10-0.20 (similar magnitude to waypoint_cap)
    """

    dj_genre_coverage_power: float = 2.0
    """
    Schedule decay exponent for coverage bonus.
    - At step t, progress s = t / (L+1)
    - Anchor A weight: wA = (1-s)^power
    - Anchor B weight: wB = s^power
    Higher power = sharper decay (stronger near anchors, weaker in middle)
    Recommended: 2.0-3.0
    """

    dj_genre_presence_threshold: float = 0.01
    """
    Threshold for "presence" in coverage computation.
    Candidate genre weight >= threshold counts as "present" for that genre.
    Recommendation: 0.01 (lenient) to 0.05 (strict)
    """
```

---

### Interactions with Existing Keys

**1. `dj_ladder_use_smoothed_waypoint_vectors` (existing)**
   - **Scope:** Only relevant when `dj_ladder_target_mode == "onehot"` (legacy mode)
   - **Behavior:** If onehot mode, controls whether to use smoothed vectors for label-based waypoints
   - **With mode=vector:** Ignored (vector mode doesn't use label-based waypoints)

**2. `dj_route_shape` (existing)**
   - **Values:** `"linear" | "arc" | "ladder"`
   - **Interaction:** When `route_shape == "ladder"` AND `target_mode == "vector"`:
     - Bypass shortest path logic entirely
     - Use direct interpolation with IDF weighting
   - **When `route_shape == "linear"` or `"arc"`:** Already uses vector interpolation, IDF can be applied

**3. `dj_waypoint_weight`, `dj_waypoint_cap`, `dj_waypoint_floor`, `dj_waypoint_penalty` (existing)**
   - **Scope:** Apply to waypoint scoring (cosine similarity to genre target)
   - **Interaction with IDF:** Waypoint scoring uses IDF-weighted targets and IDF-weighted candidate matrix (if enabled)
   - **Interaction with coverage:** Coverage bonus is **separate additive term**, doesn't affect waypoint_delta calculation

**4. Config validation:**
   - If `dj_genre_use_idf == True` but `X_genre_raw` missing → log warning, disable IDF
   - If `dj_genre_use_coverage == True` but IDF disabled → coverage still works (uses base weights)

---

## B) Vector Mode Behavior

### High-Level Flow

**When `dj_ladder_target_mode == "vector"`:**

```python
# In _build_genre_targets()
if cfg.dj_ladder_target_mode == "vector":
    # 1. Select source matrix
    if cfg.dj_genre_vector_source == "smoothed":
        X_genre_base = X_genre_smoothed  # Similarity-weighted
    else:
        X_genre_base = X_genre_raw       # Binary/count-based

    # 2. Extract anchor vectors
    vA = X_genre_base[pier_a]  # (G,) vector
    vB = X_genre_base[pier_b]  # (G,) vector

    # 3. Apply IDF weighting (optional)
    if cfg.dj_genre_use_idf:
        idf = _compute_genre_idf(X_genre_raw, cfg)  # Compute once, cache
        vA = _apply_idf(vA, idf)
        vB = _apply_idf(vB, idf)

    # 4. Normalize anchors
    vA = _normalize_vec(vA)
    vB = _normalize_vec(vB)

    # 5. Interpolate step targets
    g_targets = []
    for i in range(interior_length):
        s = _step_fraction(i, interior_length)  # s ∈ [0, 1]
        g = (1.0 - s) * vA + s * vB
        g_targets.append(_normalize_vec(g))

    return g_targets
```

**Key differences from onehot mode:**
- ❌ NO `_select_top_genre_labels()` call
- ❌ NO `_shortest_genre_path()` call
- ❌ NO `_label_to_genre_vector()` or `_label_to_smoothed_vector()`
- ✅ Direct multi-genre vector interpolation
- ✅ Preserves full signature (shoegaze, dreampop, slowcore all retained)

---

### Example: Slowdive → Deerhunter

**Anchor Vectors (after IDF weighting):**
```python
# Slowdive (IDF-weighted, smoothed)
vA = {
    shoegaze:   0.60,  # Rare genre, boosted by IDF
    dreampop:   0.45,  # Rare genre, boosted
    slowcore:   0.30,  # Rare genre, boosted
    indie rock: 0.08,  # Common genre, suppressed by IDF
    ambient:    0.25,
    ...
}

# Deerhunter (IDF-weighted, smoothed)
vB = {
    post-punk:  0.50,
    noise rock: 0.40,
    indie rock: 0.10,  # Common, suppressed
    shoegaze:   0.20,
    art rock:   0.35,
    ...
}
```

**Step Targets (interior_length=8):**
```python
# Step 0 (near Slowdive, s=0.00):
g_target[0] = normalize(1.00*vA + 0.00*vB)
            = {shoegaze: 0.60, dreampop: 0.45, slowcore: 0.30, ...}

# Step 4 (middle, s=0.50):
g_target[4] = normalize(0.50*vA + 0.50*vB)
            = {shoegaze: 0.40, dreampop: 0.23, post-punk: 0.25, ...}

# Step 7 (near Deerhunter, s=0.88):
g_target[7] = normalize(0.12*vA + 0.88*vB)
            = {post-punk: 0.44, noise rock: 0.35, art rock: 0.31, ...}
```

**Result:** Early steps prioritize shoegaze/dreampop, late steps prioritize post-punk/noise rock, no "indie rock" collapse!

---

## C) IDF Weighting

### Computation (once per run)

**Function:** `_compute_genre_idf()`

```python
def _compute_genre_idf(
    X_genre_raw: np.ndarray,  # (N, G)
    cfg: PierBridgeConfig,
) -> np.ndarray:  # (G,)
    """
    Compute IDF (inverse document frequency) for each genre.

    Formula:
        df[g] = count(tracks where genre[g] > 0)
        idf[g] = log((N + 1) / (df[g] + 1))  # +1 smoothing
        idf = idf ** cfg.dj_genre_idf_power
        idf = normalize(idf, method=cfg.dj_genre_idf_norm)

    Returns:
        idf: (G,) array where idf[g] ∈ [0, 1] (after normalization)
             High values = rare genres (shoegaze, dreampop)
             Low values = common genres (rock, indie rock)
    """
    N, G = X_genre_raw.shape

    # Count tracks per genre (document frequency)
    df = (X_genre_raw > 0).sum(axis=0)  # (G,)

    # Compute raw IDF
    idf = np.log((N + 1) / (df + 1))  # +1 smoothing

    # Apply power scaling
    power = float(cfg.dj_genre_idf_power)
    if power != 1.0 and power > 0:
        idf = idf ** power

    # Normalize
    norm_method = str(cfg.dj_genre_idf_norm).strip().lower()
    if norm_method == "max1":
        idf = idf / np.max(idf)  # Scale to [0, 1]
    elif norm_method == "sum1":
        idf = idf / np.sum(idf)  # Sum to 1.0
    # else: "none" - keep raw values

    return idf
```

**Caching strategy:**
- Compute once at start of `build_pier_bridge_sequence()`
- Cache in local variable, reuse across all segments
- Pass as parameter to `_build_genre_targets()` and `_build_segment_candidate_pool()`

**Expected IDF distribution (example):**
```python
# Sorted by IDF (high = rare, low = common)
idf = {
    "shoegaze":      1.00,  # Appears in ~500 tracks (rare)
    "dreampop":      0.95,
    "slowcore":      0.88,
    "noise rock":    0.82,
    "post-punk":     0.75,
    "alternative":   0.45,
    "indie rock":    0.15,  # Appears in ~15,000 tracks (common)
    "rock":          0.10,
    ...
}
```

---

### Application to Genre Vectors

**Function:** `_apply_idf_weighting()`

```python
def _apply_idf_weighting(
    genre_vec: np.ndarray,  # (G,) or (N, G)
    idf: np.ndarray,        # (G,)
) -> np.ndarray:
    """
    Apply IDF weighting element-wise.

    For 1D vector:
        result = genre_vec * idf  # Element-wise
        result = normalize(result)

    For 2D matrix:
        result = genre_vec * idf[np.newaxis, :]  # Broadcasting
        result = normalize_rows(result)
    """
    weighted = genre_vec * idf

    # Normalize
    if weighted.ndim == 1:
        return _normalize_vec(weighted)
    else:  # 2D matrix
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return weighted / norms
```

**Where to apply:**
1. **Anchor vectors in `_build_genre_targets()`:** `vA = _apply_idf(vA, idf)`
2. **Candidate matrix for S3 pooling:** `X_genre_idf = _apply_idf(X_genre_smoothed, idf)`
3. **Candidate matrix for beam scoring:** Reuse same `X_genre_idf`

**Why consistency matters:**
- Target and candidate must use same IDF weighting
- Otherwise: `np.dot(X_genre_idf[cand], g_target)` meaningless if scales differ

---

### IDF Statistics Logging

**Per segment, log:**
```python
if cfg.dj_genre_use_idf:
    logger.info(
        "IDF enabled: power=%.2f norm=%s | min=%.3f median=%.3f max=%.3f",
        cfg.dj_genre_idf_power,
        cfg.dj_genre_idf_norm,
        float(np.min(idf)),
        float(np.median(idf)),
        float(np.max(idf)),
    )

    # Show top rare genres
    top_rare_idx = np.argsort(-idf)[:10]
    top_rare = [(genre_vocab[i], idf[i]) for i in top_rare_idx]
    logger.debug("Top rare genres: %s", top_rare)
```

---

## D) Coverage Bonus Schedule

### Concept

**Goal:** Reward candidates that match the **most expressive genres** of nearby anchors.

**Mechanism:**
1. Extract top-K genres from each anchor (after IDF weighting)
2. For each candidate, check how many of those top-K genres are "present"
3. Apply weighted coverage bonus based on position in segment

**Why "coverage" not "exact match":**
- Exact match: `candidate_genre == "shoegaze"` → too strict, brittle
- Coverage: `candidate has {shoegaze, dreampop, slowcore} from anchor's top-8` → flexible, robust

---

### Computation

**Step 1: Extract anchor top-K genres (once per segment)**

```python
def _extract_top_genres(
    genre_vec: np.ndarray,  # (G,) after IDF weighting
    genre_vocab: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Extract top-K genres by weight.

    Returns:
        List of (genre_idx, weight) tuples, sorted descending.
    """
    indices = np.argsort(-genre_vec)[:top_k]
    return [(int(i), float(genre_vec[i])) for i in indices if genre_vec[i] > 0]

# In _build_genre_targets() or beam search setup:
topk_A = _extract_top_genres(vA_idf, genre_vocab, cfg.dj_genre_coverage_top_k)
topk_B = _extract_top_genres(vB_idf, genre_vocab, cfg.dj_genre_coverage_top_k)
```

**Example:**
```python
# Slowdive top-8 (after IDF):
topk_A = [
    (idx_shoegaze, 0.60),
    (idx_dreampop, 0.45),
    (idx_slowcore, 0.30),
    (idx_ambient, 0.25),
    (idx_ethereal, 0.20),
    (idx_atmospheric, 0.18),
    (idx_noise, 0.15),
    (idx_experimental, 0.12),
]
```

---

**Step 2: Compute candidate coverage (per candidate, per step)**

```python
def _compute_coverage(
    candidate_genre_vec: np.ndarray,  # (G,) candidate's genre vector
    topk_genres: list[tuple[int, float]],
    threshold: float,
) -> float:
    """
    Compute fraction of top-K genres "present" in candidate.

    Presence: candidate_genre_vec[g_idx] >= threshold

    Returns:
        coverage ∈ [0, 1]: fraction of top-K genres present
    """
    if not topk_genres:
        return 0.0

    present_count = 0
    for g_idx, _ in topk_genres:
        if candidate_genre_vec[g_idx] >= threshold:
            present_count += 1

    return float(present_count) / float(len(topk_genres))

# In beam search loop:
coverage_A = _compute_coverage(X_genre_idf[cand], topk_A, cfg.dj_genre_presence_threshold)
coverage_B = _compute_coverage(X_genre_idf[cand], topk_B, cfg.dj_genre_presence_threshold)
```

**Example:**
```python
# Candidate: Cocteau Twins
# Genre vector (IDF-weighted): {shoegaze: 0.55, dreampop: 0.50, ethereal: 0.40, ...}

# Check against Slowdive top-8:
# shoegaze (0.55) >= 0.01 ✅
# dreampop (0.50) >= 0.01 ✅
# slowcore (0.02) >= 0.01 ✅
# ambient (0.30) >= 0.01 ✅
# ethereal (0.40) >= 0.01 ✅
# atmospheric (0.25) >= 0.01 ✅
# noise (0.08) >= 0.01 ✅
# experimental (0.05) >= 0.01 ✅

# coverage_A = 8/8 = 1.00 (perfect match!)
```

---

**Step 3: Apply schedule weights**

```python
def _compute_coverage_bonus(
    step: int,
    interior_length: int,
    coverage_A: float,
    coverage_B: float,
    cfg: PierBridgeConfig,
) -> float:
    """
    Compute coverage bonus with decay schedule.

    Schedule:
        s = step / (interior_length + 1)  # Progress ∈ [0, 1]
        wA = (1 - s) ** power              # Strong near A (s=0)
        wB = s ** power                    # Strong near B (s=1)
        bonus = weight * (wA * coverage_A + wB * coverage_B)

    Returns:
        bonus ∈ [0, weight] (additive score adjustment)
    """
    if interior_length == 0:
        return 0.0

    s = float(step) / float(interior_length + 1)
    power = float(cfg.dj_genre_coverage_power)

    wA = (1.0 - s) ** power
    wB = s ** power

    bonus = float(cfg.dj_genre_coverage_weight) * (
        wA * float(coverage_A) + wB * float(coverage_B)
    )

    return bonus
```

**Schedule visualization (power=2.0, weight=0.15):**
```
Step  s     wA      wB     Max Bonus (coverage=1.0)
---------------------------------------------------
0     0.00  1.00    0.00   0.150  (100% anchor A)
1     0.11  0.79    0.01   0.119
2     0.22  0.61    0.05   0.098
3     0.33  0.45    0.11   0.084
4     0.44  0.31    0.20   0.076  (middle)
5     0.56  0.19    0.31   0.076
6     0.67  0.11    0.45   0.084
7     0.78  0.05    0.61   0.098
8     0.89  0.01    0.79   0.119
9     1.00  0.00    1.00   0.150  (100% anchor B)
```

**Key property:** Bonus is **strongest near anchors**, decays toward middle (power=2.0 gives smooth quadratic falloff).

---

**Step 4: Integrate into beam scoring (additive)**

```python
# In pier_bridge_builder.py beam search loop (around line 2404):

# ... existing scoring ...
waypoint_delta_val = _waypoint_delta(waypoint_sim)
combined_score += waypoint_delta_val

# NEW: Coverage bonus
if cfg.dj_genre_use_coverage and topk_A and topk_B:
    coverage_A = _compute_coverage(X_genre_idf[cand], topk_A, cfg.dj_genre_presence_threshold)
    coverage_B = _compute_coverage(X_genre_idf[cand], topk_B, cfg.dj_genre_presence_threshold)
    coverage_bonus = _compute_coverage_bonus(
        step, interior_length, coverage_A, coverage_B, cfg
    )
    combined_score += coverage_bonus

    # Log for diagnostics
    if step_is_sampled:
        step_candidates_for_ranking.append((
            ...,
            coverage_bonus,  # Track for winner_changed analysis
        ))
```

**Why additive:**
- Same pattern as waypoint_delta (additive bonus/penalty)
- Doesn't gate candidates (soft preference)
- Magnitude similar to waypoint_cap (0.10-0.20 range)

---

## E) Logging / Diagnostics

### Per-Segment Logging

**At segment start (in `_build_genre_targets()`):**

```python
logger.info(
    "Segment %d→%d | target_mode=%s source=%s idf=%s coverage=%s",
    pier_a, pier_b,
    cfg.dj_ladder_target_mode,
    cfg.dj_genre_vector_source,
    "enabled" if cfg.dj_genre_use_idf else "disabled",
    "enabled" if cfg.dj_genre_use_coverage else "disabled",
)

if cfg.dj_ladder_target_mode == "vector":
    # Log anchor top genres (after IDF weighting)
    top_A = _extract_top_genres(vA_idf, genre_vocab, 10)
    top_B = _extract_top_genres(vB_idf, genre_vocab, 10)

    logger.info(
        "  Anchor A (pier %d) top-10 genres: %s",
        pier_a,
        ", ".join([f"{genre_vocab[i]}({w:.3f})" for i, w in top_A])
    )
    logger.info(
        "  Anchor B (pier %d) top-10 genres: %s",
        pier_b,
        ", ".join([f"{genre_vocab[i]}({w:.3f})" for i, w in top_B])
    )
```

**Example output:**
```
INFO: Segment 0→1 | target_mode=vector source=smoothed idf=enabled coverage=enabled
INFO:   Anchor A (pier 0) top-10 genres: shoegaze(0.601), dreampop(0.452), slowcore(0.301), ambient(0.254), ethereal(0.198), ...
INFO:   Anchor B (pier 1) top-10 genres: post-punk(0.502), noise rock(0.398), art rock(0.352), shoegaze(0.201), indie rock(0.098), ...
```

---

### Per-Step Diagnostics (sampled)

**For first/middle/last step, log top candidates by genre alignment:**

```python
# In beam search, collect candidates at sampled steps
if step_is_sampled:
    step_debug_info = []
    for cand in top_N_candidates:
        waypoint_sim = np.dot(X_genre_idf[cand], g_target)
        coverage_A = _compute_coverage(X_genre_idf[cand], topk_A, cfg.dj_genre_presence_threshold)
        coverage_B = _compute_coverage(X_genre_idf[cand], topk_B, cfg.dj_genre_presence_threshold)
        coverage_bonus = _compute_coverage_bonus(step, interior_length, coverage_A, coverage_B, cfg)

        step_debug_info.append({
            "idx": cand,
            "artist": bundle.track_artists[cand],
            "title": bundle.track_titles[cand],
            "waypoint_sim": waypoint_sim,
            "coverage_A": coverage_A,
            "coverage_B": coverage_B,
            "coverage_bonus": coverage_bonus,
        })

    # Log top 5
    step_debug_info.sort(key=lambda x: x["coverage_bonus"], reverse=True)
    logger.debug(
        "  Step %d top-5 by coverage: %s",
        step,
        "\n    ".join([
            f"{d['artist']} - {d['title']}: waypoint={d['waypoint_sim']:.3f} cvgA={d['coverage_A']:.2f} cvgB={d['coverage_B']:.2f} bonus={d['coverage_bonus']:.3f}"
            for d in step_debug_info[:5]
        ])
    )
```

**Example output:**
```
DEBUG:   Step 0 top-5 by coverage:
    Cocteau Twins - Cherry-Coloured Funk: waypoint=0.852 cvgA=1.00 cvgB=0.25 bonus=0.148
    Slowdive - Souvlaki Space Station: waypoint=0.901 cvgA=0.88 cvgB=0.13 bonus=0.130
    My Bloody Valentine - When You Sleep: waypoint=0.798 cvgA=0.75 cvgB=0.38 bonus=0.115
    ...
```

---

### Winner Changed Tracking (extend existing diagnostic)

**Current diagnostic:** `waypoint_rank_impact` tracks if waypoint scoring changes the winner

**Extension:** Track coverage bonus impact separately

```python
# In pier_bridge_diagnostics.py (SegmentDiagnostics dataclass):
@dataclass
class SegmentDiagnostics:
    # ... existing fields ...

    # Coverage bonus impact (NEW)
    coverage_rank_impact_enabled: bool = False
    coverage_rank_impact_sample_steps: int = 0
    coverage_rank_impact_winner_changed: int = 0
    coverage_rank_impact_topk_reordered: float = 0.0
    coverage_rank_impact_mean_rank_delta: float = 0.0
```

**Computation logic (same pattern as waypoint rank impact):**

```python
# For sampled steps:
# 1. Rank candidates by base_score (without coverage bonus)
# 2. Rank candidates by full_score (with coverage bonus)
# 3. Compare rankings:
#    - winner_changed: Did #1 candidate change?
#    - topk_reordered: How many top-10 positions swapped?
#    - mean_rank_delta: Average rank shift across all candidates

# Log result:
logger.info(
    "Coverage rank impact: winner_changed=%d/%d topK_reordered=%.1f/10 mean_rank_delta=%.1f",
    coverage_winner_changed,
    sampled_steps,
    coverage_topk_reordered,
    coverage_mean_rank_delta,
)
```

---

### Summary Stats (per segment)

**Log final stats:**
```python
logger.info(
    "Segment %d→%d complete | interior=%d chosen_from_genre=%d/%d (%.1f%%)",
    pier_a, pier_b,
    interior_length,
    chosen_from_genre_count,
    interior_length,
    100.0 * chosen_from_genre_count / interior_length if interior_length > 0 else 0.0,
)

if cfg.dj_genre_use_idf or cfg.dj_genre_use_coverage:
    logger.info(
        "  Genre features enabled: idf=%s coverage=%s | waypoint_impact=%d/%d coverage_impact=%d/%d",
        "yes" if cfg.dj_genre_use_idf else "no",
        "yes" if cfg.dj_genre_use_coverage else "no",
        waypoint_winner_changed, sampled_steps,
        coverage_winner_changed, sampled_steps,
    )
```

---

## F) Implementation Plan

### Order of Implementation (minimal-change diffs)

**1. Add IDF helpers (pier_bridge_builder.py)**
   - Add `_compute_genre_idf()` function
   - Add `_apply_idf_weighting()` function
   - Add caching logic in `build_pier_bridge_sequence()`

**2. Add config keys (pier_bridge_builder.py)**
   - Update `PierBridgeConfig` dataclass with new fields
   - Add validation logic (warn if IDF enabled but X_genre_raw missing)

**3. Modify `_build_genre_targets()` for vector mode**
   - Add `if cfg.dj_ladder_target_mode == "vector":` branch
   - Bypass shortest path logic
   - Apply IDF weighting to anchor vectors
   - Return interpolated targets

**4. Update S3 pooling (segment_pool_builder.py)**
   - Add `X_genre_idf` parameter to `SegmentPoolConfig`
   - In `_build_dj_union_pool()`, use `X_genre_idf` if provided
   - Otherwise fall back to `X_genre_norm`

**5. Add coverage bonus helpers (pier_bridge_builder.py)**
   - Add `_extract_top_genres()` function
   - Add `_compute_coverage()` function
   - Add `_compute_coverage_bonus()` function

**6. Integrate coverage into beam search**
   - Compute topk_A, topk_B at segment start
   - In beam loop, compute coverage bonus per candidate
   - Add to `combined_score`
   - Collect for diagnostics

**7. Add logging**
   - Per-segment: mode, IDF stats, anchor top genres
   - Per-step (sampled): top candidates by coverage
   - Per-segment summary: winner_changed stats

**8. Update diagnostics dataclass (pier_bridge_diagnostics.py)**
   - Add coverage rank impact fields
   - Update logging output

---

### Files Modified Summary

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `pier_bridge_builder.py` | ~200-300 | IDF helpers, vector mode, coverage bonus, logging |
| `segment_pool_builder.py` | ~20-30 | IDF matrix passthrough in S3 pooling |
| `pier_bridge_diagnostics.py` | ~10-15 | Add coverage rank impact fields |

---

## G) Verification Strategy

### Test Case: Slowdive → Deerhunter

**Baseline (current behavior):**
```yaml
dj_ladder_target_mode: onehot  # Default
dj_genre_use_idf: false
dj_genre_use_coverage: false
```

**Expected result:**
- Waypoints collapse to "indie rock" (shortest path)
- S3 genre candidates: generic indie rock tracks
- `chosen_from_genre_count`: 0-1 per segment
- Tracks near Slowdive: Sweet Trip, Ramones (genre mismatch)

---

**With vector mode + IDF + coverage:**
```yaml
dj_ladder_target_mode: vector
dj_genre_vector_source: smoothed
dj_genre_use_idf: true
dj_genre_idf_power: 1.5
dj_genre_idf_norm: max1
dj_genre_use_coverage: true
dj_genre_coverage_top_k: 8
dj_genre_coverage_weight: 0.15
dj_genre_coverage_power: 2.0
dj_genre_presence_threshold: 0.01
```

**Expected result:**
- Anchor A top genres: `shoegaze(0.60), dreampop(0.45), slowcore(0.30), ...`
- Anchor B top genres: `post-punk(0.50), noise rock(0.40), ...`
- S3 genre candidates at step 0: Cocteau Twins, My Bloody Valentine, Lush, ...
- `chosen_from_genre_count`: 3-5 per segment (increased)
- Tracks near Slowdive: shoegaze/dreampop artists (genre match!)
- Coverage bonus logs show high scores for genre-aligned tracks

---

### Success Criteria

✅ **Waypoints don't collapse to "indie rock"**
- Log shows top genres include shoegaze, dreampop, slowcore (not just indie rock)

✅ **S3 picks become more relevant**
- `chosen_from_genre_count` increases from 0-1 to 3-5+ per segment
- S3 candidates in pool match anchor signatures

✅ **Winner_changed metrics improve**
- `coverage_rank_impact_winner_changed` > 0 (coverage changes selections)
- `waypoint_rank_impact` increases (vector targets have stronger signal)

✅ **Logs clearly show why selections changed**
- Coverage bonus values visible per candidate
- Top candidates by coverage align with anchor genres
- IDF stats show rare genres boosted, common genres suppressed

---

## H) Config Migration

### Backward Compatibility

**Defaults preserve existing behavior:**
- `dj_ladder_target_mode: "onehot"` → uses shortest path (legacy)
- `dj_genre_use_idf: false` → no IDF weighting
- `dj_genre_use_coverage: false` → no coverage bonus

**User must explicitly opt-in to new features:**
```yaml
pier_bridge:
  dj_bridging:
    # Enable vector mode
    dj_ladder_target_mode: vector
    dj_genre_vector_source: smoothed

    # Enable IDF
    dj_genre_use_idf: true
    dj_genre_idf_power: 1.5
    dj_genre_idf_norm: max1

    # Enable coverage
    dj_genre_use_coverage: true
    dj_genre_coverage_weight: 0.15
```

**Recommended starting values (moderate impact):**
- `idf_power: 1.5` (amplifies rare genres moderately)
- `coverage_weight: 0.15` (similar to waypoint_cap=0.10, slightly stronger)
- `coverage_power: 2.0` (smooth decay, not too sharp)

---

## I) Edge Cases & Validation

### Edge Case 1: IDF division by zero
**Scenario:** Genre appears in 0 tracks (df[g] = 0)
**Mitigation:** Use `df + 1` smoothing in IDF formula
**Result:** `idf[g] = log((N+1) / 1) = log(N+1)` (maximum IDF)

### Edge Case 2: All genres equally common
**Scenario:** Uniform distribution, all df[g] ≈ N/G
**Result:** IDF ≈ constant, no differentiation
**Impact:** System behaves like IDF disabled (no harm)

### Edge Case 3: Missing genre metadata
**Scenario:** `X_genre_raw` or `X_genre_smoothed` is all zeros for anchor
**Mitigation:** Existing fallback in `_build_genre_targets()` (lines 1047-1076)
**Result:** Falls back to sonic-only mode (logs warning)

### Edge Case 4: Coverage with zero top genres
**Scenario:** `topk_A = []` (anchor has no genres above threshold)
**Mitigation:** `_compute_coverage()` returns 0.0
**Result:** No coverage bonus applied (graceful degradation)

### Edge Case 5: Vector mode with route_shape=linear
**Scenario:** User enables vector mode with linear route
**Result:** Already uses vector interpolation! IDF can still be applied
**Benefit:** IDF + coverage work even without ladder mode

---

## J) Performance Considerations

### Computational Cost

**IDF computation (once per run):**
```python
df = (X_genre_raw > 0).sum(axis=0)  # O(N*G) sparse operation
idf = np.log((N+1) / (df+1))        # O(G)
# Total: ~10ms for N=32k, G=1000
```

**IDF matrix weighting (once per run):**
```python
X_genre_idf = X_genre_smoothed * idf[np.newaxis, :]  # O(N*G)
X_genre_idf = normalize_rows(X_genre_idf)            # O(N*G)
# Total: ~50ms for N=32k, G=1000
```

**Coverage computation (per candidate per step):**
```python
for g_idx, _ in topk_genres:  # O(K) where K=8
    if candidate_genre_vec[g_idx] >= threshold:
        present_count += 1
# Total per candidate: ~10 array lookups = <1μs
# Per step (200 candidates): ~200μs
```

**Total overhead per segment:**
- IDF computation: amortized ~0 (once per run)
- Coverage computation: ~200μs * interior_length = ~2ms per segment
- **Negligible impact** (<1% of beam search time)

---

## K) Alternatives Considered

### Alternative 1: Weighted label selection
**Idea:** Instead of shortest path, weight path by IDF
**Problem:** Still collapses to single label, loses multi-genre signature
**Verdict:** Rejected - doesn't address onehot lossiness

### Alternative 2: Multiplicative coverage bonus
**Idea:** `score *= (1 + coverage_bonus)` instead of additive
**Problem:** Can dominate other signals, harder to tune
**Verdict:** Rejected - additive is more controllable

### Alternative 3: Hard coverage gate
**Idea:** Reject candidates with coverage < threshold
**Problem:** Too strict, may make segments infeasible
**Verdict:** Rejected - soft bonus is more flexible

### Alternative 4: Per-genre IDF in waypoint_delta
**Idea:** Weight each genre dimension separately in cosine similarity
**Problem:** Complex, breaks interpretation of cosine similarity
**Verdict:** Rejected - simpler to apply IDF to vectors once

---

## L) Future Enhancements (Out of Scope)

**1. Adaptive IDF power:**
   - Increase `idf_power` when anchors are far apart in genre space
   - Decrease when anchors are similar (less need for differentiation)

**2. Genre path diversity scoring:**
   - Prefer paths through varied genres instead of repetitive genres
   - E.g., `shoegaze → dreampop → slowcore` better than `shoegaze → indie rock → indie rock`

**3. User-specified waypoint hints:**
   - Allow user to force route through specific genres
   - E.g., `force_waypoints: ["dreampop", "slowcore"]`

**4. Precompute IDF in artifact:**
   - Save to NPZ offline, load at runtime
   - Avoids recomputation, but loses flexibility for power/norm tuning

**5. Coverage bonus for non-anchor constraints:**
   - Apply coverage bonus relative to seed track signature (not just piers)
   - Useful for maintaining seed artist "vibe" throughout playlist

---

## M) Rollout Plan

**Phase 1: Implement (this PR)**
- Add config keys, IDF helpers, vector mode, coverage bonus
- Default behavior unchanged (opt-in)

**Phase 2: Test with Slowdive scenario**
- Enable vector+IDF+coverage in user config
- Verify logs show expected behavior
- Confirm `chosen_from_genre_count` increases

**Phase 3: A/B comparison**
- Generate playlists with both modes
- Compare genre diversity, waypoint quality, user satisfaction
- Tune `idf_power`, `coverage_weight`, `coverage_power`

**Phase 4: Documentation**
- Update `config.example.yaml` with new keys
- Add troubleshooting guide for genre mismatch issues
- Document recommended settings for different use cases

---

## N) Summary

**Problem:** Hub genre collapse due to shortest-path label selection and onehot vectors

**Solution:**
1. **mode=vector:** Bypass label selection, use direct multi-genre interpolation
2. **IDF weighting:** Down-weight common genres (rock, indie rock) like stop-words
3. **Coverage bonus:** Reward matching anchor's expressive genres near segment boundaries

**Impact:**
- Waypoints preserve full genre signature (shoegaze+dreampop, not just indie rock)
- S3 genre candidates become relevant (Cocteau Twins instead of Ramones near Slowdive)
- `chosen_from_genre_count` increases significantly (3-5x improvement expected)

**Implementation scope:**
- ~250 lines in `pier_bridge_builder.py`
- ~30 lines in `segment_pool_builder.py`
- ~15 lines in `pier_bridge_diagnostics.py`
- Fully backward compatible (opt-in via config)

**Next:** Proceed to Phase 3 implementation.
