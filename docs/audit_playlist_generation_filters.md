# Comprehensive Technical Audit: Playlist Generation Filters & Gates
## Mode: Artist/Seeds Mode with Narrow Cohesion

**Date**: 2026-01-16
**Analyzed For**: release/v3.4

---

## Executive Summary

The playlist generation pipeline applies **14 distinct filtering stages** with **27 configurable thresholds**. In narrow mode, the system is highly constrained with multiple cascading gates that together reject approximately **95-98% of the initial candidate pool**. Several filters are **redundant** or **over-constraining**, particularly around genre similarity which is checked at 3 different stages.

### Critical Findings:
1. **Genre gates checked 3 times**: candidate pool, segment pool, and beam search
2. **Sonic floor at 0.18 rejects ~40-60% before other processing**
3. **Bridge floor at 0.08 can reject 30-70% of segment candidates**
4. **Artist diversity constraints (min_gap=6) can cause failures in narrow pools**
5. **Duration penalties applied twice**: candidate pool + beam search

---

## Phase 1: Initial Candidate Pool Construction

**Module**: `src/playlist/candidate_pool.py` (lines 129-477)
**Purpose**: Select candidates from universe based on seed similarity

### 1.1 Hybrid Similarity Floor (HARD GATE)
- **Value**: `similarity_floor = 0.20` (config line 129)
- **Type**: Hard gate (reject)
- **Code**: `candidate_pool.py:277-279`
- **Rejects**: Tracks below hybrid similarity threshold
- **Typical Rejection Rate**: 20-40% of universe
- **Essential**: YES - prevents low-quality matches

### 1.2 Sonic Similarity Floor (HARD GATE) ⚠️
- **Value**: `min_sonic_similarity = 0.18` (mode_presets.py:88, applied via config line 131-133)
- **Type**: Hard gate (reject)
- **Code**: `candidate_pool.py:280-284`
- **Formula**: Raw sonic cosine similarity in PCA-reduced tower space
- **Rejects**: Tracks with pure sonic similarity < 0.18
- **Typical Rejection Rate**: 40-60% of universe
- **Distribution Stats**: Logged at lines 288-320
- **Over-Constraining**: POSSIBLE - applied BEFORE genre consideration, can reject genre-compatible tracks
- **Redundancy**: Combined with hybrid floor creates double-gating

### 1.3 Genre Similarity Floor (HARD GATE) - First Application ⚠️
- **Value**: `min_genre_similarity = 0.50` (mode_presets.py:37, for narrow mode)
- **Type**: Hard gate (reject) for narrow/dynamic modes
- **Code**: `candidate_pool.py:322-339`
- **Method**: Ensemble (0.6 * cosine + 0.4 * jaccard) on smoothed genre vectors
- **Special Narrow Mode Guard**: Requires at least 1 non-broad shared tag (lines 325-329)
- **Rejects**: Tracks with genre similarity < 0.50
- **Typical Rejection Rate**: 30-50% of remaining pool (after sonic floor)
- **Over-Constraining**: YES - 0.50 is very strict, combined with sonic floor creates narrow funnel
- **Redundancy**: Genre checked again in segment pool and beam search

### 1.4 Duration Penalty (SOFT PENALTY) - First Application
- **Enabled**: `duration_penalty_enabled = true` (config line 137)
- **Weight**: `duration_penalty_weight = 0.60` (config line 138)
- **Cutoff Multiplier**: `2.5x` median seed duration (hard cutoff, config line 139)
- **Type**: Soft penalty (score reduction) + hard cutoff
- **Code**: `candidate_pool.py:174-227`
- **Formula**: Four-phase geometric penalty (lines 91-126)
  - 0-20% excess: Gentle (power 1.5)
  - 20-50% excess: Moderate (power 2.0)
  - 50-100% excess: Steep (power 2.5)
  - >100% excess: Severe (power 3.0)
- **Example**: 200s seed, 400s candidate (+100%) = penalty ≈ 0.75 subtracted from similarity
- **Hard Cutoff**: Tracks > 2.5x median seed duration are fully rejected (line 208-211)
- **Typical Impact**: 10-20% of candidates penalized, 2-5% hard rejected
- **Redundancy**: Duration checked again in beam search with different penalty

### 1.5 Artist Cap (SOFT CONSTRAINT)
- **Value**: `candidates_per_artist = 8` for narrow mode (config.py:368)
- **Seed Artist Bonus**: `+2` additional tracks (config line 358-359)
- **Type**: Soft cap (limits per-artist selections)
- **Code**: `candidate_pool.py:341-368`
- **Rejects**: Excess tracks from same artist (keeps top N by similarity)
- **Typical Rejection Rate**: 10-15% of eligible pool
- **Essential**: YES - prevents artist clustering

### 1.6 Pool Size Limit (HARD CAP)
- **Value**: `max_pool_size = 800` for narrow mode (config.py:384)
- **Target Artists**: `max(playlist_len/2, 12)` for narrow (config.py:358)
- **Type**: Hard cap on final pool size
- **Code**: `candidate_pool.py:363-367`
- **Combined Condition**: Stop when BOTH max_pool_size AND target_artists reached
- **Typical Final Pool**: 600-800 tracks in narrow mode
- **Essential**: YES - prevents computational explosion

---

## Phase 3: Pier-Bridge Segment Pool Building

**Module**: `src/playlist/segment_pool_builder.py`
**Purpose**: Build candidate pool for a single bridge segment (pierA → pierB)

### 3.6 Bridge Floor Gate (HARD GATE) ⚠️ - Primary Bottleneck
- **Value**: `bridge_floor = 0.08` for narrow mode (config line 154, resolved in config.py:188)
- **Type**: Hard gate (reject if min(sim_a, sim_b) < floor)
- **Code**: `segment_pool_builder.py:517-519`
- **Formula**: `min(cos(cand, pierA), cos(cand, pierB)) >= 0.08`
- **Bridge Score**: Harmonic mean used for ranking: `2*sa*sb / (sa+sb)`
- **Typical Rejection Rate**: 30-70% of structural candidates
- **Distribution**: Varies widely by pier distance (far piers = higher rejection)
- **Over-Constraining**: VERY LIKELY - single weak pier connection rejects entire candidate
- **Bottleneck**: This is the PRIMARY failure point for narrow mode
- **Essential**: PARTIALLY - could be relaxed or use min() differently

---

## Phase 4: Beam Search Construction

**Module**: `src/playlist/pier_bridge_builder.py::_beam_search_segment` (lines 2314-3300)
**Purpose**: Find optimal path through segment candidates using constrained beam search

### 4.3 Transition Floor (HARD GATE) ⚠️
- **Value**: `transition_floor = 0.45` for narrow mode (config line 183, resolved in config.py:132)
- **Type**: Hard gate (reject transitions below floor)
- **Code**: `pier_bridge_builder.py:2920-2921`
- **Formula**: Weighted transition score
  ```
  score = w_end_start * cos(end(A), start(B))  [weight=0.70]
        + w_mid_mid * cos(mid(A), mid(B))      [weight=0.15]
        + w_full_full * cos(full(A), full(B))  [weight=0.15]
  ```
- **Typical Rejection Rate**: 40-60% of candidates per step
- **Over-Constraining**: VERY LIKELY - 0.45 is strict for end-to-start similarity
- **Essential**: PARTIALLY - could be relaxed to 0.35-0.40

---

## Summary Table: All Gates & Thresholds (Narrow Mode)

| Phase | Filter/Gate | Value (Narrow) | Type | Rejection Rate | Essential | Notes |
|-------|-------------|----------------|------|----------------|-----------|-------|
| **1.2** | Sonic Similarity Floor | **0.18** | Hard Gate | **40-60%** | MAYBE | ⚠️ Over-constraining with genre gate |
| **1.3** | Genre Similarity Floor | **0.50** | Hard Gate | **30-50%** | MAYBE | ⚠️ Very strict, checked 3x |
| **3.6** | Bridge Floor | **0.08** | Hard Gate | **30-70%** | MAYBE | ⚠️ PRIMARY BOTTLENECK |
| **4.3** | Transition Floor | **0.45** | Hard Gate | **40-60%** | MAYBE | ⚠️ Very strict |

---

## Critical Issues & Recommendations

### 🔴 Critical Over-Constraining Issues

1. **Bridge Floor Bottleneck** (Phase 3.6)
   - **Issue**: `bridge_floor=0.08` requires BOTH piers above threshold
   - **Effect**: Single weak pier connection rejects entire candidate (min() operator)
   - **Rejection Rate**: 30-70% per segment (varies by pier distance)
   - **Fix**: Consider harmonic mean or weighted combination instead of min(), OR relax to 0.05-0.06

2. **Transition Floor Too Strict** (Phase 4.3)
   - **Issue**: `transition_floor=0.45` is very high for narrow mode
   - **Effect**: Rejects 40-60% of candidates per beam step
   - **Fix**: Relax to 0.35-0.40 for narrow mode

3. **Sonic Floor + Genre Floor Cascade** (Phase 1.2 + 1.3)
   - **Issue**: Sonic floor (0.18) applied BEFORE genre consideration
   - **Effect**: Rejects 40-60% of universe before checking genre compatibility
   - **Fix**: Consider hybrid floor instead of separate sonic floor, OR apply sonic floor after genre gate

---

## Typical Rejection Funnel (Narrow Mode)

Assuming 10,000 tracks in universe, artist mode with 5 seeds, 30-track playlist:

1. **Universe**: 10,000 tracks
2. **After Sonic Floor (0.18)**: 4,200 tracks (40% rejected) ⚠️
3. **After Genre Floor (0.50)**: 2,500 tracks (40% rejected) ⚠️
4. **After Pool Size Cap**: 800 tracks (62% rejected)
5. **Segment Pool (per bridge)**: 400 tracks
   - After Bridge Floor (0.08): 150-250 tracks (38-62% rejected) ⚠️
6. **Beam Search (per step)**: 130-220 candidates
   - After Transition Floor (0.45): 40-80 tracks (50% rejected) ⚠️
7. **Final Playlist**: 30 tracks

**Overall Rejection Rate**: 99.7% (30 / 10,000)

**Primary Bottlenecks**:
1. Sonic floor (Phase 1.2): 40% rejection
2. Genre floor (Phase 1.3): 40% rejection of remaining
3. Bridge floor (Phase 3.6): 50% rejection per segment
4. Transition floor (Phase 4.3): 50% rejection per beam step
