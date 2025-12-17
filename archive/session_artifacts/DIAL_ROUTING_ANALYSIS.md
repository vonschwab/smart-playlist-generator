# Dial Routing Analysis: Why Only sonic_weight Works

## Executive Summary

**Finding:** Only `sonic_weight` affects playlist generation. The other dials (`min_genre_similarity`, `genre_method`, `transition_strictness`) are:
- Accepted by `generate_playlist_ds()`
- Logged for verification
- But **never applied to the actual generation pipeline**

**Root Cause:** The DS pipeline was designed as a simplified hybrid embedding path that doesn't implement the full similarity calculator machinery.

---

## Detailed Analysis

### 1. sonic_weight / genre_weight ✅ WORKING

**Route:**
```
tune_dial_grid.py
  → generate_playlist_ds(sonic_weight=0.55, genre_weight=0.45)
  → pipeline.py line 210-252: compute effective weights
  → build_hybrid_embedding(w_sonic=0.55, w_genre=0.45)
  → Affects candidate pool scoring via embedding distances
```

**Verification:** Different sonic_weights produce different metrics (edge_hybrid_mean, edge_genre_min, etc.)

**Code Flow:**
- `src/playlist/pipeline.py:210-240` - Effective weights computed and passed to `build_hybrid_embedding()`
- `src/similarity/hybrid.py` - Uses weights in PCA combination: `w_sonic * X_sonic_pca + w_genre * X_genre_pca`

---

### 2. min_genre_similarity ❌ NOT WIRED

**Issue:** Parameter is accepted (line 75) and logged (line 99) but **never used**.

**Where it SHOULD be applied:**
- Option A: In candidate pool filtering (like `similarity_floor` threshold)
- Option B: As a hard gate in scoring when `mode == "dynamic"`
- Currently: NOT implemented for DS pipeline

**Current code:**
```python
def generate_playlist_ds(
    ...,
    min_genre_similarity: Optional[float] = None,  # ← Accepted
    ...,
) -> DSPipelineResult:
    if any([sonic_weight, genre_weight, min_genre_similarity, genre_method]):
        logger.info(...)  # ← Logged
    # NOTHING HAPPENS WITH min_genre_similarity AFTER THIS
    # ↓
    # Direct jump to embedding building (line 210)
```

**The missing wiring:** Would need to pass `min_genre_similarity` to:
- `build_candidate_pool()` as a filter threshold, OR
- `construct_playlist()` as a gating function in transition scoring

**Evidence:** `consolidated_results.csv` shows identical metrics for min_genre_similarity=0.20, 0.25, 0.30

---

### 3. genre_method ❌ NOT WIRED

**Issue:** Parameter is accepted (line 76) and logged (line 99) but **never used**.

**Where it SHOULD be applied:**
- In genre similarity computation (normally via `SimilarityCalculator.genre_calc.calculate_similarity(method=...)`)
- DS pipeline doesn't use SimilarityCalculator—it uses pure hybrid embeddings
- No genre similarity computation happens in DS pipeline at all

**Current code:**
- Same as above: accepted, logged, ignored

**Why it was added:**
- Tuning harness designed to expose ALL available dials
- But DS pipeline doesn't support genre method selection (only uses X_genre_smoothed in PCA)

**Evidence:** `consolidated_results.csv` shows identical metrics for genre_method=ensemble, weighted_jaccard

---

### 4. transition_strictness ⚠️ PARTIALLY WORKING

**Status:** Overrides ARE applied, but effect is too subtle to detect in 30-track playlists.

**Code Flow:**
```
tune_dial_grid.py:run_single_dial()
  → creates overrides dict {"construct": {"transition_floor": 0.6, ...}}
  → passes to generate_playlist_ds(overrides=...)
  → pipeline.py:_apply_overrides() merges overrides into cfg
  → construct_playlist() uses cfg.construct.transition_floor
  → WORKS ✓
```

**Why no visible effect:**
- Default baseline `transition_floor=0.3` (from dynamic mode)
- Raising to `strictish 0.6` is a significant change but:
  - The existing candidates often satisfy both floors
  - Repair phase can recover candidates that drop below higher floor
  - 30-track playlists have some slack—difficult to force different selections

**Evidence:**
- `consolidated_results.csv` shows `below_floor_count=0` for both baseline and strictish
- Override IS applied in config (verified in test)
- But tracklists don't differ (constraint not binding enough to force different selections)

---

## Root Cause Summary

| Dial | Status | Issue | Location in Code |
|------|--------|-------|------------------|
| sonic_weight | ✅ | Fully implemented | `src/playlist/pipeline.py:210-252` |
| genre_weight | ✅ | Fully implemented | `src/playlist/pipeline.py:210-252` |
| min_genre_similarity | ❌ | Accepted but unused | Parameter never referenced after logging |
| genre_method | ❌ | Accepted but unused | Parameter never referenced after logging |
| transition_strictness | ⚠️ | Overrides work, effect too subtle | Constraints apply but don't force different selections |

---

## Why This Happened

1. **Design:** DS pipeline is a simplified hybrid embedding path
   - Doesn't use full `SimilarityCalculator` machinery
   - No genre similarity computation (only X_genre_smoothed in PCA)
   - Parameters added to `generate_playlist_ds()` signature for future flexibility

2. **Implementation Gap:** Parameters accepted but routing never completed
   - sonic_weight/genre_weight: routing COMPLETED
   - min_genre_similarity: routing ABANDONED (would need similarity gating)
   - genre_method: routing ABANDONED (would need similarity calculator integration)

3. **Transition Strictness:** Working but non-binding
   - Constraint framework works correctly
   - But effect is too subtle for playlist-level differences in 30-track case

---

## Data Verification

From `consolidated_results.csv` grouping by seed + sonic_weight:

**Seed: 89f67bd1, sonic_weight=0.55:**
- All 12 dial combos (3 × min_genre_sim, 2 × genre_method, 2 × transition_strictness)
- ALL metrics identical: edge_hybrid_mean=0.8679, edge_genre_min=0.8067, unique_artists=20
- ✓ Confirms: other dials have zero effect

**Seed: 89f67bd1, sonic_weight=0.65:**
- All 12 dial combos again
- Identical metrics: edge_hybrid_mean=0.8374 (different from 0.55 case)
- ✓ Confirms: only sonic_weight changes results

**Seed: 89f67bd1, sonic_weight=0.75:**
- All 12 dial combos
- Identical metrics: edge_hybrid_mean=0.807 (different again)
- ✓ Confirms: sonic_weight is the ONLY effective variable

---

## Fix Strategy

### Option 1: Remove Non-Functional Parameters (Minimal)
- Delete `min_genre_similarity` and `genre_method` from `generate_playlist_ds()` signature
- They add clutter without effect
- Tuning harness still has the flags, just won't pass them

### Option 2: Implement Missing Wiring (Moderate)
- Route `min_genre_similarity` as hard gate in dynamic mode
- Implement genre method switching (would require SimilarityCalculator integration)
- More complex, may have performance implications

### Option 3: Hybrid Approach (Recommended)
- Keep parameters for future-proofing (signed APIs shouldn't shrink)
- Add documentation that they don't affect DS pipeline
- Focus on making transition_strictness more impactful if needed

### Option 4: Force-Binding Experiment (Testing)
- Create a test that verifies these parameters SHOULD affect output
- If they don't, document it as a known limitation
- Provides evidence for stakeholders

