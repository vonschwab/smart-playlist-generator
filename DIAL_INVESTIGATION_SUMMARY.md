# Dial Investigation: Root Cause & Fix Summary

## TL;DR

**Finding:** 432 dial combinations collapse to only 3 unique outcomes—determined by sonic_weight alone.

**Root Causes:**
1. **min_genre_similarity** — NOT WIRED: Parameter accepted, logged, then ignored. No code paths use it.
2. **genre_method** — NOT WIRED: Same as above. DS pipeline doesn't use genre similarity computation.
3. **transition_strictness** — WORKING but NON-BINDING: Overrides apply correctly, but constraint is too weak to force different selections in 30-track playlists.

**Status:**
- ✅ sonic_weight/genre_weight: Fully functional
- ❌ min_genre_similarity: Non-functional (abandoned during implementation)
- ❌ genre_method: Non-functional (abandoned during implementation)
- ⚠️ transition_strictness: Functional but ineffective (constraint not binding)

---

## Root Cause Details

### 1. min_genre_similarity Parameter

**What happens:**
```python
# In generate_playlist_ds() at pipeline.py:75-76
def generate_playlist_ds(
    ...,
    min_genre_similarity: Optional[float] = None,  # ← Accepted
    ...
):
    if any([sonic_weight, ..., min_genre_similarity, ...]):
        logger.info("DS pipeline tuning dials: ..., min_genre_sim=%s, ...", min_genre_similarity)  # ← Logged

    # [~120 lines of code]

    embedding_model = build_hybrid_embedding(...)  # ← Never uses min_genre_similarity
    pool = build_candidate_pool(...)              # ← Never uses min_genre_similarity
    playlist = construct_playlist(...)            # ← Never uses min_genre_similarity
```

**Why it's not wired:**
- DS pipeline = simplified hybrid embedding path (no SimilarityCalculator)
- No genre similarity computation in candidate pool
- No genre-based filtering/gating in constructor
- Would need to add: similarity floor enforcement in `build_candidate_pool()` or gating in `construct_playlist()`

**Evidence from consolidated_results.csv:**
```
Seed: 89f67bd1, sonic_weight=0.55
- min_genre_sim=0.20: edge_genre_min=0.8067
- min_genre_sim=0.25: edge_genre_min=0.8067 (IDENTICAL)
- min_genre_sim=0.30: edge_genre_min=0.8067 (IDENTICAL)
All other metrics also identical → Parameter has zero effect
```

---

### 2. genre_method Parameter

**What happens:** Same as min_genre_similarity—accepted, logged, unused.

```python
# In generate_playlist_ds() at pipeline.py:76
def generate_playlist_ds(
    ...,
    genre_method: Optional[str] = None,  # ← Accepted
    ...
):
    # Logged at line 99
    # Never referenced again
```

**Why it's not wired:**
- DS pipeline doesn't compute genre similarity (uses only X_genre_smoothed in PCA)
- Genre method would only matter if there was similarity computation
- Would need SimilarityCalculator integration to switch between methods

**Evidence from consolidated_results.csv:**
```
Seed: 89f67bd1, sonic_weight=0.55
- genre_method=ensemble: edge_genre_min=0.8067
- genre_method=weighted_jaccard: edge_genre_min=0.8067 (IDENTICAL)
All metrics identical → Parameter has zero effect
```

---

### 3. transition_strictness Override

**Status:** ✓ Wiring works, ⚠️ Constraint not binding enough

**What happens:**
```python
# tune_dial_grid.py line 160-180
overrides = {}
if transition_strictness == "strictish":
    overrides["construct"] = {
        "transition_floor": 0.5,
        "hard_floor": True,
    }

# Passed to pipeline
result = generate_playlist_ds(
    ...,
    overrides=overrides,  # ← Applied correctly
)
```

**In pipeline.py:**
```python
cfg = _apply_overrides(cfg, overrides)  # ← Works ✓
# cfg.construct.transition_floor now = 0.5 instead of default 0.3
# Used in construct_playlist() ✓
```

**Why no visible effect:**
- Baseline mode (dynamic) has default transition_floor = 0.3
- Strictish raises it to 0.5 (modest increase)
- All transitions in successful playlists are ~0.998 (very high)
- Even stricter floor (0.8) would exclude almost nothing
- Constraint is non-binding: doesn't force different track selections

**Evidence:**
```
consolidated_results.csv for sonic_weight=0.55:
- transition_strictness=baseline: mean_transition=0.9988, below_floor_count=0
- transition_strictness=strictish: mean_transition=0.9988, below_floor_count=0
Playlists: IDENTICAL (constraint not binding)
```

---

## Comparison: What's Wired vs What's Not

| Dial | Wired | Evidence |
|------|-------|----------|
| sonic_weight | ✅ YES | Metrics change: 0.8679 → 0.8374 → 0.807 |
| genre_weight | ✅ YES | (inverse of sonic_weight, affects embedding) |
| min_genre_similarity | ❌ NO | All 12 combos per sonic_weight → identical metrics |
| genre_method | ❌ NO | All 12 combos per sonic_weight → identical metrics |
| transition_strictness | ⚠️ PARTIAL | Wired but constraint never violated → no effect |

---

## Data Verification

**Grouping consolidated_results.csv by (seed, sonic_weight):**

```python
# Seed 89f67bd1:
# - sonic_weight=0.55: 12 rows (all dial combos) → 1 unique edge_hybrid_mean value
# - sonic_weight=0.65: 12 rows (all dial combos) → 1 different edge_hybrid_mean value
# - sonic_weight=0.75: 12 rows (all dial combos) → 1 different edge_hybrid_mean value

# × 12 seeds = 36 rows (3 sonic_weights only)
# ÷ 432 total runs = 1 run per unique outcome
```

**Interpretation:** 432 runs compress to 36 outcomes (sonic_weight only), proving the other 11 dials (3×2×2 combos per seed) have zero effect.

---

## Recommended Fix Strategy

### Option A: Remove Non-Functional Parameters (MINIMAL)
**Pros:**
- Reduces API surface
- Honest about what DS pipeline supports
- Easiest to implement

**Cons:**
- Breaking change to function signature
- Tuning harness flags become dead code

**Implementation:** 2-3 lines in pipeline.py to delete parameters

---

### Option B: Implement Missing Wiring (MODERATE)
**For min_genre_similarity:**
```python
# In build_candidate_pool():
if min_genre_similarity is not None and mode == "dynamic":
    # Hard gate: exclude candidates with genre_sim < threshold
    # Requires genre similarity computation (not currently done)
    # Would need SimilarityCalculator integration (~100 lines)
```

**For genre_method:**
```python
# Would require changing:
# - hybrid.py to compute genre_sim (not just use X_genre_smoothed)
# - candidate_pool.py to apply genre_method parameter
# - Plus SimilarityCalculator integration
# Effort: ~200-300 lines
```

**Cons:** Significant effort, possible performance impact

---

### Option C: Document as Known Limitation (RECOMMENDED)
**Action:**
1. Add comment to `generate_playlist_ds()`:
   ```python
   # NOTE: min_genre_similarity, genre_method not implemented for DS pipeline
   # (DS uses simplified hybrid embedding, not full SimilarityCalculator)
   # These parameters accepted for future compatibility but currently ignored.
   ```

2. Mark tuning harness flags as deprecated:
   ```python
   parser.add_argument(
       "--min-genre-sim",
       default="0.20",
       help="(DEPRECATED: no effect in DS pipeline)"
   )
   ```

3. Update regression test to skip these assertions:
   ```python
   # Skip genre_method and min_genre_similarity tests (not implemented)
   ```

**Pros:**
- Honest about current capabilities
- No code changes needed
- Sets user expectations
- Room for future implementation

**Cons:** Users might expect these to work

---

### Option D: Make transition_strictness Binding (TECHNICAL)
To make transition_strictness produce visible effects:

**Current Issue:** Floor=0.3→0.5 is too lenient; transitions ~0.998 always pass

**Solution:**
```python
# In tune_dial_grid.py: Make "strictish" much stricter
transition_configs = {
    "baseline": {"transition_floor": 0.3},
    "strictish": {"transition_floor": 0.9},  # Was 0.5 - not binding
}
```

**Would require verification** that this doesn't break other modes.

---

## Verification Commands

### Run Force-Binding Test
This script confirms the root causes by running extreme configurations:

```bash
cd "C:\Users\Dylan\Desktop\PLAYLIST GENERATOR"
python scripts/force_bind_test.py
```

**Expected output:**
```
TEST 1: min_genre_similarity Effect
Config A (0.15) hash: abc123def456
Config B (0.60) hash: abc123def456  ← IDENTICAL
⚠️ RESULT: min_genre_similarity is NOT WIRED

TEST 2: genre_method Effect
Config A (ensemble) hash: abc123def456
Config B (weighted_jaccard) hash: abc123def456  ← IDENTICAL
⚠️ RESULT: genre_method is NOT WIRED

TEST 3: transition_strictness Effect
Config A (baseline) hash: abc123def456
Config B (strict floor=0.8) hash: abc123def456  ← IDENTICAL
⚠️ RESULT: transition_strictness constraint is NOT BINDING
```

### Minimal Reproducibility Test
Run a 2×2×2 dial grid with two seeds to see sonic_weight work while others don't:

```bash
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds 1c347ff04e65adf7923a9e3927ab667a,d41d8cd98f00b204e9800998ecf8427e \
    --mode dynamic \
    --sonic-weight 0.50,0.80 \
    --min-genre-sim 0.20 \
    --genre-method ensemble \
    --transition-strictness baseline \
    --output-dir diagnostics/verify_dials \
    --length 30
```

**Expected outcome:**
- Run 4 combos (2 seeds × 2 sonic_weights)
- `consolidated_results.csv` shows only 2 unique metric sets
- Metrics differ only between sonic_weight values

---

## Files Generated

1. **DIAL_ROUTING_ANALYSIS.md** - Detailed technical analysis
2. **DIAL_INVESTIGATION_SUMMARY.md** - This file
3. **scripts/force_bind_test.py** - Verification script

---

## Conclusion

The dial investigation revealed that the DS pipeline implementation is incomplete:
- **sonic_weight/genre_weight**: Fully functional ✅
- **min_genre_similarity/genre_method**: Abandoned during implementation ❌
- **transition_strictness**: Framework exists but constraint is non-binding ⚠️

**Recommended action:** Document as known limitation (Option C) and focus on features that are fully functional. Full implementation would require SimilarityCalculator integration (~300+ lines) and potential performance impact.

