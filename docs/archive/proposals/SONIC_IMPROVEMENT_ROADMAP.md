# Sonic Improvement Roadmap: Phases 2-4

**Created**: 2025-12-16
**Status**: Phase 2 (Beat-Sync) in progress
**Next Phases**: Phase C (Transitions) → Phase D (Rebalance Dynamic Mode)

---

## Overview: The Complete Fix

### Root Problem (Identified ✅)
- Sonic similarity completely non-informative (flatness 0.039, gap 0.019)
- BPM dimension 30,000,000x larger than other features
- Result: All songs appear nearly identical (0.9998 similarity)
- Impact: Playlists have poor transitions, strange genre combinations

### Solution Strategy (Phases 2-4)
| Phase | Goal | Timeline | Status |
|-------|------|----------|--------|
| Phase 2 | Make sonic similarity informative | 6-8 hours | 🔄 IN PROGRESS |
| Phase 3 | Improve transition quality | 2-3 hours | ⏳ PENDING |
| Phase 4 | Rebalance dynamic mode (genre gate) | 1-2 hours | ⏳ PENDING |

---

## PHASE 2: BEAT-SYNC FEATURES (Currently Running)

### Status: 🔄 IN PROGRESS

**What's happening**: Full artifact rebuild with beat-synchronized feature extraction

### Command Running
```bash
python scripts/update_sonic.py --beat-sync --workers 8
```

### Expected Timeline
- Duration: 6-8 hours for ~34,100 tracks
- Speed: ~1.1 tracks/second
- Success rate: 95-99% (1-5% automatic fallback)
- Progress output: Every 10 tracks
- Log file: `sonic_analysis.log`

### What Gets Built
```
Before (Windowed - Broken):
- 27 dimensions
- Similarity gap: 0.0414
- Flatness: 0.039 (avg)
- Result: 0/12 seeds pass validation

After (Beat-Sync - Fixed):
- 71 dimensions (+163%)
- Similarity gap: 0.1800 (4.34x improvement)
- Flatness: 0.52 (projected)
- Result: 7+/12 seeds expected to pass
```

### Phase 2 Completion Checklist
- [ ] Sonic analysis completes (~34,100 tracks analyzed)
- [ ] Log shows: "Analysis Complete! Total analyzed: 34100"
- [ ] Database has updated sonic_features with beat-sync metadata
- [ ] Next: Rebuild artifact matrices

---

## PHASE 2.5: REBUILD ARTIFACT MATRICES

### When to Run
After `update_sonic.py` completes successfully

### Command
```bash
python scripts/analyze_library.py --stages sonic,artifacts --beat-sync
```

### Expected Output
```
Rebuilt artifact: data_matrices.npz
- X_sonic: (34100, 71) with beat-sync features
- Includes: start/mid/end segments for future transition work
Duration: 30-60 minutes
```

### Completion Checklist
- [ ] Script completes without errors
- [ ] Output confirms: "X_sonic=(34100, 71)"
- [ ] New data_matrices.npz created
- [ ] Next: Full 12-seed validation

---

## PHASE 2.6: FULL VALIDATION (Prove Improvements)

### When to Run
After artifact rebuild completes

### Commands

**Test all 12 validation seeds** (using `scripts/sonic_validation_suite.py`):

```bash
# You'll need to get the 12 seed track IDs from the original validation
# Or get them from database:
sqlite3 data/metadata.db "SELECT track_id, artist, title FROM tracks LIMIT 12;"

# Run validation for each seed (example with one seed):
python scripts/sonic_validation_suite.py \
  --artifact data_matrices.npz \
  --seed-track-id <track_id_1> \
  --k 30 \
  --output-dir diagnostics/sonic_validation/phase2_results/

# Repeat for all 12 seeds
```

### Expected Results
```
Phase 2 Validation Results (Beat-Sync Features):

Expected improvements from baseline (0.039 flatness, 0.019 gap):
- Flatness: 0.039 → ~0.52 (PASS ✅, need ≥0.5)
- TopK Gap: 0.019 → ~0.19 (PASS ✅, need ≥0.15)
- Pass Rate: 0/12 → 7+/12 (PASS ✅, need 60%+)

Example output for each seed:
  Seed 1: Flatness=0.52, Gap=0.18 → PASS ✅
  Seed 2: Flatness=0.48, Gap=0.14 → PARTIAL (close)
  ...
  Seed 7: Flatness=0.55, Gap=0.20 → PASS ✅
  ... (7+ seeds should pass)
```

### Success Criteria: Phase 2 Complete ✅
- [ ] All 12 seeds tested
- [ ] CSV generated: `diagnostics/sonic_validation/phase2_results/sonic_validation_metrics.csv`
- [ ] Report generated: `diagnostics/sonic_validation/phase2_results/sonic_validation_report.md`
- [ ] Flatness ≥0.5 on 7+ seeds
- [ ] TopK Gap ≥0.15 on 7+ seeds
- [ ] If ANY metric fails: Debug and re-run

### If Phase 2 Passes
→ Proceed to Phase 3 (Transitions)

### If Phase 2 Fails
→ Review results, check fallback rates in database:
```bash
sqlite3 data/metadata.db "SELECT sonic_source, COUNT(*) FROM tracks GROUP BY sonic_source;"
```

---

## PHASE 3: TRANSITION SCORING IMPROVEMENTS

### Goal
Use segment-based features (start/end) to improve transition quality

### Current Problem
- Transitions use full X_sonic (loses flow information)
- A song's "ending" might be very different from its "beginning"
- Using average features loses this nuance

### Solution Design
Use X_sonic_start and X_sonic_end to compute end→start transitions:
```
Song A ending features (X_sonic_end_A)
    ↓ (compute cosine similarity)
Song B starting features (X_sonic_start_B)
    ↓
Better transition score!
```

### Phase 3 Implementation Steps

#### Step 3.1: Add Transition Helpers
**File**: `src/playlist/constructor.py`

**Add function** (after existing imports):
```python
def _compute_segment_transition(X_end, X_start, prev_idx, cand_indices):
    """
    Compute transition scores using end/start segments.

    Args:
        X_end: Shape (n_tracks, n_dims) - end-of-track features
        X_start: Shape (n_tracks, n_dims) - start-of-track features
        prev_idx: Index of previous track
        cand_indices: Indices of candidate tracks

    Returns:
        Transition scores (higher = better flow)
    """
    # Normalize vectors
    prev_vec = X_end[prev_idx]
    prev_norm = np.linalg.norm(prev_vec) + 1e-12

    cand_mat = X_start[cand_indices]
    cand_norms = np.linalg.norm(cand_mat, axis=1, keepdims=True) + 1e-12

    # Cosine similarity
    dots = cand_mat @ prev_vec
    scores = dots / (cand_norms.squeeze() * prev_norm)

    return scores
```

#### Step 3.2: Load Segment Features in Constructor
**File**: `src/playlist/constructor.py`

**Modify** `PlaylistConstructor.__init__()`:
```python
def __init__(self, bundle, ...):
    # ... existing code ...

    # Add segment features if available
    self.X_sonic_start = bundle.get('X_sonic_start')
    self.X_sonic_end = bundle.get('X_sonic_end')
    self.use_segment_transitions = (self.X_sonic_start is not None
                                    and self.X_sonic_end is not None)

    if self.use_segment_transitions:
        logger.info("Segment-based transitions enabled")
    else:
        logger.warning("Segment features not found, using full-song features for transitions")
```

#### Step 3.3: Use Segment Transitions in Candidate Scoring
**File**: `src/playlist/constructor.py`

**Modify** `_score_candidates()` method:
```python
def _score_candidates(self, prev_idx, cand_indices, ...):
    # ... existing scoring code ...

    # Compute transition score
    if self.use_segment_transitions:
        # Use end→start segments for better flow detection
        transition_scores = _compute_segment_transition(
            self.X_sonic_end,
            self.X_sonic_start,
            prev_idx,
            cand_indices
        )
    else:
        # Fallback: use full-song features
        X_norm = self._normalize_vectors(self.X_sonic)
        prev_vec = X_norm[prev_idx]
        cand_mat = X_norm[cand_indices]
        transition_scores = cand_mat @ prev_vec

    # ... rest of scoring logic ...
```

#### Step 3.4: Add Diagnostic Counters
**File**: `src/playlist/constructor.py`

**Modify** `construct_playlist()` to track transitions:
```python
def construct_playlist(self, ...):
    # ... existing code ...

    stats = {
        'total_candidates_evaluated': 0,
        'below_floor_count': 0,
        'transition_scores': [],
        'segment_transitions_used': self.use_segment_transitions,
    }

    # During candidate evaluation:
    for candidate in remaining:
        stats['total_candidates_evaluated'] += 1
        transition_score = _get_transition_score(prev_idx, candidate)
        stats['transition_scores'].append(transition_score)

        if transition_score < transition_floor:
            stats['below_floor_count'] += 1
            # ... gate logic ...

    # After playlist construction:
    if stats['transition_scores']:
        stats['transition_scores_min'] = min(stats['transition_scores'])
        stats['transition_scores_p10'] = np.percentile(stats['transition_scores'], 10)
        stats['transition_scores_p50'] = np.percentile(stats['transition_scores'], 50)

    return PlaylistResult(..., stats=stats)
```

#### Step 3.5: Test Segment Transitions
**File**: `scripts/tune_dial_grid.py`

**Command to test**:
```bash
python scripts/tune_dial_grid.py \
  --artifact data_matrices.npz \
  --seeds <seed_1>,<seed_2>,<seed_3> \
  --mode dynamic \
  --transition-strictness strictish \
  --output-dir diagnostics/transition_test/

# Check output for stats:
# - transition_scores_min (should be near transition_floor)
# - segment_transitions_used (should be True)
```

#### Step 3.6: Compare Before/After
**File**: Create `scripts/compare_transitions.py`

```python
"""Compare transition scores with and without segment features."""

import json
from pathlib import Path

# Load results from two runs:
# 1. Without segment features (old way)
# 2. With segment features (new way)

old_results = json.load(open('diagnostics/transition_test/old/results.json'))
new_results = json.load(open('diagnostics/transition_test/new/results.json'))

print("Transition Score Comparison:")
print(f"  Old (full-song): min={old_results['transition_scores_min']:.4f}, "
      f"p10={old_results['transition_scores_p10']:.4f}")
print(f"  New (segments):  min={new_results['transition_scores_min']:.4f}, "
      f"p10={new_results['transition_scores_p10']:.4f}")
print(f"  Improvement: {(new_results['transition_scores_min'] / old_results['transition_scores_min']):.2f}x better minimum")
```

### Phase 3 Success Criteria ✅
- [ ] Segment transition function implemented and tested
- [ ] Constructor loads X_sonic_start and X_sonic_end
- [ ] Diagnostic counters show segment transitions are being used
- [ ] `transition_scores_min` increases with segment approach
- [ ] Playlists still generate without errors
- [ ] Manual listening: Transitions feel smoother

### Phase 3 Acceptance Test
```bash
# Generate a test playlist with segment transitions
python scripts/playlist_generator.py \
  --seed-track "Fela Kuti" \
  --mode dynamic \
  --count 50 \
  --output test_playlist_segments.m3u

# Listen to the playlist and assess:
# - Do consecutive tracks flow well together?
# - Any jarring jumps in tempo/energy?
# - Better than before segment transitions?
```

---

## PHASE 4: REBALANCE DYNAMIC MODE (Genre as Hard Gate)

### Goal
Make genre a hard gate (filter), let sonic drive ranking and transitions

### Current Problem
- Dynamic mode: 60% sonic + 40% genre (hybrid embedding)
- Genre weight (40%) in hybrid is too high
- Genre dominates ranking, dilutes sonic quality

### Solution
- Genre: Hard gate only (minimum similarity threshold)
- Sonic: Drives all ranking and transitions within gated pool
- Result: Better genre coherence + better sonic flow

### Phase 4 Implementation Steps

#### Step 4.1: Create Genre-Gated Candidate Pool
**File**: `src/playlist/candidate_pool.py`

**Add new function**:
```python
def build_candidate_pool_genre_gated(
    bundle,
    seed_idx,
    min_genre_similarity=0.30,
    pool_size=500,
):
    """
    Genre as hard gate, sonic-only ranking within gated pool.

    Args:
        bundle: Artifact with X_sonic and X_genre_smoothed
        seed_idx: Index of seed track
        min_genre_similarity: Minimum genre similarity to include (0-1)
        pool_size: Number of candidates to return

    Returns:
        Indices of candidate tracks (ranked by sonic similarity only)
    """
    # Step 1: Compute genre similarity for all tracks
    seed_genre = bundle['X_genre_smoothed'][seed_idx]
    genre_sim_all = compute_genre_similarity(
        seed_genre,
        bundle['X_genre_smoothed'],
        method='ensemble'  # cosine + jaccard blend
    )

    # Step 2: Hard gate by genre similarity
    genre_eligible_idx = np.where(genre_sim_all >= min_genre_similarity)[0]

    if len(genre_eligible_idx) == 0:
        # Fallback: use top 1000 by genre if gate is too strict
        logger.warning(f"Genre gate too strict (threshold={min_genre_similarity}), using top genres")
        genre_eligible_idx = np.argsort(genre_sim_all)[::-1][:1000]

    # Step 3: Rank eligible tracks by SONIC similarity only
    X_sonic_eligible = bundle['X_sonic'][genre_eligible_idx]
    X_sonic_norm = X_sonic_eligible / (np.linalg.norm(X_sonic_eligible, axis=1, keepdims=True) + 1e-12)

    seed_sonic_norm = bundle['X_sonic'][seed_idx] / (np.linalg.norm(bundle['X_sonic'][seed_idx]) + 1e-12)
    sonic_sim = X_sonic_norm @ seed_sonic_norm

    # Step 4: Return top candidates by sonic similarity
    topk_sonic_idx = np.argsort(sonic_sim)[::-1][:pool_size]
    candidate_indices = genre_eligible_idx[topk_sonic_idx]

    return {
        'candidate_indices': candidate_indices,
        'sonic_sim': sonic_sim[topk_sonic_idx],
        'genre_sim': genre_sim_all[candidate_indices],
        'gate_count': len(genre_eligible_idx),
        'mode': 'genre_gated'
    }
```

#### Step 4.2: Add CLI Flag for Genre-Gated Mode
**File**: `scripts/tune_dial_grid.py`

**Add argument**:
```python
parser.add_argument(
    '--genre-as-gate',
    action='store_true',
    help='Use genre as hard gate only, sonic drives ranking'
)
```

#### Step 4.3: Use Genre-Gated in Constructor
**File**: `src/playlist/constructor.py`

**Modify candidate pool selection**:
```python
def _build_candidate_pool(self, seed_idx, ...):
    """Build candidate pool based on mode."""

    if self.genre_as_gate:
        # New: Genre gate + sonic ranking
        pool = build_candidate_pool_genre_gated(
            bundle=self.bundle,
            seed_idx=seed_idx,
            min_genre_similarity=self.min_genre_similarity,
            pool_size=self.pool_size
        )
        return pool
    else:
        # Old: Hybrid embedding (60% sonic + 40% genre)
        pool = build_candidate_pool_hybrid(
            bundle=self.bundle,
            seed_idx=seed_idx,
            w_sonic=0.60,
            w_genre=0.40,
            pool_size=self.pool_size
        )
        return pool
```

#### Step 4.4: Increase Transition Floor
**File**: `src/playlist/config.py`

**Current values**:
```python
STRICTNESS_LEVELS = {
    'loose': {'transition_floor': 0.25, 'hard_floor': False},
    'relaxed': {'transition_floor': 0.30, 'hard_floor': True},
    'balanced': {'transition_floor': 0.35, 'hard_floor': True},
    'strictish': {'transition_floor': 0.35, 'hard_floor': True},
    'strict': {'transition_floor': 0.40, 'hard_floor': True},
}
```

**Update to**:
```python
STRICTNESS_LEVELS = {
    'loose': {'transition_floor': 0.25, 'hard_floor': False},
    'relaxed': {'transition_floor': 0.30, 'hard_floor': True},
    'balanced': {'transition_floor': 0.35, 'hard_floor': True},
    'strictish': {'transition_floor': 0.45, 'hard_floor': True},  # +0.10
    'strict': {'transition_floor': 0.55, 'hard_floor': True},     # +0.15
}
```

#### Step 4.5: Rebalance Alpha (Seed Weight) in Dynamic Mode
**File**: `src/playlist/config.py`

**Current dynamic mode**:
```python
'dynamic': {
    'alpha_start': 0.65,  # High seed weight early
    'alpha_mid': 0.45,    # Low seed weight middle
    'alpha_end': 0.60,    # Medium seed weight end
    'beta': 0.45,         # Low transition weight
}
```

**Update to**:
```python
'dynamic': {
    'alpha_start': 0.50,  # Reduced seed weight (more exploration)
    'alpha_mid': 0.40,    # Keep low (transitions drive)
    'alpha_end': 0.55,    # Reduced end weight
    'beta': 0.60,         # Increased transition weight (better flow)
}
```

#### Step 4.6: A/B Test: Genre-Gated vs Hybrid
**File**: Create test script `scripts/ab_test_genre_gated.py`

```bash
# Test 1: Current hybrid mode (baseline)
python scripts/tune_dial_grid.py \
  --artifact data_matrices.npz \
  --seeds <seed_1>,<seed_2>,<seed_3> \
  --mode dynamic \
  --sonic-weight 0.60 \
  --genre-weight 0.40 \
  --transition-strictness strictish \
  --output-dir diagnostics/ab_test/hybrid/

# Test 2: New genre-gated mode
python scripts/tune_dial_grid.py \
  --artifact data_matrices.npz \
  --seeds <seed_1>,<seed_2>,<seed_3> \
  --mode dynamic \
  --genre-as-gate \
  --min-genre-similarity 0.30 \
  --transition-strictness strictish \
  --output-dir diagnostics/ab_test/genre_gated/

# Compare results
python scripts/compare_results.py \
  --baseline diagnostics/ab_test/hybrid/ \
  --variant diagnostics/ab_test/genre_gated/
```

#### Step 4.7: Manual Listening Test
**Generate playlists** and listen:

```bash
# Hybrid mode (current)
python scripts/playlist_generator.py \
  --seed-track "Fela Kuti - Kalakuta Show" \
  --mode dynamic \
  --count 50 \
  --output test_hybrid.m3u

# Genre-gated mode (new)
python scripts/playlist_generator.py \
  --seed-track "Fela Kuti - Kalakuta Show" \
  --mode dynamic \
  --genre-as-gate \
  --count 50 \
  --output test_genre_gated.m3u
```

**Assessment criteria**:
- [ ] No "strange combos" (indie + afrobeat mixing)
- [ ] Genre stays coherent (similar artists/eras)
- [ ] Sonic flow smooth (no jarring transitions)
- [ ] Playlist feels more intentional/curated
- [ ] Prefer genre-gated over hybrid? (80%+ yes)

### Phase 4 Success Criteria ✅
- [ ] `build_candidate_pool_genre_gated()` implemented
- [ ] CLI flag `--genre-as-gate` working
- [ ] Constructor uses genre-gated when flag set
- [ ] Transition floors increased (0.35→0.45, 0.40→0.55)
- [ ] Alpha weights rebalanced (reduced seed weight, higher beta)
- [ ] A/B test shows genre-gated is better on metrics
- [ ] Manual listening confirms better playlists
- [ ] No "strange combos" in genre-gated mode

### Phase 4 Acceptance Test
```bash
# Generate 5 playlists with different seeds
for seed in "Fela Kuti" "Miles Davis" "Kendrick Lamar" "Björk" "Radiohead"; do
  python scripts/playlist_generator.py \
    --seed-artist "$seed" \
    --mode dynamic \
    --genre-as-gate \
    --count 50 \
    --output test_${seed// /_}.m3u
done

# Listen to all 5 and count issues:
# - Strange genre combinations: Should be 0
# - Jarring transitions: Should be < 2 per playlist
# - Overall coherence: Should feel "intentional"
```

---

## MASTER CHECKLIST

### Phase 2: Beat-Sync Features
- [ ] Run `python scripts/update_sonic.py --beat-sync --workers 8`
- [ ] Monitor progress in `sonic_analysis.log`
- [ ] Wait for completion: "Analysis Complete!"
- [ ] Run `python scripts/analyze_library.py --stages sonic,artifacts --beat-sync`
- [ ] Run 12-seed validation with `sonic_validation_suite.py`
- [ ] Verify: Flatness ≥0.5, Gap ≥0.15, 7+/12 seeds pass
- [ ] **Phase 2 Done** ✅

### Phase 3: Transition Scoring
- [ ] Add `_compute_segment_transition()` function to `constructor.py`
- [ ] Load X_sonic_start/X_sonic_end in constructor init
- [ ] Modify `_score_candidates()` to use segments
- [ ] Add diagnostic counters for transition tracking
- [ ] Test with `tune_dial_grid.py` on 3-5 seeds
- [ ] Compare transition scores before/after
- [ ] Generate test playlist and listen
- [ ] Verify: Transitions feel smoother
- [ ] **Phase 3 Done** ✅

### Phase 4: Rebalance Dynamic Mode
- [ ] Add `build_candidate_pool_genre_gated()` function
- [ ] Add `--genre-as-gate` CLI flag
- [ ] Modify constructor to use genre-gated mode
- [ ] Increase transition floors (0.35→0.45, 0.40→0.55)
- [ ] Rebalance alpha weights (reduce seed weight)
- [ ] Increase beta (0.45→0.60)
- [ ] Run A/B tests (hybrid vs genre-gated)
- [ ] Manual listening: 5 different artists
- [ ] Verify: No strange genre combos, better flow
- [ ] **Phase 4 Done** ✅

---

## Timeline Estimate

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 2 | Sonic analysis rebuild | 6-8 hours | 🔄 RUNNING |
| 2.5 | Artifact rebuild | 1 hour | ⏳ After 2 |
| 2.6 | Full validation | 30 min | ⏳ After 2.5 |
| 3 | Segment transitions | 2-3 hours | ⏳ After 2.6 |
| 4 | Genre gating + rebalance | 2-3 hours | ⏳ After 3 |
| **Total** | **End-to-end** | **~14-16 hours** | **Sequential** |

---

## Commands Quick Reference

### Phase 2 (Currently Running)
```bash
# Main rebuild
python scripts/update_sonic.py --beat-sync --workers 8

# Artifact rebuild (after Phase 2)
python scripts/analyze_library.py --stages sonic,artifacts --beat-sync

# Validation (after artifact)
python scripts/sonic_validation_suite.py --artifact data_matrices.npz --seed-track-id <id> --k 30
```

### Phase 3 (Implementation)
```bash
# Test transitions
python scripts/tune_dial_grid.py --artifact data_matrices.npz --seeds <id1>,<id2> --mode dynamic --transition-strictness strictish

# Generate test playlist
python scripts/playlist_generator.py --seed-track "Artist - Song" --mode dynamic --count 50
```

### Phase 4 (Testing)
```bash
# Test genre-gated mode
python scripts/tune_dial_grid.py --artifact data_matrices.npz --seeds <id1>,<id2> --mode dynamic --genre-as-gate --min-genre-similarity 0.30

# A/B test both modes
python scripts/compare_results.py --baseline diagnostics/ab_test/hybrid/ --variant diagnostics/ab_test/genre_gated/
```

---

## Notes & Caveats

### About Fallback Rate
- Beat-sync has ~1-5% automatic fallback to windowed
- If fallback > 10%, investigate audio file issues
- Check: `SELECT COUNT(*) FROM tracks WHERE sonic_source='windowed'`

### About Phase 3 Implementation
- Segment features (X_sonic_start, X_sonic_end) already in artifact
- Just need to wire them into constructor for transitions
- Should improve transition quality by ~2-3x

### About Phase 4 Rebalancing
- Genre gate with 0.30 threshold keeps ~60-70% of tracks
- If gate excludes too many, reduce threshold to 0.25
- Alpha/beta changes are conservative; can adjust if needed

### About Validation Timing
- After each phase, wait for full validation before proceeding
- If any phase fails, debug before moving to next
- Success criteria are concrete (metrics + listening test)

---

## Rollback Plan

If any phase has issues:

### Phase 2 Rollback
```bash
# Revert to windowed features
sqlite3 data/metadata.db "UPDATE tracks SET sonic_features=NULL WHERE sonic_source='beat_sync';"
# Then rebuild with windowed: python scripts/analyze_library.py --stages sonic,artifacts
```

### Phase 3 Rollback
```bash
# Revert to full-song transitions
# (Just don't use segment features in constructor)
# Remove changes to _score_candidates()
```

### Phase 4 Rollback
```bash
# Revert to hybrid mode
# Set --genre-as-gate=False or remove flag usage
# Revert alpha/beta to original values
```

---

## Success Indicators

### After Phase 2 ✅
- "Sonic similarity is now informative"
- Flatness and gap metrics meet targets
- Test playlist sounds coherent

### After Phase 3 ✅
- "Transitions are smoother"
- `transition_scores_min` increased
- No more jarring jumps between tracks

### After Phase 4 ✅
- "Playlists feel more intentional"
- Genre stays coherent (no strange combos)
- Sonic flow within genre is excellent
- Listeners prefer new playlists over old

---

## Documentation References

- `SONIC_FIX_DEPLOYMENT_READINESS.md` - Current status and Phase 2
- `SONIC_FEATURE_REDESIGN_PLAN.md` - Technical design details
- `SONIC_FIX_IMPLEMENTATION_GUIDE.md` - Implementation steps
- `SONIC_FIX_VALIDATION_RESULTS.md` - Phase 2 test results

---

**Next Step**: Monitor Phase 2 rebuild. Check back when it completes to proceed with Phase 2.5 (artifact rebuild).

