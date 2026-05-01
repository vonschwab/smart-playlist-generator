# Filter Optimization Plan: Rebalancing Mode Thresholds

**Date**: 2026-01-16
**Based on**: `audit_playlist_generation_filters.md`
**Branch**: release/v3.4

---

## Executive Summary

The current narrow mode is **over-constrained** with multiple cascading hard gates creating 95-98% rejection rates. This plan proposes careful threshold recalibration to align each mode with its intended purpose while maintaining the system's careful balance.

**Key Philosophy**: Strict mode should be the only mode that flirts with over-filtering. Narrow should allow justified exploration. Dynamic should enable fun inclusions. Discover should be open with big arcs between piers.

**Risk Assessment**: This plan prioritizes conservative, phased changes with validation at each step. High-risk changes are clearly marked and should be implemented last.

---

## Mode Philosophy & Intended Behavior

### Current State vs. Intended Purpose

| Mode | Current Behavior | Intended Purpose | Gap Analysis |
|------|------------------|------------------|--------------|
| **Strict** | Not explicitly defined (inherits narrow) | Ultra-cohesive, can over-filter | ✅ No strict mode exists yet |
| **Narrow** | 95-98% rejection, frequent failures | Cohesive but allows justified exploration | ⚠️ Too restrictive - should be 90-93% rejection |
| **Dynamic** | Moderately open (85-90% rejection) | Fun inclusions, balanced exploration | ⚠️ Slightly too conservative |
| **Discover** | Open (75-85% rejection) | Very open, big arcs between piers | ✅ Close to target |

---

## Critical Bottlenecks Identified

From the audit, four primary bottlenecks are over-constraining narrow mode:

1. **Bridge Floor (0.08)**: 30-70% rejection per segment - PRIMARY BOTTLENECK
2. **Transition Floor (0.45)**: 40-60% rejection per beam step
3. **Sonic Floor (0.18)**: 40-60% rejection before genre consideration
4. **Genre Floor (0.50)**: 30-50% rejection after sonic

**Cumulative Effect**: These cascade to create impossibly narrow funnels where even 300+ candidates cannot produce valid paths.

---

## Proposed Threshold Recalibrations

### Phase 1: Add Explicit Strict Mode (Low Risk)

**Rationale**: Currently, there's no explicit "strict" mode - narrow is effectively acting as strict. Adding explicit strict mode creates headroom to relax narrow.

**Changes**:

```yaml
# src/playlist/mode_presets.py
MODE_PRESETS = {
    "strict": {
        "weight": 0.75,
        "min_sonic_similarity": 0.20,  # Keep current narrow sonic floor
        "min_genre_similarity": 0.45,
        "min_genre_similarity_narrow": 0.55,  # Keep current narrow genre floor
    },
    "narrow": {
        # Values will be relaxed in Phase 2
    }
}

# config.yaml - pier_bridge section
pier_bridge:
  bridge_floor_strict: 0.10      # Keep current narrow bridge floor
  bridge_floor_narrow: 0.08      # Will be relaxed in Phase 3
  bridge_floor_dynamic: 0.03
```

**Expected Impact**:
- No behavior change to existing playlists (narrow stays same initially)
- Creates explicit "strict" option for users who want ultra-cohesive playlists
- Establishes headroom for relaxing narrow mode

**Risk**: **LOW** - Additive only, no changes to existing modes

---

### Phase 2A: Relax Sonic Floor (Medium Risk)

**Rationale**: The sonic floor (0.18) is applied BEFORE genre consideration, rejecting 40-60% of the universe. Many genre-compatible tracks are rejected due to sonic differences that aren't perceptually problematic (e.g., different production styles within same genre).

**Changes**:

```yaml
# src/playlist/mode_presets.py
MODE_PRESETS = {
    "strict": {
        "min_sonic_similarity": 0.20,  # Strict keeps high bar
    },
    "narrow": {
        "min_sonic_similarity": 0.12,  # Relaxed from 0.18 (-33%)
    },
    "dynamic": {
        "min_sonic_similarity": 0.05,  # Relaxed from 0.10 (-50%)
    },
    "discover": {
        "min_sonic_similarity": 0.00,  # Relaxed from 0.02 (disabled)
    }
}
```

**Rationale for Values**:
- **Narrow 0.18 → 0.12**: Allows more production style variety while still maintaining sonic coherence
- **Dynamic 0.10 → 0.05**: Opens door to interesting sonic departures (e.g., acoustic → electronic versions)
- **Discover 0.02 → 0.00**: Fully disabled - genre should drive discover mode, not sonic gates

**Expected Impact**:
- Narrow: Candidate pool increases by ~15-25% (more tracks pass initial filter)
- Dynamic: Candidate pool increases by ~20-30%
- Discover: Candidate pool increases by ~5-10% (already quite open)

**Cascade Effects**:
- Genre floor will now have more candidates to evaluate (good - genre is more musically meaningful)
- Bridge and transition floors will see larger pools (good - more flexibility in path construction)

**Risk**: **MEDIUM** - Changes initial candidate selection, but genre floor still provides quality gate

---

### Phase 2B: Relax Genre Floor (Medium Risk)

**Rationale**: The genre floor (0.50 for narrow) is very strict and is checked at 3 different stages (redundant). With relaxed sonic floor, genre becomes the primary quality gate.

**Changes**:

```yaml
# config.yaml - genre_similarity section
genre_similarity:
  min_genre_similarity: 0.25              # Dynamic baseline (was 0.30)
  min_genre_similarity_narrow: 0.42       # Narrow gate (was 0.50, was 0.55 in mode_presets)

# Resolve the narrow genre floor to use min_genre_similarity_narrow consistently
```

**Note**: There's currently a discrepancy between `mode_presets.py` (0.50) and `config.yaml` (0.55). This should be resolved to a single value: 0.42.

**Rationale for Values**:
- **Narrow 0.50 → 0.42**: Allows justified exploration within adjacent genres (e.g., shoegaze → dream pop → indie pop)
- **Dynamic 0.30 → 0.25**: Opens door to interesting cross-genre bridges while still maintaining relatedness
- **Discover**: Keep 0.20 or lower (already permissive)

**Expected Impact**:
- Narrow: Candidate pool increases by ~10-15% after genre filtering
- Dynamic: Candidate pool increases by ~15-20%
- Reduces redundancy (genre checked 3x → can simplify in future phases)

**Cascade Effects**:
- More genre-adjacent candidates reach bridge and transition stages
- Bridge floor will have more diverse pool to work with

**Risk**: **MEDIUM** - Genre is musically meaningful, but 0.42 still maintains strong genre coherence

---

### Phase 3A: Relax Bridge Floor (High Risk)

**Rationale**: Bridge floor (0.08 for narrow) is the PRIMARY BOTTLENECK, rejecting 30-70% of candidates per segment. The min() operator means a single weak pier connection rejects the entire candidate.

**Changes**:

```yaml
# config.yaml - pier_bridge.artist_style section
artist_style:
  bridge_floor:
    strict: 0.10      # New explicit strict value
    narrow: 0.05      # Relaxed from 0.08 (-37.5%)
    dynamic: 0.02     # Relaxed from 0.03 (-33%)

# config.yaml - pier_bridge section (for non-artist-style mode)
pier_bridge:
  bridge_floor_strict: 0.10
  bridge_floor_narrow: 0.05     # Relaxed from 0.08
  bridge_floor_dynamic: 0.02    # Relaxed from 0.03
```

**Rationale for Values**:
- **Narrow 0.08 → 0.05**: Allows weaker connections to one pier if other pier is strong
- **Dynamic 0.03 → 0.02**: Already quite permissive, small reduction for consistency
- **Strict 0.10**: New explicit mode keeps high bar

**Expected Impact**:
- Narrow: 20-30% more candidates pass bridge floor per segment
- This is the BIGGEST impact change - should dramatically reduce "no valid continuations" failures
- Radio Dept playlist should now succeed with this change alone

**Cascade Effects**:
- Beam search will have significantly more candidates per step
- Transition floor becomes the next primary gate (intended behavior)

**Alternative Approach** (for future consideration):
Instead of min(), use weighted combination:
```python
bridge_score = 0.7 * min(sim_a, sim_b) + 0.3 * max(sim_a, sim_b)
```
This would reward strong single connections without fully rejecting weak ones.

**Risk**: **HIGH** - Bridge floor is architecturally critical. Changes affect path quality directly.

**Validation Required**:
- Generate 10 test playlists in narrow mode before/after
- Measure transition quality metrics (mean, p10, p25)
- Check for genre whiplash increases

---

### Phase 3B: Relax Transition Floor (High Risk)

**Rationale**: Transition floor (0.45 for narrow) rejects 40-60% of candidates per beam step. With relaxed bridge floor, this becomes the primary quality gate.

**Changes**:

```yaml
# config.yaml - constraints section
constraints:
  transition_floor_strict: 0.50     # New explicit strict value
  transition_floor_narrow: 0.38     # Relaxed from 0.45 (-15.6%)
  transition_floor_dynamic: 0.28    # Relaxed from 0.35 (-20%)
  transition_floor_discover: 0.20   # New explicit discover value
```

**Rationale for Values**:
- **Narrow 0.45 → 0.38**: Still maintains strong transitions, but allows more exploration
- **Dynamic 0.35 → 0.28**: Opens door to interesting jumps while keeping flow
- **Discover 0.20**: Explicit very-open value for big arcs

**Expected Impact**:
- Narrow: 15-20% more candidates per beam step
- Should eliminate remaining "no valid continuations" failures after bridge floor relaxation
- Slightly more perceptible transitions, but still smooth

**Cascade Effects**:
- With both bridge and transition floor relaxed, beam search has much more flexibility
- Artist diversity (min_gap) and progress constraints become primary quality gates

**Risk**: **HIGH** - Transition quality is perceptually critical. Too-low floor creates jarring playlists.

**Validation Required**:
- Generate 10 test playlists in narrow mode before/after
- Measure transition percentile distribution (p10, p25, p50)
- Subjective listening test for transition quality

---

### Phase 4: Reduce Duration Penalty Redundancy (Low Risk)

**Rationale**: Duration penalty is applied TWICE - once in candidate pool (weight 0.60) and once in beam search (bridge duration penalty weight 0.30). This is redundant and overly constraining.

**Changes**:

```yaml
# config.yaml - candidate_pool section
candidate_pool:
  duration_penalty_weight: 0.60      # Keep (primary gate)
  duration_cutoff_multiplier: 2.5    # Keep (hard cutoff)

# config.yaml - pier_bridge section
# Reduce bridge-specific duration penalty
pier_bridge:
  bridge_duration_penalty_weight_narrow: 0.15    # Relaxed from 0.30
  bridge_duration_penalty_weight_dynamic: 0.10   # Relaxed from 0.20
```

**Rationale**: The candidate pool already heavily penalizes long tracks. The bridge-specific penalty is redundant and over-constraining. Reduce but don't eliminate (slight penalty during bridge selection is still useful).

**Expected Impact**:
- Minor: ~2-5% more long tracks in final playlists
- Reduces "duration penalty stacking" where long tracks are penalized twice

**Risk**: **LOW** - Candidate pool penalty still prevents extreme duration mismatches

---

### Phase 5: Simplify Genre Floor Redundancy (Future Consideration)

**Rationale**: Genre floor is currently checked at 3 stages:
1. Candidate pool (Phase 1.3)
2. Segment pool (implied via bridge scoring)
3. Beam search (implied via transition scoring)

**Proposal** (not for immediate implementation):
- Keep genre floor in candidate pool only (Phase 1.3)
- Remove genre floor from beam search (use genre as tie-breaker, not gate)
- Rely on transition floor to maintain quality during path construction

**Expected Impact**:
- Reduces redundancy
- Simplifies system
- Allows more genre-adjacent exploration in narrow mode

**Risk**: **MEDIUM-HIGH** - Need to validate that transition floor alone maintains quality

**Recommendation**: Defer to future release after Phases 1-4 are validated

---

## Implementation Strategy

### Recommended Phased Rollout

1. **Phase 1** (Low Risk): Add explicit strict mode
   - Validate: No behavior change to existing modes

2. **Phase 2A+2B** (Medium Risk): Relax sonic + genre floors
   - Validate: Generate 5 test playlists per mode, check candidate pool sizes
   - Metric: Candidate pool should increase 20-30% for narrow

3. **Phase 3A** (High Risk): Relax bridge floor
   - Validate: Generate 10 test playlists in narrow mode
   - Metric: "No valid continuations" failures should drop to <5%
   - Metric: Transition quality (p10) should stay above 0.35

4. **Phase 3B** (High Risk): Relax transition floor
   - Validate: Generate 10 test playlists in narrow mode
   - Metric: Transition quality (p10) should stay above 0.30
   - Subjective: Listening test for perceptual quality

5. **Phase 4** (Low Risk): Reduce duration penalty redundancy
   - Validate: Generate 5 test playlists, check duration distribution

### Alternative: Conservative Approach

If full rollout is too risky, implement **Phase 3A only** (bridge floor relaxation):
- Narrow: `bridge_floor: 0.08 → 0.05`
- This single change should fix the Radio Dept failures

Then validate for 1-2 weeks before considering other phases.

---

## Test Cases for Validation

### Test Case 1: Radio Dept (Same-Artist Seeds Mode)
- **Mode**: Narrow
- **Seeds**: 5 tracks, all The Radio Dept
- **Expected**: Should succeed (currently fails)
- **Validation**: Check genre coherence (should stay shoegaze/dream pop/indie)

### Test Case 2: Electronic → Indie (Cross-Genre DJ Bridging)
- **Mode**: Dynamic
- **Seeds**: Squarepusher → The Radio Dept
- **Expected**: Should succeed with interesting bridge
- **Validation**: Check for smooth genre progression (no whiplash)

### Test Case 3: Tight Genre Cluster (Narrow Mode)
- **Mode**: Narrow
- **Seeds**: 5 post-punk tracks (Joy Division, Interpol, etc.)
- **Expected**: Should stay in post-punk/indie genre cluster
- **Validation**: Genre similarity distribution (p10, p25, median)

### Test Case 4: Wide Genre Spread (Discover Mode)
- **Mode**: Discover
- **Seeds**: Jazz → Classical → Electronic → Rock
- **Expected**: Should create interesting arcs with big leaps
- **Validation**: Check for genre diversity and progression smoothness

---

## Metrics for Success

### Quantitative Metrics

1. **Failure Rate**:
   - Current (narrow): ~15-20% "no valid continuations" failures
   - Target (narrow after Phase 3A): <5% failures

2. **Candidate Pool Size**:
   - Current (narrow): 600-800 tracks
   - Target (narrow after Phase 2A+2B): 800-1000 tracks (+25%)

3. **Transition Quality (p10)**:
   - Current (narrow): ~0.38-0.42
   - Target (narrow after Phase 3B): ≥0.30 (acceptable threshold)

4. **Genre Coherence**:
   - Current (narrow): Mean genre_sim ~0.55-0.65
   - Target (narrow after Phase 2B): Mean genre_sim ≥0.48 (acceptable threshold)

### Qualitative Metrics

1. **Perceptual Smoothness**: Subjective listening test for transition quality
2. **Genre Coherence**: Playlist "feels" cohesive within intended genre space
3. **Exploration**: Dynamic/discover modes include "interesting" tracks that wouldn't be in narrow

---

## Risk Mitigation

### Rollback Plan

If any phase degrades quality:
1. Immediately revert threshold changes
2. Use git to restore previous config values
3. Re-run test cases to confirm restoration

### A/B Testing

For high-risk phases (3A, 3B), consider generating paired playlists:
- Playlist A: Current thresholds
- Playlist B: Relaxed thresholds
- Compare metrics and subjective quality

### Gradual Tuning

For high-risk phases, consider intermediate steps:
- Bridge floor: 0.08 → 0.065 → 0.05 (two steps instead of one)
- Transition floor: 0.45 → 0.42 → 0.38 (two steps instead of one)

---

## Summary of Proposed Changes

| Parameter | Current (Narrow) | Proposed (Narrow) | Change | Risk | Phase |
|-----------|------------------|-------------------|--------|------|-------|
| **Sonic Floor** | 0.18 | 0.12 | -33% | Medium | 2A |
| **Genre Floor** | 0.50 | 0.42 | -16% | Medium | 2B |
| **Bridge Floor** | 0.08 | 0.05 | -37% | **High** | 3A |
| **Transition Floor** | 0.45 | 0.38 | -16% | **High** | 3B |
| **Bridge Duration Penalty** | 0.30 | 0.15 | -50% | Low | 4 |

**Expected Overall Impact**:
- Narrow mode rejection rate: 95-98% → 90-93% (intended target)
- Failure rate: 15-20% → <5%
- Transition quality: Maintained within acceptable range (p10 ≥ 0.30)
- Genre coherence: Maintained within acceptable range (mean ≥ 0.48)

---

## Conclusion

This plan proposes careful, phased recalibration of filtering thresholds to align each mode with its intended purpose. The focus is on relaxing the **primary bottlenecks** (bridge floor, transition floor) while maintaining quality through validation at each step.

**Recommendation**: Start with **Phase 3A only** (bridge floor relaxation) as a conservative first step. This single change should fix the Radio Dept failures and can be validated before proceeding to other phases.

The key is to maintain the system's careful balance while giving narrow mode the headroom it needs for "justified exploration" - and reserving over-filtering for an explicit "strict" mode.
