# Pier-Bridge Global Gap Constraint Enforcement

## Overview

This document describes the fix for the pier-bridge post-order filtering bug that was causing length mismatch errors across all pier-bridge playlists (artist mode, genre mode, and default mode).

## The Problem

### Original Design Violation

The pier-bridge builder had a fundamental architectural flaw: it was **dropping tracks AFTER ordering** via `_enforce_min_gap_global()` at finalization time (lines 1927-1933 in the original code).

**Violating Code (REMOVED):**
```python
# Convert to track IDs (after enforcing cross-segment min_gap to avoid back-to-back repeats)
final_indices, dropped = _enforce_min_gap_global(
    final_indices, bundle.artist_keys, min_gap=1, bundle=bundle
)
if dropped:
    logger.debug(
        "Pier+Bridge: dropped %d tracks to enforce cross-segment min_gap", dropped
    )
```

### Impact

This violated the core DS pipeline design rule: **no post-order filtering ever**.

**Symptoms:**
- Length mismatch errors: `final=29 expected=30`
- Non-deterministic results (gap enforcement was non-deterministic)
- Promises N tracks but delivers N-k tracks
- Affected ALL pier-bridge playlists, not just genre mode

**Example Error:**
```
ERROR | src.playlist.pipeline | Validation errors detected: ['length_mismatch final=29 expected=30']
```

### Root Cause

The pier-bridge builder enforces "one track per artist per segment" during beam search, but this constraint is segment-local. Adjacent segments could end with artist X and start with artist X, creating back-to-back repeats at segment boundaries.

The original "solution" was to drop violating tracks after concatenation, but this:
1. Breaks length guarantees
2. Violates DS design principles
3. Produces non-deterministic results

## The Solution

### Boundary-Aware Constraint Enforcement

The fix enforces cross-segment min_gap **DURING generation** by making each segment aware of the global playlist prefix built so far.

### Implementation

**1. Added boundary context tracking:**

```python
# Initialize boundary tracking before segment loop
MIN_GAP_GLOBAL = 1  # Cross-segment min_gap constraint
recent_boundary_artists: List[str] = []
```

**2. Pass context to beam search:**

Each segment receives the artist keys from the last `MIN_GAP_GLOBAL` positions of the concatenated result from previous segments:

```python
segment_path = _beam_search_segment(
    ...,
    recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
)
```

**3. Merge into beam search initial state:**

The `_beam_search_segment()` function now accepts `recent_global_artists` and merges them into the initial `used_artists` set:

```python
# Add boundary context from previous segments (cross-segment min_gap enforcement)
if recent_global_artists:
    for artist_key in recent_global_artists:
        if artist_key:
            used_artists_init.add(str(artist_key))
```

This prevents the beam search from selecting any track whose artist appears in the recent boundary context.

**4. Update boundary context after each segment:**

After each segment is built and appended, extract the last `MIN_GAP_GLOBAL` artists for the next segment:

```python
# Update boundary context for next segment (cross-segment min_gap enforcement)
current_concat: List[int] = []
for concat_seg_idx, concat_seg in enumerate(all_segments):
    if concat_seg_idx == 0:
        current_concat.extend(concat_seg)
    else:
        current_concat.extend(concat_seg[1:])  # Drop duplicate pier

# Extract artist keys from the last MIN_GAP_GLOBAL positions
recent_boundary_artists = []
start_pos = max(0, len(current_concat) - MIN_GAP_GLOBAL)
for pos in range(start_pos, len(current_concat)):
    try:
        artist_key = identity_keys_for_index(bundle, int(current_concat[pos])).artist_key
        if artist_key:
            recent_boundary_artists.append(str(artist_key))
    except Exception:
        continue
```

**5. Removed post-order filter:**

Deleted the `_enforce_min_gap_global()` call at finalization. Cross-segment min_gap is now enforced during generation, not as a post-filter.

**6. Added strict length validation:**

```python
# Strict length validation: pier-bridge must return EXACTLY the requested number of tracks
if len(final_track_ids) != total_tracks:
    failure_msg = (
        f"Pier-bridge length mismatch: generated {len(final_track_ids)} tracks "
        f"but expected exactly {total_tracks}. This indicates a bug in segment generation."
    )
    logger.error(failure_msg)
    return PierBridgeResult(
        track_ids=[],
        track_indices=[],
        seed_positions=[],
        segment_diagnostics=diagnostics,
        stats={"error": "length_mismatch", "expected": total_tracks, "actual": len(final_track_ids)},
        success=False,
        failure_reason=failure_msg,
    )
```

If the final length doesn't match the target, the builder now fails with a clear error message instead of silently returning fewer tracks.

## Design Properties

### Guarantees

1. **Exact length:** Pier-bridge returns EXACTLY N tracks or fails with clear error
2. **No post-order filtering:** All constraints enforced during generation
3. **Deterministic:** Given same random_seed, produces same result
4. **Cross-segment min_gap:** No artist repeats across segment boundaries (min_gap=1)
5. **Sonic cohesion:** Preserved via transition/bridge scoring

### Constraints Enforced During Generation

**Per-segment constraints:**
- One track per artist per segment (prevents intra-segment repeats)
- Bridge floor (similarity to both piers)
- Transition floor (between consecutive tracks)
- Progress monotonicity (optional)

**Cross-segment constraints (NEW):**
- Global min_gap via boundary-aware beam search
- Artist in last position of segment N cannot appear in first positions of segment N+1

## Testing

### Test Results

**Unit tests:** All 144 tests pass
```
======================= 144 passed, 2 warnings in 2.25s =======================
```

**Integration tests:**
- Genre mode: `Pier+Bridge complete: 30 tracks, 3 segments, 3 successful` ✅
- Artist mode: `Pier+Bridge complete: 30 tracks, 5 segments, 5 successful` ✅

No length mismatch errors in either test.

### Verification Commands

```bash
# Genre mode
python main_app.py --genre "ambient" --tracks 30 --dry-run --log-level INFO

# Artist mode
python main_app.py --artist "Radiohead" --tracks 30 --dry-run --log-level INFO
```

Expected log output:
```
INFO | src.playlist.pier_bridge_builder | Pier+Bridge complete: 30 tracks, N segments, N successful
```

## Files Modified

1. **src/playlist/pier_bridge_builder.py**
   - Updated module docstring (lines 1-19)
   - Modified `_beam_search_segment()` signature (line 993)
   - Added boundary context merging (lines 1056-1060)
   - Added boundary tracking initialization (lines 1505-1508)
   - Pass boundary context to beam search (line 1692)
   - Update boundary context after each segment (lines 1920-1938)
   - Removed post-order filter (deleted old lines 1965-1972)
   - Added strict length validation (lines 1970-1985)
   - Updated seed position comment (line 2019)

## Migration Notes

### For Users

No action required. The fix is transparent - playlists will now be generated with exact lengths and no back-to-back artist repeats across segments.

### For Developers

If you're working on pier-bridge code:

1. **Never add post-order filtering** - all constraints must be enforced during generation
2. **Maintain boundary awareness** - segments must know about the global prefix
3. **Preserve determinism** - all randomness controlled by `random_seed`
4. **Test length guarantees** - verify final length exactly matches target

## Future Work

### Potential Enhancements

1. **Configurable min_gap:** Allow min_gap > 1 for stricter artist spacing
2. **Album-level gap:** Prevent same album appearing too close together
3. **Genre-level gap:** Prevent rapid genre shifts at boundaries
4. **Hybrid constraints:** Combine multiple gap types (artist + album + genre)

### Implementation Notes

All future gap constraints should follow the boundary-aware pattern:
1. Track constraint state in global context
2. Pass context to segment builder
3. Merge into beam search initial state
4. Update context after each segment

**Never use post-order filtering.**

## References

- Original bug report: Genre mode length mismatch errors
- Design document: `docs/GENRE_MODE_DESIGN.md`
- Audit report: `docs/GENRE_MODE_AUDIT.md`
- Implementation: `src/playlist/pier_bridge_builder.py`
