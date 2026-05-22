# Phase 3: Seed Ordering Implementation

**Date**: 2026-01-12
**Phase**: 3 - Seed Stability, Auto-ordering, and QA

## Overview

Phase 3 implements real auto-ordering for seed chips that visually reorders the UI and matches the generation order passed to the backend.

## Ordering Approach

### Decision: GUI-Side Heuristic

The backend's `_order_seeds_by_bridgeability()` function in `pier_bridge_builder.py` provides optimal ordering but requires feature vectors (sonic embeddings, genre vectors) that are only available at generation time.

For immediate visual feedback in the GUI, we implemented a **simple artist-based heuristic** that:
1. Works without feature vectors
2. Provides instant visual feedback when seeds change
3. Creates a reasonable approximation of optimal ordering

### Algorithm: Round-Robin Artist Interleaving

**Location**: `src/playlist_gui/widgets/seed_chips.py::compute_seed_order()`

```python
def compute_seed_order(chips: List[SeedChip]) -> List[int]:
    """
    Compute an ordering of seeds to maximize bridging potential.

    Strategy:
    1. Group seeds by artist_key
    2. Interleave groups round-robin style to maximize variety
    3. Creates sequence like: A1, B1, C1, A2, B2, C2, ...
    """
```

**Rationale**:
- Maximizing artist variety at each position creates natural "bridge points"
- The backend can further refine order if needed (with "auto" seed_ordering mode)
- Simple and deterministic - same inputs always produce same output

### Example

Input seeds (in add order):
1. "Song A" - Artist X
2. "Song B" - Artist X
3. "Song C" - Artist Y
4. "Song D" - Artist Z

After auto-ordering:
1. "Song A" - Artist X (first from X)
2. "Song C" - Artist Y (first from Y)
3. "Song D" - Artist Z (first from Z)
4. "Song B" - Artist X (second from X)

## Implementation Details

### When Ordering is Applied

Auto-ordering is triggered when:
1. **Auto-order checkbox toggled ON**: `set_auto_order(True)` calls `_apply_auto_order()`
2. **New seed added with auto-order ON**: `add_seed()` calls `_apply_auto_order()` if enabled
3. **Seeds replaced with auto-order ON**: `set_seeds()` calls `_apply_auto_order()` if enabled

### Key Methods

| Method | File | Purpose |
|--------|------|---------|
| `compute_seed_order()` | seed_chips.py | Pure function computing new order |
| `_apply_auto_order()` | seed_chips.py | Applies computed order to internal list |
| `set_auto_order()` | seed_chips.py | Handles toggle, triggers reorder if ON |
| `add_seed()` | seed_chips.py | Adds seed, auto-orders if enabled |

### Backend Integration

The GUI passes `seed_ordering = "auto"` to the backend when auto-order is enabled. This tells the backend's `_order_seeds_by_bridgeability()` to further optimize the order using feature vectors.

Flow:
```
GUI auto-order (artist heuristic)
         │
         ▼
Backend receives seeds in GUI order
         │
         ▼ (if seed_ordering == "auto")
_order_seeds_by_bridgeability() refines using features
         │
         ▼
Final order used for playlist generation
```

## Limitations

1. **No feature-based ordering**: GUI heuristic uses only artist metadata, not sonic/genre similarity
2. **Backend may further reorder**: The visual order shown may differ from final generation order when backend applies feature-based optimization
3. **Single-artist edge case**: If all seeds are from the same artist, original order is preserved (no interleaving possible)

## Future Improvements

Potential enhancements (not in Phase 3 scope):
- Fetch precomputed similarity data for GUI-side ordering
- Show "backend optimized" indicator after generation completes
- Allow user to see/accept backend's recommended order

## Files Modified

| File | Changes |
|------|---------|
| `src/playlist_gui/widgets/seed_chips.py` | Added `compute_seed_order()`, `_apply_auto_order()`, modified `add_seed()`, `set_auto_order()`, `set_seeds()` |
| `src/playlist_gui/widgets/mode_panels.py` | Added `get_seed_display_strings()` |
| `src/playlist_gui/widgets/generate_panel.py` | Added `get_seed_display_strings()` |

## Testing

The auto-ordering logic can be verified by:
1. Adding seeds from multiple artists with auto-order ON
2. Observing visual chip reordering
3. Checking logs for "Auto-ordered N seeds" debug message
