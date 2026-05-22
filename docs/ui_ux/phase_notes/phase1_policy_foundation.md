# Phase 1: Policy Foundation Notes

**Date**: 2026-01-10
**Phase**: 1 of the GUI "Just Works" implementation

## Verified Semantics

### Seed Ordering

**Source**: `pier_bridge_builder.py:3720-3741`

The config key `playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering` accepts:
- `"auto"` = Optimize seed order for bridgeability (calls `_order_seeds_by_bridgeability`)
- `"fixed"` = Preserve user-specified order

**Policy Mapping**:
- `ui.seed_auto_order = True` → `seed_ordering = "auto"` (optimize)
- `ui.seed_auto_order = False` → `seed_ordering = "fixed"` (preserve)

### Genre Pool Coupling

**Source**: `pier_bridge_builder.py:3912-3914`, `segment_pool_builder.py:197-202`

**Finding**: Genre pool enrichment (S3 in the dj_union strategy) exists **ONLY inside DJ bridging** today.

The pooling strategy `dj_union` requires:
1. `dj_bridging.enabled = true`
2. `pooling.strategy = "dj_union"`
3. `pooling.k_genre > 0`

If DJ bridging is disabled, setting `pooling.strategy = "dj_union"` has no effect because the entire DJ bridging code path is skipped.

**Policy Implication**: When `cohesion == "discover"` but DJ bridging conditions aren't met, the policy:
1. Sets `genre_pool_enabled = True` (tracking user intent)
2. Falls back to `pooling.strategy = "baseline"`
3. Adds explanatory note: "Genre pool desired but unavailable: dj_union pooling requires DJ bridging to be enabled"

This design tracks the user's *desire* for genre pool expansion separately from whether it can actually be delivered, enabling future features where genre pool might become independently available.

## Policy Rules Implemented

### 1. Cohesion Mapping (Monotonic)

| UI Cohesion | genre_mode | sonic_mode |
|-------------|------------|------------|
| tight       | strict     | strict     |
| balanced    | narrow     | narrow     |
| wide        | dynamic    | dynamic    |
| discover    | discover   | discover   |

### 2. DJ Bridging Gating

Enabled only when ALL conditions met:
- `mode == "seeds"`
- `len(seed_track_ids) >= 2`
- `unique_artists(seed_artist_keys) >= 2`

Conservative fallback: If `seed_artist_keys` is not provided, DJ bridging is disabled with note.

### 3. Genre Pool Gating

- `genre_pool_enabled = (cohesion == "discover")`
- Actual effect requires DJ bridging to be enabled (see coupling above)

### 4. Artist Spacing

| UI Setting | min_gap |
|------------|---------|
| normal     | 6       |
| strong     | 9       |

### 5. Recency Filter (Hard Exclude)

Maps directly to:
- `playlists.recently_played_filter.enabled`
- `playlists.recently_played_filter.lookback_days`
- `playlists.recently_played_filter.min_playcount_threshold`

## Assumptions for Phase 2+

1. **Seed Track IDs**: The Seeds UI must resolve track names to stable database IDs before populating `seed_track_ids`. Display strings are not acceptable.

2. **Artist Key Resolution**: For accurate DJ bridging gating, the UI layer must resolve each seed track ID to its artist key before calling `derive_runtime_config()`. Without this, DJ bridging is conservatively disabled.

3. **Policy Owned Keys**: The Advanced Panel (if retained) must respect `POLICY_OWNED_KEYS` - policy always wins for these keys to ensure the simplified UI controls take precedence.

4. **No Worker Protocol Changes**: Phase 1 does not modify the worker/NDJSON protocol. The `PolicyDecisions.overrides` dict is designed to merge cleanly with existing config structure.

## Files Created

- `src/playlist_gui/ui_state.py` - UIStateModel dataclass
- `src/playlist_gui/policy.py` - PolicyLayer with derive_runtime_config
- `tests/unit/test_gui_policy.py` - 37 unit tests (all passing)

## Test Coverage

- UIStateModel defaults and helpers
- Cohesion mapping (monotonic progression verified)
- Recency override mapping
- DJ bridging gating (all edge cases)
- Genre pool gating and dj_union coupling
- Seed ordering semantics
- Artist spacing mapping
- merge_overrides policy precedence
- Integration tests (full flow scenarios)
