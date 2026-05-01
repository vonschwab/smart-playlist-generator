# Phase 2: UI Wiring Notes

**Date**: 2026-01-12
**Phase**: 2 of the GUI "Just Works" implementation

## Overview

Phase 2 implements the new "Generate" UI that replaces the scattered generation controls with a cohesive, mode-based panel integrated with the Phase 1 PolicyLayer.

## Files Created

| File | Purpose |
|------|---------|
| `src/playlist_gui/widgets/cohesion_dial.py` | 4-notch cohesion selector (Tight/Balanced/Wide/Discover) |
| `src/playlist_gui/widgets/seed_chips.py` | Draggable seed track chip list with auto-ordering |
| `src/playlist_gui/widgets/mode_panels.py` | Mode-specific panels (Artist/History/Seeds) |
| `src/playlist_gui/widgets/generate_panel.py` | Main generation panel composing all controls |
| `src/playlist_gui/seed_resolver.py` | Resolve autocomplete selections to SeedChip objects |

## Files Modified

| File | Changes |
|------|---------|
| `src/playlist_gui/main_window.py` | Added GeneratePanel integration, `_on_generate_v2`, feature flag |
| `src/playlist_gui/widgets/advanced_panel.py` | Added `disable_policy_owned_controls()` method |

## Feature Flag

The new UI is controlled by `_USE_GENERATE_PANEL_V2 = True` at the top of `main_window.py`.

Set to `False` to revert to the legacy UI for comparison or debugging.

## Seed Track Resolution

### How seed_track_ids are resolved

1. User types in the Seeds mode track search input
2. Autocomplete shows matches from `DatabaseCompleter` (format: "Title - Artist (Album)")
3. User selects a track and clicks "Add Seed"
4. `seed_resolver.resolve_track_from_display()` is called:
   - Parses display string to extract title, artist, album
   - Queries the metadata database (`tracks` table) by title + artist
   - Returns a `SeedChip` with `track_id`, `display`, `artist_key`
5. The SeedChip is added to the `SeedChipsList`

### How seed_artist_keys are looked up

1. Before calling `derive_runtime_config()`, the main window calls:
   ```python
   seed_artist_keys = self._generate_panel.get_seed_artist_keys()
   ```
2. This returns the `artist_key` field from each `SeedChip` in the list
3. The artist_key is normalized using `normalize_artist_key()` from `string_utils.py`

### Database Query

```sql
SELECT id, title, artist, album, artist_key
FROM tracks
WHERE title = ? AND artist = ?
LIMIT 1
```

If the `artist_key` column is NULL (legacy data), it's computed on-the-fly:
```python
artist_key = db_artist_key or normalize_artist_key(db_artist)
```

## Discover "Genre Pool Intent vs Availability"

### The Coupling Reality

Per Phase 1 findings, genre pool enrichment (dj_union pooling strategy) exists **ONLY** inside DJ bridging. If DJ bridging is disabled, setting `pooling.strategy = "dj_union"` has no effect.

### How the Hint is Handled

1. In `SeedsModePanel`, there's a hint label:
   ```
   "Genre enrichment requires 2+ seeds from different artists."
   ```

2. The hint visibility is controlled by `_update_dj_hint()`:
   ```python
   def _update_dj_hint(self):
       count = self._chips_list.seed_count()
       unique_artists = self._chips_list.unique_artist_count()
       show_hint = count > 0 and (count < 2 or unique_artists < 2)
       self._dj_hint_label.setVisible(show_hint)
   ```

3. This method is called whenever seeds change

4. The hint is styled with a warning appearance:
   ```css
   color: #856404; background: #fff3cd;
   border: 1px solid #ffc107; border-radius: 4px;
   ```

### Policy Behavior

When cohesion == "discover" but DJ bridging conditions aren't met:
- `PolicyDecisions.genre_pool_enabled = True` (tracks user intent)
- `pooling.strategy = "baseline"` (fallback, since dj_union requires DJ bridging)
- Note added: "Genre pool desired but unavailable: dj_union pooling requires DJ bridging to be enabled"

## PolicyLayer Integration

### Generation Flow

```
User clicks "Generate"
         │
         ▼
GeneratePanel._on_generate()
         │
         ▼
build_ui_state() → UIStateModel
         │
         ▼
emit generate_requested(asdict(ui_state))
         │
         ▼
main_window._on_generate_v2(ui_state_dict)
         │
         ├─► UIStateModel(**ui_state_dict)
         │
         ├─► get_seed_artist_keys() (Seeds mode only)
         │
         ├─► derive_runtime_config(ui_state, seed_artist_keys)
         │            │
         │            ▼
         │      PolicyDecisions (overrides, dj_bridging_enabled, notes)
         │
         ├─► merge_overrides(user_overrides, policy.overrides)
         │
         ▼
worker_client.generate_playlist(overrides=final_overrides, ...)
```

### Policy-Owned Keys

The following keys are controlled by the main UI and disabled in Advanced Panel:

- `playlists.genre_mode`
- `playlists.sonic_mode`
- `playlists.recently_played_filter.enabled`
- `playlists.recently_played_filter.lookback_days`
- `playlists.recently_played_filter.min_playcount_threshold`
- `playlists.ds_pipeline.constraints.min_gap`
- `playlists.tracks_per_playlist`
- `playlists.ds_pipeline.pier_bridge.dj_bridging.enabled`
- `playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering`
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy`
- `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre`

## Error Handling

### Generation Failure Dialog

When generation fails (`ok=False` in `_on_worker_done`):
- Shows blocking QMessageBox with error reason
- Detailed text includes suggestions

### Incomplete Generation Dialog

When fewer tracks than requested are returned:
- Triggered by checking `playlist.requested_count` vs `len(tracks)`
- Shows suggestions: reduce track count, widen cohesion, relax recency, try different seeds

## QA Checklist

- [ ] Seed(s) mode: 1 seed → DJ bridging OFF, hint visible
- [ ] Seed(s) mode: 2+ seeds same artist → DJ bridging OFF, hint visible
- [ ] Seed(s) mode: 2+ seeds different artists → DJ bridging ON, hint hidden
- [ ] Cohesion mapping: tight/balanced/wide/discover map correctly
- [ ] Recency defaults: ON, 14 days, 1 play threshold
- [ ] Auto-order toggle: ON shows "Auto-ordering enabled" message
- [ ] Advanced Panel: policy-owned controls disabled with tooltip
- [ ] Export buttons: only enabled when tracks exist, not auto-triggered
- [ ] Failure dialog: shown on generation failure
- [ ] Incomplete dialog: shown when tracks < requested

## Known Limitations

1. **Genre mode removed**: The Phase 2 UI removes Genre mode entirely (was Artist/Genre/History, now Artist/History/Seeds)

2. **Single artist only**: Artist mode accepts chips but backend uses only the first artist until multi-artist journeys are implemented

3. **Auto-ordering visualization**: The auto-order toggle affects drag-drop behavior but actual reordering for bridging would need backend integration to compute optimal order

4. **Track ID format**: Currently using database `id` column as track_id. May need alignment with `rating_key` used by Plex library
