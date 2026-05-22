# Phase 3: QA Checklist

**Date**: 2026-01-12
**Phase**: 3 - Seed Stability, Auto-ordering, and QA

## Pre-Generation Checks

### Seeds Mode - DJ Bridging Gating

- [ ] **1 seed**: DJ bridging should be OFF, hint label visible
  - Add a single seed track
  - Verify hint shows: "Genre enrichment requires 2+ seeds from different artists."
  - Generate and verify DJ bridging is disabled in logs

- [ ] **2 seeds, same artist**: DJ bridging should be OFF, hint visible
  - Add two tracks from the same artist
  - Verify hint still shows
  - Generate and verify DJ bridging is disabled

- [ ] **2 seeds, different artists**: DJ bridging should be ON, hint hidden
  - Add two tracks from different artists
  - Verify hint is hidden
  - Generate and verify DJ bridging is enabled in logs

### Discover Mode - Genre Pool Behavior

- [ ] **Cohesion at Discover with DJ bridging conditions met**:
  - Set cohesion to "Discover"
  - Add 2+ seeds from different artists
  - Generate and verify `pooling.strategy = "dj_union"` in config

- [ ] **Cohesion at Discover without DJ bridging**:
  - Set cohesion to "Discover"
  - Add only 1 seed (or 2 same-artist seeds)
  - Verify hint about genre enrichment is visible
  - Generate and verify `pooling.strategy = "baseline"` (fallback)

### Recency Filter Defaults

- [ ] **Defaults are correct**:
  - Open the app fresh
  - Verify "Exclude recent" checkbox is ON by default
  - Verify days spinner shows 14
  - Verify plays threshold shows 1

- [ ] **Recency exclusion works**:
  - Generate a playlist
  - Verify recently played tracks are excluded (check logs)

## Seed Resolution Stability

- [ ] **Autocomplete selection uses cached data**:
  - Type in track search, select from autocomplete dropdown
  - Click "Add Seed"
  - Verify NO "fallback triggered" warning in application logs
  - Verify track is added with correct display string

- [ ] **Fallback logs warning when triggered**:
  - (Developer test) Manually enter a track string not in cache
  - Verify warning log: "Seed resolution fallback triggered..."

## Auto-Ordering

- [ ] **Auto-order ON reorders chips visually**:
  - Enable "Auto-order seeds for optimal bridging" checkbox
  - Add 3+ seeds from different artists (e.g., A1, B1, A2, C1)
  - Verify chips reorder to interleave artists (e.g., A1, B1, C1, A2)

- [ ] **Auto-order OFF preserves user order**:
  - Disable auto-order checkbox
  - Add seeds in specific order
  - Verify order is preserved
  - Verify drag-drop reordering is enabled

- [ ] **Toggling auto-order applies reordering**:
  - Add seeds with auto-order OFF
  - Toggle auto-order ON
  - Verify chips reorder immediately

## Export Behavior

- [ ] **No auto-export on Generate**:
  - Generate a playlist
  - Verify playlist is NOT automatically exported to Plex
  - Verify export buttons are enabled after generation

- [ ] **No auto-export on Regenerate**:
  - Click Regenerate
  - Verify no automatic export occurs

- [ ] **No auto-export on New Seeds**:
  - Click New Seeds
  - Verify no automatic export occurs

## Error Handling

- [ ] **Generation failure shows dialog**:
  - Trigger a generation failure (e.g., invalid config)
  - Verify blocking dialog appears with error message

- [ ] **Incomplete generation shows dialog**:
  - Request more tracks than available
  - Verify dialog shows suggestions for improvement

## UI State Consistency

- [ ] **Mode switching preserves global controls**:
  - Set cohesion, length, recency values
  - Switch between Artist/History/Seeds modes
  - Verify global control values are preserved

- [ ] **Advanced panel controls are disabled for policy-owned keys**:
  - Open Advanced panel
  - Verify policy-owned controls show disabled state
  - Verify tooltip explains "Controlled by main UI"

## Regression Tests

- [ ] **Artist mode still works**:
  - Select Artist mode
  - Enter an artist name
  - Generate playlist
  - Verify playlist contains expected artist tracks


---

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| 1 seed DJ gating | | |
| 2 seeds same artist | | |
| 2 seeds different artists | | |
| Discover with DJ bridging | | |
| Discover without DJ bridging | | |
| Recency defaults | | |
| Seed resolution stability | | |
| Auto-order ON | | |
| Auto-order OFF | | |
| No auto-export | | |
| Error dialogs | | |
| Mode switching | | |
| Advanced panel disabled | | |
| Artist mode regression | | |

**Tested by**: _______________
**Date**: _______________
