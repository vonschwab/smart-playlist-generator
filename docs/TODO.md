# TODO & Known Issues

**Last Updated:** 2026-01-09

---

## 🔴 High Priority

### Artist Identity Resolution Not Working
**Status:** Bug
**Severity:** High
**Impact:** Playlists contain duplicate artists despite artist diversity constraints

**Observed Behavior:**
From `docs/dj_union_ladder_playlist_log.txt`:
- Herbie Hancock appears **3 times** (tracks 2, 13, 28)
- Kaidi Tatham appears **2 times** (tracks 29, 30)
- Prince appears **2 times** (tracks 10, 17)
- Maxwell appears **2 times** (tracks 9, 25)
- Fela Kuti variants appear **2 times** (tracks 12, 15):
  - "Fela Ransome-Kuti and the Africa '70"
  - "Fela Kuti"

**Expected Behavior:**
Artist identity resolution should normalize artist names and prevent duplicates across the entire playlist.

**Investigation Needed:**
1. Check if `artist_identity_cfg.enabled` is actually enabled in the config
2. Verify artist identity mappings in the identity database
3. Check if pier-bridge is using artist identity resolution correctly
4. Verify `min_gap` constraint is being enforced across segment boundaries
5. Look at `disallow_pier_artists_in_interiors` setting (currently `false`)

**Files to Investigate:**
- `src/playlist/pier_bridge_builder.py` (lines ~2306-2429: artist diversity checks)
- `config.yaml` (artist identity settings)
- Artist identity database/mappings

**Potential Fixes:**
- Enable artist identity resolution if disabled
- Add artist name normalization rules for common variants ("Fela Kuti" → "Fela Ransome-Kuti")
- Increase `min_gap` constraint
- Add cross-segment artist tracking (global used_artists set)

---

## 🟡 Medium Priority

### Pool-to-Selection Gap for Genre Waypoint Tracks
**Status:** Tuning needed
**Severity:** Medium
**Impact:** Genre waypoint tracks added to pool but rarely selected

**Observed Behavior:**
From recent diagnostics:
```
Segment 0: chosen_from_genre_count=1 (out of 9 tracks)
Segment 1: chosen_from_genre_count=1 (out of 9 tracks)
Segment 2: chosen_from_genre_count=0 (out of 8 tracks)
```

**Analysis:**
- Genre waypoint tracks are in the pool (S3 source)
- Beam search prefers "toward" and "local" tracks
- Waypoint scoring influence increased (0.15→0.25) helps but may need further tuning

**Potential Solutions:**
- Further increase `waypoint_weight` (0.25 → 0.30-0.35)
- Increase `waypoint_cap` (0.10 → 0.15)
- Add genre source preference bonus in beam search scoring
- Adjust pool source balancing (increase `k_genre` relative to `k_toward`)

---

## 🟢 Low Priority / Future Enhancements

### DJ Bridging Diagnostics Enhancements
**Status:** Enhancement
**Severity:** Low

**Completed:**
- ✅ Pool diagnostics clarity (pool_before_gating, pool_after_gating)
- ✅ Waypoint rank impact metric
- ✅ Provenance tracking fix (mutually exclusive counting)

**Potential Future Enhancements:**
- Add per-step detailed tables to log output (currently only stored in waypoint_stats)
- Add diagnostic for genre waypoint similarity distribution
- Track why genre candidates are being rejected (score too low? other candidates better?)
- Add A/B comparison mode (baseline vs dj_union side-by-side)

---

### Ladder Route Planning Improvements
**Status:** Enhancement
**Severity:** Low

**Current Behavior:**
Ladder route planning uses shortest path in genre graph with onehot or smoothed vectors.

**Potential Enhancements:**
- Add weighted path planning (prefer high-similarity genre edges)
- Add genre path diversity scoring (prefer paths through varied genres)
- Add user-specified waypoint hints (force route through specific genres)
- Add genre path visualization/logging

---

### Config Validation & Warnings
**Status:** Enhancement
**Severity:** Low

**Potential Enhancements:**
- Validate config on startup (catch invalid values early)
- Warn when deprecated flat-key config is used (`dj_pooling_strategy`)
- Warn when conflicting settings are detected
- Add config migration tool for deprecated keys

---

## ✅ Recently Completed

### Waypoint Rank Impact Diagnostic (2026-01-09)
- ✅ Added opt-in diagnostic to measure waypoint scoring influence
- ✅ Samples evenly-spaced beam steps, compares base_score vs full_score
- ✅ Metrics: winner_changed, topK_reordered, mean_rank_delta
- ✅ Gate behind `dj_bridging.diagnostics.waypoint_rank_impact_enabled` (default: false)
- ✅ 10 unit tests passing

### Pool Diagnostics Clarity (2026-01-09)
- ✅ Fixed pool_before_gating / pool_after_gating tracking (was showing 0)
- ✅ Renamed "Pool sources" → "Chosen edge provenance" in logs
- ✅ Added invariant checks (WARNINGs for inconsistencies)

### Provenance Tracking Bug Fix (2026-01-09)
- ✅ Fixed double-counting of tracks in multiple source sets
- ✅ Implemented mutually exclusive counting with priority: genre > toward > local > baseline_only
- ✅ Sum now equals interior_length (was 2x before)
- ✅ 3 unit tests added

### DJ Bridging Config Parsing Fix (Phase 1)
- ✅ Added flat-key fallback for `dj_pooling_strategy` (backward compatibility)
- ✅ Deprecation warning when flat key is used
- ✅ 7 unit tests passing

### Waypoint Scoring Influence Increase (Phase 2)
- ✅ Removed waypoint tie-break band suppression (always apply waypoint scoring)
- ✅ Added waypoint diagnostics collection (mean_sim, p50, p90, delta_applied)
- ✅ Extended SegmentDiagnostics dataclass with waypoint stats fields

---

## 📝 Notes

### Test Coverage
- DJ/pier-bridge tests: **40 passing** (as of 2026-01-09)
- Diagnostic tests: **13 passing** (pool clarity: 5, rank impact: 5, provenance fix: 3)

### Key Files Modified Recently
- `src/playlist/pier_bridge_builder.py` (diagnostics, provenance fix, waypoint scoring)
- `src/playlist/pipeline.py` (config parsing, diagnostic flags)
- `src/playlist/pier_bridge_diagnostics.py` (waypoint stats fields)
- `config.example.yaml` (diagnostic settings documentation)
- `config.yaml` (waypoint_weight: 0.15→0.25, waypoint_cap: 0.05→0.10)

### Branch
Current work branch: `dj-ordering`

---

## 🔗 Related Documentation
- `docs/diagnostics/06_waypoint_rank_impact.md` - Rank impact diagnostic spec
- `docs/diagnostics/dj_bridging_status_audit.md` - DJ bridging audit
- `docs/diagnostics/dj_ladder_route_audit.md` - Ladder route planning audit
- `docs/dj_union_ladder_playlist_log.txt` - Latest playlist generation logs
