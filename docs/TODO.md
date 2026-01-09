# TODO & Known Issues

**Last Updated:** 2026-01-09

---

## 🔴 High Priority

### ✅ FIXED: Artist Identity Resolution Not Working
**Status:** Fixed (2026-01-09)
**Severity:** High (was)
**Impact:** Playlists contained duplicate artists despite artist diversity constraints

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

**Fix Applied (2026-01-09):**
- ✅ Enabled artist identity resolution in `config.yaml`:
  ```yaml
  constraints:
    artist_identity:
      enabled: true
      strip_trailing_ensemble_terms: true
  ```
- ✅ Enabled pier artist exclusion in `config.yaml`:
  ```yaml
  pier_bridge:
    disallow_pier_artists_in_interiors: true   # Pier artists excluded from their own bridges
    disallow_seed_artist_in_interiors: false   # Seed artist allowed everywhere
  ```
- ✅ Updated `config.example.yaml` with documentation and examples
- ✅ All 40 DJ/pier-bridge tests passing

**Expected Results:**
- "Fela Kuti" and "Fela Ransome-Kuti and the Africa '70" now recognized as same artist
- "Herbie Hancock Trio" normalized to "Herbie Hancock"
- Pier artists excluded from their own bridge segments (e.g., Ramones excluded from Ramones→Replacements)
- Per-segment constraint (1 track per artist) still enforced
- Cross-segment MIN_GAP=1 (allows artist repeating across segments if not adjacent, per user preference)

---

## 🟡 Medium Priority

### ✅ RESOLVED: Hub Genre Collapse in DJ Bridging
**Status:** Fixed (2026-01-09 - Phase 2)
**Severity:** Medium (was)
**Impact:** Genre waypoints collapsed to hub genres ("indie rock") instead of respecting specific genres ("shoegaze", "dreampop", "slowcore")

**Problem:**
Shortest-path label selection in ladder mode picked generic hub genres:
- Slowdive→Deerhunter bridge: all waypoints = "indie rock"
- Lost nuanced genre signatures (shoegaze, dreampop, noise pop)
- S3 genre pool candidates rarely selected

**Root Cause:**
1. **Onehot mode**: Shortest-path algorithm selects single-label waypoints → loss of multi-genre information
2. **Equal genre weighting**: Common genres (indie rock) weighted same as rare genres (shoegaze)
3. **Weak genre signal**: Waypoint scoring alone insufficient to prefer genre-aligned candidates

**Fix Applied (Phase 2 - 2026-01-09):**
- ✅ **Vector Mode**: Direct multi-genre interpolation bypasses shortest-path
  - Preserves full genre signatures throughout bridge
  - Interpolates between anchor vectors: `g = (1-s)*vA + s*vB`
- ✅ **IDF Weighting**: Down-weights common genres like stop-words
  - Formula: `idf = log((N+1)/(df+1))^power`
  - Rare genres (shoegaze) → high weight (0.8-1.0)
  - Common genres (indie rock) → low weight (0.1-0.3)
- ✅ **Coverage Bonus**: Rewards candidates matching anchor's top-K genres
  - Schedule decay: `wA=(1-s)^power, wB=s^power`
  - Bonus weight: 0.15 (additive to score)
  - Tracks top-8 genres per anchor

**Results (from logs):**
```
Segment 0 (Slowdive→Beach House):
  Anchor A topK: shoegaze=0.383, dream pop=0.323, psychedelic=0.272, noise pop=0.228
  Target [Step 0]: shoegaze=0.386, dream pop=0.327, psychedelic=0.252, noise pop=0.222

Coverage bonus impact: winner_changed=1/3 mean_bonus=0.104
IDF stats: min=0.052 median=0.641 max=1.000
Mode: vector (not onehot)
```

**Config Added:**
```yaml
dj_ladder_target_mode: vector
dj_genre_use_idf: true
dj_genre_use_coverage: true
dj_genre_coverage_weight: 0.15
```

**Files Modified:**
- `pier_bridge_builder.py` (~350 lines): IDF helpers, vector mode, coverage bonus, logging
- `segment_pool_builder.py` (~25 lines): IDF parameter passthrough
- `pipeline.py` (~40 lines): Phase 2 config parsing
- `config.yaml`: Phase 2 settings enabled

**Documentation:**
- See `docs/dj_bridge_architecture.md` for complete Phase 2 design

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

### Phase 2: Genre Bridging Vector Mode + IDF + Coverage (2026-01-09)
- ✅ Implemented vector mode for genre targets (bypasses shortest-path label selection)
- ✅ Added IDF (Inverse Document Frequency) weighting to emphasize rare genres
- ✅ Added coverage bonus to reward matching anchor's top-K genres with schedule decay
- ✅ Added comprehensive diagnostic logging (per-segment, per-step, winner impact)
- ✅ Fixed hub genre collapse (Slowdive→Deerhunter now uses shoegaze/dreampop, not indie rock)
- ✅ Config parsing for 10 new Phase 2 parameters
- ✅ All existing tests passing, no regressions
- **Impact:** Multi-genre signatures preserved, rare genres emphasized, better genre alignment
- **Files:** `pier_bridge_builder.py` (~350 lines), `segment_pool_builder.py` (~25 lines), `pipeline.py` (~40 lines)
- **Documentation:** `docs/dj_bridge_architecture.md`

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
