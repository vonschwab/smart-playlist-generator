# Artist Duplication Issue - Diagnostic Analysis

**Date:** 2026-01-09
**Branch:** `dj-ordering`
**Playlist Log:** `docs/dj_union_ladder_playlist_log.txt`

---

## 🔴 Problem: Excessive Artist Duplication

### Observed Duplicates (30-track playlist)

From latest playlist generation:

| Artist | Occurrences | Track Numbers | Issue |
|--------|-------------|---------------|-------|
| **Herbie Hancock** | **3** | 2, 13, 28 | Same artist 3 times in 30 tracks |
| **Kaidi Tatham** | **2** | 29, 30 | Same artist, adjacent tracks |
| **Prince** | **2** | 10, 17 | Same artist twice |
| **Maxwell** | **2** | 9, 25 | Same artist twice |
| **Fela Kuti variants** | **2** | 12, 15 | Name variants not normalized |

**Specific Fela Kuti variants:**
- Track 12: "Fela Ransome-Kuti and the Africa '70"
- Track 15: "Fela Kuti"

These are the **same artist** but not being recognized as such.

---

## 🔍 Root Cause Analysis

### 1. Artist Identity Resolution is DISABLED ❌

**Current State:**
```yaml
# config.yaml
# NO artist_identity section found
```

**Pipeline Default (src/playlist/pipeline.py:148):**
```python
enabled=bool(artist_identity_overrides.get("enabled", False))
# Defaults to False when no config provided
```

**Impact:**
- "Fela Kuti" and "Fela Ransome-Kuti" are treated as **different artists**
- No normalization of ensemble variants ("Orchestra", "Trio", etc.)
- No collaboration splitting ("Artist A & Artist B")

**Expected Behavior:**
When enabled, artist identity resolution:
- Strips trailing ensemble terms: "Duke Ellington Orchestra" → "Duke Ellington"
- Splits collaborations: "Herbie Hancock & Wayne Shorter" → ["Herbie Hancock", "Wayne Shorter"]
- Normalizes case and whitespace

---

### 2. Cross-Segment MIN_GAP is TOO WEAK ⚠️

**Current Setting (src/playlist/pier_bridge_builder.py:3092):**
```python
MIN_GAP_GLOBAL = 1  # Cross-segment min_gap constraint
```

**What this means:**
- Artist can repeat as long as they're **not adjacent**
- Example: `[Herbie (pos 2), ..., Herbie (pos 13), ..., Herbie (pos 28)]` is ALLOWED
- Only prevents: `[Herbie, Herbie]` (adjacent positions)

**Why Herbie Hancock appeared 3 times:**
- Track 2 (Herbie) → Track 13 (Herbie): Gap = 11 positions ✅ Allowed
- Track 13 (Herbie) → Track 28 (Herbie): Gap = 15 positions ✅ Allowed

**User's expectation:** Artists should not repeat in a 30-track playlist, or at least have a much larger gap (e.g., 10-15 tracks).

---

### 3. Pier/Seed Artist Exclusion is DISABLED ❌

**Current Setting (from log line 67):**
```
disallow_seed_artist_in_interiors=False disallow_pier_artists_in_interiors=False
```

**Impact:**
- Pier artists CAN appear in their own bridge segments
- Seed artists CAN appear in ANY segment

**User's Requirement:**
> "Ramones cannot appear in a Ramones -> Replacements bridge, but they could appear in a subsequent Replacements -> Pavement bridge"

**Current Behavior:** Seed/pier artists are allowed everywhere (no exclusion).

---

### 4. Per-Segment Artist Constraint is CORRECT ✅

**Current Implementation:**
Each segment enforces **one track per artist** via `BeamState.used_artists` set.

**Verification from code (src/playlist/pier_bridge_builder.py:2310-2325):**
```python
# Artist diversity: check if candidate artist already used
if artist_key_by_idx is not None:
    cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
    if cand_artist and cand_artist in state.used_artists:
        continue  # Reject candidate
```

**This is working correctly** - no artist repeats within a single segment.

---

## 📊 Current Artist Diversity Mechanisms

### Tier 1: Per-Segment Enforcement (WORKING ✅)
- **Scope:** Within each bridge segment (e.g., Ramones → Replacements)
- **Rule:** One track per artist maximum
- **Implementation:** `BeamState.used_artists` set in beam search
- **Status:** ✅ Correct

### Tier 2: Cross-Segment Enforcement (WEAK ⚠️)
- **Scope:** Across segment boundaries
- **Rule:** Artist cannot repeat within last `MIN_GAP_GLOBAL` positions
- **Current Setting:** `MIN_GAP_GLOBAL = 1` (only prevents adjacent duplicates)
- **Status:** ⚠️ Too weak

### Tier 3: Seed/Pier Exclusion (DISABLED ❌)
- **Scope:** Bridge segment interiors
- **Rule:** Seed/pier artists excluded from their own bridge segments
- **Current Setting:** `disallow_seed_artist_in_interiors=False`, `disallow_pier_artists_in_interiors=False`
- **Status:** ❌ Disabled

### Tier 4: Artist Identity Resolution (DISABLED ❌)
- **Scope:** All artist comparisons
- **Rule:** Normalize artist names, strip ensemble terms, split collaborations
- **Current Setting:** `enabled: False` (default)
- **Status:** ❌ Disabled

---

## 🛠️ Recommended Fixes

### Priority 1: Enable Artist Identity Resolution (CRITICAL)

**Add to `config.yaml`:**
```yaml
playlists:
  ds_pipeline:
    constraints:
      artist_identity:
        enabled: true
        strip_trailing_ensemble_terms: true
        split_delimiters:
          - ","
          - " & "
          - " and "
          - " feat. "
          - " feat "
          - " featuring "
          - " ft. "
          - " ft "
          - " with "
          - " x "
          - " + "
```

**Expected Impact:**
- "Fela Kuti" and "Fela Ransome-Kuti and the Africa '70" → same identity
- "Herbie Hancock Trio" → "Herbie Hancock"
- "Artist A & Artist B" → both artists tracked separately

---

### Priority 2: Increase Cross-Segment MIN_GAP

**Option A: Moderate (Recommended)**
```python
MIN_GAP_GLOBAL = 10  # Artist cannot repeat within 10 positions
```

**Option B: Strict**
```python
MIN_GAP_GLOBAL = 15  # Artist cannot repeat within 15 positions
```

**Option C: Playlist-wide (Strictest)**
```python
MIN_GAP_GLOBAL = 999  # Effectively prevents any artist repeating in playlist
```

**Implementation:**
Modify `src/playlist/pier_bridge_builder.py:3092`:
```python
MIN_GAP_GLOBAL = 10  # Cross-segment min_gap constraint (was: 1)
```

**Expected Impact (MIN_GAP=10):**
- Herbie Hancock at track 2 → cannot appear again until track 12+
- Significantly reduces artist repetition in 30-track playlists

---

### Priority 3: Enable Seed/Pier Artist Exclusion (OPTIONAL)

**Add to `config.yaml`:**
```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      disallow_pier_artists_in_interiors: true  # Prevent pier artists in their own bridges
```

**Expected Behavior:**
- Ramones (pier A) → Replacements (pier B): Ramones **excluded** from interior
- Replacements (pier B) → Pavement (pier C): Ramones **allowed** (not a pier in this segment)

**Note:** This is OPTIONAL because cross-segment MIN_GAP already provides significant protection.

---

## 🧪 Testing Strategy

### Step 1: Enable Artist Identity Resolution
```yaml
# config.yaml
playlists:
  ds_pipeline:
    constraints:
      artist_identity:
        enabled: true
```

**Expected Log:**
```
INFO: Artist identity resolution enabled for min_gap enforcement
```

**Test:** Generate playlist, check if "Fela Kuti" variants are deduplicated.

### Step 2: Increase MIN_GAP_GLOBAL

**Code Change:**
```python
# src/playlist/pier_bridge_builder.py:3092
MIN_GAP_GLOBAL = 10  # Increased from 1
```

**Test:** Generate playlist, verify no artist repeats within 10 positions.

### Step 3: Verify Logging

**Add debug logging to track artist dedupe decisions:**
```python
logger.debug("Artist %s blocked by min_gap (distance=%d)", artist_key, gap_distance)
```

### Step 4: Edge Cases to Test

1. **Ensemble variants:** "Duke Ellington Orchestra" should match "Duke Ellington"
2. **Collaborations:** "A & B" should track both A and B separately
3. **Case sensitivity:** "The beatles" should match "The Beatles"
4. **Whitespace:** "Herbie  Hancock" (double space) should match "Herbie Hancock"

---

## 📝 Code Locations

### Artist Identity Resolution
- **Config:** `src/playlist/artist_identity_resolver.py:31-90` (ArtistIdentityConfig dataclass)
- **Parser:** `src/playlist/pipeline.py:142-159` (config parsing)
- **Usage:** `src/playlist/pier_bridge_builder.py:2224, 2311, 2421, 2479, 4118` (identity checks)

### Cross-Segment MIN_GAP
- **Setting:** `src/playlist/pier_bridge_builder.py:3092` (`MIN_GAP_GLOBAL = 1`)
- **Boundary Tracking:** `src/playlist/pier_bridge_builder.py:4105-4137` (recent_boundary_artists update)
- **Enforcement:** Via `recent_global_artists` parameter in beam search

### Seed/Pier Exclusion
- **Config Fields:** `src/playlist/pier_bridge_builder.py:109-110` (PierBridgeConfig dataclass)
- **Enforcement:** `src/playlist/pier_bridge_builder.py:2226-2246` (used_artists_init)

---

## 🎯 Success Criteria

After implementing fixes, a 30-track playlist should:
1. ✅ Have **0 exact artist name duplicates** (artist identity resolution working)
2. ✅ Have **no artists within MIN_GAP positions** (e.g., 10 tracks)
3. ✅ Recognize "Fela Kuti" and "Fela Ransome-Kuti" as same artist
4. ✅ Strip ensemble terms: "Herbie Hancock Trio" → "Herbie Hancock"
5. ✅ (Optional) Exclude pier artists from their own bridge segments

---

## 🚀 Quick Fix Commands

### 1. Enable Artist Identity (config.yaml)
Add under `playlists.ds_pipeline.constraints`:
```yaml
artist_identity:
  enabled: true
```

### 2. Increase MIN_GAP (code change)
```python
# src/playlist/pier_bridge_builder.py:3092
MIN_GAP_GLOBAL = 10  # Increased from 1
```

### 3. Restart GUI and regenerate playlist

**Expected Result:** Herbie Hancock should appear **at most once**, Fela Kuti variants deduplicated.
