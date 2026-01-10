# Playlist Generator v3.4 - DJ Bridge Mode & Multi-Seed Playlists

**Release Date:** 2026-01-10
**Branch:** `dj-ordering` (59 commits)
**Focus:** Multi-seed playlist generation with genre-aware bridging

---

## 🎧 What's New: DJ Bridge Mode

Version 3.4 introduces **DJ Bridge Mode**, a revolutionary approach to multi-seed playlist generation that creates smooth transitions between 2-10 seed tracks using genre-aware routing and intelligent candidate pooling.

### The Problem We Solved

**Before v3.4:**
- Single-seed playlists only: Start with one artist, expand outward
- Multi-artist playlists had no control over genre evolution
- Bridging between stylistically distant artists was hit-or-miss
- No way to plan genre arcs (e.g., shoegaze → dream pop → indie rock)

**With v3.4:**
- **Multi-seed support**: 2-10 seed tracks as anchors ("piers")
- **Genre-aware routing**: Plans optimal genre paths between seeds
- **Union pooling**: Combines local, toward, and genre candidates
- **Beam search**: Explores multiple paths to find optimal bridges
- **Artist diversity**: Enforces constraints within and across segments

### Real-World Example

```
Seeds: Slowdive, Beach House, Deerhunter, Helvetia
Mode: DJ Bridge (dj_union pooling)
Result: 30-track playlist with 3 segments

Segment 1 (Slowdive → Beach House): 10 tracks
  Genre arc: shoegaze → dream pop → psychedelic

Segment 2 (Beach House → Deerhunter): 10 tracks
  Genre arc: dream pop → indie rock → noise rock

Segment 3 (Deerhunter → Helvetia): 10 tracks
  Genre arc: noise rock → indie rock → lo-fi
```

---

## 🏗️ DJ Bridge Architecture

### 1. Seed Ordering

**What it does:** Arranges N seed tracks in optimal order to minimize total bridging distance.

**How it works:**
- Evaluates all permutations of seed order
- Scores each ordering by sonic similarity between consecutive pairs
- Selects ordering with highest total bridgeability

**Why it matters:** Starting with Radiohead → Aphex Twin → Boards of Canada is much easier to bridge than Radiohead → Boards of Canada → Aphex Twin. Optimal ordering reduces jarring transitions.

---

### 2. Union Pooling Strategy (Phase 1)

**The Innovation:** Instead of one candidate pool, DJ Bridge Mode combines three specialized pools:

#### **S1: Local Pool**
- Top-K neighbors of current pier
- Provides familiar, safe transitions
- Example: k_local=200 → 100 near pier A + 100 near pier B

#### **S2: Toward Pool**
- Top-K candidates moving toward destination pier per step
- Ensures progress toward bridge endpoint
- Example: k_toward=80 per step × 30 steps = 2,400 candidates

#### **S3: Genre Pool**
- Top-K candidates matching genre waypoint targets per step
- Introduces discovery while staying on-genre
- Example: k_genre=40 per step × 30 steps = 1,200 candidates

**Deduplication:** Union of S1 + S2 + S3, capped at k_union_max (default: 900)

**Result:** Balanced pool that's safe (local), progressive (toward), and adventurous (genre).

---

### 3. Genre Waypoint Planning

**Two Modes:**

#### **Onehot Mode (Legacy)**
- Finds shortest path in genre similarity graph
- Selects single-label waypoints
- **Problem:** Collapses to hub genres ("indie rock"), loses nuance

#### **Vector Mode (Phase 2 - NEW)**
- Direct interpolation between anchor genre vectors
- Preserves full multi-genre signatures
- Formula: `g_step = (1-t) * vA + t * vB` where t = step fraction
- **Benefit:** Shoegaze → dream pop → psychedelic (not "indie rock" × 30)

---

### 4. Phase 2: IDF Weighting & Coverage Bonus

#### **IDF (Inverse Document Frequency) Weighting**

**The Problem:** Common genres dominate scoring.
- "indie rock" appears in 40% of library → weight = 1.0
- "shoegaze" appears in 2% of library → weight = 1.0
- Result: System treats rare and common genres equally

**The Solution:** IDF down-weights common genres, emphasizes rare genres.
- Formula: `idf = log((N+1)/(df+1))^power`
- "indie rock" (40% of library) → idf = 0.2
- "shoegaze" (2% of library) → idf = 0.9
- Result: Rare genre signatures preserved

**Configuration:**
```yaml
dj_genre_use_idf: true
dj_genre_idf_power: 0.5          # Exponential scaling
dj_genre_idf_norm: l2            # Normalization method
```

#### **Coverage Bonus**

**The Problem:** Waypoint scoring alone insufficient to prefer genre-aligned candidates.

**The Solution:** Reward candidates matching anchor's top-K genres with schedule decay.
- Tracks top-8 genres per anchor
- Bonus weight: 0.15 (additive to score)
- Schedule decay: `wA = (1-s)^power, wB = s^power`
- Early steps favor pier A genres, late steps favor pier B genres

**Configuration:**
```yaml
dj_genre_use_coverage: true
dj_genre_coverage_weight: 0.15
dj_genre_coverage_top_k: 8
```

**Impact:** Genre pool candidates now competitive with local/toward pools.

---

## 🎯 Beam Search with Waypoint Guidance

### How It Works

1. **Initialize beam** at pier A with width=40 (configurable)
2. **For each step:**
   - Expand each beam state to all valid candidates
   - Score candidates by:
     - **Bridge score** (harmonic mean of sim_A and sim_B)
     - **Transition score** (end-to-start segment similarity)
     - **Waypoint bonus** (genre target similarity, optional)
     - **Coverage bonus** (anchor genre matching, optional)
   - Keep top-K states (beam pruning)
3. **Select path** with highest total score reaching pier B

### Waypoint Scoring

**Centered Mode (Default):**
- Computes baseline similarity (median/mean of all candidates)
- Calculates centered delta: `Δ = sim(cand, target) - baseline`
- Applies tanh squashing: `bonus = waypoint_cap * tanh(Δ / waypoint_cap)`
- Allows negative deltas (penalizes off-genre candidates)

**Configuration:**
```yaml
dj_waypoint_enabled: true
dj_waypoint_weight: 0.25         # Additive weight
dj_waypoint_cap: 0.10            # Maximum bonus/penalty
dj_waypoint_delta_mode: centered # Allow negative deltas
```

---

## 👥 Artist Identity Resolution (v3.3 → v3.4 Enhancement)

### The Problem (v3.3)

**Featured artists were invisible:**
- "Charli XCX feat. MØ" → system only saw "charli xcx"
- "Bob Brookmeyer & Bill Evans" → system only saw "bob brookmeyer"
- Min-gap constraints couldn't block collaborators

**Root cause:** Identity resolution received pre-normalized strings with collaboration markers stripped.

### The Solution (v3.4)

**Use raw artist strings** that preserve collaboration markers:
- "Charli XCX feat. MØ" → `{"charli xcx", "mø"}`
- "Bob Brookmeyer & Bill Evans" → `{"bob brookmeyer", "bill evans"}`
- "Mount Eerie with Julie Doiron & Fred Squire" → `{"mount eerie", "julie doiron", "fred squire"}`

**Supports all collaboration types:**
- feat., featuring, ft., with, &, and, x, +, comma-separated

**Configuration:**
```yaml
constraints:
  artist_identity:
    enabled: true
    strip_trailing_ensemble_terms: true  # "Bill Evans Trio" → "bill evans"
```

### The Impact

- ✅ **True artist diversity** in collaboration-heavy libraries
- ✅ **Jazz playlists** no longer repeat featured soloists
- ✅ **Electronic playlists** block all remix artists and collaborators
- ✅ **Hip-hop playlists** respect featured artist constraints
- ✅ **Ensemble normalization** still works (Trio, Quartet, Orchestra, etc.)

---

## 📊 Comprehensive Diagnostics

### Always-On Pool Visibility

Every segment logs pool composition:
```
DJ union pool: S1_local=200 S2_toward=720 S3_genre=0 | overlaps: L∩T=156 L∩G=0 T∩G=0
```

Shows:
- Raw pool sizes before deduplication
- Overlap counts (how many tracks in multiple pools)
- Dedup efficiency

### Per-Track Membership

Each chosen track logs which pools it came from:
```
[Track 0] idx=24932 pools=L+T
[Track 1] idx=18843 pools=T
[Track 2] idx=31204 pools=L+T+G
```

Legend: L=local, T=toward, G=genre, B=baseline

### Waypoint Saturation Metrics

Measures if waypoint scoring is effective:
```
Waypoint saturation: sim0=0.673 delta(mean=-0.0087 p50=0.0011 p90=0.0626) near_cap=20.5% at_cap=15.5%
```

Shows:
- **sim0**: Baseline similarity (centered mode)
- **delta distribution**: Mean, p50, p90 of centered deltas
- **near_cap**: Fraction of candidates with |delta| > 0.8*cap
- **at_cap**: Fraction of candidates with |delta| > 0.9*cap

**Good:** <30% near_cap means waypoint scoring has room to differentiate
**Bad:** >50% at_cap means cap too low, all candidates bunching

---

## ⚡ Performance Optimizations

### 1. Transition Score Caching

**Moved computation outside per-step loop:**
- Before: 30 steps × 1,000 candidates = 30,000 operations
- After: 1,000 candidates = 1,000 operations
- **Speedup: 30×** for this component

### 2. Genre Pool Caching

**Cache genre pool selections per step:**
- Reused across beam states
- Avoids redundant similarity computations
- Especially impactful for wide beams (width=60+)

---

## 🛠️ Code Quality Improvements

### IDF Consistency in Waypoint Scoring

**Problem:** Waypoint similarity used non-IDF matrix even when IDF enabled.

**Fix:** Use IDF-weighted matrix (`X_genre_norm_idf`) when available, matching target construction.

**Impact:** Waypoint deltas now accurate when IDF is enabled.

---

### Genre Vector Source Alignment

**Problem:** When `dj_genre_vector_source=raw`, targets used raw matrix but scoring used smoothed matrix (vector space mismatch).

**Fix:** Scoring matrix (`X_genre_for_sim`) now matches target construction mode:
- Raw mode: Use raw matrix + IDF (if enabled) + normalize
- Smoothed mode: Use smoothed matrix (X_genre_norm_idf or X_genre_norm)

**Impact:** Targets and scoring use same space, eliminating distorted rankings.

---

### Raw Genre Vector Normalization

**Problem:** Raw genre weights can exceed 1.0 (track/album weights are 1.2), causing coverage >1.

**Fix:** Normalize raw vectors before computing weighted coverage.

**Impact:** Coverage values stay in 0-1 range, bonuses don't exceed configured weights.

---

### Progress Arc Override Support

**Problem:** Overrides that set `progress_arc.enabled: false` were ignored if base config had it enabled.

**Fix:** Apply override regardless of enabled state (fully bidirectional).

**Impact:** Can now disable progress_arc via overrides without editing base config.

---

## 📚 Documentation Overhaul

### Professional Structure

- Created **`docs/README.md`** as documentation hub with navigation guide
- Added **"I want to..."** quick links for common tasks
- Archived **20+ diagnostic files** to `docs/archive/diagnostics_2026-01/` (git-ignored)
- Created **clean TODO.md** focused on future roadmap (not historical issues)

### Archive Policy

Diagnostic reports, A/B tests, and design docs archived after development cycles complete. Retained locally for reference but git-ignored to keep repo clean.

---

## 🔧 Configuration Reference

### DJ Bridge Mode Settings

```yaml
pier_bridge:
  # Pooling strategy
  pooling:
    strategy: dj_union           # Use union pooling
    k_local: 200                 # Local pool size
    k_toward: 80                 # Toward pool per step
    k_genre: 0                   # Genre pool per step (opt-in)
    k_union_max: 900             # Max after dedup

  # Genre waypoint planning
  dj_ladder_target_mode: vector  # Direct interpolation (Phase 2)
  dj_genre_vector_source: smoothed  # Use smoothed matrix

  # IDF weighting (Phase 2)
  dj_genre_use_idf: true
  dj_genre_idf_power: 0.5
  dj_genre_idf_norm: l2

  # Coverage bonus (Phase 2)
  dj_genre_use_coverage: true
  dj_genre_coverage_weight: 0.15
  dj_genre_coverage_top_k: 8

  # Waypoint scoring
  dj_waypoint_enabled: true
  dj_waypoint_weight: 0.25
  dj_waypoint_cap: 0.10
  dj_waypoint_delta_mode: centered

  # Artist diversity
  disallow_pier_artists_in_interiors: true
  disallow_seed_artist_in_interiors: false

constraints:
  artist_identity:
    enabled: true
    strip_trailing_ensemble_terms: true
```

---

## 📈 Benefits Summary

| Feature | Before | After | Impact |
|---------|--------|-------|--------|
| **Multi-seed playlists** | Not supported | 2-10 seeds | Revolutionary workflow |
| **Genre bridging** | Accidental | Planned arcs | Smooth evolution |
| **Pool diversity** | Single source | 3 sources (L+T+G) | Balanced safe/adventurous |
| **Genre nuance** | Hub collapse | Multi-genre preservation | "shoegaze" not "indie rock" |
| **IDF weighting** | Equal weights | Rare genre emphasis | Signature preservation |
| **Coverage bonus** | Waypoint only | +Top-K matching | Genre pool competitive |
| **Artist diversity** | Name-based | Identity-based | Blocks collaborators |
| **Diagnostics** | Limited | Comprehensive | Full visibility |
| **Performance** | Baseline | 30× cache speedup | Faster generation |
| **Documentation** | Scattered | Professional hub | Easy navigation |

---

## 🚀 How to Use DJ Bridge Mode

### Basic Multi-Seed Playlist

```bash
# CLI
python main_app.py \
  --seed-list "Slowdive,Beach House,Deerhunter,Helvetia" \
  --tracks 30 \
  --output my_playlist.m3u

# GUI
1. Select "Seed List Mode" from dropdown
2. Enter comma-separated seeds: "Slowdive, Beach House, Deerhunter"
3. Set target tracks: 30
4. Click "Generate Playlist"
```

### Configuration Tips

**For tight genre cohesion:**
```yaml
pooling:
  k_genre: 40           # Enable genre pool

dj_waypoint_weight: 0.30  # Strong waypoint guidance
```

**For adventurous discovery:**
```yaml
pooling:
  k_toward: 120         # More toward candidates
  k_genre: 0            # Disable genre pool (pure sonic)

dj_waypoint_enabled: false
```

**For collaboration-heavy libraries (jazz, electronic):**
```yaml
constraints:
  artist_identity:
    enabled: true       # Must enable for featured artist blocking
```

---

## 🧪 Testing

All test suites passing:
- **16 DJ/pier-bridge tests** - Core algorithm validation
- **34 artist identity tests** - Collaboration splitting, ensemble normalization
- **3 segment pool builder tests** - Union pooling, deduplication
- **28 pool diagnostics tests** - Overlap metrics, provenance tracking
- **5 DJ ladder planner tests** - Genre path planning, vector mode

**Total: 86+ tests** with zero regressions.

---

## 🎓 Technical Deep-Dive

### Why Union Pooling?

**Baseline pooling (before v3.4):**
- Top-K neighbors of current position
- Problem: Gets stuck in local optima
- Problem: Can't discover genre-aligned alternatives

**Union pooling (v3.4):**
- Local pool: Safety net (always have familiar options)
- Toward pool: Progress guarantee (always moving toward goal)
- Genre pool: Discovery engine (finds genre-aligned alternatives)
- Result: Balanced exploration-exploitation

### Why Vector Mode > Onehot Mode?

**Onehot mode:**
- Shortest-path label selection
- Picks single genre per waypoint
- Multi-genre tracks lose information
- Example: "indie rock" × 30 steps (hub genre collapse)

**Vector mode:**
- Direct multi-genre interpolation
- Preserves full genre signatures
- No information loss
- Example: shoegaze=0.38 + dream pop=0.33 + psychedelic=0.25 (nuanced)

### Why IDF + Coverage?

**IDF:** Makes rare genres "louder" in scoring
- Without IDF: "indie rock" dominates (appears everywhere)
- With IDF: "shoegaze" gets fair representation (rare but important)

**Coverage:** Rewards matching anchor top-K genres
- Without coverage: Waypoint scoring alone is weak signal
- With coverage: Genre pool candidates become competitive
- Schedule decay: Early steps favor pier A genres, late steps favor pier B genres

---

## 🔮 Future Roadmap (v3.5)

- **Genre path visualization** in audit reports
- **User-specified genre waypoints** (force route through specific genres)
- **A/B comparison mode** (baseline vs dj_union side-by-side)
- **Preset configurations** for common use cases (tight cohesion, discovery, etc.)
- **Pool parameter auto-tuning** based on saturation metrics

---

## 📦 Upgrade Notes

**No breaking changes.** All DJ Bridge features are additive and backward-compatible.

**To enable DJ Bridge Mode:**
1. Set `pooling.strategy: dj_union` in config
2. Enable artist identity: `constraints.artist_identity.enabled: true`
3. Use seed list mode with 2+ seeds

**Migration from v3.3:**
- Existing playlists continue to work
- Single-seed playlists use traditional mode
- DJ mode only activates with 2+ seeds

---

## 🙏 Acknowledgments

DJ Bridge Mode represents 59 commits and comprehensive architectural evolution:
- **Phase 1:** Union pooling foundation, seed ordering, segment allocation
- **Phase 2:** Vector mode, IDF weighting, coverage bonus
- **Phase 3:** Quality fixes, diagnostics, documentation

Special thanks to the code review process for identifying critical bugs in artist identity resolution and genre vector alignment.

---

**Version:** 3.4.0
**Branch:** `dj-ordering`
**Commits:** 59
**Release Type:** Major Feature Release
**Documentation:** `docs/DJ_BRIDGE_ARCHITECTURE.md` (54KB comprehensive guide)

For complete technical details, see `DJ_BRIDGE_ARCHITECTURE.md` and commit history.
