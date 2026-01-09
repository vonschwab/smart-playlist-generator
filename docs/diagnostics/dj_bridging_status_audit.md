# DJ Bridging End-to-End Audit

**Date:** 2026-01-08
**Status:** Partial implementation; transitions still sub-optimal in real runs
**Auditor:** Claude Sonnet 4.5

---

## Executive Summary

DJ-style bridging infrastructure is **90% complete** with route planning, union pooling, waypoint scoring, and diagnostics all implemented. However, **user-perceived transition quality** has not materially improved, especially with semi-disparate anchors. Root cause analysis reveals **3 critical issues**:

1. **Config Parsing Bug (HIGH)**: User config key `dj_bridging.dj_pooling_strategy` is **ignored**; must use nested `dj_bridging.pooling.strategy` instead.
2. **Waypoint Scoring Suppression (MEDIUM)**: Tie-break bands cause waypoint scores to be applied only on close-call edges, limiting genre guidance influence.
3. **Pool-to-Selection Gap (MEDIUM)**: dj_union populates pool with genre candidates, but beam scoring often selects local/toward candidates instead (diagnostic shows `chosen_from_genre_count` near zero).

---

## What Works Today (✅)

### 1. Multi-Seed Pier-Bridge Architecture ✅
- **Location**: `src/playlist/pier_bridge_builder.py:2600-3800`
- Seeds become fixed piers; beam search builds bridge segments
- Single-seed arc mode works (seed as both start/end)
- Segment-local pools with 1-per-artist enforcement
- Cross-segment min_gap enforcement via boundary tracking

### 2. DJ Config Parsing (Partial) ⚠️
- **Location**: `src/playlist/pipeline.py:883-1114`
- Parses `dj_bridging` block from config
- **Works**: nested keys like `dj_bridging.pooling.strategy`
- **Broken**: flat keys like `dj_bridging.dj_pooling_strategy` (not in expected structure)
- **Evidence**: User logs show `dj_pooling_strategy=baseline` despite user setting `dj_union`

### 3. Route Planning (Linear / Arc / Ladder) ✅
- **Location**: `src/playlist/pier_bridge_builder.py:1012-1244` (`_build_genre_targets`)
- **Linear**: Simple interpolation `(1-t)*g_a + t*g_b`
- **Arc**: Arc-shaped progress curve
- **Ladder**: Genre graph shortest-path with waypoint labels
  - Loads `data/genre_similarity.yaml` graph
  - Computes ladder labels via shortest-path (BFS)
  - Converts labels to one-hot or smoothed vectors
  - Interpolates between waypoints for per-step targets
- **Diagnostics**: Waypoint labels logged; A/B runner shows hash divergence between route shapes
- **Evidence**: `docs/diagnostics/dj_ladder_route_ab.md` shows ladder finds waypoints and changes tracklist

### 4. Seed Auto-Ordering ✅
- **Location**: `src/playlist/pier_bridge_builder.py:2853-2862`
- Evaluates all permutations by bridgeability heuristic
- Uses sonic + genre + bridge-harmonic-mean weighted score
- User can override with `seed_ordering: fixed`

### 5. DJ Union Pooling ✅
- **Location**: `src/playlist/segment_pool_builder.py:563-702` (`_build_dj_union_pool`)
- **Strategy**: `dj_union` builds union of 3 sources:
  - **S1 (local)**: top-K neighbors to pier_a and pier_b
  - **S2 (toward-B)**: top-K toward interpolated sonic targets per step
  - **S3 (genre)**: top-K toward interpolated genre waypoint targets per step
- **Per-step caching**: Reuses top-K computations within a segment (stride control)
- **Union cap**: Applies global cap after dedupe (scores by harmonic bridge mean)
- **Diagnostics**: Tracks source counts, overlap, attribution (chosen-from-local/toward/genre)
- **Evidence**: Stress A/B scripts show leverage under stress (more baseline-only pool collapses)

### 6. Waypoint Scoring in Beam Search ✅
- **Location**: `src/playlist/pier_bridge_builder.py:2345-2361, 2404-2408`
- Computes `waypoint_sim = dot(X_genre_norm[cand], g_target)` for each step
- Applies `_waypoint_delta()`:
  - Base: `waypoint_weight * sim` (capped by `waypoint_cap`)
  - Penalty if `sim < waypoint_floor`: subtracts `waypoint_penalty * (floor - sim)`
- **Tie-break band**: Only applies waypoint delta when `(best_score - combined_score) <= waypoint_tie_break_band`
- **Evidence**: Code exists and is called; diagnostics show `g_targets` computed

### 7. Connector Bias ✅
- **Location**: Config `dj_bridging.connector_bias.*`, parsed in `pipeline.py:992-1013`
- **Purpose**: Cap internal-connector usage per segment (linear vs adventurous modes)
- **Evidence**: Diagnostics show pool composition changes when enabled
- **Caveat**: User reports it doesn't always change ordering/hashes

### 8. Diagnostics & Visibility ✅
- **Location**: `src/playlist/pier_bridge_diagnostics.py`, segment stats in builder
- Tracks:
  - Pool sizes (initial, final, before/after gates)
  - Chosen-from source counts (local, toward, genre, baseline, connectors)
  - Waypoint labels per segment (route_shape, vector mode, label list)
  - Ladder warnings (fallback to linear, missing labels, smoothed failures)
  - Progress arc metrics (mean abs dev, p50/p90, max jump)
  - Overlap diagnostics (Jaccard, baseline-only, union-only counts)
- **A/B runners**:
  - `run_dj_union_pooling_stress_ab.py`: Pool strategy stress test
  - `run_dj_connector_bias_ab.py`: Connector bias impact
  - `run_dj_ladder_route_ab.py`: Route shape variants (linear/ladder/smoothed)
  - `run_dj_relaxation_micro_pier_demo.py`: Micro-pier fallback

---

## What's Missing / Weak (❌ or ⚠️)

### 1. Config Parsing: Flat Key Aliases Missing ❌ (HIGH PRIORITY)
- **Problem**: User config snippet:
  ```yaml
  pier_bridge:
    dj_bridging:
      enabled: true
      route_shape: ladder
      dj_pooling_strategy: dj_union  # ← IGNORED!
  ```
- **Why**: Parser expects `dj_bridging.pooling.strategy`, not flat `dj_pooling_strategy`
- **Impact**: User's `dj_union` setting is not applied; logs show `dj_pooling_strategy=baseline`
- **Evidence**:
  - `config.example.yaml:253` shows correct nested path: `pooling.strategy: baseline`
  - `pipeline.py:963-981` parses `dj_raw.get("pooling")` → `pooling_raw.get("strategy")`
  - Flat key `dj_pooling_strategy` is only in PierBridgeConfig dataclass defaults, not parsed
- **Fix**: Add flat-key fallback aliases in `pipeline.py:883-1114` parsing block

### 2. Waypoint Scoring: Tie-Break Band Suppresses Influence ⚠️ (MEDIUM)
- **Problem**: `waypoint_tie_break_band` (default `null`) means waypoint scoring is applied to ALL edges, BUT if set, it's only applied to edges within the band of best_score
- **Code**: `pier_bridge_builder.py:2405-2408`
  ```python
  if waypoint_enabled:
      if waypoint_tie_break_band is None:
          combined_score += _waypoint_delta(waypoint_sim)
      elif (best_score - combined_score) <= float(waypoint_tie_break_band):
          combined_score += _waypoint_delta(waypoint_sim)
  ```
- **Impact**: If bridge/transition scores are already decisive, waypoint never influences selection
- **Evidence**: User reports "transitions still feel off" despite ladder + dj_union enabled
- **Hypothesis**: Waypoint weight (0.15 default) is too low vs bridge weight (0.6) + transition weight (0.4)
- **Fix Options**:
  - Remove tie-break band for waypoint (always apply)
  - Increase waypoint_weight to 0.25-0.30
  - Add "waypoint bonus pool" that guarantees top-K waypoint-aligned candidates make it into beam
  - Add per-step diagnostics showing waypoint_sim distribution for chosen vs rejected edges

### 3. Pool-to-Selection Gap: Genre Candidates Not Chosen ❌ (MEDIUM)
- **Problem**: dj_union adds genre candidates to pool, but beam search often selects local/toward candidates instead
- **Evidence**: Diagnostics show:
  ```
  chosen_from_local_count: 13
  chosen_from_toward_count: 24
  chosen_from_genre_count: 1  ← NEARLY ZERO!
  ```
- **Why**: Genre candidates may have:
  - Lower transition scores (end-start similarity)
  - Lower bridge scores (harmonic mean of sim_a, sim_b)
  - Waypoint bonus is capped and may not rescue them
- **Implication**: Even with `dj_union` enabled, the **effective pool** used by beam search is mostly baseline-like
- **Fix Options**:
  - **Option A (Increase waypoint influence)**: Raise waypoint_weight, remove tie-break band, increase waypoint_cap
  - **Option B (Improve pool construction)**: In `dj_union`, score candidates by `bridge_score + waypoint_bonus` BEFORE taking top-K, not just bridge_score
  - **Option C (Guarantee genre representation)**: Reserve top-N beam slots for genre-aligned candidates (diversity constraint)

### 4. Connector Bias: Visible in Pool, Not in Ordering ⚠️ (LOW)
- **Problem**: Diagnostics show connector bias changes pool composition, but user reports it doesn't always change final ordering/hashes
- **Evidence**: User notes from connector bias A/B runner
- **Why**: If connectors have same scoring profile as external candidates, bias doesn't help
- **Fix**: Add connector priority scoring bonus (small, e.g., +0.05 to combined_score) when enabled

### 5. Redundant Gating May Collapse Pool Too Early ⚠️ (LOW)
- **Problem**: Multiple hard gates applied in sequence:
  1. Used track IDs
  2. Allowed set
  3. Seed/pier artist policies
  4. Track-key collision
  5. Bridge floor gate (min(sim_a, sim_b) >= bridge_floor)
  6. Transition floor gate (per-edge during beam expansion)
- **Evidence**: Pool diagnostics show steep drops: `base_universe=1200 → eligible_after_structural=800 → pass_bridge_floor=200 → final=80`
- **Impact**: By the time we select top-K, we may have removed the very connectors/waypoint tracks we need
- **Fix**: Relax bridge_floor for genre/connector candidates (two-tier gating: strict for baseline, lenient for genre)

### 6. No Per-Step Waypoint Diagnostics ❌ (HIGH for debugging)
- **Problem**: We log segment-level "chosen_from_genre_count" but not per-edge waypoint_sim values for chosen edges
- **Impact**: Can't debug why waypoint scoring isn't influencing selection
- **Fix**: Add per-edge diagnostics in beam search final path:
  - `edge_waypoint_sim`: waypoint similarity for chosen candidate
  - `edge_waypoint_delta`: actual waypoint delta applied to score
  - `edge_best_waypoint_sim_rejected`: best waypoint_sim among rejected candidates
  - Aggregate stats: `mean_waypoint_sim_chosen`, `mean_waypoint_sim_rejected`, `waypoint_rescue_count`

---

## Minimal Next Steps Plan

### Phase 1: Fix Config Parsing + Add Waypoint Diagnostics (Highest Impact, Lowest Risk)

**Goal**: Ensure user's `dj_pooling_strategy: dj_union` takes effect AND add visibility into waypoint scoring

**Changes**:
1. **Config parsing fix** (`src/playlist/pipeline.py:883-1114`):
   - Add fallback: if `dj_raw.get("dj_pooling_strategy")` exists, use it when nested `pooling.strategy` is absent
   - Log warning when flat key is used (deprecation path)
   - Update `config.example.yaml` to show ONLY nested form (remove flat key from defaults)

2. **Waypoint diagnostics** (`src/playlist/pier_bridge_builder.py:_beam_search_segment`):
   - Collect per-edge waypoint_sim for chosen path
   - Aggregate stats: mean, p50, p90 for chosen vs rejected candidates
   - Add to segment diagnostics: `waypoint_influence_stats` dict

**Tests**:
- Unit test: `tests/unit/test_dj_config_parsing.py`
  - Verify flat key `dj_pooling_strategy` is parsed correctly (with deprecation warning)
  - Verify nested key `pooling.strategy` takes precedence
  - Verify default remains `baseline` when neither is set
- Unit test: `tests/unit/test_dj_waypoint_diagnostics.py`
  - Verify waypoint_sim is collected for chosen edges
  - Verify stats are aggregated correctly (mean, p50, p90)

**Diagnostic script**: Extend `run_dj_ladder_route_ab.py`
- Add column: `mean_waypoint_sim_chosen`
- Add column: `waypoint_applied_count` (edges where delta != 0)
- Add column: `waypoint_rescue_count` (edges where waypoint delta changed ranking)

**Acceptance**:
- User can set `dj_bridging.dj_pooling_strategy: dj_union` OR `dj_bridging.pooling.strategy: dj_union` and logs show correct strategy
- Diagnostic output shows per-segment waypoint influence stats
- A/B runner shows whether waypoint scoring is actually moving the needle

---

### Phase 2: Increase Waypoint Influence (Conditional on Phase 1 Diagnostics)

**Goal**: If Phase 1 diagnostics show waypoint_sim is high but delta is capped/suppressed, increase influence

**Changes**:
1. **Remove tie-break band for waypoints** (`pier_bridge_builder.py:2404-2408`):
   - Always apply waypoint delta (remove conditional on `waypoint_tie_break_band`)
   - OR: Set default `waypoint_tie_break_band = 0.10` (wide enough to matter)

2. **Increase default waypoint_weight** (`config.example.yaml:247`):
   - Change from `0.15` to `0.25` (if diagnostics show capping isn't the issue)

3. **Increase waypoint_cap** (`config.example.yaml:251`):
   - Change from `0.05` to `0.10` (allow larger bonuses)

**Tests**:
- Unit test: `tests/unit/test_dj_waypoint_scoring.py`
  - Verify waypoint delta is applied without tie-break suppression
  - Verify cap is respected (no unbounded bonuses)
  - Verify penalty is applied when sim < floor

**Diagnostic script**: Re-run `run_dj_ladder_route_ab.py`
- Compare `chosen_from_genre_count` before/after
- Compare `mean_waypoint_sim_chosen` before/after
- Compare tracklist hashes (should change if waypoint influence increases)

**Acceptance**:
- `chosen_from_genre_count` increases from ~0-1 to ~5-10 per segment
- `waypoint_rescue_count` > 0 (waypoint scoring changed edge rankings)
- Listener-perceived transitions improve (qualitative test)

---

### Phase 3: Improve Pool Construction Scoring (Optional, If Phase 2 Insufficient)

**Goal**: Make dj_union pool construction prioritize genre-aligned candidates BEFORE taking top-K

**Changes**:
1. **Hybrid pool scoring** (`src/playlist/segment_pool_builder.py:673-687`):
   - Current: scores by `bridge_score` (harmonic mean)
   - New: scores by `bridge_score + waypoint_bonus_preview`
   - `waypoint_bonus_preview = waypoint_weight * dot(X_genre_norm[cand], g_target_midpoint)`
   - Apply BEFORE union cap selection

2. **Genre-aware k_union_max** (optional):
   - Reserve top-N genre candidates (e.g., top 20% of union) regardless of bridge score
   - Ensures genre candidates survive union cap

**Tests**:
- Unit test: `tests/unit/test_dj_union_pool_scoring.py`
  - Verify hybrid scoring ranks genre-aligned candidates higher
  - Verify reserved slots are filled with top genre candidates

**Diagnostic script**: Extend `run_dj_union_pooling_stress_ab.py`
- Add column: `pool_top_k_genre_fraction` (fraction of top-K pool that are genre candidates)
- Compare before/after hybrid scoring

**Acceptance**:
- `pool_top_k_genre_fraction` increases from ~0.10 to ~0.30
- `chosen_from_genre_count` increases further (compounding with Phase 2)

---

### Phase 4: Add Per-Edge Waypoint Diagnostics to Run Audits (Optional, for Deep Debugging)

**Goal**: Expose per-edge waypoint_sim in run audit reports for manual inspection

**Changes**:
1. **Extend edge_scores dict** (`pier_bridge_builder.py:3435-3465`):
   - Add `waypoint_sim` to each edge dict
   - Add `waypoint_delta_applied` to each edge dict
   - Add `waypoint_step_fraction` (which step target was used)

2. **Extend run audit report** (`src/playlist/run_audit.py`):
   - Add "Waypoint Scoring" section to markdown report
   - Show per-edge table: `step | track_id | waypoint_sim | delta_applied | bridge_score | final_score`
   - Aggregate stats: mean waypoint_sim across all edges

**Acceptance**:
- Run audit markdown shows per-edge waypoint details
- User can inspect specific edges to understand why waypoint didn't influence

---

## Recommended Immediate Action

**Implement Phase 1** (config fix + waypoint diagnostics) in this session:
1. Fix config parsing to accept flat key `dj_pooling_strategy`
2. Add waypoint influence diagnostics to segment stats
3. Write unit tests for both
4. Extend `run_dj_ladder_route_ab.py` to show waypoint stats
5. Run diagnostic and review output

**Expected outcome**: We'll see whether:
- User's `dj_union` setting now takes effect (pool sources change)
- Waypoint scoring is being applied (delta != 0)
- Waypoint scoring is influencing selection (chosen_from_genre_count > 0)

If diagnostics show waypoint_sim is high but delta is capped, proceed to Phase 2.
If diagnostics show waypoint_sim is low (candidates don't align with targets), investigate route planning or genre metadata quality.

---

## Evidence Summary

| Component | Status | Evidence |
|-----------|--------|----------|
| Pier-bridge architecture | ✅ Working | Code review, existing runs succeed |
| Seed auto-ordering | ✅ Working | Logs show ordered seeds, diagnostics show bridgeability scores |
| Route planning (linear/arc/ladder) | ✅ Working | `dj_ladder_route_ab.md` shows waypoint labels, hash divergence |
| dj_union pooling sources | ✅ Working | Diagnostics show S1/S2/S3 counts, overlap metrics |
| Waypoint scoring implementation | ✅ Exists | Code review confirms `_waypoint_delta()` is called |
| Waypoint scoring influence | ❌ Weak | `chosen_from_genre_count` near zero in real runs |
| Config parsing (nested keys) | ✅ Working | Nested `pooling.strategy` is parsed correctly |
| Config parsing (flat keys) | ❌ Broken | Flat `dj_pooling_strategy` is ignored (user's issue) |
| Connector bias | ⚠️ Partial | Changes pool composition but not always ordering |
| Diagnostics visibility | ✅ Good | Segment stats, A/B runners, run audits all exist |
| Per-edge waypoint visibility | ❌ Missing | No per-edge waypoint_sim in diagnostics |

---

## File/Line Reference Index

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Config parsing (dj_bridging block) | `src/playlist/pipeline.py` | 883-1114 | Parses dj_bridging from pier_bridge overrides |
| Config example (nested keys) | `config.example.yaml` | 237-297 | Shows `dj_bridging.pooling.strategy` (correct) |
| PierBridgeConfig dataclass | `src/playlist/pier_bridge_builder.py` | 64-176 | Defaults for all DJ config |
| Pier-bridge main builder | `src/playlist/pier_bridge_builder.py` | 2600-3800 | Orchestrates segments, seed ordering, global state |
| Beam search segment | `src/playlist/pier_bridge_builder.py` | 1941-2484 | Core beam search with scoring |
| Waypoint scoring logic | `src/playlist/pier_bridge_builder.py` | 2345-2361, 2404-2408 | Computes waypoint_sim, applies delta |
| Waypoint delta function | `src/playlist/pier_bridge_builder.py` | 2040-2051 | Caps, penalty, floor logic |
| Route planning (genre targets) | `src/playlist/pier_bridge_builder.py` | 1012-1244 | Builds per-step genre waypoint vectors |
| Ladder waypoint planning | `src/playlist/pier_bridge_builder.py` | 1080-1244 | Shortest-path, label-to-vector, interpolation |
| DJ union pool builder | `src/playlist/segment_pool_builder.py` | 563-702 | S1/S2/S3 sources, caching, union cap |
| Segment pool config | `src/playlist/segment_pool_builder.py` | 27-135 | Config dataclass with dj_union params |
| Segment diagnostics | `src/playlist/pier_bridge_diagnostics.py` | 24-109 | Dataclass with route_shape, waypoint labels, etc. |
| A/B runner (ladder route) | `scripts/run_dj_ladder_route_ab.py` | 1-100 | Compares linear vs ladder vs smoothed |
| A/B runner (union pooling) | `scripts/run_dj_union_pooling_stress_ab.py` | Full | Stress test for dj_union vs baseline |
| A/B runner (connector bias) | `scripts/run_dj_connector_bias_ab.py` | Full | Connector bias impact |
| Unit tests (ladder planner) | `tests/unit/test_dj_ladder_planner.py` | Full | Tests route planning logic |
| Unit tests (relaxation/micro-pier) | `tests/unit/test_dj_relaxation_micro_pier.py` | Full | Tests fallback mechanisms |

---

## Conclusion

DJ bridging is **architecturally sound** with all major components implemented. The gap between implementation and perceived quality stems from:
1. **Configuration bug** preventing user's pooling strategy from taking effect
2. **Weak waypoint scoring influence** due to low weight, tight cap, and tie-break suppression
3. **Pool-to-selection gap** where genre candidates are added but not chosen

**Phase 1 fix (config + diagnostics) is highest priority** and will immediately clarify whether the issue is configuration vs scoring vs pool construction. Estimated effort: 2-3 hours (code + tests + diagnostic run).
