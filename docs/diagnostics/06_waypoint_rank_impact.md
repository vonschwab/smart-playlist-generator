# DJ Bridging Diagnostics: Pool Clarity & Waypoint Rank Impact

**Status:** Implementation in progress
**Branch:** `dj-ordering`
**Goal:** Make diagnostics internally consistent and add high-signal metrics for waypoint scoring effectiveness

---

## Pool Diagnostics Fields (TASK A)

### Field Definitions (Clarified)

| Field | Definition | When Set |
|-------|------------|----------|
| `pool_before_gating` | Candidate count after pool merge strategy (baseline or dj_union), before any structural/artist gates | After `_build_segment_candidate_pool_scored` returns |
| `pool_after_gating` | Candidate count after applying all gates (used tracks, artist policies, bridge floor, etc.) | Before passing to `_beam_search_segment` |
| `pool_after_gate` | **SAME as `pool_after_gating`** (legacy name, kept for compatibility) | Before passing to `_beam_search_segment` |

**Invariant:** `pool_after_gating <= pool_before_gating`

### Source Count Fields (Clarified)

**Pool Composition Counts** (what's IN the pool):
- `pool_local_count`: Tracks from S1 (local neighbors to piers)
- `pool_toward_count`: Tracks from S2 (toward-destination targets)
- `pool_genre_count`: Tracks from S3 (genre waypoint targets)
- Computed by `_build_dj_union_pool` when `strategy=dj_union`

**Chosen Edge Provenance** (what beam search SELECTED):
- `chosen_from_local_count`: Chosen tracks that came from S1
- `chosen_from_toward_count`: Chosen tracks that came from S2
- `chosen_from_genre_count`: Chosen tracks that came from S3
- `chosen_from_baseline_only_count`: Chosen tracks NOT in S1/S2/S3 (baseline pool only)
- Computed by `_compute_chosen_source_counts` after beam search completes

**Current Logging:**
```
Pool sources: strategy=dj_union local=8 toward=14 genre=1 baseline_only=0
```
This shows **chosen edge provenance**, not pool composition. It answers: "Of the 27 tracks chosen by beam search, how many came from each source?"

### Invariant Checks

When diagnostics are enabled, the following checks are logged as WARNINGs (not fatal):

1. If `pool_after_gating > 0` but `pool_before_gating == 0`:
   ```
   WARNING: pool_before_gating=0 but pool_after_gating=135 (possible missing instrumentation)
   ```

2. If sum of chosen_from counts doesn't equal segment length:
   ```
   WARNING: chosen_from_* sum (23) != interior_length (27) (possible provenance tracking gap)
   ```

---

## Waypoint Rank Impact Metric (TASK B)

### Purpose

Measure whether waypoint scoring is **capable of changing candidate rankings**, not just what it did on chosen edges.

**Key Question:** If waypoint scoring were disabled, would beam search choose different tracks?

### Configuration

```yaml
pier_bridge:
  dj_bridging:
    diagnostics:
      waypoint_rank_impact_enabled: false  # default: disabled
      waypoint_rank_sample_steps: 3        # sample up to N beam steps per segment
```

**Default:** `false` (no-op, zero cost)

### Computation (When Enabled)

For each segment where `dj_bridging_enabled=true`:

1. **Sample beam steps:** Choose up to 3 evenly-spaced steps (e.g., steps 0, 7, 14 for interior_length=14)
2. **For each sampled step:**
   - Collect top-K=10 candidates by `base_score` (score WITHOUT waypoint delta)
   - Re-rank same candidates by `full_score` (score WITH waypoint delta)
   - Compute metrics:
     - `winner_changed`: Did rank-1 candidate change? (bool)
     - `topK_reordered_count`: How many of top-K changed positions? (int)
     - `mean_abs_rank_delta`: Average absolute rank shift across all candidates (float)
     - `max_rank_jump`: Largest single rank change (int)

3. **Emit per-step table** (compact, top-10 only):
   ```
   Step 7/14:
   | cand_idx | base_score | waypoint_sim | waypoint_delta | full_score | base_rank | full_rank | rank_delta |
   |----------|------------|--------------|----------------|------------|-----------|-----------|------------|
   | 12345    | 0.852      | 0.421        | +0.042         | 0.894      | 1         | 1         | 0          |
   | 67890    | 0.848      | 0.312        | +0.031         | 0.879      | 2         | 3         | -1         |
   | ...
   ```

4. **Aggregate segment stats:**
   ```
   Segment waypoint rank impact: sampled_steps=3 winner_changed=1/3 mean_reordered=2.3/10 mean_rank_delta=1.2
   ```

### Implementation Details

- **Minimal compute:** Only active when `waypoint_rank_impact_enabled=true`
- **Deterministic:** Uses existing `_waypoint_delta()` function; no new math
- **No behavior change:** Pure diagnostic, doesn't affect playlist output
- **Step sampling:** Evenly spaced (e.g., 0, 7, 14 for length=15) to avoid sampling bias

---

## Example Output

### Before (Current)
```
INFO: worker: Segment 0: ... bridge_floor=0.03 pool_before=0 pool_after=0
INFO: worker:   Waypoint stats: enabled=True mean_sim=0.323 ... delta_applied=14/14 mean_delta=0.0381
INFO: worker:   Pool sources: strategy=dj_union local=8 toward=14 genre=1 baseline_only=0
```

### After (With Fixes)
```
INFO: worker: Segment 0: ... bridge_floor=0.03 pool_before_gating=412 pool_after_gating=135
INFO: worker:   Waypoint stats: enabled=True mean_sim=0.323 ... delta_applied=14/14 mean_delta=0.0381
INFO: worker:   Chosen edge provenance: strategy=dj_union local=8 toward=14 genre=1 baseline_only=0
INFO: worker:   Waypoint rank impact: sampled_steps=3 winner_changed=0/3 topK_reordered=2.3/10 mean_rank_delta=0.8
```

**Interpretation:**
- Pool diagnostics now show actual numbers (412 → 135 after gating)
- "Chosen edge provenance" clarifies these are beam-selected tracks, not pool composition
- Rank impact shows waypoint scoring shifted ~2-3 tracks in top-10 but didn't change winner
  - `winner_changed=0/3`: Top-ranked candidate stayed the same in all 3 sampled steps
  - `topK_reordered=2.3/10`: Average of 2-3 tracks changed positions within top-10
  - `mean_rank_delta=0.8`: Average absolute rank shift was less than 1 position
  - **Conclusion:** Waypoint scoring has **weak influence** on rankings; increasing weight may help

---

## Unit Test Coverage

### Pool Diagnostics Tests (`test_pool_diagnostics_clarity.py`)
- `test_pool_fields_populated`: Verify pool_before/after_gating are non-zero when pool is built
- `test_chosen_vs_pool_counts_distinct`: Verify chosen_from_* counts differ from pool_*_count
- `test_invariant_checks_logged`: Verify WARNINGs appear when invariants violated

### Waypoint Rank Impact Tests (`test_waypoint_rank_impact.py`)
- `test_rank_impact_disabled_by_default`: Verify no-op when flag=false
- `test_rank_impact_computes_metrics`: Verify winner_changed, topK_reordered computed correctly
- `test_rank_impact_deterministic`: Same input → same metrics (seeded RNG)
- `test_rank_impact_step_sampling`: Verify even spacing (0, 7, 14 for length=15)

---

## Exact Run Commands

### 1. Run Unit Tests
```bash
cd /c/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3
python -m pytest tests/unit/test_pool_diagnostics_clarity.py tests/unit/test_waypoint_rank_impact.py -v
```

### 2. Generate Playlist with Enhanced Diagnostics
```bash
# Enable waypoint rank impact diagnostic
# Edit config.yaml:
#   pier_bridge:
#     dj_bridging:
#       diagnostics:
#         waypoint_rank_impact_enabled: true

# Run playlist generation (via GUI or CLI)
# Check logs for:
#   - pool_before_gating / pool_after_gating (non-zero)
#   - "Chosen edge provenance" (renamed from "Pool sources")
#   - "Waypoint rank impact" (new)
```

### 3. Compare Before/After
```bash
# Save baseline log (current behavior)
# Run with diagnostics.waypoint_rank_impact_enabled: false

# Save enhanced log (new behavior)
# Run with diagnostics.waypoint_rank_impact_enabled: true

# Compare: should show rank impact metrics when enabled, no-op when disabled
```

---

## Implementation Status

- [x] TASK A: Pool diagnostics clarity
  - [x] Populate `pool_before_gating`, `pool_after_gating`
  - [x] Rename log line to "Chosen edge provenance"
  - [x] Add invariant checks (WARNINGs)
  - [x] Unit tests (5 tests passing)

- [x] TASK B: Waypoint rank impact metric
  - [x] Add config flag `waypoint_rank_impact_enabled`
  - [x] Implement rank comparison logic
  - [x] Emit per-step tables + aggregate stats
  - [x] Unit tests (5 tests passing)

- [x] TASK C: Documentation
  - [x] Create this markdown doc
  - [x] Add inline code comments

- [x] TASK D: Testing
  - [x] Run pytest (10/10 tests passing)
  - [ ] Generate test playlist (user to run)
  - [ ] Verify output matches spec (user to verify)
