# Feature Flag Migration Plan
## Gradual Enablement Strategy for Refactored Code

**Last Updated:** 2026-01-04
**Status:** Ready for Production Rollout

---

## Overview

This document provides a detailed plan for gradually enabling refactored code in production using the feature flag system. The strategy follows a **low-risk, incremental approach** with validation breaks between each migration step.

**Key Principles:**
1. **One flag at a time:** Enable only one feature flag per migration step
2. **Monitor actively:** Watch for behavior changes, errors, and performance impact
3. **Validate thoroughly:** Run tests and compare with golden files before proceeding
4. **Rollback easily:** Disable flag immediately if issues arise
5. **Document results:** Track metrics and decisions for each migration

---

## Migration Order (Low Risk → High Risk)

### Phase 1: Low-Risk Flags (Week 1-2)

These flags have minimal impact and extensive test coverage:

1. **use_variant_cache** (Phase 5.3)
   - **Risk:** LOW
   - **Impact:** Performance improvement only (3-5x faster variant computation)
   - **Rollback:** Disable flag, no data loss
   - **Validation:** Compare variant computation times, verify cache hit rates

2. **use_config_resolver** (Phase 5.1)
   - **Risk:** LOW
   - **Impact:** Configuration resolution logic (thoroughly tested)
   - **Rollback:** Disable flag
   - **Validation:** Verify config values match old implementation

3. **use_unified_genre_normalization** (Phase 2.1)
   - **Risk:** LOW-MEDIUM
   - **Impact:** Genre normalization (55 tests, bug fixes included)
   - **Rollback:** Disable flag
   - **Validation:** Compare normalized genre tokens with golden data

4. **use_unified_artist_normalization** (Phase 2.2)
   - **Risk:** LOW-MEDIUM
   - **Impact:** Artist normalization (19 tests)
   - **Rollback:** Disable flag
   - **Validation:** Compare artist clustering results

---

### Phase 2: Medium-Risk Flags (Week 3-4)

These flags affect core algorithms but have good test coverage:

5. **use_extracted_pier_bridge_diagnostics** (Phase 3.3)
   - **Risk:** LOW
   - **Impact:** Optional diagnostics collection
   - **Rollback:** Disable flag
   - **Validation:** Verify diagnostics output identical

6. **use_pipeline_builder** (Phase 5.2)
   - **Risk:** LOW
   - **Impact:** API usability improvement, delegates to existing pipeline
   - **Rollback:** Disable flag
   - **Validation:** Compare playlist results with old API

7. **use_extracted_pier_bridge_scoring** (Phase 3.1)
   - **Risk:** MEDIUM
   - **Impact:** Core transition/bridge scoring logic (21 tests)
   - **Rollback:** Disable flag immediately if ANY differences
   - **Validation:** Run golden file tests, compare transition scores

8. **use_extracted_segment_pool** (Phase 3.2)
   - **Risk:** MEDIUM
   - **Impact:** Candidate pool building logic (18 tests)
   - **Rollback:** Disable flag immediately if ANY differences
   - **Validation:** Compare pool sizes and candidate composition

---

### Phase 3: High-Risk Flags (Week 5-6)

These flags significantly change playlist generation logic:

9. **use_new_candidate_generator** (Phase 4.2)
   - **Risk:** HIGH
   - **Impact:** Candidate pool generation strategy
   - **Rollback:** Disable flag, monitor closely
   - **Validation:** A/B test playlists, compare diversity metrics

10. **use_filtering_pipeline** (Phase 4.3)
    - **Risk:** MEDIUM
    - **Impact:** Filter application order (should be identical)
    - **Rollback:** Disable flag
    - **Validation:** Verify filter statistics match

11. **use_history_repository** (Phase 4.4)
    - **Risk:** LOW-MEDIUM
    - **Impact:** Data access abstraction
    - **Rollback:** Disable flag
    - **Validation:** Verify recency filtering behaves identically

---

### Phase 4: Experimental Flags (Week 7+)

These flags are for future enhancements:

12. **use_playlist_factory** (Phase 4.1)
    - **Risk:** HIGH (main entry point)
    - **Impact:** Strategy pattern for playlist generation
    - **Rollback:** Disable flag immediately if issues
    - **Validation:** Test all 4 modes (artist, genre, batch, history)

13. **use_unified_genre_similarity** (Phase 2.3)
    - **Risk:** MEDIUM-HIGH
    - **Impact:** Genre similarity calculation method
    - **Rollback:** Disable flag
    - **Validation:** Compare similarity scores, genre coherence

14. **use_typed_config** (Phase 2.4)
    - **Risk:** LOW
    - **Impact:** Type-safe configuration access
    - **Rollback:** Disable flag
    - **Validation:** Verify config values identical

---

## Migration Steps (Per Flag)

### Step 1: Pre-Migration Validation

**Before enabling any flag:**

1. **Run full test suite:**
   ```bash
   pytest tests/unit/ tests/integration/ -v
   ```
   All tests must pass.

2. **Run golden file tests:**
   ```bash
   pytest tests/integration/test_playlist_golden_files.py -v
   ```
   Capture baseline metrics.

3. **Check current behavior:**
   ```bash
   # Generate test playlist with flag disabled
   python -m src.playlist_generator --mode dynamic --seed <seed_id> --tracks 30
   ```

4. **Document baseline metrics:**
   - Playlist diversity
   - Mean transition score
   - Candidate pool size
   - Execution time

---

### Step 2: Enable Flag in Config

**Edit `config.yaml`:**

```yaml
experimental:
  use_variant_cache: true  # ← Enable one flag
  # All other flags remain false
```

**Restart service:**
```bash
# If running as service
systemctl restart playlist-generator

# Or reload config programmatically
```

---

### Step 3: Validation Tests

**Run validation suite:**

1. **Golden file comparison:**
   ```bash
   pytest tests/integration/test_playlist_golden_files.py -v
   ```
   **Expected:** Exact match (or acceptable tolerance documented)

2. **Generate test playlists:**
   ```bash
   # Same seed as baseline
   python -m src.playlist_generator --mode dynamic --seed <seed_id> --tracks 30
   ```

3. **Compare results:**
   - Track IDs should match (or explain differences)
   - Metrics within acceptable range
   - No errors in logs

4. **Performance check:**
   ```bash
   # Measure execution time
   time python -m src.playlist_generator --mode dynamic --seed <seed_id> --tracks 30
   ```

---

### Step 4: Monitor Production

**Monitoring checklist:**

- [ ] Check error logs for deprecation warnings
- [ ] Monitor playlist quality metrics
- [ ] Compare with historical baselines
- [ ] Verify no user complaints
- [ ] Check performance metrics (response time, memory usage)

**Monitoring duration:** 2-3 days minimum

**Key Metrics:**
- Error rate (should be 0)
- Mean transition score (should match baseline ± 0.01)
- Playlist diversity (should match baseline ± 0.05)
- Execution time (should match or improve)

---

### Step 5: Decision Point

**After monitoring period:**

**If successful:**
- ✅ Keep flag enabled
- ✅ Document success in migration log
- ✅ Proceed to next flag after 2-3 days

**If issues detected:**
- ⚠️ Disable flag immediately
- ⚠️ Investigate root cause
- ⚠️ Fix issues, re-test
- ⚠️ Retry migration after fixes

**Rollback procedure:**
```yaml
# config.yaml
experimental:
  use_variant_cache: false  # ← Disable problematic flag
```

---

## Validation Criteria

### Success Criteria (per flag)

- [ ] All tests pass (166/166)
- [ ] Golden file tests produce identical or acceptable results
- [ ] No errors in production logs
- [ ] Performance within 5% of baseline (or improved)
- [ ] No user-reported issues
- [ ] Metrics within documented tolerances

### Acceptance Tolerances

**Exact Match Required:**
- Genre normalization output
- Artist normalization output
- Config resolution values

**Tolerance Allowed:**
- Transition scores: ± 0.01
- Playlist diversity: ± 0.05
- Execution time: +10% / -∞ (improvements welcome)
- Pool sizes: ± 5% (due to random sampling)

**Documented Differences:**
- Bug fixes in new implementation (e.g., substring matching)
- Performance improvements
- Optional features (caching, diagnostics)

---

## Rollback Scenarios

### Scenario 1: Test Failures

**Symptoms:**
- Golden file tests fail
- Unit tests fail after enabling flag

**Action:**
1. Disable flag immediately
2. Capture failure details
3. Fix issues in refactored code
4. Re-test in development
5. Retry migration

---

### Scenario 2: Production Errors

**Symptoms:**
- Errors in production logs
- User complaints
- Playlist quality degradation

**Action:**
1. **IMMEDIATELY disable flag**
2. Alert team
3. Investigate logs
4. Fix root cause
5. Test thoroughly before retry

---

### Scenario 3: Performance Degradation

**Symptoms:**
- Execution time > 10% slower
- Memory usage increased significantly
- Timeout errors

**Action:**
1. Disable flag if impact > 20%
2. Profile performance bottleneck
3. Optimize refactored code
4. Benchmark improvements
5. Retry migration

---

## Migration Log Template

Use this template to track each migration:

```markdown
## Flag: use_variant_cache

**Date:** 2026-01-XX
**Status:** ✅ Success / ⚠️ Rolled Back

### Pre-Migration Baseline
- Test Pass Rate: 166/166
- Mean Transition Score: 0.42
- Execution Time: 8.2s
- Pool Size: 2,450 tracks

### Post-Migration Results
- Test Pass Rate: 166/166
- Mean Transition Score: 0.42 (exact match)
- Execution Time: 2.8s (66% improvement! ✅)
- Pool Size: 2,450 tracks
- Cache Hit Rate: 85%

### Monitoring (72 hours)
- Errors: 0
- User Complaints: 0
- Performance: Improved significantly

### Decision
✅ Keep enabled. Proceed to next flag.

### Notes
- Significant performance improvement from caching
- No behavioral changes detected
- Cache hit rate as expected
```

---

## Final Migration Checklist

**Before July 2026 (when deprecated code is removed):**

- [ ] All 14 feature flags enabled in production
- [ ] All flags validated and monitored
- [ ] Migration log complete for each flag
- [ ] No outstanding issues
- [ ] Team comfortable with new implementations
- [ ] Documentation updated
- [ ] Deprecated code can be safely removed

---

## Emergency Contacts

**If critical issues arise:**

1. **Disable all flags immediately:**
   ```yaml
   experimental:
     # Set all flags to false
   ```

2. **Rollback to last stable configuration**

3. **Contact development team**

4. **Document incident for post-mortem**

---

## Document Version

**Version:** 1.0
**Last Updated:** 2026-01-04
**Status:** Ready for Execution
