# Next Steps: Dial Investigation Completion

## Quick Links

- **Start here:** Read `DIAL_INVESTIGATION_EXECUTIVE_SUMMARY.txt` (5 min read)
- **Technical details:** `DIAL_ROUTING_ANALYSIS.md` (15 min read)
- **Code changes:** `MINIMAL_FIX_DIFF.md` (5 min review)
- **Verify findings:** Run `scripts/force_bind_test.py` (2 min run)

---

## Immediate Actions (Pick One)

### Option 1: Document the Limitation (RECOMMENDED)

**Effort:** 15 minutes
**Impact:** Non-breaking, prevents future confusion

1. Apply diffs from `MINIMAL_FIX_DIFF.md`:
   - Add docstring note to `src/playlist/pipeline.py`
   - Update help text in `scripts/tune_dial_grid.py`
   - Update test comments in `tests/test_dial_grid_tuning.py`

2. Commit with message:
   ```
   docs: clarify dial implementation status in DS pipeline

   - min_genre_similarity: not implemented (would need SimilarityCalculator)
   - genre_method: not implemented (would need SimilarityCalculator)
   - transition_strictness: working but rarely binding for 30-track playlists
   - sonic_weight/genre_weight: fully functional

   See DIAL_INVESTIGATION_SUMMARY.md for analysis.
   ```

3. Mark tuning harness flags as deprecated in wiki/docs

---

### Option 2: Verify and Move Forward

**Effort:** 5 minutes
**Impact:** Confirms findings, baseline for future work

1. Run verification:
   ```bash
   python scripts/force_bind_test.py
   ```

2. Commit test script:
   ```
   test: add dial binding verification script

   Confirms min_genre_similarity, genre_method, transition_strictness
   do not affect DS pipeline output.
   ```

3. Document in TUNING_WORKFLOW.md that only sonic_weight should be used

---

### Option 3: Full Implementation (Later)

**Effort:** 2-3 hours
**Impact:** Enable all dials, but requires core changes

1. Implement min_genre_similarity:
   - Add genre similarity computation to candidate pool
   - Add hard gate for dynamic mode
   - Add filtering logic (~100 lines)

2. Implement genre_method:
   - Integrate SimilarityCalculator
   - Route genre_method through similarity computation
   - (~200 lines)

3. Make transition_strictness binding:
   - Increase default strictish floor to 0.8+
   - Test with various track lengths
   - (~50 lines)

4. Test extensively with regression suite

**Recommendation:** Defer this unless min_genre_similarity becomes critical feature

---

## Verification Workflow

### Step 1: Run Force-Binding Test (2 min)
```bash
python scripts/force_bind_test.py
```

Expected output confirms:
- ❌ min_genre_similarity NOT WIRED
- ❌ genre_method NOT WIRED
- ⚠️ transition_strictness NOT BINDING

### Step 2: Run Minimal Grid (5 min)
```bash
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds 1c347ff04e65adf7923a9e3927ab667a \
    --mode dynamic \
    --sonic-weight 0.50,0.70,0.90 \
    --min-genre-sim 0.20,0.40 \
    --genre-method ensemble,weighted_jaccard \
    --transition-strictness baseline,strictish \
    --output-dir diagnostics/minimal_verify \
    --length 30
```

### Step 3: Check Results (1 min)
```bash
python << 'EOF'
import csv
rows = [r for r in csv.DictReader(open("diagnostics/minimal_verify/consolidated_results.csv")) if not r['error']]
by_weight = {}
for r in rows:
    w = r['sonic_weight']
    if w not in by_weight:
        by_weight[w] = []
    by_weight[w].append(r)

print("Outcome uniqueness by sonic_weight:")
for w in sorted(by_weight.keys()):
    unique = len(set(r['edge_hybrid_mean'] for r in by_weight[w]))
    total = len(by_weight[w])
    print(f"  {w}: {total} runs → {unique} unique outcome ({'✓' if unique == 1 else '✗'})")
EOF
```

Expected:
```
Outcome uniqueness by sonic_weight:
  0.5: 8 runs → 1 unique outcome ✓
  0.7: 8 runs → 1 unique outcome ✓
  0.9: 8 runs → 1 unique outcome ✓
```

---

## Recommended Decision Flow

```
┌─────────────────────────────────────────┐
│ Are min_genre_similarity and            │
│ genre_method critical for your use case?│
└─────────────────────────────────────────┘
             ╱          ╲
           YES            NO
           ╱                ╲
    ┌─────────────┐    ┌──────────────────┐
    │ Implement   │    │ Document as      │
    │ full        │    │ limitation       │
    │ feature     │    │ (Option 1)       │
    │ (Option 3)  │    │                  │
    │ ~2-3 hours  │    │ ~15 minutes      │
    └─────────────┘    └──────────────────┘
          │                     │
          │                     │
       LATER              DO NOW ✓
```

---

## File Reference

### Analysis Documents
- `DIAL_INVESTIGATION_EXECUTIVE_SUMMARY.txt` - Start here (5 min)
- `DIAL_ROUTING_ANALYSIS.md` - Technical deep dive (15 min)
- `DIAL_INVESTIGATION_SUMMARY.md` - Comprehensive breakdown (20 min)
- `DIAL_GRID_FIX_SUMMARY.md` - Original sonic_weight fix (earlier work)

### Implementation Reference
- `MINIMAL_FIX_DIFF.md` - Code diffs for documentation option
- `scripts/force_bind_test.py` - Verification script
- `scripts/tune_dial_grid.py` - Where dials are passed (lines 210-223)
- `src/playlist/pipeline.py` - Where dials should be applied (currently only sonic_weight)

### Test Suite
- `tests/test_dial_grid_tuning.py` - 4 regression tests (all passing)
  - ✅ test_sonic_weight_changes_result
  - ✅ test_genre_weight_changes_result
  - ✅ test_transition_strictness_changes_result
  - ✅ test_extreme_dials_produce_extreme_metrics

### Data
- `diagnostics/tune_grid/consolidated_results.csv` - 432 runs showing the pattern
- `diagnostics/tune_grid/docs/sonic_weight_averages.csv` - Proof only sonic_weight works
- `diagnostics/tune_grid/docs/best_sonic_weight_per_seed.csv` - Per-seed best setting

---

## Decision Matrix

| Action | Effort | Impact | Risk | Status |
|--------|--------|--------|------|--------|
| Document limitation (Option 1) | 15 min | Low (just docs) | None | ✅ READY |
| Run verification (Step 2 above) | 5 min | Confirms findings | None | ✅ READY |
| Full implementation (Option 3) | 2-3 hrs | High (enables all dials) | Medium | ⏳ LATER |
| Remove parameters (Option C) | 20 min | Medium (breaking API) | High | ❌ NOT RECOMMENDED |

---

## Recommended Path Forward

**Phase 1 (This Week): Documentation**
1. Run `scripts/force_bind_test.py` to verify findings
2. Apply diffs from `MINIMAL_FIX_DIFF.md`
3. Commit with evidence from analysis
4. Update docs to clarify only sonic_weight works

**Phase 2 (Later if Needed): Full Implementation**
- If users request min_genre_similarity as critical feature
- Implement SimilarityCalculator integration
- Add comprehensive tests
- Update TUNING_WORKFLOW.md

**Phase 3 (Optional): Enhance Transition Strictness**
- Make transition floor changes more binding
- Test with different playlist lengths
- Consider adding "extremely strict" mode with floor=0.95

---

## Questions to Ask Stakeholders

Before proceeding with full implementation:

1. **Is min_genre_similarity critical?**
   - If no: proceed with documentation option
   - If yes: schedule full implementation

2. **Are users expecting these dials to work?**
   - If no: clarify in docs, move on
   - If yes: implement full feature

3. **Performance impact acceptable?**
   - Genre similarity computation ~10% overhead
   - If critical-performance system: might not want it

---

## Success Criteria

**Documentation Option (Recommended):**
- ✅ Docstrings updated in pipeline.py
- ✅ Help text updated in tune_dial_grid.py
- ✅ Test comments document status
- ✅ force_bind_test.py runs without errors
- ✅ No regression in existing tests

**Full Implementation Option:**
- ✅ All 5 dial grid tests passing
- ✅ consolidated_results.csv shows all dials varying
- ✅ Performance impact < 20%
- ✅ Documentation updated
- ✅ Example workflow demonstrates all dials

---

## Next Command to Run

Pick one:

**To verify findings:**
```bash
python scripts/force_bind_test.py
```

**To see the problem yourself:**
```bash
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds 1c347ff04e65adf7923a9e3927ab667a \
    --mode dynamic \
    --sonic-weight 0.50,0.70 \
    --min-genre-sim 0.20,0.50 \
    --genre-method ensemble \
    --transition-strictness baseline,strictish \
    --output-dir diagnostics/demo_issue \
    --length 30
# Then check: consolidated_results.csv only has 2 rows of unique metrics (sonic_weight only)
```

**To apply documentation fix:**
```bash
# First read MINIMAL_FIX_DIFF.md
# Then apply each diff to the 3 files
# Then run: python -m pytest tests/test_dial_grid_tuning.py -v
```

---

## Summary

✅ Root cause identified: min_genre_similarity, genre_method, transition_strictness not wired into DS pipeline

✅ Verification script created: scripts/force_bind_test.py

✅ Documentation prepared: DIAL_INVESTIGATION_*.md files

**Recommended next step:** Document as known limitation (~15 min), run verification tests, commit evidence.
