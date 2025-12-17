# Minimal Fix: Document Non-Functional Dials

## Problem
432 dial grid runs compress to 36 outcomes. Analysis shows only sonic_weight/genre_weight affect generation. Other dials (min_genre_similarity, genre_method, transition_strictness) are non-functional:
- min_genre_similarity: accepted, logged, ignored
- genre_method: accepted, logged, ignored
- transition_strictness: constraint applies but never binding

## Solution: Document Limitation (Option C - Recommended)

Add comments to clearly indicate what's wired and what's not. This is the minimal change that prevents future confusion.

---

## Diff 1: src/playlist/pipeline.py

```diff
@@ -75,13 +75,21 @@ def generate_playlist_ds(
     single_artist: bool = False,
     sonic_variant: Optional[str] = None,
     # Hybrid-level tuning dials (not part of DSPipelineConfig)
     sonic_weight: Optional[float] = None,
     genre_weight: Optional[float] = None,
     min_genre_similarity: Optional[float] = None,
     genre_method: Optional[str] = None,
 ) -> DSPipelineResult:
     """
     Orchestrate:
     - load_artifact_bundle
     - build embedding (use smoothed genres for embedding)
     - build candidate pool
     - construct playlist (segment-aware transitions if available)
     Returns ordered track_ids and stats.

     Optional hybrid-level tuning dials:
-    - sonic_weight, genre_weight: control hybrid embedding balance
-    - min_genre_similarity: gate threshold
-    - genre_method: which genre similarity algorithm to use
+    - sonic_weight, genre_weight: control hybrid embedding balance ✓ IMPLEMENTED
+
+    NOTE: The following parameters are accepted for API compatibility but NOT
+    currently implemented in the DS pipeline (uses hybrid embedding, not
+    full SimilarityCalculator):
+    - min_genre_similarity: NOT WIRED (would need genre similarity computation)
+    - genre_method: NOT WIRED (would need SimilarityCalculator integration)
     """
     bundle = load_artifact_bundle(artifact_path)
     if seed_track_id not in bundle.track_id_to_index:
         raise ValueError(f"Seed track_id not found in artifact: {seed_track_id}")
     seed_idx = bundle.track_id_to_index[seed_track_id]

     # Log requested tuning dials for verification
     if any([sonic_weight, genre_weight, min_genre_similarity, genre_method]):
         logger.info(
-            "DS pipeline tuning dials: sonic_weight=%s, genre_weight=%s, min_genre_sim=%s, genre_method=%s",
+            "DS pipeline applied dials: sonic_weight=%s, genre_weight=%s "
+            "(note: min_genre_similarity=%s, genre_method=%s not implemented)",
             sonic_weight, genre_weight, min_genre_similarity, genre_method
         )
```

---

## Diff 2: scripts/tune_dial_grid.py (Documentation)

```diff
@@ -386,13 +386,16 @@ def main():
     )
     parser.add_argument(
         "--min-genre-sim",
         default="0.20,0.25,0.30",
-        help="Comma-separated min_genre_similarity values"
+        help="Comma-separated min_genre_similarity values (NOTE: not implemented in DS pipeline)"
     )
     parser.add_argument(
         "--genre-method",
         default="ensemble,weighted_jaccard",
-        help="Comma-separated genre similarity methods"
+        help="Comma-separated genre similarity methods (NOTE: not implemented in DS pipeline)"
     )
     parser.add_argument(
         "--transition-strictness",
         default="baseline,strictish",
-        help="Comma-separated transition strictness levels (baseline, strictish, lenient, strict)"
+        help="Comma-separated transition strictness levels (NOTE: framework exists but "
+             "constraint rarely binding in 30-track playlists)"
     )
```

---

## Diff 3: tests/test_dial_grid_tuning.py (Update Docstring)

```diff
@@ -1,8 +1,12 @@
 """
 Regression test for dial grid tuning.

 Verifies that tuning dials (sonic_weight, genre_weight, min_genre_similarity, etc.)
 actually affect playlist generation and metrics.

-This test failed before the fix (all dials produced identical playlists).
+IMPLEMENTATION STATUS:
+✓ sonic_weight, genre_weight: Fully implemented - tests verify they change playlists
+✗ min_genre_similarity, genre_method: NOT implemented - would require SimilarityCalculator integration
+⚠ transition_strictness: Framework works but constraint rarely binding in 30-track case
 """
```

---

## Alternate Diff (If Implementing min_genre_similarity)

If you decide to implement min_genre_similarity later, here's where it would go:

```diff
 # src/playlist/pipeline.py - Add in build_candidate_pool call (line ~254)
 pool = build_candidate_pool(
     seed_idx=seed_idx,
     embedding=embedding_model.embedding,
     artist_keys=bundle.artist_keys,
     cfg=cfg.candidate,
+    min_genre_similarity=min_genre_similarity,  # NEW: pass parameter
     random_seed=random_seed,
 )
```

Then in `src/playlist/candidate_pool.py`, add filtering logic:

```diff
-def build_candidate_pool(seed_idx, embedding, artist_keys, cfg, random_seed):
+def build_candidate_pool(seed_idx, embedding, artist_keys, cfg, random_seed, min_genre_similarity=None):
     """Build candidate pool with optional genre gate."""
     # ... existing code ...

+    if min_genre_similarity is not None:
+        # Filter by genre similarity to seed
+        # This would require:
+        # 1. Compute X_genre_similarity between candidates and seed
+        # 2. Apply threshold: keep only candidates with genre_sim >= min_genre_similarity
+        # (implementation ~20 lines)
```

**BUT:** This is NOT included in minimal fix because:
- Requires genre similarity computation (currently not in DS pipeline)
- Needs SimilarityCalculator or equivalent
- Might impact performance
- Current sonic_weight/genre_weight are already working well

---

## Verification Commands

### Run Analysis Script (No Changes Needed)
```bash
python scripts/force_bind_test.py
# Output will show confirmed: min_genre_sim and genre_method produce identical results
```

### Run Quick Verification Grid
```bash
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds 1c347ff04e65adf7923a9e3927ab667a \
    --mode dynamic \
    --sonic-weight 0.45,0.65,0.85 \
    --min-genre-sim 0.20,0.50 \
    --genre-method ensemble,weighted_jaccard \
    --transition-strictness baseline,strictish \
    --output-dir diagnostics/verify_unimplemented \
    --length 30

# Then check: consolidated_results.csv should have only 3 unique outcome rows
# (one per sonic_weight value, all 12 other dial combos identical)
```

### Verify Only sonic_weight Creates Unique Outcomes
```bash
python << 'EOF'
import csv

with open("diagnostics/verify_unimplemented/consolidated_results.csv") as f:
    rows = [r for r in csv.DictReader(f) if not r['error']]

# Group by sonic_weight
by_weight = {}
for r in rows:
    w = r['sonic_weight']
    if w not in by_weight:
        by_weight[w] = []
    by_weight[w].append(r)

print("Unique outcome rows per sonic_weight value:")
for weight in sorted(by_weight.keys()):
    rows_for_weight = by_weight[weight]
    unique_metrics = len(set(r['edge_hybrid_mean'] for r in rows_for_weight))
    print(f"  sonic_weight={weight}: {len(rows_for_weight)} runs → {unique_metrics} unique metric value(s)")

print("\nExpected: 1 unique metric value per sonic_weight (others have no effect)")
EOF
```

---

## Summary

| Change | File | Lines | Impact |
|--------|------|-------|--------|
| Add docstring note | `pipeline.py` | +5 | Clarifies what's implemented |
| Update help text | `tune_dial_grid.py` | +3 | Manages user expectations |
| Update test docs | `test_dial_grid_tuning.py` | +4 | Documents status |
| **Total** | **3 files** | **~12 lines** | **Non-breaking documentation** |

**No breaking changes.** All existing code continues to work. This change simply documents the current (partial) implementation status and prevents future confusion about why some dials don't affect output.

