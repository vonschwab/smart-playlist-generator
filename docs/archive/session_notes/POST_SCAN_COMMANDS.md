# Post-Scan Commands: 3-Tower Sonic Redesign

Execute these commands **after the beat-sync scan completes**.

---

## Step 1: Validate Current State (Phase 0)

First, establish a baseline with the current beat-sync features:

```bash
# Find the current artifact path
ARTIFACT_PATH="experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz"

# Get some seed track IDs from your library (Fela Kuti, electronic, ballad examples)
# You can find track IDs with:
python -c "
import sqlite3
conn = sqlite3.connect('data/metadata.db')
cursor = conn.cursor()

# Find Fela Kuti or afrobeat
cursor.execute(\"\"\"
    SELECT track_id, artist, title FROM tracks
    WHERE artist LIKE '%Fela%' OR artist LIKE '%Afrobeat%'
    LIMIT 3
\"\"\")
print('AFROBEAT SEEDS:')
for row in cursor.fetchall():
    print(f'  {row[0]} - {row[1]} - {row[2]}')

# Find electronic
cursor.execute(\"\"\"
    SELECT track_id, artist, title FROM tracks
    WHERE artist LIKE '%Electronic%' OR album LIKE '%Electronic%'
    LIMIT 3
\"\"\")
print('ELECTRONIC SEEDS:')
for row in cursor.fetchall():
    print(f'  {row[0]} - {row[1]} - {row[2]}')

# Get random seeds
cursor.execute('SELECT track_id, artist, title FROM tracks ORDER BY RANDOM() LIMIT 5')
print('RANDOM SEEDS:')
for row in cursor.fetchall():
    print(f'  {row[0]} - {row[1]} - {row[2]}')
"

# Run baseline validation (replace SEED1,SEED2,SEED3 with actual track IDs)
python scripts/validate_sonic_quality.py \
    --artifact "$ARTIFACT_PATH" \
    --seeds "SEED1,SEED2,SEED3" \
    --k 30 \
    --output diagnostics/sonic_validation/baseline/
```

**Expected Output**:
- `diagnostics/sonic_validation/baseline/validation_results.json`
- `diagnostics/sonic_validation/baseline/validation_report.md`

**Review the report** to see:
- Flatness score (likely < 0.5 = FAIL)
- TopK gap (likely < 0.15 = FAIL)
- Within-artist coherence
- Top 10 neighbors for each seed (do they make sense?)

---

## Step 2: Test 3-Tower Extractor

Verify the extractor works on sample tracks:

```bash
# Test on a single file
python src/features/beat3tower_extractor.py "/path/to/sample/track.flac"

# Expected output:
# Extraction successful!
#   Beats detected: 243
#   Duration: 242.5s
# Feature dimensions:
#   Rhythm: (21,)
#   Timbre: (83,)
#   Harmony: (33,)
#   Total: (137,)
```

---

## Step 3: Re-Scan with 3-Tower Features (Future)

**NOT YET IMPLEMENTED** - Requires integration of extractor into scan pipeline.

Once integrated, the scan command will be:

```bash
# Re-scan with 3-tower extraction (future)
python scripts/update_sonic.py --beat3tower --workers 8

# This will:
# 1. Extract 3-tower features per track
# 2. Store as sonic_features JSON in database
# 3. Include segment features (start/mid/end)
```

---

## Step 4: Build 3-Tower Artifacts

**IMPLEMENTED** - Phase 2 complete.

Build 3-tower artifacts from tracks that have beat3tower features:

```bash
# Build 3-tower artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz

# Options:
#   --no-pca           Disable PCA whitening (use robust standardization only)
#   --pca-variance 0.95  Fraction of variance to retain in PCA
#   --clip-sigma 3.0   Sigma for outlier clipping
#   --max-tracks 1000  Limit tracks for testing
#   -v                 Verbose output
```

The artifact builder:
- Loads tracks with beat3tower features from database
- Separates into rhythm/timbre/harmony towers
- Applies per-tower robust normalization with optional PCA whitening
- Computes calibration statistics for weighted combination
- Saves all required matrices for playlist generation

---

## Step 5: Validate 3-Tower Quality

After artifacts are built:

```bash
# Validate 3-tower features
python scripts/validate_sonic_quality.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_beat3tower.npz \
    --seeds "SEED1,SEED2,SEED3" \
    --output diagnostics/sonic_validation/beat3tower/

# Compare baseline vs beat3tower
diff diagnostics/sonic_validation/baseline/validation_report.md \
     diagnostics/sonic_validation/beat3tower/validation_report.md
```

**Success Criteria**:
- `sonic_flatness >= 0.5` (up from ~0.05)
- `topK_gap >= 0.15` (up from ~0.02)
- `within_artist_coherence > 0`

---

## Quick Validation (Single Command)

Run baseline validation with 5 random seeds:

```bash
python scripts/validate_sonic_quality.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --random-seeds 5 \
    --output diagnostics/sonic_validation/quick_baseline/
```

---

## Troubleshooting

### "Artifact not found"
```bash
# Check artifact exists
ls -la experiments/genre_similarity_lab/artifacts/*.npz

# If missing, the scan may not be complete. Check scan status:
tail -f logs/scan_library.log
```

### "Track not found in artifact"
The track ID may be from a different scan. Use `--random-seeds 5` instead.

### "InsufficientBeatsError"
Track has < 4 beats detected. Expected for ambient/drone music. The extractor will skip these.

### Very low flatness/gap in baseline
**This is expected.** The baseline uses current beat-sync features which are known to be flat. The 3-tower redesign should fix this.

---

## Next Steps After Validation

1. **If baseline metrics are bad** (flatness < 0.1, gap < 0.05):
   - Confirms the problem exists
   - Proceed with Phase 2-4 implementation

2. **If baseline metrics are OK** (flatness > 0.3, gap > 0.10):
   - Current features may be better than expected
   - Still worth proceeding with 3-tower for rhythm emphasis

3. **After 3-tower artifacts built**:
   - Run same validation
   - Compare metrics
   - Human sniff test: Check Fela neighbors

---

## Files Created

| File | Purpose |
|------|---------|
| `docs/SONIC_3TOWER_ARCHITECTURE.md` | Design document (2 pages) |
| `docs/SONIC_3TOWER_IMPLEMENTATION_PHASES.md` | Phased PR plan |
| `scripts/validate_sonic_quality.py` | Phase 0 validation harness |
| `src/features/beat3tower_types.py` | Data classes for features |
| `src/features/beat3tower_extractor.py` | Phase 1 extractor |
| `src/features/beat3tower_normalizer.py` | Phase 2 per-tower normalizer |
| `scripts/build_beat3tower_artifacts.py` | Phase 2 artifact builder |
| `tests/unit/test_beat3tower_normalizer.py` | Phase 2 normalizer tests |
| `tests/unit/test_beat3tower_types.py` | Phase 1/2 type tests |

---

## Remaining Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ DONE | Validation infrastructure |
| Phase 1 | ✅ DONE | 3-tower extractor skeleton |
| Phase 2 | ✅ DONE | Normalization + artifact builder |
| Phase 3 | 🔜 TODO | Calibrated multi-tower similarity |
| Phase 4 | 🔜 TODO | Transition scoring (end->start) |
| Phase 5 | 🔜 OPTIONAL | Weak supervision weight tuning |
