# Playlist Tuning Workflow

This document describes the initial dial grid, recommended seed composition, and listening workflow for tuning the playlist generator.

## Quick Start

```bash
# 1. Create a seeds file with 8-15 diverse tracks
echo "track_id_1
track_id_2
track_id_3" > diagnostics/tune_seeds.txt

# 2. Run the starter grid
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds-file diagnostics/tune_seeds.txt \
    --mode dynamic \
    --length 30 \
    --sonic-weight 0.55,0.65,0.75 \
    --min-genre-sim 0.20,0.25,0.30 \
    --genre-method ensemble,weighted_jaccard \
    --transition-strictness baseline,strictish \
    --export-m3u-dir diagnostics/tune_m3u \
    --output-dir diagnostics/tune_grid

# 3. Review results in diagnostics/tune_grid/consolidated_results.csv
# 4. Listen to M3U files in diagnostics/tune_m3u/
```

---

## Initial Dial Grid (Starter)

This grid is designed to reduce **genre leakage** in dynamic mode while preserving flow.

### Grid Parameters

| Dial | Values | Rationale |
|------|--------|-----------|
| `sonic_weight` | 0.55, 0.65, **0.75** | Higher sonic emphasis may help when genres are similar |
| `genre_weight` | 0.45, 0.35, **0.25** | Computed as 1 - sonic_weight |
| `min_genre_similarity` | **0.20**, 0.25, 0.30 | Raising this is the primary leakage control |
| `genre_method` | **ensemble**, weighted_jaccard | Ensemble combines multiple methods |
| `transition_strictness` | **baseline**, strictish | baseline uses mode defaults; strictish raises floor +0.05 |
| `sonic_variant` | raw | Keep simple initially |

**Total combinations**: 3 × 3 × 2 × 2 × 1 = **36 per seed**

### Recommended Starter Grid CLI

```bash
python scripts/tune_dial_grid.py \
    --artifact experiments/genre_similarity_lab/artifacts/data_matrices_step1.npz \
    --seeds-file diagnostics/tune_seeds.txt \
    --mode dynamic \
    --sonic-weight 0.55,0.65,0.75 \
    --min-genre-sim 0.20,0.25,0.30 \
    --genre-method ensemble,weighted_jaccard \
    --transition-strictness baseline,strictish \
    --export-m3u-dir diagnostics/tune_m3u
```

---

## Recommended Seed Set Composition

Choose 8-15 seeds that span different:
- **Genres** (rock, electronic, jazz, hip-hop, classical, metal, etc.)
- **Eras** (60s, 80s, 00s, 2020s)
- **Energy levels** (ambient vs. aggressive)
- **Popularity** (well-tagged mainstream vs. obscure)

### Example Seed Categories

| Category | Example Style | Why Include |
|----------|---------------|-------------|
| Classic Rock | Led Zeppelin, Pink Floyd | Well-tagged, common case |
| Metal | Black Sabbath, Metallica | Genre leakage is common here |
| Indie/Alt | Radiohead, The National | Often has cross-genre tags |
| Electronic | Aphex Twin, Boards of Canada | Different sonic profile |
| Soul/R&B | Marvin Gaye, D'Angelo | Tests genre boundaries |
| Hip-Hop | Kendrick Lamar, MF DOOM | Another distinct genre |
| Jazz | Miles Davis, John Coltrane | Tests classical vs. jazz |
| Classical | Bach, Debussy | Very different from all else |

### Finding Good Seeds

```bash
# List tracks with high genre coverage for testing
sqlite3 data/metadata.db "
SELECT t.track_id, t.artist, t.title, COUNT(DISTINCT ag.genre) as n_genres
FROM tracks t
LEFT JOIN artist_genres ag ON ag.artist = t.artist
GROUP BY t.track_id
HAVING n_genres > 3
ORDER BY n_genres DESC
LIMIT 50
"
```

---

## Listening Workflow

### Step 1: Generate Playlists

Run the dial grid to generate M3U files:

```bash
python scripts/tune_dial_grid.py \
    --artifact path/to/artifact.npz \
    --seeds-file diagnostics/tune_seeds.txt \
    --export-m3u-dir diagnostics/tune_m3u
```

### Step 2: Organize for Listening

M3U files are named: `{seed_id_prefix}__{dial_suffix}.m3u8`

Example: `abc12345__sw65_mgs25_gm-ense_ts-base.m3u8`
- `sw65` = sonic_weight=0.65
- `mgs25` = min_genre_similarity=0.25
- `gm-ense` = genre_method=ensemble
- `ts-base` = transition_strictness=baseline

### Step 3: A/B Listening Protocol

For each seed, compare 2-3 dial settings:

1. **Baseline vs. Higher Genre Gate**
   - Compare `mgs20` vs `mgs30` with same sonic_weight
   - Listen for: genre intrusions, diversity, flow

2. **Sonic vs. Genre Emphasis**
   - Compare `sw55` vs `sw75` with same min_genre_sim
   - Listen for: sonic texture consistency vs. genre coherence

3. **Transition Strictness**
   - Compare `ts-base` vs `ts-stri`
   - Listen for: abrupt transitions, "whiplash" moments

### What to Listen For

| Issue | Symptom | Likely Dial to Adjust |
|-------|---------|----------------------|
| **Genre leakage** | Metal in a soul playlist | ↑ min_genre_similarity |
| **Boring/repetitive** | All tracks sound the same | ↓ sonic_weight, ↓ min_genre_sim |
| **Whiplash transitions** | Jarring jumps | ↑ transition_strictness or ↑ beta |
| **Not enough seed artist** | Seed buried in others | Check artist constraints |
| **Too many same-artist runs** | 3+ in a row | Check min_gap, constraints |

### Step 4: Record Observations

Create a listening log:

```markdown
# Listening Log - 2024-12-15

## Seed: Black Sabbath - Paranoid

### sw65_mgs20_gm-ense_ts-base
- Genre leakage: YES - track 12 was jazz fusion (?!)
- Transitions: Mostly smooth, one jarring at 8→9
- Overall vibe: 6/10 - too diverse

### sw65_mgs30_gm-ense_ts-base
- Genre leakage: NO - stayed in heavy rock/metal
- Transitions: All smooth
- Overall vibe: 8/10 - good balance

### Winner: mgs30
```

### Step 5: Aggregate Results

After listening to all seeds, tally wins:

| Setting | Wins | Notes |
|---------|------|-------|
| min_genre_sim=0.30 | 6/8 | Best for rock/metal |
| min_genre_sim=0.25 | 4/8 | Good for mixed |
| sonic_weight=0.65 | 5/8 | Sweet spot |
| ensemble method | 7/8 | Consistently better |

---

## Avoiding Overfitting

- **Don't tune to one genre**: Ensure improvements work across categories
- **Test edge cases**: Include at least one "difficult" seed (e.g., ambient, classical)
- **Use multiple seeds per style**: 2-3 rock seeds, not just one
- **Validate on new seeds**: After tuning, test on 5 new seeds not in original set

---

## Metrics to Watch

From `consolidated_results.csv`:

| Metric | Good Range | Bad Sign |
|--------|------------|----------|
| `edge_genre_min` | > 0.15 | Very low = leakage |
| `edges_with_very_low_genre` | 0-2 | > 5 = widespread leakage |
| `edge_transition_min` | > 0.25 | Very low = whiplash |
| `unique_artists` | 15-25 for 30 tracks | < 10 = repetitive |
| `adjacency_violations` | 0 | > 0 = constraint bug |
| `below_floor_count` | 0-3 | > 5 = floor too strict |

### Quick Analysis Commands

```bash
# Find worst genre leakage runs
csvtool col 1,8,27,28 diagnostics/tune_grid/consolidated_results.csv | sort -t, -k4 -n | head

# Compare min_genre_sim settings
grep "0.20" diagnostics/tune_grid/consolidated_results.csv | csvstat --mean -c edge_genre_min
grep "0.30" diagnostics/tune_grid/consolidated_results.csv | csvstat --mean -c edge_genre_min
```

---

## Advanced: Second-Round Tuning

After initial round, narrow the grid:

```bash
# If mgs=0.30 and sw=0.65 performed best, fine-tune:
python scripts/tune_dial_grid.py \
    --seeds-file diagnostics/tune_seeds.txt \
    --sonic-weight 0.60,0.65,0.70 \
    --min-genre-sim 0.27,0.30,0.33 \
    --genre-method ensemble \
    --transition-strictness baseline,strictish \
    --export-m3u-dir diagnostics/tune_m3u_round2
```

---

## Config Integration

Once you find winning dials, update `config.yaml`:

```yaml
playlists:
  genre_similarity:
    weight: 0.35           # 1 - sonic_weight
    sonic_weight: 0.65     # Winner from tuning
    min_genre_similarity: 0.30  # Winner from tuning
    method: ensemble       # Winner from tuning

  ds_pipeline:
    mode: dynamic
    # Optionally add custom overrides
```

---

## Summary

1. **Start with the default grid** (36 combinations × N seeds)
2. **Listen systematically** with A/B comparisons
3. **Track wins per setting** in a simple log
4. **Avoid overfitting** by testing diverse seeds
5. **Validate on new seeds** before committing changes
6. **Update config** with winning parameters
