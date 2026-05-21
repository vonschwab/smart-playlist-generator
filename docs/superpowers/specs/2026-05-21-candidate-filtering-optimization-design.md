# Candidate Filtering Optimization — Design

**Date:** 2026-05-21
**Scope:** Targeted cleanup + IDF-weighted admission genre similarity. Larger items (subgenre diversity arcs, easy-out sonic prevention, bridge-floor cleanup) are parked in `docs/CANDIDATE_FILTERING_BACKLOG.md`.

## Goal

Make candidate admission filter more intentionally — fix three real bugs, consolidate two duplicate systems, and reward rare-tag genre matches so admission no longer treats `indie` and `slowcore` as equally informative.

## Non-goals

- Adding subgenre-diversity arc planning (deferred)
- Preventing ambient/noise/drone runs (deferred)
- Tightening the per-segment `bridge_floor` (deferred)
- Backward compatibility for renamed/removed config keys (the user is the only operator and is in extended testing; old keys get deleted, not aliased)

## Architectural overview

A single shared IDF service used by both the existing dj_bridging waypoint code and the new admission filter. Today the IDF is computed inside `pier_bridge_builder.py` from the raw genre matrix; this lifts it into `src/playlist/genre_idf.py` so admission can use the identical values.

The candidate-pool admission flow becomes:

```
seed_idx + anchor_seed_ids
  ↓
sonic max-over-seeds          (unchanged)
hybrid max-over-seeds          (unchanged)
title-quality filter           (consolidated — uses detect_title_artifacts with config flag set)
genre similarity                (fixed — max-over-seeds, IDF-weighted in narrow/dynamic/strict)
overlap guard                   (unchanged)
genre compatibility penalty     (renamed from genre_conflict_*, dead gate code removed)
duration penalty + cutoff       (unchanged)
artist diversity cap            (unchanged)
```

Per-segment filtering (inside pier-bridge) is unchanged.

## Components

### 1. `src/playlist/genre_idf.py` (new)

Pure-function module. Single source of truth for IDF computation.

```python
def compute_genre_idf(
    *,
    X_genre_raw: np.ndarray,
    power: float = 1.0,
    norm: str = "max1",   # "max1" | "sum1" | "none"
) -> np.ndarray:
    """Return one IDF weight per genre column.

    Higher weights for rare tags, lower for common ones. The result is
    a 1D array whose length matches the genre vocabulary.
    """
```

No I/O. No logging at module level. Used by:
- `pier_bridge_builder.py` (dj_bridging waypoint code, refactored to call this)
- `candidate_pool.py` (new admission usage)

### 2. `pier_bridge_builder.py` refactor

The existing `_compute_genre_idf` body is deleted. Replaced by an import + call into `genre_idf.compute_genre_idf`. The dj_bridging code is unchanged from a behavior perspective — same parameters, same outputs.

### 3. `candidate_pool.py:_compute_genre_similarity` accepts IDF weights

Function signature becomes:

```python
def _compute_genre_similarity(
    seed_genres: np.ndarray,
    candidate_genres: np.ndarray,
    method: str = "cosine",
    idf_weights: Optional[np.ndarray] = None,
) -> np.ndarray
```

When `idf_weights` is provided, the seed and candidate vectors are multiplied elementwise by the weights before computing cosine/jaccard/ensemble. Rare-tag matches contribute more; common-tag matches less. When `None`, behavior is unchanged.

The three similarity methods all support IDF weighting:
- **cosine**: applies weights to vectors before computing cosine similarity
- **weighted_jaccard**: applies weights to binary presence vectors, computes weighted intersection / weighted union
- **ensemble**: combines weighted cosine and weighted jaccard with the existing 0.6/0.4 mix

### 4. Genre similarity + overlap guard max-over-seeds (bug A1)

Two lines in `candidate_pool.py` currently reference only the primary seed for genre filtering. Both need the same fix.

The genre similarity reference vector:
```python
# OLD
seed_genres = genre_matrix[seed_idx]
# NEW
seed_genres = np.max(genre_matrix[seed_list], axis=0)
```

The overlap guard's seed reference (a separate code path further down that rejects candidates with zero raw-tag overlap):
```python
# OLD
seed_binary = (genre_raw_matrix[seed_idx] > 0).astype(float)
# NEW
seed_binary = (np.max(genre_raw_matrix[seed_list], axis=0) > 0).astype(float)
```

Both changes match what the existing genre-compatibility code already does at `candidate_pool.py:327` — internal consistency. Multi-seed playlists no longer have their genre filter or overlap guard biased to seed #1.

### 5. Title quality consolidation

`src/playlist/title_quality.py:detect_title_artifacts` gains three new flags: `interlude`, `skit`, `acapella`. Same word-boundary regex approach as the existing flags. The `acapella` pattern also matches `a cappella` and `a capella` (the three variants currently in `title_exclusion_words`).

`candidate_pool.py` replaces the existing `is_title_excluded(title, words_list)` call with:

```python
flags = detect_title_artifacts(track_titles[i])
if flags & cfg.title_hard_exclude_flags:
    title_exclusion_rejected += 1
    continue
```

The old `is_title_excluded` function is deleted from `src/playlist/filtering.py`.

Config: `candidate_pool.title_hard_exclude_flags` replaces `candidate_pool.title_exclusion_words`. Default `{"interlude", "skit", "acapella"}`. The old `title_exclusion_words` key is removed from the config schema (no alias).

### 6. Genre conflict → genre compatibility rename + dead code removal

The current `genre_conflict_*` naming is misleading because the math measures both *compatibility* and *conflict* mass; "conflict" is half the picture. Renames:

| Old | New |
|---|---|
| `genre_conflict_enabled` | `genre_compatibility_enabled` |
| `genre_conflict_penalty_strength` | `genre_compatibility_penalty_strength` |
| `genre_conflict_compatible_threshold` | `genre_compatibility_compatible_threshold` |
| `genre_conflict_conflict_threshold` | `genre_compatibility_conflict_threshold` |
| `genre_conflict_min_confidence` | (deleted — see below) |

The `min_confidence` hard gate code path is deleted entirely from `candidate_pool.py`. The v4.1 work proved that the gate at any positive value over a 764-dim identity-affinity vocab rejects ~50% of legitimate candidates. There's no scenario where re-enabling it is the right answer with the current vocabulary; keeping the code around as dead-but-available was just confusing surface area. The soft penalty (renamed `genre_compatibility_penalty_strength`) remains and continues to demote off-axis tracks.

### 7. Mode preset integration

`src/playlist/mode_presets.py` (or wherever mode→config translation lives) gets the new `genre_idf_enabled` key wired into each preset:

| Mode | `genre_idf_enabled` |
|---|---|
| `strict` | true |
| `narrow` | true |
| `dynamic` | true |
| `discover` | false |
| `off` | n/a |

`discover` keeps unweighted similarity because that mode is for exploration — rewarding narrow tag matches would defeat the point.

## Config knobs (final user-facing surface)

**One new knob:**
- `playlists.ds_pipeline.candidate_pool.genre_idf_enabled: bool` — admission applies IDF weighting to genre similarity. Default mode-driven (true for narrow/dynamic/strict, false for discover). User can override.

**Existing knobs reused:**
- `playlists.ds_pipeline.pier_bridge.dj_genre_idf_power: 1.0` — same value used by admission.
- `playlists.ds_pipeline.pier_bridge.dj_genre_idf_norm: max1` — same.

**Renamed (old keys deleted, no alias):**
- `genre_conflict_*` → `genre_compatibility_*` (4 keys renamed, 1 deleted).
- `title_exclusion_words` → `title_hard_exclude_flags` (data type changes from list of substrings to set of flag names).

## Data flow

The IDF weights are computed once per pipeline run (from the artifact's full genre matrix) and shared between admission and dj_bridging. No per-call recomputation.

The admission filter computes genre similarity as follows when IDF is enabled:

```
seed_genres = np.max(genre_matrix[seed_indices], axis=0)
idf = compute_genre_idf(X_genre_raw=raw_matrix, power=cfg.idf_power, norm=cfg.idf_norm)
seed_w = seed_genres * idf
cand_w = candidate_genres * idf
sim = cosine(cand_w, seed_w)  # or jaccard / ensemble
```

When IDF is disabled, `seed_genres * idf` is skipped and the math reverts to the current behavior.

## Testing

### New unit tests

1. **`tests/unit/test_genre_idf.py`**: `compute_genre_idf` returns higher weights for rare tags than common tags; `max1` / `sum1` / `none` normalization produce expected output ranges; power exponent works as documented.

2. **`tests/unit/test_candidate_pool_idf.py`**:
   - With IDF enabled, two candidates that have identical raw-cosine similarity to seeds but differ on rare-vs-common tag matches produce different ranking — rare-match candidate ranks higher.
   - With IDF disabled, both candidates produce equal scores (sanity check).
   - End-to-end: a candidate that matches only on `indie/rock/pop` scores lower than one that matches on `slowcore/shoegaze`, given seeds that have all of those tags.

3. **`tests/unit/test_candidate_pool_max_over_seeds.py`**:
   - Multi-seed playlist with seeds in different genre profiles. Before the fix, a candidate strongly matching seed #2's genre but unrelated to seed #1 is rejected. After the fix, it's admitted because max-over-seeds picks up the seed #2 match.

4. **`tests/unit/test_candidate_pool_title_consolidation.py`**:
   - Tracks titled with `interlude`, `skit`, `acapella` are still excluded (functional parity with old behavior).
   - User can add `medley` to `title_hard_exclude_flags` and a track titled `"X (Medley)"` is excluded.
   - `a cappella` and `a capella` spelling variants are still caught (via the consolidated `acapella` flag's pattern).

5. **`tests/unit/test_title_quality_new_flags.py`**: The three new flags (`interlude`, `skit`, `acapella`) are detected by `detect_title_artifacts` with correct word boundaries (no false positives on `interludial`, `skitter`, etc.).

### Golden updates

Pipeline smoke goldens (`tests/unit/goldens/pipeline/*.json`) will see new genre-similarity numbers because IDF weighting changes them. Inspect each diff to confirm:
- Rare-tag candidates moved up in eligible artists ranking
- Common-tag-only candidates moved down or rejected
- Pool sizes haven't collapsed

If the changes look as designed, update goldens. If a golden diff doesn't make sense, investigate before accepting.

### End-to-end validation

After implementation, manually regenerate two reference playlists and compare admission pool composition before/after:
- A multi-seed mix spanning different genres (the Toro Y Moi-style chillwave/pop/R&B/electronic combo)
- An artist-mode narrow-style playlist (e.g., Tiger Trap)

Looking for:
- Pool size approximately stable (we're re-weighting, not adding/removing)
- Rare-tag-matching artists more prominent
- Multi-seed cases: tracks aligned to seed #2/#3 now showing up

## What gets deleted

- `src/playlist/filtering.py:is_title_excluded` function (replaced by `detect_title_artifacts` flag-set check)
- `candidate_pool.py:genre_conflict_min_confidence` config field and the dead gate code path that uses it
- The duplicate IDF computation inside `pier_bridge_builder.py` (moved to shared module)
- `candidate_pool.title_exclusion_words` config field (replaced by `title_hard_exclude_flags`)
- All `genre_conflict_*` config field names (replaced by `genre_compatibility_*`)
- Any code reading the deleted config keys

## What's parked for follow-up

See `docs/CANDIDATE_FILTERING_BACKLOG.md` for the full backlog. Items deferred from this design:

- **A3** — genre-neighbor pool primary-seed-only filter (already mostly correct in single-seed-artist case)
- **C2** — overlap guard redundancy investigation with `broad_filters`
- **D1, D2, D3** — positive-pressure subgenre diversity, easy-out sonic prevention, subgenre arc planning
- **E1** — `bridge_floor` cleanup (currently 0.02, doing essentially no filtering)
