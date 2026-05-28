# Candidate Filtering Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the candidate-filtering optimization (Scope B from `docs/CANDIDATE_FILTERING_BACKLOG.md`) — fix the primary-seed-only genre/overlap-guard bugs, consolidate title quality detection, add IDF-weighted admission genre similarity, rename `genre_conflict_*` → `genre_compatibility_*`, and delete dead code. No backward compatibility.

**Architecture:** New shared `src/playlist/genre_idf.py` module used by both the existing dj_bridging waypoint code and the new admission filter. Surgical changes to `candidate_pool.py` to fix bugs and add IDF support. Title detection consolidated into the existing `title_quality.py`. Old `genre_conflict_*` config keys renamed; deprecated `min_confidence` gate code deleted.

**Tech Stack:** Python 3.11+, NumPy, existing pier-bridge infrastructure, pytest.

**Reference documents:**
- Design spec: `docs/superpowers/specs/2026-05-21-candidate-filtering-optimization-design.md`
- Backlog of deferred items: `docs/CANDIDATE_FILTERING_BACKLOG.md`

---

## File Structure

**New:**
- `src/playlist/genre_idf.py` — pure-function IDF computation, single source of truth
- `tests/unit/test_genre_idf.py`
- `tests/unit/test_candidate_pool_idf.py`
- `tests/unit/test_candidate_pool_max_over_seeds.py`
- `tests/unit/test_candidate_pool_title_consolidation.py`
- `tests/unit/test_title_quality_new_flags.py`

**Modified:**
- `src/playlist/title_quality.py` — add `interlude`/`skit`/`acapella` flags
- `src/playlist/candidate_pool.py` — consolidate title filter, fix max-over-seeds (genre + overlap guard), wire IDF into genre similarity, rename `genre_conflict_*` → `genre_compatibility_*`, delete `min_confidence` gate code
- `src/playlist/config.py` — rename `CandidatePoolConfig` fields; delete `title_exclusion_enabled`/`title_exclusion_words`/`genre_conflict_min_confidence`; add `title_hard_exclude_flags` and `genre_idf_enabled`
- `src/playlist/pier_bridge_builder.py` — delete the local IDF copy, import from `genre_idf`
- `src/playlist/filtering.py` — delete `is_title_excluded` function
- `src/config_loader.py` — parse renamed keys; delete old key parsing
- `src/playlist/mode_presets.py` — add `genre_idf_enabled` to per-mode overrides
- `config.yaml` — rename keys in the `candidate_pool` block; remove deleted keys
- `tests/unit/goldens/pipeline/*.json` (4 files) — regenerate after numeric changes settle

---

### Task 1: Shared IDF module + refactor pier_bridge to use it

Lift IDF computation out of `pier_bridge_builder.py` into a new module so admission can share it.

**Files:**
- Create: `src/playlist/genre_idf.py`
- Modify: `src/playlist/pier_bridge_builder.py` (delete local `_compute_genre_idf` body)
- Test: `tests/unit/test_genre_idf.py`

- [ ] **Step 1: Write failing tests for `compute_genre_idf`**

Create `tests/unit/test_genre_idf.py`:

```python
"""Tests for the shared genre IDF computation."""
import numpy as np
import pytest

from src.playlist.genre_idf import compute_genre_idf


def _toy_matrix() -> np.ndarray:
    # 5 tracks, 4 genres. Genre 0 appears in all 5 (common).
    # Genre 1 in 4. Genre 2 in 2 (rare). Genre 3 in 1 (rarest).
    return np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=float)


def test_rare_genres_get_higher_weights():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    # Rarer genre should have strictly higher weight.
    assert idf[3] > idf[2] > idf[1] > idf[0]


def test_max1_normalization_caps_at_one():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    assert float(np.max(idf)) == pytest.approx(1.0)
    assert float(np.min(idf)) > 0.0


def test_sum1_normalization_sums_to_one():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="sum1")
    assert float(np.sum(idf)) == pytest.approx(1.0)


def test_none_normalization_returns_raw_idf_values():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="none")
    # Rare genre weights should be > 1.0 with no normalization in this construction.
    assert float(np.max(idf)) > 1.0


def test_power_zero_collapses_to_uniform_weights():
    idf = compute_genre_idf(X_genre_raw=_toy_matrix(), power=0.0, norm="none")
    # With power=0, every IDF value becomes 1.0 before any normalization.
    assert np.allclose(idf, 1.0)


def test_power_two_amplifies_rare_more_than_power_one():
    idf_p1 = compute_genre_idf(X_genre_raw=_toy_matrix(), power=1.0, norm="max1")
    idf_p2 = compute_genre_idf(X_genre_raw=_toy_matrix(), power=2.0, norm="max1")
    # Rarest tag still wins at 1.0 in both, but common tag should be more demoted at p=2.
    assert idf_p2[0] < idf_p1[0]


def test_empty_matrix_returns_empty_array():
    idf = compute_genre_idf(
        X_genre_raw=np.zeros((0, 4), dtype=float),
        power=1.0,
        norm="max1",
    )
    assert idf.shape == (4,)
```

- [ ] **Step 2: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_genre_idf.py -q --basetemp .pytest-tmp-idf -o cache_dir=.pytest-tmp-cache-idf
```
Expected: ImportError for `src.playlist.genre_idf`.

- [ ] **Step 3: Implement `compute_genre_idf`**

Create `src/playlist/genre_idf.py`:

```python
"""Shared IDF weight computation for the genre vocabulary.

Single source of truth used by:
 - the dj_bridging waypoint scoring in src/playlist/pier_bridge_builder.py
 - the candidate-pool admission genre similarity in src/playlist/candidate_pool.py

Pure function; no I/O, no logging.
"""
from __future__ import annotations

import numpy as np


def compute_genre_idf(
    *,
    X_genre_raw: np.ndarray,
    power: float = 1.0,
    norm: str = "max1",
) -> np.ndarray:
    """Return one IDF weight per genre column.

    Args:
        X_genre_raw: (N, V) raw genre presence matrix (binary or float >0).
        power: exponent applied to raw IDF before normalization. 0 collapses to
            uniform weights. 1.0 is standard IDF. Higher values amplify rarity.
        norm: "max1" (largest weight is 1.0), "sum1" (weights sum to 1.0), or
            "none" (raw IDF values, smallest is 1.0 by the smoothed-log formula).

    Returns:
        (V,) array of weights, higher for rare tags, lower for common tags.

    The IDF formula is the smoothed log-IDF:
        idf_g = log((N + 1) / (df_g + 1)) + 1
    where df_g is the count of tracks where genre g is present.
    """
    presence = (np.asarray(X_genre_raw) > 0).astype(float)
    n = float(presence.shape[0])
    if presence.size == 0:
        return np.zeros(presence.shape[1], dtype=float)
    df = presence.sum(axis=0)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    if float(power) != 1.0:
        idf = idf ** float(power)

    method = str(norm).strip().lower()
    if method == "max1":
        max_val = float(np.max(idf))
        if max_val > 1e-12:
            idf = idf / max_val
    elif method == "sum1":
        total = float(np.sum(idf))
        if total > 1e-12:
            idf = idf / total
    elif method == "none":
        pass
    else:
        # Unknown norm — fall back to max1 to keep behavior bounded.
        max_val = float(np.max(idf))
        if max_val > 1e-12:
            idf = idf / max_val
    return idf
```

- [ ] **Step 4: Run tests and verify pass**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_genre_idf.py -q --basetemp .pytest-tmp-idf -o cache_dir=.pytest-tmp-cache-idf
```
Expected: 7 passed.

- [ ] **Step 5: Refactor `pier_bridge_builder.py` to use the shared module**

In `src/playlist/pier_bridge_builder.py`, find the existing `_compute_genre_idf` function definition (search the file for `def _compute_genre_idf`). It accepts `X_genre_raw`, `idf_power`, `idf_norm`. Replace its body to delegate:

```python
def _compute_genre_idf(
    X_genre_raw,
    idf_power,
    idf_norm,
):
    """Wrapper kept for ABI compatibility with internal call sites.

    Delegates to src.playlist.genre_idf.compute_genre_idf — the single source
    of truth for IDF weights. New code should import compute_genre_idf directly.
    """
    from src.playlist.genre_idf import compute_genre_idf
    return compute_genre_idf(
        X_genre_raw=X_genre_raw,
        power=float(idf_power),
        norm=str(idf_norm),
    )
```

Note: there's also a thin back-compat wrapper in `src/playlist/pier_bridge/config.py` (`_compute_genre_idf`) and an implementation in `src/playlist/pier_bridge/genre.py`. Update `pier_bridge/genre.py:_compute_genre_idf` to delegate the same way, and leave the config.py wrapper alone (it already imports from `genre.py`).

- [ ] **Step 6: Run pier-bridge smoke tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pier_bridge_smoke_golden.py tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-pb-smoke -o cache_dir=.pytest-tmp-cache-pb-smoke
```
Expected: pass. The IDF values should be identical to before since the math is the same; goldens shouldn't drift.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/genre_idf.py tests/unit/test_genre_idf.py src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/genre.py
git commit -m "feat: shared genre_idf module; dedup pier-bridge IDF computation"
```

---

### Task 2: Add `interlude`/`skit`/`acapella` flags to title_quality

Bring the structural-track flags (interlude, skit, acapella) into the same detector used everywhere else.

**Files:**
- Modify: `src/playlist/title_quality.py`
- Test: `tests/unit/test_title_quality_new_flags.py`

- [ ] **Step 1: Write failing tests for new flags**

Create `tests/unit/test_title_quality_new_flags.py`:

```python
"""Tests for the new structural-track flags in detect_title_artifacts."""
from src.playlist.title_quality import detect_title_artifacts


def test_interlude_detected_with_word_boundary():
    assert "interlude" in detect_title_artifacts("Interlude")
    assert "interlude" in detect_title_artifacts("Track Interlude")
    assert "interlude" in detect_title_artifacts("Side B Interlude")
    # Word boundary: 'interludial' must NOT match 'interlude'.
    assert detect_title_artifacts("Interludial Phase") == set()


def test_skit_detected_with_word_boundary():
    assert "skit" in detect_title_artifacts("Skit")
    assert "skit" in detect_title_artifacts("Skit 1")
    assert "skit" in detect_title_artifacts("Intro Skit")
    # Word boundary: 'skitter', 'skittle' must NOT match.
    assert detect_title_artifacts("Skitter Like Light") == set()
    assert detect_title_artifacts("Skittles") == set()


def test_acapella_detected_with_spelling_variants():
    assert "acapella" in detect_title_artifacts("Song (Acapella)")
    assert "acapella" in detect_title_artifacts("Song (A Cappella)")
    assert "acapella" in detect_title_artifacts("Song (A Capella)")
    assert "acapella" in detect_title_artifacts("Acapella Version")
    # Word boundary: 'capella' alone shouldn't trigger; only the spelling variants we care about.
    assert detect_title_artifacts("Capella Star System") == set()


def test_new_flags_do_not_collide_with_existing_flags():
    # The new flags should be additive — existing detection still works.
    assert "demo" in detect_title_artifacts("Lee #2 (8 Track Demo)")
    assert "live" in detect_title_artifacts("All Of You (Live At The Village Vanguard)")
    assert "medley" in detect_title_artifacts("Rubber Ring/What She Said (Medley)")
```

- [ ] **Step 2: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_quality_new_flags.py -q --basetemp .pytest-tmp-tq -o cache_dir=.pytest-tmp-cache-tq
```
Expected: assertion failures for `interlude`, `skit`, `acapella` — those flags don't exist yet.

- [ ] **Step 3: Extend `_FLAG_PATTERNS` in `title_quality.py`**

Open `src/playlist/title_quality.py`. Find the `_FLAG_PATTERNS` dict. Add three entries at the end of the dict (before the closing brace):

```python
    "interlude": [
        r"\binterlude\b",
    ],
    "skit": [
        r"\bskit\b",
    ],
    "acapella": [
        r"\bacapella\b",
        r"\ba\s+cappella\b",
        r"\ba\s+capella\b",
    ],
```

- [ ] **Step 4: Run tests and verify pass**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_title_quality_new_flags.py tests/unit/test_title_quality.py -q --basetemp .pytest-tmp-tq -o cache_dir=.pytest-tmp-cache-tq
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/title_quality.py tests/unit/test_title_quality_new_flags.py
git commit -m "feat: title_quality detects interlude/skit/acapella flags"
```

---

### Task 3: Consolidate title filter in candidate_pool

Replace the legacy `is_title_excluded` substring matcher with the unified `detect_title_artifacts` flag detector. Add the new `title_hard_exclude_flags` config field. Delete the old `title_exclusion_enabled`/`title_exclusion_words` fields and the `is_title_excluded` helper.

**Files:**
- Modify: `src/playlist/candidate_pool.py`
- Modify: `src/playlist/config.py`
- Modify: `src/playlist/filtering.py` (delete `is_title_excluded`)
- Modify: `src/config_loader.py` (parsing update)
- Modify: `config.yaml`
- Test: `tests/unit/test_candidate_pool_title_consolidation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_candidate_pool_title_consolidation.py`:

```python
"""Title consolidation: candidate_pool uses detect_title_artifacts."""
import numpy as np
import pytest

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(*, hard_exclude_flags=frozenset({"interlude", "skit", "acapella"})):
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=hard_exclude_flags,
    )


def _embedding(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 8))


def test_interlude_track_is_excluded_by_default():
    embedding = _embedding()
    artist_keys = np.array(["a", "b", "c", "d", "e", "f"])
    titles = np.array(["Seed", "Real Track", "Some Interlude", "Another Real", "Live At Venue", "Demo Cut"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted_titles = {titles[int(i)] for i in result.pool_indices}
    assert "Some Interlude" not in admitted_titles
    # Live/demo are NOT in the default hard-exclude set; they should still admit.
    assert "Live At Venue" in admitted_titles
    assert "Demo Cut" in admitted_titles


def test_user_can_add_medley_to_hard_exclude():
    embedding = _embedding()
    artist_keys = np.array(["a", "b", "c", "d", "e", "f"])
    titles = np.array(["Seed", "Real", "Track (Medley)", "Other", "Fine", "Fine 2"])
    cfg = _make_cfg(hard_exclude_flags=frozenset({"interlude", "skit", "acapella", "medley"}))

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted_titles = {titles[int(i)] for i in result.pool_indices}
    assert "Track (Medley)" not in admitted_titles


def test_acapella_variants_all_excluded():
    embedding = _embedding(n=5)
    artist_keys = np.array(["seed", "b", "c", "d", "e"])
    titles = np.array([
        "Seed",
        "Song (Acapella)",
        "Song (A Cappella)",
        "Song (A Capella)",
        "Normal Song",
    ])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted = {titles[int(i)] for i in result.pool_indices}
    assert "Normal Song" in admitted
    assert "Song (Acapella)" not in admitted
    assert "Song (A Cappella)" not in admitted
    assert "Song (A Capella)" not in admitted


def test_empty_flag_set_admits_everything_titled():
    embedding = _embedding(n=4)
    artist_keys = np.array(["seed", "b", "c", "d"])
    titles = np.array(["Seed", "Interlude", "Skit 1", "Acapella"])
    cfg = _make_cfg(hard_exclude_flags=frozenset())

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted = {titles[int(i)] for i in result.pool_indices}
    # With empty hard-exclude set, all three structural-track titles slip through.
    assert "Interlude" in admitted
    assert "Skit 1" in admitted
    assert "Acapella" in admitted
```

- [ ] **Step 2: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_title_consolidation.py -q --basetemp .pytest-tmp-title -o cache_dir=.pytest-tmp-cache-title
```
Expected: failures (config field doesn't exist yet, or behavior wrong).

- [ ] **Step 3: Update `CandidatePoolConfig` in `src/playlist/config.py`**

Find the `CandidatePoolConfig` dataclass. Delete the existing `title_exclusion_enabled` and `title_exclusion_words` fields. Add:

```python
    title_hard_exclude_flags: frozenset[str] = frozenset({"interlude", "skit", "acapella"})
```

The field type is `frozenset[str]` (immutable, hashable, semantically right).

- [ ] **Step 4: Replace title filter logic in `candidate_pool.py`**

In `src/playlist/candidate_pool.py`, find the block (currently around line 352-358) that calls `is_title_excluded`:

```python
        if (
            cfg.title_exclusion_enabled
            and track_titles is not None
            and is_title_excluded(track_titles[i], cfg.title_exclusion_words)
        ):
            title_exclusion_rejected += 1
            continue
```

Replace with:

```python
        if (
            track_titles is not None
            and cfg.title_hard_exclude_flags
            and detect_title_artifacts(track_titles[i]) & cfg.title_hard_exclude_flags
        ):
            title_exclusion_rejected += 1
            continue
```

Also update the import at the top of the file. Find the existing:
```python
from .filtering import is_title_excluded
```
Delete it. Add:
```python
from src.playlist.title_quality import detect_title_artifacts
```

- [ ] **Step 5: Delete `is_title_excluded` from `filtering.py`**

In `src/playlist/filtering.py`, find and delete the `is_title_excluded` function entirely. If `filtering.py` becomes empty afterward, leave the module file in place but empty (do not delete the file — other tests/imports may exist). Run grep to confirm no other call sites:

```
C:\Windows\py.exe -3.13 -c "import ast, pathlib; [print(p) for p in pathlib.Path('src').rglob('*.py') if 'is_title_excluded' in p.read_text(encoding='utf-8')]"
```

If anything shows up beyond `filtering.py`, update those call sites the same way (use `detect_title_artifacts`).

- [ ] **Step 6: Update `config_loader.py` parsing**

In `src/config_loader.py` (or wherever `CandidatePoolConfig` is constructed from yaml — grep for `title_exclusion_words` or `CandidatePoolConfig(`), update the field parsing.

Replace any code that reads `title_exclusion_enabled` / `title_exclusion_words` with code that reads `title_hard_exclude_flags`:

```python
title_hard_exclude_flags = candidate_pool_cfg.get(
    "title_hard_exclude_flags",
    ["interlude", "skit", "acapella"],
)
# Normalize to frozenset of lowercase strings
title_hard_exclude_flags = frozenset(
    str(flag).strip().lower()
    for flag in (title_hard_exclude_flags or [])
    if str(flag).strip()
)
```

Pass `title_hard_exclude_flags=...` when constructing the `CandidatePoolConfig` and delete the old kwargs.

- [ ] **Step 7: Update `config.yaml`**

In `config.yaml`, find the `candidate_pool:` block. Replace:

```yaml
      title_exclusion_enabled: true
      title_exclusion_words: ["interlude", "skit", "acapella", "a cappella", "a capella"]
```

with:

```yaml
      title_hard_exclude_flags:
        - interlude
        - skit
        - acapella
```

(The `acapella` flag's pattern already catches `a cappella` and `a capella` — no need to list them.)

- [ ] **Step 8: Run focused + smoke tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_title_consolidation.py tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-title -o cache_dir=.pytest-tmp-cache-title
```
Expected: title-consolidation tests pass. Pipeline smoke goldens may fail because the effective-config JSON changed (`title_exclusion_words` removed, `title_hard_exclude_flags` added). Update goldens after confirming the JSON diff is just the field rename.

- [ ] **Step 9: Update pipeline smoke goldens**

For each of the 4 files in `tests/unit/goldens/pipeline/*.json`, find the `candidate_pool` object inside `effective.candidate_pool`. Remove the old fields and add the new one, preserving sort order if the project sorts effective-config JSON (look at how prior commits handled this — `108fa1d`, `72616ab`).

```json
"title_hard_exclude_flags": ["acapella", "interlude", "skit"]
```

(Listing the values sorted ensures determinism.)

- [ ] **Step 10: Re-run smoke tests to confirm goldens pass**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-title-smoke -o cache_dir=.pytest-tmp-cache-title-smoke
```
Expected: pass.

- [ ] **Step 11: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/config.py src/playlist/filtering.py src/config_loader.py config.yaml tests/unit/test_candidate_pool_title_consolidation.py tests/unit/goldens/pipeline/
git commit -m "refactor: consolidate title filter on detect_title_artifacts"
```

---

### Task 4: Fix max-over-seeds for genre similarity + overlap guard (bug A1)

Two lines in `candidate_pool.py` reference only the primary seed for genre filtering. Both need to use `np.max(... seed_list)` like the sonic and hybrid filters already do.

**Files:**
- Modify: `src/playlist/candidate_pool.py`
- Test: `tests/unit/test_candidate_pool_max_over_seeds.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_candidate_pool_max_over_seeds.py`:

```python
"""Multi-seed genre filtering must use max-over-seeds, not primary-seed-only."""
import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(**overrides):
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
    )
    base.update(overrides)
    return CandidatePoolConfig(**base)


def test_genre_filter_admits_candidate_aligned_to_secondary_seed():
    # 4 tracks, 3 genres.
    # Seed 0: genre 0 ("electronic"). Seed 1: genre 2 ("indie").
    # Candidate 2: genre 0 only. Aligned to seed 0. Should pass.
    # Candidate 3: genre 2 only. Aligned to seed 1 (secondary). Should ALSO pass with the fix.
    X_genre = np.array([
        [1.0, 0.0, 0.0],  # seed 0
        [0.0, 0.0, 1.0],  # seed 1
        [1.0, 0.0, 0.0],  # candidate aligned to seed 0
        [0.0, 0.0, 1.0],  # candidate aligned to seed 1
    ], dtype=float)
    # Identical embedding so similarity_floor isn't the gate.
    emb = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=float)
    artist_keys = np.array(["seed_a", "seed_b", "cand_aligned_to_a", "cand_aligned_to_b"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[1],
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.5,
        genre_method="cosine",
        genre_vocab=["electronic", "rock", "indie"],
        mode="dynamic",
    )

    admitted_artists = {artist_keys[int(i)] for i in result.pool_indices}
    assert "cand_aligned_to_a" in admitted_artists
    # With the fix, candidate aligned to the SECONDARY seed should also be admitted.
    assert "cand_aligned_to_b" in admitted_artists


def test_overlap_guard_admits_candidate_with_secondary_seed_tag_overlap():
    # 4 tracks. Seed 0 has genre 0 only. Seed 1 has genre 2 only.
    # Candidate 2 has genre 2 only (overlaps with seed 1).
    # Without fix: overlap guard would reject because no overlap with seed 0.
    # With fix: max-over-seeds picks up the overlap with seed 1.
    X_genre = np.array([
        [1.0, 0.0, 0.0],  # seed 0
        [0.0, 0.0, 1.0],  # seed 1
        [0.0, 0.0, 1.0],  # candidate
        [0.0, 1.0, 0.0],  # unrelated track
    ], dtype=float)
    emb = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=float)
    artist_keys = np.array(["seed_a", "seed_b", "cand_overlap_b", "unrelated"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[1],
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=emb,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.4,
        genre_method="ensemble",
        genre_vocab=["electronic", "rock", "indie"],
        broad_filters=("electronic",),  # triggers narrow-mode overlap guard
        mode="narrow",
    )

    admitted_artists = {artist_keys[int(i)] for i in result.pool_indices}
    assert "cand_overlap_b" in admitted_artists
```

- [ ] **Step 2: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_max_over_seeds.py -q --basetemp .pytest-tmp-mos -o cache_dir=.pytest-tmp-cache-mos
```
Expected: at least the second test fails (overlap guard rejects candidate aligned to secondary seed); the first may pass or fail depending on threshold math.

- [ ] **Step 3: Fix genre similarity reference (line ~311)**

In `src/playlist/candidate_pool.py`, find:

```python
        seed_genres = genre_matrix[seed_idx]
```

Replace with:

```python
        # Use max over all seed genre vectors so multi-seed playlists are
        # not biased to the primary seed. Matches what the genre_compatibility
        # code does at the line below for raw_seed aggregation.
        seed_genres = np.max(genre_matrix[seed_list], axis=0)
```

- [ ] **Step 4: Fix overlap guard reference (line ~413)**

In the same file, find:

```python
            seed_binary = (genre_raw_matrix[seed_idx] > 0).astype(float)
```

Replace with:

```python
            # Max over seed list so a candidate with overlap to ANY seed survives.
            seed_binary = (np.max(genre_raw_matrix[seed_list], axis=0) > 0).astype(float)
```

- [ ] **Step 5: Run focused tests + full smoke suite**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_max_over_seeds.py tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py -q --basetemp .pytest-tmp-mos -o cache_dir=.pytest-tmp-cache-mos
```
Expected: max-over-seeds tests pass. Smoke goldens *may* drift if the seed selection in goldens happens to be multi-seed and produces a different pool. Inspect any drift and update goldens if the diff looks correct.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/candidate_pool.py tests/unit/test_candidate_pool_max_over_seeds.py
git add -u tests/unit/goldens/pipeline/  # only if goldens drifted
git commit -m "fix: candidate_pool genre filter + overlap guard use max-over-seeds"
```

---

### Task 5: IDF-weighted admission genre similarity

Add an optional `idf_weights` argument to `_compute_genre_similarity` and wire it from `build_candidate_pool` when the new `genre_idf_enabled` config flag is true.

**Files:**
- Modify: `src/playlist/candidate_pool.py`
- Modify: `src/playlist/config.py` (add `genre_idf_enabled` field)
- Modify: `src/config_loader.py` (parse new field)
- Modify: `config.yaml`
- Test: `tests/unit/test_candidate_pool_idf.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_candidate_pool_idf.py`:

```python
"""IDF-weighted admission genre similarity ranks rare-tag matches higher."""
import numpy as np

from src.playlist.candidate_pool import _compute_genre_similarity, build_candidate_pool
from src.playlist.config import CandidatePoolConfig
from src.playlist.genre_idf import compute_genre_idf


def _make_cfg(*, idf_enabled: bool):
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
        genre_idf_enabled=idf_enabled,
    )


def test_idf_disabled_produces_identical_scores_for_equal_cosine():
    # Two candidates with same raw cosine to the seed; without IDF they tie.
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0, 0.0])  # matches the common tag
    b = np.array([0.0, 1.0, 0.0, 0.0])  # matches the rare tag (same shape)
    candidates = np.stack([a, b])

    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=None)
    assert sim[0] == sim[1]


def test_idf_enabled_ranks_rare_tag_match_higher():
    # Same two candidates, but now IDF weights tag 1 as rare and tag 0 as common.
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    candidates = np.stack([a, b])

    # Tag 0 is common (high df); tag 1 is rare (low df). Weights reflect this.
    idf = np.array([0.2, 1.0, 0.5, 0.5])

    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=idf)
    assert sim[1] > sim[0]


def test_idf_zero_weights_collapse_to_zero_similarity():
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    cand = np.array([1.0, 1.0, 0.0, 0.0])
    candidates = np.stack([cand])
    idf = np.zeros(4, dtype=float)

    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=idf)
    # Zero weights everywhere -> zero-norm vectors -> safe-zero similarity.
    assert sim[0] == 0.0


def test_idf_applied_in_build_candidate_pool_when_enabled():
    # 4 tracks, 3 genres. Seed: rare tag and common tag.
    # Candidate a: only common-tag match. Candidate b: only rare-tag match.
    # With IDF enabled, b should rank ahead of a in the pool.
    X_genre = np.array([
        [1.0, 1.0, 0.0],  # seed: common + rare
        [1.0, 1.0, 1.0],  # common tag is common across population
        [1.0, 0.0, 0.0],  # candidate a: common only
        [0.0, 1.0, 0.0],  # candidate b: rare only
    ], dtype=float)
    # All-equal embedding so admission similarity_floor isn't the gate.
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed", "noise", "cand_common", "cand_rare"])
    cfg = _make_cfg(idf_enabled=True)

    result = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.3,
        genre_method="cosine",
        genre_vocab=["common", "rare", "filler"],
        mode="dynamic",
    )

    pool_artists = [artist_keys[int(i)] for i in result.pool_indices]
    assert "cand_rare" in pool_artists
    # Rare-match candidate should outrank common-match in the order.
    assert pool_artists.index("cand_rare") < pool_artists.index("cand_common")
```

- [ ] **Step 2: Run failing tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_idf.py -q --basetemp .pytest-tmp-idf-pool -o cache_dir=.pytest-tmp-cache-idf-pool
```
Expected: failures — the function doesn't accept `idf_weights` yet, and `genre_idf_enabled` doesn't exist on `CandidatePoolConfig`.

- [ ] **Step 3: Add `genre_idf_enabled` to `CandidatePoolConfig`**

In `src/playlist/config.py`, find `CandidatePoolConfig` and add (near other genre-related fields):

```python
    genre_idf_enabled: bool = True
```

- [ ] **Step 4: Add `idf_weights` parameter to `_compute_genre_similarity`**

In `src/playlist/candidate_pool.py`, modify the signature and body of `_compute_genre_similarity`:

```python
def _compute_genre_similarity(
    seed_genres: np.ndarray,
    candidate_genres: np.ndarray,
    method: str = "cosine",
    idf_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute genre similarity between seed and all candidates.

    When idf_weights is provided, seed and candidate vectors are multiplied
    elementwise by the weights before computing cosine/jaccard/ensemble.
    Rare-tag matches contribute more; common-tag matches less. When None,
    behavior is the existing unweighted similarity.

    Args:
        seed_genres: (D,) 1D array of seed's genres (binary or float).
        candidate_genres: (N, D) matrix of candidate genres.
        method: "weighted_jaccard", "cosine", or "ensemble".
        idf_weights: (D,) optional per-genre weights. Apply before similarity.

    Returns:
        (N,) similarity scores [0, 1].
    """
    if idf_weights is not None:
        weights = np.asarray(idf_weights, dtype=float)
        seed_genres = seed_genres * weights
        candidate_genres = candidate_genres * weights.reshape(1, -1)

    if method == "weighted_jaccard":
        seed_binary = (seed_genres > 0).astype(float)
        cand_binary = (candidate_genres > 0).astype(float)
        intersection = np.sum(seed_binary * cand_binary, axis=1)
        union = np.sum(np.maximum(seed_binary, cand_binary), axis=1)
        union = np.maximum(union, 1e-12)
        return intersection / union

    if method == "cosine":
        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        cand_norm = candidate_genres / (
            np.linalg.norm(candidate_genres, axis=1, keepdims=True) + 1e-12
        )
        sim = np.dot(cand_norm, seed_norm)
        return np.clip(sim, 0.0, 1.0)

    if method == "ensemble":
        seed_binary = (seed_genres > 0).astype(float)
        cand_binary = (candidate_genres > 0).astype(float)
        intersection = np.sum(seed_binary * cand_binary, axis=1)
        union = np.sum(np.maximum(seed_binary, cand_binary), axis=1)
        union = np.maximum(union, 1e-12)
        jaccard_sim = intersection / union

        seed_norm = seed_genres / (np.linalg.norm(seed_genres) + 1e-12)
        cand_norm = candidate_genres / (
            np.linalg.norm(candidate_genres, axis=1, keepdims=True) + 1e-12
        )
        cosine_sim = np.clip(np.dot(cand_norm, seed_norm), 0.0, 1.0)

        return 0.6 * cosine_sim + 0.4 * jaccard_sim

    raise ValueError(f"Unknown genre_method: {method}")
```

- [ ] **Step 5: Wire IDF into `build_candidate_pool`**

In `src/playlist/candidate_pool.py`, find the block that computes genre similarity (around line 302-317 after Task 4 changes). It looks like:

```python
    if min_genre_similarity is not None and (X_genre_raw is not None or X_genre_smoothed is not None):
        # Choose matrix based on method
        ...
        seed_genres = np.max(genre_matrix[seed_list], axis=0)
        genre_sim_all = _compute_genre_similarity(seed_genres, genre_matrix, method=genre_method)
```

Update it to compute IDF weights when enabled and pass them through:

```python
    if min_genre_similarity is not None and (X_genre_raw is not None or X_genre_smoothed is not None):
        # Choose matrix based on method
        if genre_method == "weighted_jaccard" and genre_raw_matrix is not None:
            genre_matrix = genre_raw_matrix
        elif X_genre_smoothed is not None:
            genre_matrix = X_genre_smoothed[:, genre_mask] if genre_mask is not None else X_genre_smoothed
        else:
            genre_matrix = genre_raw_matrix if genre_raw_matrix is not None else X_genre_smoothed

        # Compute IDF weights from the raw matrix (matches dj_bridging's reference).
        idf_weights = None
        if (
            bool(getattr(cfg, "genre_idf_enabled", True))
            and genre_raw_matrix is not None
            and genre_raw_matrix.shape[1] == genre_matrix.shape[1]
        ):
            from src.playlist.genre_idf import compute_genre_idf
            idf_weights = compute_genre_idf(
                X_genre_raw=genre_raw_matrix,
                power=1.0,
                norm="max1",
            )

        seed_genres = np.max(genre_matrix[seed_list], axis=0)
        genre_sim_all = _compute_genre_similarity(
            seed_genres,
            genre_matrix,
            method=genre_method,
            idf_weights=idf_weights,
        )
        genre_sim_all[seed_idx] = 1.0  # seed matches itself perfectly
        logger.info(
            "Candidate pool genre gating: method=%s, min_threshold=%.3f, mode=%s, idf=%s",
            genre_method, min_genre_similarity, mode, "on" if idf_weights is not None else "off",
        )
```

(Note: `genre_matrix.shape[1] == genre_raw_matrix.shape[1]` ensures dimensions align after any broad-filter masking — both should be masked consistently. If they differ, we skip IDF for safety.)

- [ ] **Step 6: Update `config_loader.py` parsing**

In `src/config_loader.py`, find where `CandidatePoolConfig` is constructed. Add `genre_idf_enabled` parsing:

```python
genre_idf_enabled = candidate_pool_cfg.get("genre_idf_enabled", True)
```

And pass `genre_idf_enabled=bool(genre_idf_enabled)` when constructing the config.

- [ ] **Step 7: Add `genre_idf_enabled: true` to `config.yaml`**

In `config.yaml`, in the `candidate_pool:` block, add:

```yaml
      genre_idf_enabled: true
```

Place it near the other genre-related fields.

- [ ] **Step 8: Run focused tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_candidate_pool_idf.py -q --basetemp .pytest-tmp-idf-pool -o cache_dir=.pytest-tmp-cache-idf-pool
```
Expected: 4 passed.

- [ ] **Step 9: Run smoke goldens; refresh if needed**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-idf-smoke -o cache_dir=.pytest-tmp-cache-idf-smoke
```
Goldens **will** drift here — the `effective.candidate_pool` JSON gains a `genre_idf_enabled: true` field, and the playlist selections themselves may differ because IDF re-weights the candidates. Inspect each golden diff:

1. The new field appears with value `true` (or matches `dynamic` mode default).
2. Track-selection drift, if any, should favor rarer-tag candidates over common-tag ones.

Update each golden after inspecting.

- [ ] **Step 10: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/config.py src/config_loader.py config.yaml tests/unit/test_candidate_pool_idf.py
git add -u tests/unit/goldens/pipeline/
git commit -m "feat: IDF-weighted admission genre similarity"
```

---

### Task 6: Rename `genre_conflict_*` → `genre_compatibility_*` + delete dead gate code

Pure rename of the four surviving fields, with deletion of the dead `min_confidence` gate code and its config field.

**Files:**
- Modify: `src/playlist/config.py`
- Modify: `src/playlist/candidate_pool.py`
- Modify: `src/config_loader.py`
- Modify: `config.yaml`
- Modify: `tests/unit/goldens/pipeline/*.json` (4 files)

- [ ] **Step 1: Find every reference to `genre_conflict_`**

Run:
```
C:\Windows\py.exe -3.13 -c "import pathlib; [print(p) for p in pathlib.Path('.').rglob('*.py') if 'genre_conflict_' in p.read_text(encoding='utf-8') and '.pytest-tmp' not in str(p)]"
```
And:
```
C:\Windows\py.exe -3.13 -c "import pathlib; [print(p) for p in pathlib.Path('.').rglob('*.yaml') if 'genre_conflict_' in p.read_text(encoding='utf-8')]"
C:\Windows\py.exe -3.13 -c "import pathlib; [print(p) for p in pathlib.Path('.').rglob('*.json') if 'genre_conflict_' in p.read_text(encoding='utf-8') and '.pytest-tmp' not in str(p)]"
```

Make a list of every file. The rename is global.

- [ ] **Step 2: Rename `CandidatePoolConfig` fields**

In `src/playlist/config.py`, find `CandidatePoolConfig`:

```python
    genre_conflict_enabled: bool = True
    genre_conflict_min_confidence: Optional[float] = None
    genre_conflict_penalty_strength: float = 0.20
    genre_conflict_compatible_threshold: float = 0.35
    genre_conflict_conflict_threshold: float = 0.15
```

Replace with:

```python
    genre_compatibility_enabled: bool = True
    genre_compatibility_penalty_strength: float = 0.20
    genre_compatibility_compatible_threshold: float = 0.35
    genre_compatibility_conflict_threshold: float = 0.15
```

Note: `genre_conflict_min_confidence` is deleted entirely.

- [ ] **Step 3: Update `candidate_pool.py` to use the new field names**

In `src/playlist/candidate_pool.py`, find every occurrence of `genre_conflict_` and replace with `genre_compatibility_`. Then, delete the dead gate code block (currently around lines 433-447 after Task 4 changes):

```python
    if genre_conflict_result is not None and cfg.genre_conflict_min_confidence is not None:
        min_confidence = float(cfg.genre_conflict_min_confidence)
        eligible_before_conflict = len(eligible)
        eligible = [
            i for i in eligible
            if bool(genre_conflict_result.missing_or_sparse[i])
            or float(genre_conflict_result.confidence[i]) >= min_confidence
        ]
        genre_conflict_rejected = eligible_before_conflict - len(eligible)
        if genre_conflict_rejected:
            logger.info(
                "Genre conflict confidence gate applied: rejected=%d min_confidence=%.3f",
                genre_conflict_rejected,
                min_confidence,
            )
```

Delete this block entirely. Also delete the `genre_conflict_rejected` counter declaration earlier in the function (if it's no longer referenced).

In the `_first_rejection_reason` helper at the top of the file (around lines 130-165), remove the `genre_conflict_min_confidence` parameter and the corresponding check block:

```python
    if genre_conflict_result is not None and genre_conflict_min_confidence is not None:
        missing = bool(genre_conflict_result.missing_or_sparse[idx])
        confidence = float(genre_conflict_result.confidence[idx])
        if not missing and confidence < float(genre_conflict_min_confidence):
            return "genre_conflict"
```

Delete the block and remove the parameter. Update callers of `_first_rejection_reason` to drop that argument.

- [ ] **Step 4: Rename in `_first_rejection_reason` parameters**

Rename any parameter named `genre_conflict_*` to `genre_compatibility_*` to match the config field names. The "genre_conflict" rejection reason string can become "genre_compatibility" or stay "genre_conflict" — your call. Keeping the diagnostic label as `genre_conflict` is fine because it's a human-readable category and "conflict" describes *what triggered* the rejection.

Actually keep the rejection-reason string as `"genre_conflict"` since the math still measures conflict mass — only the config knobs are renamed.

- [ ] **Step 5: Update `src/config_loader.py`**

Find every `genre_conflict_` reference in `src/config_loader.py` and rename to `genre_compatibility_`. Delete any code that parses the `genre_conflict_min_confidence` field (e.g., the `if candidate_pool.get("genre_conflict_min_confidence") is None: ... else: float(...)` block).

- [ ] **Step 6: Update `config.yaml`**

In `config.yaml`, in the `candidate_pool:` block, find:

```yaml
      genre_conflict_enabled: true
      genre_conflict_min_confidence: null
      genre_conflict_penalty_strength: 0.20
      genre_conflict_compatible_threshold: 0.35
      genre_conflict_conflict_threshold: 0.15
```

Replace with:

```yaml
      genre_compatibility_enabled: true
      genre_compatibility_penalty_strength: 0.20
      genre_compatibility_compatible_threshold: 0.35
      genre_compatibility_conflict_threshold: 0.15
```

(`genre_compatibility_min_confidence` does not exist — the field is deleted.)

- [ ] **Step 7: Update pipeline smoke goldens**

For each of the 4 `tests/unit/goldens/pipeline/*.json` files, find the `effective.candidate_pool` block and apply the rename + deletion:

Remove these keys:
- `"genre_conflict_enabled"`
- `"genre_conflict_min_confidence"`
- `"genre_conflict_penalty_strength"`
- `"genre_conflict_compatible_threshold"`
- `"genre_conflict_conflict_threshold"`

Add these (with the same values, sorted to match the file's existing convention):
- `"genre_compatibility_compatible_threshold": 0.35`
- `"genre_compatibility_conflict_threshold": 0.15`
- `"genre_compatibility_enabled": true`
- `"genre_compatibility_penalty_strength": 0.2`

- [ ] **Step 8: Run focused + smoke tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py tests/unit/test_pier_bridge_smoke_golden.py tests/unit/test_candidate_pool_idf.py tests/unit/test_candidate_pool_max_over_seeds.py -q --basetemp .pytest-tmp-rename -o cache_dir=.pytest-tmp-cache-rename
```
Expected: pass.

- [ ] **Step 9: Verify no `genre_conflict_` references remain**

Run:
```
C:\Windows\py.exe -3.13 -c "import pathlib; remaining = [str(p) for p in pathlib.Path('.').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json','.md') and '.pytest-tmp' not in str(p) and 'genre_conflict_' in p.read_text(encoding='utf-8', errors='ignore')]; print('\n'.join(remaining) if remaining else 'CLEAN')"
```
Expected: `CLEAN`, with the possible exception of:
- `docs/CANDIDATE_FILTERING_BACKLOG.md` (historical references — leave intact)
- `docs/CHANGELOG.md` (historical references in v4.1 entry — leave intact)
- `docs/superpowers/specs/*.md` and `docs/superpowers/plans/*.md` (historical — leave intact)

If any *active source code* file (not docs) still has `genre_conflict_` references, fix them before commit.

- [ ] **Step 10: Commit**

```bash
git add src/playlist/config.py src/playlist/candidate_pool.py src/config_loader.py config.yaml tests/unit/goldens/pipeline/
git commit -m "refactor: rename genre_conflict_* to genre_compatibility_*; delete dead gate"
```

---

### Task 7: Mode preset integration

Wire `genre_idf_enabled` into the per-mode override logic so `discover` defaults to off.

**Files:**
- Modify: `src/playlist/mode_presets.py`
- Modify: `tests/unit/test_mode_threshold_resolution.py` (likely existing tests; add new ones)

- [ ] **Step 1: Find the mode preset structure**

Open `src/playlist/mode_presets.py`. Find the per-mode preset table (likely a dict keyed on mode name returning a dict of candidate_pool overrides).

- [ ] **Step 2: Write a failing test for mode-driven `genre_idf_enabled`**

Append to `tests/unit/test_mode_threshold_resolution.py` (or create a new test file if cleaner). The test imports the mode resolution function (look at existing tests in this file for the API).

```python
def test_genre_idf_enabled_default_per_mode():
    from src.playlist.mode_presets import apply_mode_preset_to_candidate_pool

    # Names of the function and dict may differ; adjust based on what exists.
    # The intent is: dynamic/narrow/strict default to True; discover defaults to False.
    for mode in ("strict", "narrow", "dynamic"):
        overrides = apply_mode_preset_to_candidate_pool(mode=mode, base={})
        assert overrides.get("genre_idf_enabled") is True, mode

    discover = apply_mode_preset_to_candidate_pool(mode="discover", base={})
    assert discover.get("genre_idf_enabled") is False
```

Adjust the import and function name to match what actually exists in `mode_presets.py`. Run:

```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_mode_threshold_resolution.py -q --basetemp .pytest-tmp-mode -o cache_dir=.pytest-tmp-cache-mode
```
Expected: failure — `genre_idf_enabled` isn't in any preset yet.

- [ ] **Step 3: Add `genre_idf_enabled` to each mode preset**

In `src/playlist/mode_presets.py`, for each mode preset, add `genre_idf_enabled` to the candidate_pool overrides:

- `strict`: `True`
- `narrow`: `True`
- `dynamic`: `True`
- `discover`: `False`
- `off`: omit (mode has no genre filtering anyway)
- `sonic_only`: omit (same)

If the preset format uses nested dicts like `{"candidate_pool": {"key": val, ...}}`, place `genre_idf_enabled` inside that nested block.

- [ ] **Step 4: Run focused tests**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_mode_threshold_resolution.py tests/unit/test_candidate_pool_idf.py -q --basetemp .pytest-tmp-mode -o cache_dir=.pytest-tmp-cache-mode
```
Expected: pass.

- [ ] **Step 5: Run pipeline smoke goldens; refresh if needed**

Run:
```
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_pipeline_smoke_golden.py -q --basetemp .pytest-tmp-mode-smoke -o cache_dir=.pytest-tmp-cache-mode-smoke
```

If any golden whose mode is `discover` shows `genre_idf_enabled: true` in the effective config, update it to `false`. For other modes the value should already be `true` from Task 5's golden refresh.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/mode_presets.py tests/unit/test_mode_threshold_resolution.py
git add -u tests/unit/goldens/pipeline/
git commit -m "feat: mode presets default genre_idf_enabled (true for narrow/dynamic/strict)"
```

---

### Task 8: Full test suite + backlog update

Final sweep. Confirm all 933+ tests still pass; mark Scope B as done in the backlog.

**Files:**
- Modify: `docs/CANDIDATE_FILTERING_BACKLOG.md`

- [ ] **Step 1: Run the full test suite**

Run:
```
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-final -o cache_dir=.pytest-tmp-cache-final
```
Expected: all pass. Investigate any failures.

- [ ] **Step 2: Update the backlog**

In `docs/CANDIDATE_FILTERING_BACKLOG.md`, find the "Scope chosen for the active brainstorm (2026-05-21)" section at the bottom. Replace it with:

```markdown
## Scope completed (2026-05-21)

**Scope B — Bugs + IDF-weighted admission genre filter — done.**

Items completed:
- A1 (genre admission max-over-seeds) ✓
- A2 (genre compatibility consistency check) ✓ — already correct; documented
- B1 (title quality consolidation) ✓
- B2 (IDF weighting in admission genre similarity) ✓
- C1 (genre conflict → genre compatibility rename; dead gate code removed) ✓

Deferred to follow-up brainstorms (see categories above):
- A3 (genre-neighbor pool primary-seed-only filter)
- C2 (overlap guard redundancy investigation with broad_filters)
- D1, D2, D3 (positive-pressure subgenre diversity, easy-out sonic prevention, subgenre arc planning)
- E1 (bridge_floor cleanup; currently 0.02, doing essentially no filtering)
```

- [ ] **Step 3: Commit**

```bash
git add docs/CANDIDATE_FILTERING_BACKLOG.md
git commit -m "docs: mark candidate filtering Scope B complete in backlog"
```

---

## Self-Review

**1. Spec coverage:**
- Spec component 1 (shared `genre_idf.py`) → Task 1 ✓
- Spec component 2 (refactor pier_bridge IDF) → Task 1 step 5 ✓
- Spec component 3 (`_compute_genre_similarity` accepts IDF) → Task 5 step 4 ✓
- Spec component 4 (genre + overlap guard max-over-seeds) → Task 4 ✓
- Spec component 5 (title quality consolidation) → Tasks 2 + 3 ✓
- Spec component 6 (genre_conflict → genre_compatibility + dead code removed) → Task 6 ✓
- Spec component 7 (mode preset integration) → Task 7 ✓
- "Config knobs" section (one new knob: `genre_idf_enabled`) → Task 5 step 3 + Task 7 ✓
- "What gets deleted" section → Tasks 3, 5, 6 cover every deletion ✓
- "Testing" section (5 new unit test files + goldens) → all covered ✓
- Backlog update → Task 8 ✓

**2. Placeholder scan:** All steps contain exact code or commands. The mode preset task (Task 7) calls out that the implementer needs to adjust the test imports to match the actual function names in `mode_presets.py` — that's an instruction to the implementer, not a placeholder.

**3. Type consistency:**
- `compute_genre_idf(*, X_genre_raw, power, norm) -> np.ndarray` — same signature in Tasks 1 and 5.
- `_compute_genre_similarity(seed_genres, candidate_genres, method="cosine", idf_weights=None) -> np.ndarray` — same signature in Task 5 implementation and Task 5 tests.
- `CandidatePoolConfig.title_hard_exclude_flags: frozenset[str]` — consistent across Tasks 3 and 5.
- `CandidatePoolConfig.genre_idf_enabled: bool` — consistent across Tasks 5 and 7.
- `CandidatePoolConfig.genre_compatibility_*` (four fields) — consistent across Task 6.
- Deleted: `title_exclusion_enabled`, `title_exclusion_words`, `genre_conflict_*` (all 5 keys) — consistent in Tasks 3 and 6.

**4. Test ordering:**
- Task 1 unit tests don't depend on anything else.
- Task 2 unit tests depend on `title_quality.py` extensions only.
- Task 3 unit tests depend on Tasks 1 + 2 (because `CandidatePoolConfig` is being modified). Tests call `build_candidate_pool` which uses the new field.
- Task 4 unit tests depend on Tasks 1-3 (config shape stable).
- Task 5 unit tests depend on Tasks 1-4 (uses `_compute_genre_similarity` and `genre_idf`).
- Task 6 (rename) intentionally last among code tasks — touches every preceding test file's config setup.
- Task 7 builds on the working pipeline.
- Task 8 is the final sweep.

Implementer should follow task order strictly; tests in earlier tasks may need their config kwargs updated when later tasks land if I missed a config-field churn point.
