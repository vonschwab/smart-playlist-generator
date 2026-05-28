# Artist Mode Genre Conflict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve artist-mode One Each playlists so sparse but culturally correct indie artists are recovered, while one-tag off-axis artists such as Tirzah/Suicide are penalized when their raw genre tags conflict with the seed artist envelope.

**Architecture:** Add a pure raw-genre compatibility module, then wire it into candidate admission as a tunable confidence/penalty stage that One Each fallback does not relax. Expand artist-style allowed IDs with a genre-neighbor union pool so culturally obvious artists can enter the pipeline even if the sonic-only style pool misses them. Add optional watched-artist diagnostics so future runs explain every gate.

**Tech Stack:** Python 3.11+, NumPy, existing DS pipeline modules, pytest.

---

## File Structure

- Create `src/playlist/genre_compatibility.py`
  - Pure NumPy helpers for raw genre envelope scoring.
  - No logging side effects, no DB/artifact loading.
- Modify `src/playlist/config.py`
  - Add genre-conflict and artist-style genre-union knobs to dataclasses/default config.
- Modify `src/playlist/candidate_pool.py`
  - Apply genre conflict confidence/penalty after genre similarity is computed.
  - Preserve missing/sparse metadata instead of punishing it.
- Modify `src/playlist/artist_style.py`
  - Add a genre-neighbor candidate pool builder that can be unioned with the existing sonic style pool.
- Modify `src/playlist_generator.py`
  - Parse artist-style genre-union config and include it in `style_allowed_track_ids`.
  - Pass watched artists from config/env into DS diagnostics if implemented at pipeline level.
- Optionally modify `src/playlist/pipeline/core.py`
  - Include watched-artist diagnostics in audit/preflight/fallback summaries.
- Modify `tests/test_candidate_filters.py`
  - Unit tests for conflict scoring in candidate admission and One Each fallback preserving conflict rules.
- Modify `tests/test_artist_style.py`
  - Unit tests for genre-neighbor union pool recovering sparse obvious artists and excluding conflict-heavy artists.
- Add `tests/unit/test_genre_compatibility.py`
  - Pure tests for scoring math.

---

### Task 1: Add Pure Genre Compatibility Scoring

**Files:**
- Create: `src/playlist/genre_compatibility.py`
- Test: `tests/unit/test_genre_compatibility.py`

- [ ] **Step 1: Write failing tests for sparse-match vs conflict-heavy candidates**

Create `tests/unit/test_genre_compatibility.py`:

```python
import numpy as np

from src.playlist.genre_compatibility import compute_raw_genre_compatibility


def test_sparse_candidate_with_single_compatible_tag_is_not_penalized():
    vocab = ["indie pop", "rnb", "house", "punk"]
    seed_raw = np.array([1.0, 0.0, 0.0, 1.0])
    candidates_raw = np.array([
        [1.0, 0.0, 0.0, 0.0],  # sparse but correct
    ])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
    )

    assert result.compatible_mass[0] > 0
    assert result.conflict_mass[0] == 0
    assert result.confidence[0] == 1.0


def test_one_overlap_plus_many_conflicting_tags_has_low_confidence():
    vocab = ["indie pop", "rnb", "house", "soul", "funk", "punk"]
    seed_raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    candidates_raw = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # Tirzah-like one overlap, many conflicts
    ])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
    )

    assert result.compatible_mass[0] > 0
    assert result.conflict_mass[0] > result.compatible_mass[0]
    assert result.confidence[0] < 0.5


def test_missing_candidate_raw_tags_is_uncertain_not_bad():
    vocab = ["indie pop", "punk"]
    seed_raw = np.array([1.0, 1.0])
    candidates_raw = np.array([[0.0, 0.0]])

    result = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=candidates_raw,
        genre_vocab=vocab,
    )

    assert result.compatible_mass[0] == 0
    assert result.conflict_mass[0] == 0
    assert result.confidence[0] == 1.0
    assert result.missing_or_sparse[0]
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_genre_compatibility.py -q --basetemp .pytest-tmp-genre-compat -o cache_dir=.pytest-tmp-cache-genre-compat
```

Expected: import failure for `src.playlist.genre_compatibility`.

- [ ] **Step 3: Implement the pure scoring helper**

Create `src/playlist/genre_compatibility.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GenreCompatibilityResult:
    compatible_mass: np.ndarray
    conflict_mass: np.ndarray
    neutral_mass: np.ndarray
    confidence: np.ndarray
    penalty: np.ndarray
    missing_or_sparse: np.ndarray


def _idf_weights(candidate_raw: np.ndarray) -> np.ndarray:
    presence = np.count_nonzero(candidate_raw > 0, axis=0).astype(float)
    n = max(1, int(candidate_raw.shape[0]))
    idf = np.log((1.0 + n) / (1.0 + presence)) + 1.0
    max_val = float(np.max(idf)) if idf.size else 1.0
    return idf / max(max_val, 1e-12)


def _affinity_matrix(
    genre_vocab: Sequence[str],
    genre_affinity: np.ndarray | None,
) -> np.ndarray:
    n = len(genre_vocab)
    if genre_affinity is not None and tuple(genre_affinity.shape) == (n, n):
        return np.asarray(genre_affinity, dtype=float)
    return np.eye(n, dtype=float)


def compute_raw_genre_compatibility(
    *,
    seed_raw: np.ndarray,
    candidate_raw: np.ndarray,
    genre_vocab: Sequence[str],
    genre_affinity: np.ndarray | None = None,
    compatible_threshold: float = 0.35,
    conflict_threshold: float = 0.15,
    penalty_strength: float = 1.0,
) -> GenreCompatibilityResult:
    seed = np.asarray(seed_raw, dtype=float).reshape(-1)
    candidates = np.asarray(candidate_raw, dtype=float)
    if candidates.ndim != 2:
        raise ValueError("candidate_raw must be a 2D matrix")
    if candidates.shape[1] != seed.shape[0]:
        raise ValueError("seed_raw and candidate_raw must have the same genre dimension")

    seed_active = seed > 0
    candidate_active = candidates > 0
    missing = np.count_nonzero(candidate_active, axis=1) == 0
    if not bool(np.any(seed_active)):
        zeros = np.zeros(candidates.shape[0], dtype=float)
        return GenreCompatibilityResult(
            compatible_mass=zeros,
            conflict_mass=zeros,
            neutral_mass=zeros,
            confidence=np.ones(candidates.shape[0], dtype=float),
            penalty=zeros,
            missing_or_sparse=np.ones(candidates.shape[0], dtype=bool),
        )

    affinity = _affinity_matrix(genre_vocab, genre_affinity)
    max_to_seed = np.max(affinity[:, seed_active], axis=1)
    weights = _idf_weights(candidates)

    compatible_tag = max_to_seed >= float(compatible_threshold)
    conflict_tag = max_to_seed <= float(conflict_threshold)
    neutral_tag = ~(compatible_tag | conflict_tag)

    weighted_active = candidate_active.astype(float) * weights.reshape(1, -1)
    compatible_mass = np.sum(weighted_active[:, compatible_tag], axis=1)
    conflict_mass = np.sum(weighted_active[:, conflict_tag], axis=1)
    neutral_mass = np.sum(weighted_active[:, neutral_tag], axis=1)

    denom = compatible_mass + conflict_mass
    confidence = np.where(denom > 1e-12, compatible_mass / denom, 1.0)
    penalty = float(max(0.0, penalty_strength)) * np.where(
        denom > 1e-12,
        conflict_mass / denom,
        0.0,
    )

    return GenreCompatibilityResult(
        compatible_mass=compatible_mass,
        conflict_mass=conflict_mass,
        neutral_mass=neutral_mass,
        confidence=np.clip(confidence, 0.0, 1.0),
        penalty=np.clip(penalty, 0.0, 1.0),
        missing_or_sparse=missing,
    )
```

- [ ] **Step 4: Run the tests and verify pass**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_genre_compatibility.py -q --basetemp .pytest-tmp-genre-compat -o cache_dir=.pytest-tmp-cache-genre-compat
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/genre_compatibility.py tests/unit/test_genre_compatibility.py
git commit -m "feat: add raw genre compatibility scoring"
```

---

### Task 2: Wire Genre Conflict Into Candidate Admission

**Files:**
- Modify: `src/playlist/config.py`
- Modify: `src/playlist/candidate_pool.py`
- Test: `tests/test_candidate_filters.py`

- [ ] **Step 1: Add failing candidate-pool tests**

Append to `tests/test_candidate_filters.py`:

```python
def test_genre_conflict_rejects_one_overlap_with_many_conflicts():
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    artist_keys = np.array(["seed", "sparse_correct", "conflict_heavy"])
    genre_vocab = ["indie pop", "punk", "rnb", "house", "soul", "funk"]
    X_genre_raw = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1],
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        genre_conflict_enabled=True,
        genre_conflict_min_confidence=0.50,
        genre_conflict_penalty_strength=0.0,
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=embedding,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        min_genre_similarity=0.1,
        genre_method="ensemble",
        genre_vocab=genre_vocab,
        broad_filters=(),
        mode="dynamic",
    )

    admitted_artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "sparse_correct" in admitted_artists
    assert "conflict_heavy" not in admitted_artists
    assert result.stats["genre_conflict_rejected"] == 1


def test_genre_conflict_penalty_can_demote_without_rejecting():
    embedding = np.array([
        [1.0, 0.0],
        [0.8, 0.6],
        [1.0, 0.0],
    ])
    artist_keys = np.array(["seed", "sparse_correct", "conflict_heavy"])
    genre_vocab = ["indie pop", "punk", "rnb", "house", "soul", "funk"]
    X_genre_raw = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1],
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        genre_conflict_enabled=True,
        genre_conflict_min_confidence=None,
        genre_conflict_penalty_strength=0.6,
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=embedding,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        min_genre_similarity=0.1,
        genre_method="ensemble",
        genre_vocab=genre_vocab,
        broad_filters=(),
        mode="dynamic",
    )

    admitted_artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert admitted_artists.index("sparse_correct") < admitted_artists.index("conflict_heavy")
    assert result.stats["genre_conflict_penalty_applied"] == 1
```

- [ ] **Step 2: Run tests and verify they fail on unknown config fields**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_candidate_filters.py -q --basetemp .pytest-tmp-candidate-conflict -o cache_dir=.pytest-tmp-cache-candidate-conflict
```

Expected: `CandidatePoolConfig.__init__()` rejects the new fields.

- [ ] **Step 3: Add config fields**

In `src/playlist/config.py`, extend `CandidatePoolConfig`:

```python
    genre_conflict_enabled: bool = False
    genre_conflict_min_confidence: Optional[float] = None
    genre_conflict_penalty_strength: float = 0.0
    genre_conflict_compatible_threshold: float = 0.35
    genre_conflict_conflict_threshold: float = 0.15
```

In `default_ds_config()`, read these from `candidate_pool`:

```python
        genre_conflict_enabled=bool(candidate_pool.get("genre_conflict_enabled", False)),
        genre_conflict_min_confidence=(
            None
            if candidate_pool.get("genre_conflict_min_confidence") is None
            else float(candidate_pool.get("genre_conflict_min_confidence"))
        ),
        genre_conflict_penalty_strength=float(
            candidate_pool.get("genre_conflict_penalty_strength", 0.0)
        ),
        genre_conflict_compatible_threshold=float(
            candidate_pool.get("genre_conflict_compatible_threshold", 0.35)
        ),
        genre_conflict_conflict_threshold=float(
            candidate_pool.get("genre_conflict_conflict_threshold", 0.15)
        ),
```

- [ ] **Step 4: Apply conflict scoring in `build_candidate_pool()`**

In `src/playlist/candidate_pool.py`, import:

```python
from src.playlist.genre_compatibility import compute_raw_genre_compatibility
```

After `genre_sim_all` is computed and after the raw overlap guard has zeroed incompatible smoothed matches, compute conflict scores:

```python
    genre_conflict = None
    genre_conflict_rejected = 0
    genre_conflict_penalty_applied = 0
    if (
        bool(getattr(cfg, "genre_conflict_enabled", False))
        and genre_raw_matrix is not None
        and genre_raw_matrix.shape[0] == len(seed_sim_all)
        and genre_raw_matrix.shape[1] > 0
        and genre_vocab
    ):
        genre_conflict = compute_raw_genre_compatibility(
            seed_raw=genre_raw_matrix[seed_idx],
            candidate_raw=genre_raw_matrix,
            genre_vocab=genre_vocab,
            compatible_threshold=float(getattr(cfg, "genre_conflict_compatible_threshold", 0.35)),
            conflict_threshold=float(getattr(cfg, "genre_conflict_conflict_threshold", 0.15)),
            penalty_strength=float(getattr(cfg, "genre_conflict_penalty_strength", 0.0)),
        )
        if float(getattr(cfg, "genre_conflict_penalty_strength", 0.0)) > 0:
            penalty = np.asarray(genre_conflict.penalty, dtype=float)
            apply_mask = (~seed_mask) & (penalty > 0)
            seed_sim_all = seed_sim_all - penalty
            genre_conflict_penalty_applied = int(np.count_nonzero(apply_mask))
```

Then, after the existing genre hard gate, add the confidence gate:

```python
        min_confidence = getattr(cfg, "genre_conflict_min_confidence", None)
        if genre_conflict is not None and min_confidence is not None and mode in ("dynamic", "narrow"):
            eligible_before_conflict = len(eligible)
            confidence = np.asarray(genre_conflict.confidence, dtype=float)
            missing = np.asarray(genre_conflict.missing_or_sparse, dtype=bool)
            eligible = [
                i for i in eligible
                if missing[i] or confidence[i] >= float(min_confidence)
            ]
            genre_conflict_rejected = eligible_before_conflict - len(eligible)
```

Add stats:

```python
        "genre_conflict_rejected": int(genre_conflict_rejected),
        "genre_conflict_penalty_applied": int(genre_conflict_penalty_applied),
```

Add params:

```python
    if bool(getattr(cfg, "genre_conflict_enabled", False)):
        params_effective["genre_conflict_enabled"] = True
        params_effective["genre_conflict_min_confidence"] = getattr(cfg, "genre_conflict_min_confidence", None)
        params_effective["genre_conflict_penalty_strength"] = float(getattr(cfg, "genre_conflict_penalty_strength", 0.0))
```

- [ ] **Step 5: Run candidate tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_candidate_filters.py -q --basetemp .pytest-tmp-candidate-conflict -o cache_dir=.pytest-tmp-cache-candidate-conflict
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/config.py src/playlist/candidate_pool.py tests/test_candidate_filters.py
git commit -m "feat: apply raw genre conflict in candidate pool"
```

---

### Task 3: Keep One Each Fallback From Relaxing Conflict Rules

**Files:**
- Modify: `tests/test_candidate_filters.py`
- Modify: `src/playlist/pipeline/core.py` only if the test reveals replacement drops config fields.

- [ ] **Step 1: Extend the existing One Each fallback test**

In `test_one_each_retries_with_relaxed_candidate_gate_when_pier_bridge_infeasible`, change the fake pool assertion block to include:

```python
    assert pool_calls[1][0].genre_conflict_enabled is pool_calls[0][0].genre_conflict_enabled
    assert pool_calls[1][0].genre_conflict_min_confidence == pool_calls[0][0].genre_conflict_min_confidence
    assert pool_calls[1][0].genre_conflict_penalty_strength == pool_calls[0][0].genre_conflict_penalty_strength
```

Pass conflict config through the `overrides` used by that test:

```python
        overrides={
            "candidate_pool": {
                "genre_conflict_enabled": True,
                "genre_conflict_min_confidence": 0.5,
                "genre_conflict_penalty_strength": 0.4,
            },
            "pier_bridge": {"max_non_seed_tracks_per_artist": 1},
        },
```

- [ ] **Step 2: Run the test**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_candidate_filters.py::test_one_each_retries_with_relaxed_candidate_gate_when_pier_bridge_infeasible -q --basetemp .pytest-tmp-one-each-conflict -o cache_dir=.pytest-tmp-cache-one-each-conflict
```

Expected: pass if `dataclasses.replace()` preserves fields; if it fails, fix `_relaxed_one_each_candidate_attempts()` to replace only `similarity_floor` and `min_sonic_similarity`, preserving all other fields.

- [ ] **Step 3: Commit if code or tests changed**

```bash
git add tests/test_candidate_filters.py src/playlist/pipeline/core.py
git commit -m "test: preserve genre conflict during one each relaxation"
```

---

### Task 4: Add Artist-Style Genre Neighbor Union Pool

**Files:**
- Modify: `src/playlist/artist_style.py`
- Test: `tests/test_artist_style.py`

- [ ] **Step 1: Add failing artist-style union tests**

Append to `tests/test_artist_style.py`:

```python
def test_genre_neighbor_pool_recovers_low_sonic_genre_match():
    from src.playlist.artist_style import build_genre_neighbor_candidate_pool

    X_sonic = np.array([
        [1.0, 0.0],  # seed artist pier
        [0.0, 1.0],  # obvious genre neighbor, not sonic-near
        [0.0, 1.0],  # off-axis one-overlap conflict
    ])
    bundle = DummyBundle(
        X_sonic=X_sonic,
        artist_keys=np.array(["tiger trap", "obvious", "off_axis"]),
        track_ids=np.array(["seed", "obvious_track", "off_axis_track"]),
        track_artists=np.array(["Tiger Trap", "Obvious Band", "Off Axis"]),
        track_titles=np.array(["Seed", "Obvious", "Off Axis"]),
    )
    bundle.X_genre_raw = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1],
    ], dtype=float)
    bundle.X_genre_smoothed = bundle.X_genre_raw.copy()
    bundle.genre_vocab = np.array(["indie pop", "punk", "rnb", "house", "soul"])

    pool, diag = build_genre_neighbor_candidate_pool(
        bundle=bundle,
        seed_artist_indices=[0],
        artist_key="tiger trap",
        max_candidates=10,
        min_genre_similarity=0.1,
        min_confidence=0.5,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
        broad_filters=(),
    )

    assert "obvious_track" in pool
    assert "off_axis_track" not in pool
    assert diag["genre_neighbor_conflict_rejected"] == 1
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_artist_style.py::test_genre_neighbor_pool_recovers_low_sonic_genre_match -q --basetemp .pytest-tmp-artist-genre-union -o cache_dir=.pytest-tmp-cache-artist-genre-union
```

Expected: import failure for `build_genre_neighbor_candidate_pool`.

- [ ] **Step 3: Implement `build_genre_neighbor_candidate_pool()`**

In `src/playlist/artist_style.py`, import:

```python
from src.playlist.genre_compatibility import compute_raw_genre_compatibility
from src.playlist.candidate_pool import _compute_genre_similarity
```

Add the function near `build_balanced_candidate_pool()`:

```python
def build_genre_neighbor_candidate_pool(
    *,
    bundle,
    seed_artist_indices: List[int],
    artist_key: str,
    max_candidates: int,
    min_genre_similarity: float,
    min_confidence: Optional[float],
    compatible_threshold: float,
    conflict_threshold: float,
    broad_filters: tuple[str, ...] = (),
) -> tuple[List[str], Dict[str, int]]:
    if not seed_artist_indices:
        return [], {"genre_neighbor_candidates": 0}
    X_raw = getattr(bundle, "X_genre_raw", None)
    X_smooth = getattr(bundle, "X_genre_smoothed", None)
    vocab_raw = getattr(bundle, "genre_vocab", None)
    if X_raw is None or X_smooth is None or vocab_raw is None:
        return [], {"genre_neighbor_candidates": 0, "genre_neighbor_missing_matrix": 1}

    vocab = [str(g) for g in list(vocab_raw)]
    raw = np.asarray(X_raw, dtype=float)
    smooth = np.asarray(X_smooth, dtype=float)
    if broad_filters:
        broad = {str(g).lower() for g in broad_filters}
        mask = np.array([str(g).lower() not in broad for g in vocab], dtype=bool)
        if mask.shape[0] == raw.shape[1]:
            raw = raw[:, mask]
            smooth = smooth[:, mask]
            vocab = [g for g, keep in zip(vocab, mask) if keep]

    seed_raw = np.max(raw[seed_artist_indices], axis=0)
    seed_smooth = np.max(smooth[seed_artist_indices], axis=0)
    genre_sim = _compute_genre_similarity(seed_smooth, smooth, method="ensemble")
    compatibility = compute_raw_genre_compatibility(
        seed_raw=seed_raw,
        candidate_raw=raw,
        genre_vocab=vocab,
        compatible_threshold=compatible_threshold,
        conflict_threshold=conflict_threshold,
    )

    rejected_seed_artist = 0
    rejected_similarity = 0
    rejected_conflict = 0
    candidates: list[tuple[float, int]] = []
    for i in range(raw.shape[0]):
        key = normalize_artist_key(str(bundle.artist_keys[i])) if bundle.artist_keys is not None else ""
        if key == artist_key:
            rejected_seed_artist += 1
            continue
        if float(genre_sim[i]) < float(min_genre_similarity):
            rejected_similarity += 1
            continue
        if min_confidence is not None and not bool(compatibility.missing_or_sparse[i]):
            if float(compatibility.confidence[i]) < float(min_confidence):
                rejected_conflict += 1
                continue
        score = float(genre_sim[i]) * float(compatibility.confidence[i])
        candidates.append((score, i))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [int(i) for _score, i in candidates[: max(0, int(max_candidates))]]
    return [str(bundle.track_ids[i]) for i in selected], {
        "genre_neighbor_candidates": int(len(selected)),
        "genre_neighbor_rejected_seed_artist": int(rejected_seed_artist),
        "genre_neighbor_rejected_similarity": int(rejected_similarity),
        "genre_neighbor_conflict_rejected": int(rejected_conflict),
    }
```

- [ ] **Step 4: Run artist-style tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_artist_style.py -q --basetemp .pytest-tmp-artist-genre-union -o cache_dir=.pytest-tmp-cache-artist-genre-union
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/test_artist_style.py
git commit -m "feat: add artist style genre neighbor pool"
```

---

### Task 5: Wire Genre Union Pool Into Artist Mode

**Files:**
- Modify: `src/playlist/artist_style.py`
- Modify: `src/playlist_generator.py`
- Modify: `config.yaml`
- Test: existing tests plus a new focused config/unit test if needed.

- [ ] **Step 1: Add config fields to `ArtistStyleConfig`**

In `src/playlist/artist_style.py`, extend `ArtistStyleConfig`:

```python
    genre_neighbor_pool_enabled: bool = False
    genre_neighbor_pool_size: int = 400
    genre_neighbor_min_similarity: float = 0.25
    genre_neighbor_min_confidence: Optional[float] = 0.50
    genre_neighbor_compatible_threshold: float = 0.35
    genre_neighbor_conflict_threshold: float = 0.15
```

- [ ] **Step 2: Parse config in both artist-mode blocks**

In both artist-style config constructions in `src/playlist_generator.py`, add:

```python
            genre_neighbor_pool_enabled=bool(style_cfg_raw.get("genre_neighbor_pool_enabled", False)),
            genre_neighbor_pool_size=int(style_cfg_raw.get("genre_neighbor_pool_size", 400)),
            genre_neighbor_min_similarity=float(style_cfg_raw.get("genre_neighbor_min_similarity", 0.25)),
            genre_neighbor_min_confidence=(
                None
                if style_cfg_raw.get("genre_neighbor_min_confidence", 0.50) is None
                else float(style_cfg_raw.get("genre_neighbor_min_confidence", 0.50))
            ),
            genre_neighbor_compatible_threshold=float(style_cfg_raw.get("genre_neighbor_compatible_threshold", 0.35)),
            genre_neighbor_conflict_threshold=float(style_cfg_raw.get("genre_neighbor_conflict_threshold", 0.15)),
```

- [ ] **Step 3: Union genre neighbors with existing sonic style pool**

In both artist-mode style blocks, immediately after `external_pool = build_balanced_candidate_pool(...)`, add:

```python
                genre_neighbor_pool = []
                genre_neighbor_diag = {}
                if bool(style_cfg.genre_neighbor_pool_enabled):
                    genre_neighbor_pool, genre_neighbor_diag = build_genre_neighbor_candidate_pool(
                        bundle=bundle,
                        seed_artist_indices=_artist_indices_in_bundle(
                            bundle,
                            artist_name,
                            include_collaborations=include_collaborations,
                        ),
                        artist_key=artist_key_norm,
                        max_candidates=int(style_cfg.genre_neighbor_pool_size),
                        min_genre_similarity=float(style_cfg.genre_neighbor_min_similarity),
                        min_confidence=style_cfg.genre_neighbor_min_confidence,
                        compatible_threshold=float(style_cfg.genre_neighbor_compatible_threshold),
                        conflict_threshold=float(style_cfg.genre_neighbor_conflict_threshold),
                        broad_filters=tuple(
                            str(b).lower()
                            for b in ds_cfg.get("candidate_pool", {}).get("broad_filters", ())
                        ),
                    )
                    logger.info(
                        "Artist style genre-neighbor pool: selected=%d diag=%s",
                        len(genre_neighbor_pool),
                        genre_neighbor_diag,
                    )
```

Then change:

```python
                style_allowed_track_ids = list(dict.fromkeys(pier_ids + external_pool + list(internal_connector_ids or [])))
```

to:

```python
                style_allowed_track_ids = list(dict.fromkeys(
                    pier_ids + external_pool + genre_neighbor_pool + list(internal_connector_ids or [])
                ))
```

Add to `style_summary`:

```python
                    "genre_neighbor_pool_count": int(len(genre_neighbor_pool)),
                    "genre_neighbor_pool_diagnostics": dict(genre_neighbor_diag),
```

- [ ] **Step 4: Enable conservative defaults in `config.yaml`**

Under `playlists.ds_pipeline.artist_style`, add:

```yaml
      genre_neighbor_pool_enabled: true
      genre_neighbor_pool_size: 500
      genre_neighbor_min_similarity: 0.25
      genre_neighbor_min_confidence: 0.50
      genre_neighbor_compatible_threshold: 0.35
      genre_neighbor_conflict_threshold: 0.15
```

Under `playlists.ds_pipeline.candidate_pool`, add or merge:

```yaml
      genre_conflict_enabled: true
      genre_conflict_min_confidence: 0.50
      genre_conflict_penalty_strength: 0.30
      genre_conflict_compatible_threshold: 0.35
      genre_conflict_conflict_threshold: 0.15
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_artist_style.py tests/test_candidate_filters.py -q --basetemp .pytest-tmp-artist-mode-conflict -o cache_dir=.pytest-tmp-cache-artist-mode-conflict
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/artist_style.py src/playlist_generator.py src/playlist/config.py config.yaml tests/test_artist_style.py tests/test_candidate_filters.py
git commit -m "feat: expand artist mode with genre-aware pool"
```

---

### Task 6: Add Optional Watched-Artist Diagnostics

**Files:**
- Modify: `src/playlist/candidate_pool.py`
- Modify: `src/playlist/pipeline/core.py`
- Test: `tests/test_candidate_filters.py`

- [ ] **Step 1: Add a focused diagnostics test**

Append to `tests/test_candidate_filters.py`:

```python
def test_candidate_pool_reports_genre_conflict_stats():
    embedding = np.array([[1.0, 0.0], [1.0, 0.0]])
    artist_keys = np.array(["seed", "conflict_heavy"])
    track_ids = np.array(["seed_id", "conflict_id"])
    genre_vocab = ["indie pop", "rnb", "house"]
    X_genre_raw = np.array([[1, 0, 0], [1, 1, 1]], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        genre_conflict_enabled=True,
        genre_conflict_min_confidence=0.50,
        genre_conflict_penalty_strength=0.0,
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=cfg,
        random_seed=0,
        X_sonic=embedding,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        min_genre_similarity=0.1,
        genre_method="ensemble",
        genre_vocab=genre_vocab,
        broad_filters=(),
        mode="dynamic",
    )

    assert result.stats["genre_conflict_rejected"] == 1
    assert result.stats["genre_conflict_penalty_applied"] == 0
    assert "genre_conflict_enabled" in result.params_effective
```

- [ ] **Step 2: Implement stats-only diagnostics first**

Ensure `candidate_pool.py` always includes these stats when conflict is enabled:

```python
    if bool(getattr(cfg, "genre_conflict_enabled", False)):
        stats["genre_conflict_rejected"] = int(genre_conflict_rejected)
        stats["genre_conflict_penalty_applied"] = int(genre_conflict_penalty_applied)
```

Do not add bulky per-track maps by default.

- [ ] **Step 3: Add watched-artist verbose diagnostics behind env/config**

Use environment variable `PLAYLIST_WATCHED_ARTISTS` as a low-risk first pass. Format: comma-separated artist keys/names. In `pipeline/core.py`, after pool construction and before pier-bridge, if env var is set, log rows for matching artists from the restricted bundle:

```python
    watched_raw = os.environ.get("PLAYLIST_WATCHED_ARTISTS", "")
    watched = {w.strip().casefold() for w in watched_raw.split(",") if w.strip()}
```

If implementing this step requires adding `os` import, add it at the top. The row should include only fields already available from `pool.stats`, `pool.pool_indices`, `pool.eligible_indices`, `bundle.artist_keys`, and `bundle.track_ids`. Do not recompute heavy matrices in the hot path.

- [ ] **Step 4: Run diagnostics tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/test_candidate_filters.py::test_candidate_pool_reports_genre_conflict_stats -q --basetemp .pytest-tmp-conflict-diag -o cache_dir=.pytest-tmp-cache-conflict-diag
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py tests/test_candidate_filters.py
git commit -m "feat: add genre conflict diagnostics"
```

---

### Task 7: Run Read-Only Tiger Trap Verification

**Files:**
- No source changes unless verification reveals a bug.
- Do not modify `data/metadata.db` or music files.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest tests/unit/test_genre_compatibility.py tests/test_candidate_filters.py tests/test_artist_style.py -q --basetemp .pytest-tmp-focused-conflict -o cache_dir=.pytest-tmp-cache-focused-conflict
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
C:\Windows\py.exe -3.13 -m pytest -q --basetemp .pytest-tmp-full-conflict -o cache_dir=.pytest-tmp-cache-full-conflict
```

Expected: pass.

- [ ] **Step 3: Run a dry Tiger Trap artist-mode audit**

Use the project’s existing dry-run entrypoint if available in this branch. Run with watched diagnostics enabled:

```bash
set PLAYLIST_WATCHED_ARTISTS=The Pains of Being Pure at Heart,Seapony,The Umbrellas,Chime School,Ducks Ltd.,Alvvays,Tirzah,Suicide
C:\Windows\py.exe -3.13 main_app.py --artist "Tiger Trap" --tracks 50 --dry-run --verbose
```

Expected:
- `Chime School` still reports absent from artifact until artifact rebuild occurs.
- `Seapony` and/or `The Umbrellas` appear in the allowed or eligible pool if genre metadata is present.
- `Tirzah - Sleeping` and `Suicide - Johnny` either fail conflict confidence or rank lower due to conflict penalty.
- One Each fallback, if triggered, does not lower `genre_conflict_min_confidence`.

- [ ] **Step 4: Commit verification-only test updates if any**

```bash
git status --short
git add tests src config.yaml
git commit -m "test: verify artist mode genre conflict behavior"
```

Skip this commit if there were no new changes.

---

## Self-Review

- Spec coverage: The plan covers sparse genre handling, conflict penalties, artist-mode genre pool recovery, One Each fallback behavior, and diagnostics.
- Placeholder scan: No task contains `TBD`, `TODO`, or “write tests” without concrete test code.
- Type consistency: New config fields are consistently named with `genre_conflict_*` for candidate admission and `genre_neighbor_*` for artist-style allowed-pool expansion.
- Backward compatibility: Conflict behavior defaults off in `CandidatePoolConfig`; `config.yaml` enables it for this local product behavior. Missing candidate raw genres are treated as uncertain, not bad.
