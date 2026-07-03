# Genre-Tag Steering (Artist Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After picking an artist in the web GUI, show the artist's published genre tags as chips; the user selects up to 3 and the playlist softly leans toward those flavors at the candidate pool and the artist-pier selection (stage 1 of the approved spec).

**Architecture:** One unit-norm dense target vector is resolved from the selected tag names (`genre_emb` rows from the artifact's genre sidecar). Two soft levers consume it: (1) the candidate-pool dense genre-admission centroid is blended `(1−λ)·seed_centroid + λ·target`; (2) artist pier (medoid) scoring gains a `+ weight · cos(track_genre_dense, target)` term. Tags travel exclusively on the config-override channel: `GenerateRequestBody.steering_tags` → `UIStateModel` → `policy.derive_runtime_config` → `playlists.ds_pipeline.pier_bridge.tag_steering_tags` → merged config → both consumers. No hard gates anywhere; zero tags = byte-identical behavior to today.

**Tech Stack:** Python 3.11 (FastAPI, numpy, sqlite3, pytest), React + TypeScript + Tailwind (web/), existing gui_fidelity test harness.

**Spec:** `docs/superpowers/specs/2026-07-02-tag-steering-design.md` (approved 2026-07-02). Two wiring-reality deviations already synced into the spec: config keys live at `playlists.ds_pipeline.pier_bridge.tag_steering_*` (rides the existing `build_ds_overrides` wholesale pass-through of the `pier_bridge` dict — `src/playlist_generator.py:58`, consumed as `pb_overrides` at `src/playlist/pipeline/core.py:304`), and the endpoint is `GET /api/genres/for_artist` (matches the existing `/api/genres/for_album` naming).

## Global Constraints

- **Session placement:** execute from a session launched WITH cwd = a dedicated worktree on its own branch (subagents and hooks anchor to the LAUNCH directory — entering a worktree mid-session leaks commits into the main checkout). The worktree setup links `data/`; copy `config.yaml` from the main checkout by hand (gitignored, not carried).
- **Pytest:** always `python -m pytest -q -m "not slow" <path>` with the Bash tool's timeout parameter. NEVER pipe pytest through `tail`/`head` (hangs sessions).
- **`data/metadata.db` is production — read-only.** All DB tests use a `tmp_path` sqlite file built by the test. Never write/migrate the real DB.
- **Genre reads: authority only.** `release_effective_genres` via `src/genre/authority.py`. Raw `artist_genres`/`track_genres` are forbidden for new consumers.
- **No hard gates.** Every steering effect is a soft re-ranking or blended floor input. A configured tag that can't act must WARN loudly, never silently no-op.
- **Zero-tag behavior must be byte-identical to today** — every new code path is behind `if steering_target is not None` / non-empty tag list.
- **Commit per task**, message style `feat(tag-steering): ...` / `test(tag-steering): ...`. Stage explicit paths only (never `git add -A`).
- **Front-end changes require `npm --prefix web run build`** (stale `web/dist` trap) and the worker must be RESTARTED before any live verification (`@lru_cache` + process-state trap).

## File Structure

| File | Role |
|---|---|
| `src/genre/authority.py` | +`ArtistGenreTag`, +`resolved_genres_for_artist()` (authority aggregation) |
| `src/playlist_web/app.py` | +`GET /api/genres/for_artist` endpoint |
| `src/playlist/tag_steering.py` | **NEW** — tag names → unit-norm dense target (single resolver, both levers) |
| `src/playlist/candidate_pool.py` | pool lever: `steering_target`/`steering_blend` params, centroid blend, affinity log |
| `src/playlist/pipeline/core.py` | resolve target from `pb_overrides` + bundle; pass into `build_candidate_pool` |
| `src/playlist/artist_style.py` | pier lever: `_medoids_for_cluster` tag term; `cluster_artist_tracks` threading; `ArtistStyleConfig.medoid_tag_weight` |
| `src/playlist_generator.py` | artist path: resolve target once, thread into `cluster_artist_tracks`, per-pier affinity log, `medoid_tag_weight` in the `ArtistStyleConfig` builder |
| `src/playlist_gui/ui_state.py` | +`UIStateModel.steering_tags` |
| `src/playlist_gui/policy.py` | emit `playlists.ds_pipeline.pier_bridge.tag_steering_tags` override |
| `src/playlist_web/schemas.py` | +`GenerateRequestBody.steering_tags` |
| `web/src/lib/api.ts`, `web/src/lib/types.ts` (or wherever `GenerateRequestBody` TS type lives — grep `cohesion_mode` under `web/src/lib/`), `web/src/components/GenerateControls.tsx` | chips UI |
| `config.yaml` + `config.example.yaml` | `tag_steering_pool_blend: 0.5`, `tag_steering_pier_weight: 0.3` under `playlists.ds_pipeline.pier_bridge` |
| `docs/PLAYLIST_ORDERING_TUNING.md`, `.claude/skills/genre-data-authority/SKILL.md` | tuning recipe + authority-recipe row |
| Tests | `tests/unit/test_authority_artist_genres.py`, `tests/unit/test_tag_steering.py`, `tests/unit/test_tag_steering_pool_lever.py`, `tests/unit/test_tag_steering_pier_lever.py`, additions to `tests/test_gui_fidelity.py`, `tests/integration/test_tag_steering_behavioral.py` |

---

### Task 1: Authority — `resolved_genres_for_artist`

**Files:**
- Modify: `src/genre/authority.py` (append after `display_genre_names_for_album`, line 161)
- Test: `tests/unit/test_authority_artist_genres.py` (new)

**Interfaces:**
- Consumes: existing `release_effective_genres` / `genre_graph_canonical_genres` / `tracks` tables.
- Produces: `ArtistGenreTag(genre_id: str, name: str, release_count: int, max_confidence: float)` and `resolved_genres_for_artist(conn: sqlite3.Connection, artist_name: str) -> list[ArtistGenreTag]` — Task 2's endpoint calls this exact signature.

- [ ] **Step 1: Write the failing test**

```python
"""Authority-side aggregation of an artist's published genres (tag-steering chips)."""
import sqlite3

import pytest

from src.genre.authority import ArtistGenreTag, resolved_genres_for_artist


@pytest.fixture()
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "meta.db")
    db.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (
            album_id TEXT, release_key TEXT, genre_id TEXT,
            assignment_layer TEXT, confidence REAL, source TEXT
        );
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        """
    )
    db.executemany(
        "INSERT INTO tracks VALUES (?, ?, ?)",
        [
            ("t1", "Herbie Hancock", "alb1"),
            ("t2", "Herbie Hancock", "alb1"),
            ("t3", "Herbie Hancock", "alb2"),
            ("t4", "Aretha Franklin", "alb3"),
            ("t5", "Herbie Hancock", None),  # albumless track must not crash
        ],
    )
    db.executemany(
        "INSERT INTO release_effective_genres VALUES (?, '', ?, ?, ?, 'graph')",
        [
            ("alb1", "g-jazzfunk", "observed_leaf", 0.9),
            ("alb2", "g-jazzfunk", "observed_leaf", 0.8),
            ("alb1", "g-postbop", "observed_leaf", 0.7),
            ("alb1", "g-jazz", "inferred_family", 0.95),   # hub family: excluded
            ("alb2", "g-fusion", "legacy", 0.6),            # legacy layer: included
            ("alb3", "g-soul", "observed_leaf", 0.9),       # other artist: excluded
        ],
    )
    db.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?, ?)",
        [("g-jazzfunk", "jazz-funk"), ("g-postbop", "post-bop"),
         ("g-jazz", "jazz"), ("g-fusion", "jazz fusion"), ("g-soul", "soul")],
    )
    db.commit()
    yield db
    db.close()


def test_aggregates_observed_leaf_and_legacy_across_releases(conn):
    tags = resolved_genres_for_artist(conn, "Herbie Hancock")
    names = [t.name for t in tags]
    assert names[0] == "jazz-funk"                       # 2 releases: strongest first
    assert set(names) == {"jazz-funk", "post-bop", "jazz fusion"}
    jf = tags[0]
    assert jf == ArtistGenreTag("g-jazzfunk", "jazz-funk", 2, 0.9)


def test_excludes_inferred_families_and_other_artists(conn):
    names = {t.name for t in resolved_genres_for_artist(conn, "Herbie Hancock")}
    assert "jazz" not in names   # inferred_family carries no steering signal
    assert "soul" not in names   # different artist


def test_artist_match_is_case_insensitive_exact(conn):
    assert resolved_genres_for_artist(conn, "  herbie hancock ")
    assert resolved_genres_for_artist(conn, "Herbie") == []  # no substring matching


def test_unknown_artist_and_missing_tables_return_empty(conn, tmp_path):
    assert resolved_genres_for_artist(conn, "Nobody") == []
    bare = sqlite3.connect(tmp_path / "bare.db")
    assert resolved_genres_for_artist(bare, "Herbie Hancock") == []
    bare.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_authority_artist_genres.py`
Expected: FAIL — `ImportError: cannot import name 'ArtistGenreTag'`

- [ ] **Step 3: Implement** (append to `src/genre/authority.py`)

```python
@dataclass(frozen=True)
class ArtistGenreTag:
    genre_id: str
    name: str
    release_count: int
    max_confidence: float


def resolved_genres_for_artist(
    conn: sqlite3.Connection, artist_name: str
) -> list[ArtistGenreTag]:
    """Published observed-leaf/legacy genres across an artist's releases.

    Feeds the tag-steering chips. The input comes from the artist autocomplete,
    which reads ``tracks.artist`` — so an exact case-insensitive match on the
    same column is the correct key (no substring matching). ``inferred_family``
    rows are excluded: hub families carry no steering signal (hub-saturation
    incident 2026-06-12). Ordered strongest-first by (release_count,
    max_confidence). Returns [] for unknown artists or when the authority
    tables are absent — callers render an empty chip row, they don't crash.
    """
    name = (artist_name or "").strip()
    if not name:
        return []
    try:
        rows = conn.execute(
            "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) AS display_name, "
            "       COUNT(DISTINCT reg.album_id) AS n_releases, "
            "       MAX(reg.confidence) AS max_conf "
            "FROM release_effective_genres reg "
            "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
            "WHERE reg.assignment_layer IN ('observed_leaf', 'legacy') "
            "  AND reg.album_id IN ("
            "      SELECT DISTINCT album_id FROM tracks "
            "      WHERE LOWER(TRIM(artist)) = LOWER(TRIM(?)) "
            "        AND album_id IS NOT NULL"
            "  ) "
            "GROUP BY reg.genre_id "
            "ORDER BY n_releases DESC, max_conf DESC, display_name ASC",
            (name,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [ArtistGenreTag(r[0], r[1], int(r[2]), float(r[3])) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_authority_artist_genres.py`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/genre/authority.py tests/unit/test_authority_artist_genres.py
git commit -m "feat(tag-steering): authority-side artist genre aggregation for steering chips"
```

---

### Task 2: Web endpoint `GET /api/genres/for_artist` + api.ts client

**Files:**
- Modify: `src/playlist_web/app.py` (insert directly after the `genres_for_album` endpoint, line 521)
- Modify: `web/src/lib/api.ts` (after `albumGenres`, line 99-102)

**Interfaces:**
- Consumes: Task 1's `resolved_genres_for_artist(conn, artist)`.
- Produces: `GET /api/genres/for_artist?artist=<name>` → `{"genres": [{"name": str, "release_count": int, "confidence": float}]}` (top 12); TS client `api.artistGenres(artist: string)` — Task 7 consumes both.

- [ ] **Step 1: Add the endpoint** (pattern copied from `genres_for_album` at `app.py:502-521` — same ro connection, same fail-soft shape)

```python
    @app.get("/api/genres/for_artist")
    async def genres_for_artist(artist: str = "") -> dict:
        """Published observed-leaf genres across an artist's releases (steering chips)."""
        if not artist.strip() or not DB_PATH.exists():
            return {"genres": []}
        from src.genre.authority import resolved_genres_for_artist
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                tags = resolved_genres_for_artist(conn, artist)
                return {"genres": [
                    {"name": t.name, "release_count": t.release_count,
                     "confidence": round(t.max_confidence, 3)}
                    for t in tags[:12]
                ]}
            finally:
                conn.close()
        except sqlite3.Error:
            return {"genres": []}
```

- [ ] **Step 2: Add the TS client method** (in the `api` object in `web/src/lib/api.ts`, mirroring `albumGenres`)

```ts
  async artistGenres(artist: string): Promise<{ genres: { name: string; release_count: number; confidence: number }[] }> {
    const params = new URLSearchParams({ artist });
    return jsonOrThrow(await fetch(`/api/genres/for_artist?${params}`));
  },
```

- [ ] **Step 3: Verify manually against the real DB** (endpoint test is covered live; the SQL itself is unit-tested in Task 1)

Run: `python -c "import sqlite3; from src.genre.authority import resolved_genres_for_artist as f; c = sqlite3.connect('file:data/metadata.db?mode=ro', uri=True); print(f(c, 'Herbie Hancock')[:5])"`
Expected: a non-empty list of `ArtistGenreTag` rows with plausible jazz tags. If empty, STOP — check the artist exists in `tracks` and has published releases before proceeding.

- [ ] **Step 4: Commit**

```bash
git add src/playlist_web/app.py web/src/lib/api.ts
git commit -m "feat(tag-steering): /api/genres/for_artist endpoint + web client"
```

---

### Task 3: Target resolver — `src/playlist/tag_steering.py`

**Files:**
- Create: `src/playlist/tag_steering.py`
- Test: `tests/unit/test_tag_steering.py` (new)

**Interfaces:**
- Consumes: artifact fields `genre_vocab: np.ndarray (G,)`, `genre_emb: Optional[np.ndarray] (V, dim)` (`src/features/artifacts.py:58,69`).
- Produces: `resolve_tag_steering_target(tags, *, genre_vocab, genre_emb) -> tuple[Optional[np.ndarray], list[str], list[str]]` returning `(unit-norm target | None, mapped_tags, unmapped_tags)` — Tasks 4 and 5 call this exact signature.

- [ ] **Step 1: Write the failing test**

```python
"""Tag-name -> dense steering target resolution."""
import logging

import numpy as np

from src.playlist.tag_steering import resolve_tag_steering_target

VOCAB = ["jazz-funk", "post-bop", "soul"]
EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_maps_case_insensitively_and_normalizes():
    target, mapped, unmapped = resolve_tag_steering_target(
        ["Jazz-Funk", " post-bop "], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert mapped == ["Jazz-Funk", "post-bop"]  # resolver stores stripped inputs
    assert unmapped == []
    np.testing.assert_allclose(target, np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5]))
    assert abs(np.linalg.norm(target) - 1.0) < 1e-9


def test_unmapped_tags_warn_and_are_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk", "vaporwave"], genre_vocab=VOCAB, genre_emb=EMB
        )
    assert unmapped == ["vaporwave"]
    assert target is not None
    assert any("not in the artifact genre vocabulary" in r.message for r in caplog.records)


def test_missing_genre_emb_warns_and_disables(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk"], genre_vocab=VOCAB, genre_emb=None
        )
    assert target is None and mapped == [] and unmapped == ["jazz-funk"]
    assert any("genre_emb" in r.message for r in caplog.records)


def test_no_tags_is_silent_none():
    target, mapped, unmapped = resolve_tag_steering_target(
        [], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert target is None and mapped == [] and unmapped == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_tag_steering.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.playlist.tag_steering'`

- [ ] **Step 3: Implement `src/playlist/tag_steering.py`**

```python
"""User-selected genre-tag steering: tag names -> dense target vector.

Single resolver shared by the candidate-pool lever (pipeline/core.py) and the
artist pier lever (playlist_generator -> artist_style). Soft-bias only:
callers blend or re-rank with the target; nothing here gates or excludes.
A selected tag that cannot act WARNS loudly — never a silent no-op.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def resolve_tag_steering_target(
    tags: Sequence[str],
    *,
    genre_vocab: Sequence[str],
    genre_emb: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], list[str], list[str]]:
    """Map tag names to a unit-norm mean of their vocabulary embeddings.

    Returns ``(target | None, mapped_tags, unmapped_tags)``. Matching is
    case-insensitive on the artifact ``genre_vocab``. Returns ``None`` when
    nothing maps or the dense vocabulary embedding is absent.
    """
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return None, [], []
    if genre_emb is None:
        logger.warning(
            "Tag steering requested (%s) but the artifact's dense genre sidecar "
            "has no vocabulary embedding (genre_emb) — steering disabled for this run.",
            wanted,
        )
        return None, [], list(wanted)
    vocab_lower = {str(v).strip().lower(): i for i, v in enumerate(genre_vocab)}
    mapped: list[str] = []
    unmapped: list[str] = []
    rows: list[int] = []
    for tag in wanted:
        idx = vocab_lower.get(tag.lower())
        if idx is None or idx >= int(genre_emb.shape[0]):
            unmapped.append(tag)
        else:
            mapped.append(tag)
            rows.append(int(idx))
    if unmapped:
        logger.warning(
            "Tag steering: %d/%d selected tags not in the artifact genre vocabulary: %s",
            len(unmapped), len(wanted), unmapped,
        )
    if not rows:
        logger.warning("Tag steering: no selected tags mapped — steering disabled for this run.")
        return None, mapped, unmapped
    target = np.asarray(genre_emb, dtype=np.float64)[rows].mean(axis=0)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        logger.warning("Tag steering: degenerate zero-norm target — steering disabled for this run.")
        return None, mapped, unmapped
    target = target / norm
    logger.info("Tag steering target: tags=%s (mapped %d/%d)", mapped, len(rows), len(wanted))
    return target, mapped, unmapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/unit/test_tag_steering.py`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/playlist/tag_steering.py tests/unit/test_tag_steering.py
git commit -m "feat(tag-steering): tag-name -> unit-norm dense target resolver"
```

---

### Task 4: Pool lever — blended admission centroid

**Files:**
- Modify: `src/playlist/candidate_pool.py` (signature at `:518-561`; `_agg` resolution at `:777-779`; centroid branch at `:810-833`)
- Modify: `src/playlist/pipeline/core.py` (after the `_genre_admission_aggregate` block ending `:542`; `_build_pool` call at `:561-600`)
- Modify: `config.yaml` AND `config.example.yaml` (`playlists.ds_pipeline.pier_bridge` section)
- Test: `tests/unit/test_tag_steering_pool_lever.py` (new)

**Interfaces:**
- Consumes: Task 3's `resolve_tag_steering_target`.
- Produces: `build_candidate_pool(..., steering_target: Optional[np.ndarray] = None, steering_blend: float = 0.5)`; config keys `tag_steering_tags` (list, per-request via policy) and `tag_steering_pool_blend` (float, static) read from `pb_overrides` in core.py. Task 6's policy emission targets `tag_steering_tags`; Task 8 asserts the log lines added here.

- [ ] **Step 1: Write the failing test**

```python
"""Pool lever: steering target blends into the dense genre-admission centroid."""
import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _pool(steering_target=None, steering_blend=0.5):
    n = 5  # 0 = seed; 1,2 aligned with seed genre; 3,4 aligned with target genre
    embedding = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # sonically identical
    embedding += np.arange(n).reshape(-1, 1) * 1e-6         # break exact ties
    artist_keys = np.array([f"artist{i}" for i in range(n)])
    x_genre_dense = np.array([
        [1.0, 0.0],   # seed
        [1.0, 0.0],   # near-seed genre
        [0.9, 0.1],
        [0.0, 1.0],   # on-target genre
        [0.1, 0.9],
    ])
    x_genre_dense /= np.linalg.norm(x_genre_dense, axis=1, keepdims=True)
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=3,
        seed_artist_bonus=1,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
        title_exclusion_enabled=False,
    )
    return build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_dense=x_genre_dense,
        genre_admission_percentile=0.5,   # floor at the median non-seed sim
        mode="dynamic",
        steering_target=steering_target,
        steering_blend=steering_blend,
    )


def test_unsteered_pool_prefers_seed_genre_neighbors():
    res = _pool(steering_target=None)
    assert {1, 2} <= set(res.pool_indices.tolist())
    assert not {3, 4} <= set(res.pool_indices.tolist())


def test_full_blend_flips_admission_toward_target():
    res = _pool(steering_target=np.array([0.0, 1.0]), steering_blend=1.0)
    assert {3, 4} <= set(res.pool_indices.tolist())
    assert not {1, 2} <= set(res.pool_indices.tolist())


def test_zero_blend_equals_unsteered():
    base = _pool(steering_target=None)
    zero = _pool(steering_target=np.array([0.0, 1.0]), steering_blend=0.0)
    assert base.pool_indices.tolist() == zero.pool_indices.tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_tag_steering_pool_lever.py`
Expected: FAIL — `TypeError: build_candidate_pool() got an unexpected keyword argument 'steering_target'`

- [ ] **Step 3: Implement in `candidate_pool.py`**

3a. Add to the `build_candidate_pool` signature (after `popularity_rank_cutoff: Optional[int] = None,` at `:560`):

```python
    # Tag steering (soft): blend a user-selected genre target into the dense
    # admission centroid. None = feature off (byte-identical legacy behavior).
    steering_target: Optional[np.ndarray] = None,
    steering_blend: float = 0.5,
```

3b. In the `_use_dense` block, extend the `_agg` resolution (currently `:777-779`):

```python
        _agg = str(genre_admission_aggregate or "centroid").strip().lower()
        if _agg not in {"centroid", "per_seed"}:
            _agg = "centroid"
        if steering_target is not None and _agg == "per_seed":
            logger.info("Tag steering: forcing genre_admission_aggregate=centroid (was per_seed)")
            _agg = "centroid"
```

3c. In the centroid branch, after the existing seed-centroid normalization (`seed_dense = seed_dense / seed_dense_norm`, `:816`) and BEFORE `genre_sim_all = (X_genre_dense @ seed_dense)`:

```python
            if steering_target is not None:
                _blend = float(np.clip(steering_blend, 0.0, 1.0))
                _steered = (1.0 - _blend) * seed_dense + _blend * np.asarray(
                    steering_target, dtype=seed_dense.dtype
                )
                _steered_norm = np.linalg.norm(_steered)
                if _steered_norm > 1e-12:
                    seed_dense = _steered / _steered_norm
                logger.info(
                    "Tag steering pool lever: blend=%.2f applied to admission centroid",
                    _blend,
                )
```

3d. In the same branch, immediately after `effective_genre_floor` is computed (after the `else: effective_genre_floor = min_genre_similarity` at `:832-833`):

```python
            if steering_target is not None and effective_genre_floor is not None:
                _aff = (X_genre_dense @ np.asarray(steering_target, dtype=np.float64)).astype(np.float64)
                _adm = _aff[genre_sim_all >= float(effective_genre_floor)]
                if _adm.size:
                    logger.info(
                        "Tag steering pool affinity (genre-admitted set): "
                        "p10=%.3f p50=%.3f p90=%.3f n=%d",
                        float(np.percentile(_adm, 10)), float(np.percentile(_adm, 50)),
                        float(np.percentile(_adm, 90)), int(_adm.size),
                    )
```

- [ ] **Step 4: Wire core.py.** After the `_genre_admission_aggregate` block (ends `:542`), insert:

```python
    # Tag steering (user-selected genre tags): resolve the dense target once per run.
    _tag_steering_tags = [
        str(t) for t in (pb_overrides.get("tag_steering_tags") or []) if str(t).strip()
    ]
    _tag_steering_target = None
    try:
        _tag_steering_blend = float(pb_overrides.get("tag_steering_pool_blend", 0.5))
    except (TypeError, ValueError):
        _tag_steering_blend = 0.5
    if _tag_steering_tags:
        from src.playlist.tag_steering import resolve_tag_steering_target
        _tag_steering_target, _, _ = resolve_tag_steering_target(
            _tag_steering_tags,
            genre_vocab=[str(v) for v in getattr(bundle, "genre_vocab", [])],
            genre_emb=getattr(bundle, "genre_emb", None),
        )
```

and add to the `build_candidate_pool(...)` call inside `_build_pool` (after `popularity_rank_cutoff=popularity_rank_cutoff,` at `:599`):

```python
            steering_target=_tag_steering_target,
            steering_blend=_tag_steering_blend,
```

- [ ] **Step 5: Add config keys.** In BOTH `config.yaml` and `config.example.yaml`, under `playlists: → ds_pipeline: → pier_bridge:` (alongside keys like `tail_dp_floor`), add:

```yaml
      # Tag steering (artist-mode GUI chips; soft lean, never a gate).
      # Per-request tag list arrives via policy override (tag_steering_tags).
      tag_steering_pool_blend: 0.5    # 0=ignore tags, 1=pool centroid is pure target
      tag_steering_pier_weight: 0.3   # on-tag bonus in artist pier (medoid) scoring
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest -q tests/unit/test_tag_steering_pool_lever.py tests/unit/test_tag_steering.py`
Expected: all passed
Run: `python -m pytest -q -m "not slow" tests/test_gui_fidelity.py`
Expected: all passed (zero-tag path unchanged)

- [ ] **Step 7: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py config.example.yaml tests/unit/test_tag_steering_pool_lever.py
git commit -m "feat(tag-steering): pool lever — blend steering target into dense admission centroid"
```

(`config.yaml` is gitignored — edit it, don't stage it.)

---

### Task 5: Pier lever — on-tag medoid bonus

**Files:**
- Modify: `src/playlist/artist_style.py` (`_medoids_for_cluster` signature `:398-414` + scoring `:447-465`; `cluster_artist_tracks` signature `:529-541` + cluster loop `:657-682`; `ArtistStyleConfig` — grep `class ArtistStyleConfig`, likely `src/playlist/config.py` or top of `artist_style.py`)
- Modify: `src/playlist_generator.py` (`ArtistStyleConfig` builder `:1709-1743`; bundle load `:1766`; the `cluster_artist_tracks(` call site — grep, ~`:1850`; pier logging near `pier_ids` at `:1936`)
- Test: `tests/unit/test_tag_steering_pier_lever.py` (new)

**Interfaces:**
- Consumes: Task 3's `resolve_tag_steering_target`; config key `tag_steering_pier_weight` (Task 4 Step 5).
- Produces: `_medoids_for_cluster(..., tag_weight: float = 0.0, tag_affinity: Optional[np.ndarray] = None)`; `cluster_artist_tracks(..., steering_target: Optional[np.ndarray] = None)`; `ArtistStyleConfig.medoid_tag_weight: float = 0.0`.

- [ ] **Step 1: Write the failing test**

```python
"""Pier lever: tag affinity shifts medoid choice within a cluster."""
import numpy as np

from src.playlist.artist_style import _medoids_for_cluster


def _select(tag_weight, tag_affinity):
    # Three cluster members; member 0 is closest to the centroid.
    x = np.array([[1.0, 0.0], [0.98, 0.02], [0.96, 0.04]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return _medoids_for_cluster(
        x,
        [0, 1, 2],
        x[0],
        ["t0", "t1", "t2"],
        1,                       # per_cluster
        np.random.default_rng(0),
        1,                       # top_k
        None,                    # artist_duration_stats
        None,                    # track_durations_ms
        1.0,                     # similarity_weight
        0.0,                     # duration_weight
        0.0,                     # energy_weight
        None,                    # energy_proximity
        0.0,                     # popularity_weight
        None,                    # popularity_values
        tag_weight,
        tag_affinity,
    )


def test_zero_weight_keeps_sonic_medoid():
    assert _select(0.0, np.array([0.0, 0.0, 1.0])) == [0]


def test_tag_affinity_promotes_on_tag_member():
    # Sonic gap 0->2 is ~0.04; affinity gap is 1.0 at weight 0.5 -> member 2 wins.
    assert _select(0.5, np.array([0.0, 0.0, 1.0])) == [2]


def test_none_affinity_is_inert_even_with_weight():
    assert _select(0.5, None) == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_tag_steering_pier_lever.py`
Expected: FAIL — `TypeError: _medoids_for_cluster() takes ... positional arguments but 17 were given`

- [ ] **Step 3: Implement in `artist_style.py`**

3a. Append to the `_medoids_for_cluster` signature (after `popularity_values: Optional[np.ndarray] = None,` at `:413`):

```python
    tag_weight: float = 0.0,
    tag_affinity: Optional[np.ndarray] = None,
```

3b. After the popularity-bias block in the scoring section (ends `:465`), mirroring its shape exactly:

```python
    # Tag steering: prefer the artist's on-tag tracks WITHIN this cluster's slot.
    if tag_affinity is not None and tag_weight > 0:
        ta = np.asarray(tag_affinity, dtype=float)
        if ta.shape[0] == len(indices):
            scores = scores + ta * tag_weight
```

3c. Add field to `ArtistStyleConfig` (grep `class ArtistStyleConfig`; append with the other `medoid_*` weights):

```python
    medoid_tag_weight: float = 0.0  # tag-steering on-tag bonus in pier scoring
```

3d. Add keyword to `cluster_artist_tracks` (after `metadata_db_path: Optional[str] = None,` at `:540`):

```python
    steering_target: Optional[np.ndarray] = None,
```

3e. In the cluster loop (`:657-682`), after the `pop_slice` computation (`:663-665`), add the affinity slice and extend the positional call:

```python
        tag_slice: Optional[np.ndarray] = None
        _xgd = getattr(bundle, "X_genre_dense", None)
        if steering_target is not None and _xgd is not None and cfg.medoid_tag_weight > 0:
            tag_slice = np.asarray(_xgd, dtype=float)[members_local] @ np.asarray(
                steering_target, dtype=float
            )
        medoid_list = _medoids_for_cluster(
            X_norm,
            members_local,
            centroids[c],
            track_ids,
            medoid_top_k,
            rng,
            medoid_top_k,
            artist_duration_stats,
            bundle.durations_ms,
            cfg.medoid_similarity_weight,
            cfg.medoid_duration_weight,
            cfg.medoid_energy_weight,
            energy_prox,
            cfg.medoid_popularity_weight,
            pop_slice,
            cfg.medoid_tag_weight,
            tag_slice,
        )
```

- [ ] **Step 4: Wire `playlist_generator.py` (artist path).**

4a. In the `ArtistStyleConfig` builder (`:1709-1743`), add:

```python
        medoid_tag_weight=float(
            (ds_cfg.get("pier_bridge", {}) or {}).get("tag_steering_pier_weight", 0.3)
        ),
```

4b. After the bundle load (`bundle = load_artifact_bundle(artifact_path)`, `:1766`), resolve the target once:

```python
        # Tag steering (artist mode): resolve the user's selected tags once per run.
        _pb_cfg_dict = ds_cfg.get("pier_bridge", {}) or {}
        _steering_tag_list = [
            str(t) for t in (_pb_cfg_dict.get("tag_steering_tags") or []) if str(t).strip()
        ]
        steering_target = None
        if _steering_tag_list:
            from src.playlist.tag_steering import resolve_tag_steering_target
            steering_target, _, _ = resolve_tag_steering_target(
                _steering_tag_list,
                genre_vocab=[str(v) for v in getattr(bundle, "genre_vocab", [])],
                genre_emb=getattr(bundle, "genre_emb", None),
            )
```

4c. Thread `steering_target=steering_target,` into the `cluster_artist_tracks(` call (grep the call site in `create_playlist_for_artist`; it already passes `energy_values=`/`popularity_values=` keywords).

4d. After the final pier list exists (`pier_ids = [str(bundle.track_ids[m]) for m in ordered_medoids]`, `:1936`), log per-pier affinity:

```python
        if steering_target is not None and getattr(bundle, "X_genre_dense", None) is not None:
            _pier_aff = [
                round(float(np.dot(bundle.X_genre_dense[m], steering_target)), 3)
                for m in ordered_medoids
            ]
            logger.info("Tag steering piers: affinity per selected pier = %s", _pier_aff)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q tests/unit/test_tag_steering_pier_lever.py`
Expected: 3 passed
Run: `python -m pytest -q -m "not slow"` (bounded, tool timeout 600000)
Expected: full fast suite green — quote the real counts.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/artist_style.py src/playlist_generator.py src/playlist/config.py tests/unit/test_tag_steering_pier_lever.py
git commit -m "feat(tag-steering): pier lever — on-tag medoid bonus in artist pier scoring"
```

(If `ArtistStyleConfig` lives in `artist_style.py` rather than `config.py`, stage accordingly.)

---

### Task 6: Request → policy threading

**Files:**
- Modify: `src/playlist_web/schemas.py` (`GenerateRequestBody`, `:12-57`)
- Modify: `src/playlist_web/app.py` (UIStateModel construction, `:173-190`)
- Modify: `src/playlist_gui/ui_state.py` (`UIStateModel`, after `seed_track_ids` block `:127-131`)
- Modify: `src/playlist_gui/policy.py` (`derive_runtime_config`, after the mode `_set_nested` block `:259-267`)
- Test: add to `tests/test_gui_fidelity.py`

**Interfaces:**
- Consumes: config channel established in Task 4 (`pb_overrides["tag_steering_tags"]`).
- Produces: `GenerateRequestBody.steering_tags: list[str]`; `UIStateModel.steering_tags: List[str]`; policy override `playlists.ds_pipeline.pier_bridge.tag_steering_tags`. Task 7's GUI sends `steering_tags`; Task 8's harness sets `UIStateModel.steering_tags`.

Note: `GeneratePlaylistRequest`/worker args need NO change — tags ride the policy-overrides channel only (worker merges overrides into config at `worker.py:1078` before the generator is constructed; both engine consumers read merged config).

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_fidelity.py`)

```python
def test_steering_tags_flow_into_pier_bridge_overrides():
    """Tag steering rides the policy->config channel (never hand-carried)."""
    ui = gui_ui_state(steering_tags=["jazz-funk", "soul jazz"])
    overrides = resolve_gui_overrides(ui)
    assert overrides["pier_bridge"]["tag_steering_tags"] == ["jazz-funk", "soul jazz"]


def test_no_steering_tags_leaves_config_untouched():
    overrides = resolve_gui_overrides(gui_ui_state())
    assert "tag_steering_tags" not in overrides["pier_bridge"]


def test_steering_tags_capped_at_three():
    ui = gui_ui_state(steering_tags=["a", "b", "c", "d"])
    overrides = resolve_gui_overrides(ui)
    assert overrides["pier_bridge"]["tag_steering_tags"] == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_gui_fidelity.py -k steering`
Expected: FAIL — `TypeError: UIStateModel.__init__() got an unexpected keyword argument 'steering_tags'` (via `replace`)

- [ ] **Step 3: Implement the four hops**

3a. `ui_state.py` — add after the `seed_track_ids` field block:

```python
    steering_tags: List[str] = field(default_factory=list)
    """
    Genre-tag steering: canonical tag names selected in the GUI (≤3; the GUI
    only surfaces the picker in artist mode, but the engine honors tags in any
    mode). Empty = steering off. Single writer of the runtime knob:
    policy.derive_runtime_config() → playlists.ds_pipeline.pier_bridge.tag_steering_tags.
    """
```

3b. `policy.py` — in `derive_runtime_config`, after the cohesion `_set_nested` block (`:266-267`):

```python
    steering_tags = [
        str(t).strip() for t in getattr(ui, "steering_tags", []) if str(t).strip()
    ][:3]
    if steering_tags:
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.tag_steering_tags",
            steering_tags,
        )
        notes.append(f"Tag steering: {', '.join(steering_tags)}")
```

3c. `schemas.py` — add to `GenerateRequestBody` (after `seed_track_ids`):

```python
    steering_tags: list[str] = Field(default_factory=list)
```

3d. `app.py` — add to the `UIStateModel(...)` construction (`:173-190`):

```python
            steering_tags=list(body.steering_tags),
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q tests/test_gui_fidelity.py`
Expected: all passed (new + existing)

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/ui_state.py src/playlist_gui/policy.py src/playlist_web/schemas.py src/playlist_web/app.py tests/test_gui_fidelity.py
git commit -m "feat(tag-steering): thread steering_tags request -> UIStateModel -> policy override"
```

---

### Task 7: GUI chips in GenerateControls

**Files:**
- Modify: `web/src/components/GenerateControls.tsx` (state near `:67-90`; effects near `:129-148`; `submit()` body `:150-164`; render in the artist-mode section under the artist input)
- Modify: the TS `GenerateRequestBody` type (grep `cohesion_mode` under `web/src/lib/` — add `steering_tags?: string[];`)

**Interfaces:**
- Consumes: `api.artistGenres` (Task 2), `steering_tags` API field (Task 6).
- Produces: user-visible chips; `steering_tags` in the POST body when mode is artist and ≥1 chip selected.

- [ ] **Step 1: Add state + fetch effect** (below the `artistSearch` block, `:97-103`; NOT in localStorage — tags are artist-specific and must reset on artist change)

```tsx
  // Tag steering (artist mode): the artist's published genres as selectable chips.
  const [artistTags, setArtistTags] = useState<
    { name: string; release_count: number; confidence: number }[]
  >([]);
  const [steeringTags, setSteeringTags] = useState<string[]>([]);

  // Fetch chips whenever a confirmed artist is selected; reset both on change.
  useEffect(() => {
    setSteeringTags([]);
    setArtistTags([]);
    if (mode !== "artist" || !seed.trim() || seed !== selectedRef.current) return;
    let cancelled = false;
    api.artistGenres(seed)
      .then((r) => { if (!cancelled) setArtistTags(r.genres); })
      .catch(() => { /* chips are best-effort; generation works without them */ });
    return () => { cancelled = true; };
  }, [seed, mode]);

  function toggleSteeringTag(name: string) {
    setSteeringTags((prev) =>
      prev.includes(name) ? prev.filter((t) => t !== name)
        : prev.length >= 3 ? prev
        : [...prev, name],
    );
  }
```

- [ ] **Step 2: Render the chip row** in the artist-mode controls, directly under the artist input/autocomplete block (match surrounding Tailwind tokens; this default styling is a starting point):

```tsx
      {mode === "artist" && artistTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5" data-testid="steering-chips">
          {artistTags.map((t) => {
            const on = steeringTags.includes(t.name);
            const capped = !on && steeringTags.length >= 3;
            return (
              <button
                key={t.name}
                type="button"
                disabled={capped}
                onClick={() => toggleSteeringTag(t.name)}
                title={`${t.release_count} release${t.release_count === 1 ? "" : "s"}`}
                className={
                  "rounded-full border px-2 py-0.5 text-xs transition-colors " +
                  (on
                    ? "border-primary bg-primary/15 text-primary"
                    : capped
                      ? "opacity-40"
                      : "hover:bg-muted")
                }
              >
                {t.name}
              </button>
            );
          })}
        </div>
      )}
```

- [ ] **Step 2b: Empty-state hint (spec §1.5).** Track whether the fetch returned empty for a confirmed artist, and render the hint instead of the chip row:

```tsx
  const [tagsFetched, setTagsFetched] = useState(false);
```

In the fetch effect: `setTagsFetched(false);` alongside the resets, and `setTagsFetched(true);` inside the `.then` (after `setArtistTags`). Then below the chip-row block:

```tsx
      {mode === "artist" && tagsFetched && artistTags.length === 0 && seed.trim() !== "" && (
        <p className="text-xs text-muted-foreground">
          No published genres for this artist — run enrichment publish to enable tag steering.
        </p>
      )}
```

- [ ] **Step 3: Send tags in `submit()`** — add to the `body` literal (`:151-164`):

```tsx
      steering_tags: mode === "artist" && steeringTags.length ? steeringTags : undefined,
```

and add `steering_tags?: string[];` to the TS `GenerateRequestBody` type.

- [ ] **Step 4: Build + typecheck**

Run: `npm --prefix web run build`
Expected: build succeeds, no TS errors. (Do NOT skip — stale `web/dist` is the #1 GUI trap.)

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GenerateControls.tsx web/src/lib/api.ts web/src/lib/types.ts
git commit -m "feat(tag-steering): artist-mode genre chips in GenerateControls"
```

(Adjust the types.ts path to wherever the TS `GenerateRequestBody` actually lives.)

---

### Task 8: Behavioral integration test (pool lever, real artifact)

**Files:**
- Create: `tests/integration/test_tag_steering_behavioral.py`

**Interfaces:**
- Consumes: `generate_like_gui` + `UIStateModel.steering_tags` (Task 6), pool-lever log lines (Task 4).

- [ ] **Step 1: Write the test**

```python
"""Behavioral: steering tags shift the generated playlist's genre affinity.

Runs through generate_like_gui (the production config chain) against the live
artifact. Read the log lines, not just the metric (playlist-testing skill).
"""
import logging
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, "tests")
from support.gui_fidelity import generate_like_gui, gui_ui_state  # noqa: E402

from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.tag_steering import resolve_tag_steering_target  # noqa: E402

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    return load_artifact_bundle(str(ARTIFACT))


def _seeds_and_tag(bundle):
    """Two same-artist seeds + the library's most common vocab tag (always mappable)."""
    if getattr(bundle, "genre_emb", None) is None:
        pytest.skip("dense genre sidecar absent")
    col_mass = np.asarray(bundle.X_genre_raw.sum(axis=0)).ravel()
    tag = str(bundle.genre_vocab[int(np.argmax(col_mass))])
    keys, counts = np.unique(bundle.artist_keys, return_counts=True)
    artist = keys[np.argmax(counts)]  # best-represented artist -> feasible piers
    idx = np.nonzero(bundle.artist_keys == artist)[0][:2]
    if len(idx) < 2:
        pytest.skip("no artist with 2+ tracks in artifact")
    return [str(bundle.track_ids[i]) for i in idx], tag


def _mean_affinity(bundle, track_ids, target):
    rows = [bundle.track_id_to_index[t] for t in track_ids if t in bundle.track_id_to_index]
    return float(np.mean(bundle.X_genre_dense[rows] @ target))


def test_steering_shifts_playlist_affinity_and_logs(bundle, caplog):
    seeds, tag = _seeds_and_tag(bundle)
    target, _, _ = resolve_tag_steering_target(
        [tag], genre_vocab=[str(v) for v in bundle.genre_vocab],
        genre_emb=bundle.genre_emb,
    )
    assert target is not None

    base = generate_like_gui(seeds=seeds, length=15, random_seed=0)
    with caplog.at_level(logging.INFO):
        steered = generate_like_gui(
            seeds=seeds, length=15, random_seed=0, steering_tags=[tag]
        )

    # 1) The lever FIRED (log evidence, not just a metric).
    assert any("Tag steering pool lever" in r.message for r in caplog.records)
    assert any("Tag steering target" in r.message for r in caplog.records)

    # 2) The playlist's mean tag affinity moved toward the target.
    aff_base = _mean_affinity(bundle, base.track_ids, target)
    aff_steered = _mean_affinity(bundle, steered.track_ids, target)
    assert aff_steered >= aff_base, (
        f"steered affinity {aff_steered:.3f} < baseline {aff_base:.3f} — "
        "read the gate tally + pool lines before concluding the lever is weak"
    )
```

(If `DsRunResult` exposes the playlist as something other than `.track_ids`, mirror whatever `tests/integration/test_gui_fidelity_regressions.py` reads from the result object — that file is the reference for harness result access.)

- [ ] **Step 2: Run it**

Run: `python -m pytest -q tests/integration/test_tag_steering_behavioral.py` (tool timeout 600000)
Expected: 1 passed (or skip on machines without the artifact). If the affinity assertion fails, follow the playlist-testing skill: check `Tag steering target` mapped count, the gate tally, and `pool_after_gate` before touching the lever.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_tag_steering_behavioral.py
git commit -m "test(tag-steering): behavioral pool-lever shift through the gui_fidelity harness"
```

---

### Task 9: Docs, skill maintenance, spec sync

**Files:**
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md` (new knob recipe)
- Modify: `.claude/skills/genre-data-authority/SKILL.md` (reading-recipes section)
- Verify: `docs/superpowers/specs/2026-07-02-tag-steering-design.md` still matches what shipped

- [ ] **Step 1: Tuning recipe.** Append to `docs/PLAYLIST_ORDERING_TUNING.md` following the house knob-recipe format:

```markdown
## Knob: tag steering (artist-mode genre chips)

- `playlists.ds_pipeline.pier_bridge.tag_steering_pool_blend` (default **0.5**) — how far the
  candidate-pool genre-admission centroid moves from the seed centroid toward the selected
  tags. 0 = chips do nothing to the pool; 1 = pool admission ranks purely by tag affinity.
  Symptom → move: "playlist ignores my tags" → raise toward 0.7; "playlist lost the artist's
  own character" → lower toward 0.3.
- `playlists.ds_pipeline.pier_bridge.tag_steering_pier_weight` (default **0.3**) — bonus for
  on-tag artist tracks in pier (medoid) selection, composing with the sonic/duration/energy/
  popularity terms. Raise if the anchors don't reflect the tags; 0 disables pier steering only.
- Per-request tags arrive as `tag_steering_tags` via the GUI; empty = feature fully inert.
- Diagnose with the per-playlist log: `Tag steering target` (mapped count), `Tag steering
  pool lever` (blend applied), `Tag steering pool affinity` (p10/p50/p90 of the admitted
  set), `Tag steering piers` (per-pier affinity). No lines = the knob didn't act — that's a
  bug, not a tuning problem.
```

- [ ] **Step 2: Authority skill update.** In `.claude/skills/genre-data-authority/SKILL.md`, add to the "Reading recipes" list:

```markdown
- Artist-level aggregation (tag-steering chips): `resolved_genres_for_artist(conn, artist_name)`
  — observed_leaf+legacy only (inferred hub families excluded by design), exact
  case-insensitive match on `tracks.artist`, ordered by (release_count, max_confidence).
```

- [ ] **Step 3: Spec sync check.** Re-read the spec; confirm config-key names, endpoint path, and the shipped log-line names match. Fix any drift in the spec (it documents what shipped).

- [ ] **Step 4: Commit**

```bash
git add docs/PLAYLIST_ORDERING_TUNING.md .claude/skills/genre-data-authority/SKILL.md docs/superpowers/specs/2026-07-02-tag-steering-design.md
git commit -m "docs(tag-steering): tuning recipe + authority-skill recipe row"
```

---

### Task 10: Live verification (real GUI, real logs)

No code. This is the acceptance gate — do not skip any step.

- [ ] **Step 1:** Restart the web server + worker (`python tools/serve_web.py`) — worker `@lru_cache` and process state make un-restarted verification meaningless.
- [ ] **Step 2:** In the browser: artist mode → type/select **Herbie Hancock** → confirm chips appear (jazz tags, ≤12, ordered sensibly) → select 2 (e.g. the funk-leaning ones) → generate 30 tracks.
- [ ] **Step 3:** Open the newest `logs/playlists/*_Herbie_Hancock_*.log` and verify ALL of:
  - `Policy: Tag steering: <tags>` (policy emitted),
  - `Tag steering target: tags=[...] (mapped 2/2)` (vocab mapped),
  - `Tag steering pool lever: blend=0.50 applied to admission centroid`,
  - `Tag steering pool affinity (genre-admitted set): p10=... p50=... p90=...`,
  - `Tag steering piers: affinity per selected pier = [...]`,
  - transition stats not degraded vs. a no-tag run of the same artist (`T transition:` mean/min in the same ballpark).
- [ ] **Step 4:** Generate the SAME artist with NO chips selected → verify none of the steering lines appear and the playlist generates normally (zero-tag inertness in production).
- [ ] **Step 5:** Listen check (Dylan): does the steered playlist audibly lean? This is the real acceptance; log evidence + ears together decide whether stage 2 (beam lever) is warranted.
- [ ] **Step 6:** Run the full suite once: `python -m pytest -q -m "not slow"` (timeout 600000). Quote real pass/fail counts. Fix anything red before declaring done.
