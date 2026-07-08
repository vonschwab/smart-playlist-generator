# Tag Steering — Sonic-Prototype Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give tag steering track-level resolution by letting the selected tag(s) learn a *sonic* prototype from the library (MuQ centroid), combined additively with the existing genre-dense signal, consumed at pier selection and the candidate pool.

**Architecture:** A shared row-resolver maps tag(s) → library track-rows carrying them (genre authority, seed-artist excluded). Each consumer builds a centered, normalized MuQ centroid (the "prototype") *in its own sonic space* and scores tracks by proximity. Piers add an additive sonic term to the existing genre-dense `tag_slice`; the candidate pool gets a parallel sonic-admission lever mirroring the existing genre pool lever. No beam changes.

**Tech Stack:** Python 3.11, NumPy, SQLite (`track_effective_genres`), pytest (markers: `integration`, `slow`).

## Global Constraints

- **Byte-identical when no tags are selected.** Every new path is gated on a resolved prototype, which requires selected tags. No-tag generations must be unchanged.
- **A configured lever that can't act WARNS loudly — never a silent no-op** (project gotcha). Low prototype support / degenerate prototype → warn + fall back to genre-dense only.
- **Genre labels come from the authority-consistent view** (`track_effective_genres`), per the `genre-data-authority` skill — not raw `track_genres`.
- **Generation tests mirror production** via the real artist path (live artifact + DB), per the `playlist-testing` skill. Never hand-build partial overrides dicts.
- **Space-consistency:** a prototype blended into / scored against a sonic matrix MUST be built from that same matrix (per-row L2-normalized). Piers score on `bundle.X_sonic_muq`; the pool scores on `embedding.X_sonic_for_embed`. Build each prototype from its own matrix.
- **Canonical shared checkout:** stage explicit paths only (`git add <paths>`; `git commit --only -- <paths>`), never `git add -A/.`/bare commit. Verify `git diff --cached --name-only` before each commit.
- **Defaults:** `tag_steering_sonic_weight=0.5` (pier), `tag_steering_sonic_blend=0.35` (pool), `tag_steering_prototype_min_support=25`. These are starting estimates; Task 5 calibrates them.

---

### Task 1: Pure prototype math (`sonic_prototype_from_rows`)

**Files:**
- Modify: `src/playlist/tag_steering.py`
- Test: `tests/unit/test_tag_steering.py`

**Interfaces:**
- Produces:
  - `sonic_prototype_from_rows(sonic_matrix: np.ndarray, rows: Sequence[int], *, global_mean: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], float, int]` — returns `(prototype | None, cohesion, support_n)`.
  - `sonic_global_mean(sonic_matrix: np.ndarray) -> np.ndarray` — mean of per-row L2-normalized rows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_tag_steering.py  (add to existing file)
import numpy as np
from src.playlist.tag_steering import sonic_prototype_from_rows, sonic_global_mean


def test_sonic_prototype_points_at_member_mean_direction():
    # Two tight clusters; prototype of cluster-A rows should align with A, oppose B.
    A = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1)) + 1e-3
    B = np.tile(np.array([0.0, 1.0, 0.0]), (10, 1)) + 1e-3
    M = np.vstack([A, B])
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(10)))
    assert n == 10
    assert cohesion > 0.9                      # tight cluster -> high cohesion
    # aligns with A direction, not B
    assert proto @ np.array([1.0, 0.0, 0.0]) > 0.8
    assert proto @ np.array([0.0, 1.0, 0.0]) < 0.2


def test_sonic_prototype_low_cohesion_for_scattered_rows():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((50, 16))
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(50)))
    assert proto is not None
    assert cohesion < 0.5                       # scattered -> low cohesion


def test_sonic_prototype_empty_rows_returns_none():
    M = np.eye(4)
    proto, cohesion, n = sonic_prototype_from_rows(M, [])
    assert proto is None and n == 0


def test_global_mean_centering_subtracts_common_component():
    # All rows share a big common direction + small distinguishing part.
    common = np.array([5.0, 0.0, 0.0])
    M = np.vstack([common + np.array([0, 1.0, 0]), common + np.array([0, 0, 1.0])])
    gm = sonic_global_mean(M)
    proto_centered, _, _ = sonic_prototype_from_rows(M, [0], global_mean=gm)
    # after centering, the common x-component should not dominate the prototype
    assert abs(proto_centered[0]) < abs(proto_centered[1]) + abs(proto_centered[2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_tag_steering.py -k sonic -v`
Expected: FAIL with `ImportError: cannot import name 'sonic_prototype_from_rows'`.

- [ ] **Step 3: Implement the functions**

```python
# src/playlist/tag_steering.py  (append)
def sonic_global_mean(sonic_matrix: np.ndarray) -> np.ndarray:
    """Mean of the per-row L2-normalized sonic rows (the 'generic' direction)."""
    M = np.asarray(sonic_matrix, dtype=np.float64)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn.mean(axis=0)


def sonic_prototype_from_rows(
    sonic_matrix: np.ndarray,
    rows: Sequence[int],
    *,
    global_mean: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], float, int]:
    """Centered, L2-normalized centroid of ``rows`` + intra-set cohesion.

    ``rows`` index into ``sonic_matrix`` (bundle-aligned). When ``global_mean`` is
    given it is subtracted from each normalized member row before averaging, to
    remove the generic-sonic component. ``cohesion`` is the mean cosine of member
    vectors to the prototype (low => sonically multimodal tag). Returns
    ``(prototype | None, cohesion, support_n)``.
    """
    idx = [int(r) for r in rows]
    if not idx:
        return None, 0.0, 0
    M = np.asarray(sonic_matrix, dtype=np.float64)[idx]
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    if global_mean is not None:
        Mn = Mn - np.asarray(global_mean, dtype=np.float64)
    proto = Mn.mean(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm <= 1e-12:
        return None, 0.0, len(idx)
    proto = proto / norm
    cohesion = float(np.mean(Mn @ proto))
    return proto, cohesion, len(idx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_tag_steering.py -k sonic -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/tag_steering.py tests/unit/test_tag_steering.py
git commit --only -- src/playlist/tag_steering.py tests/unit/test_tag_steering.py -m "feat(tag-steering): pure sonic-prototype centroid + cohesion"
```

---

### Task 2: Tag → library-row resolver (`resolve_tag_sonic_prototype_rows`)

**Files:**
- Modify: `src/playlist/tag_steering.py`
- Test: `tests/unit/test_tag_steering.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `resolve_tag_sonic_prototype_rows(tags: Sequence[str], *, metadata_db_path: str, track_id_to_row: dict[str, int], exclude_artist: Optional[str] = None, min_support: int = 25) -> tuple[Optional[list[int]], int, list[str]]` — returns `(rows | None, support_n, tags_used)`. `None` + WARN when support < `min_support`.

- [ ] **Step 1: Write the failing test (temp sqlite, no live DB dependency)**

```python
# tests/unit/test_tag_steering.py  (add)
import sqlite3
from src.playlist.tag_steering import resolve_tag_sonic_prototype_rows


def _make_db(tmp_path):
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT)")
    con.execute("CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT)")
    rows = [(f"t{i}", "Seed" if i < 3 else "Other") for i in range(40)]
    con.executemany("INSERT INTO tracks VALUES (?,?)", rows)
    con.executemany("INSERT INTO track_effective_genres VALUES (?,?)",
                    [(f"t{i}", "jangle pop") for i in range(40)])
    con.commit(); con.close()
    return str(db)


def test_resolver_returns_rows_excluding_seed_artist(tmp_path):
    db = _make_db(tmp_path)
    t2r = {f"t{i}": i for i in range(40)}
    rows, n, used = resolve_tag_sonic_prototype_rows(
        ["Jangle Pop"], metadata_db_path=db, track_id_to_row=t2r,
        exclude_artist="Seed", min_support=25)
    assert n == 37                       # 40 tagged minus 3 Seed tracks
    assert all(t2r_val >= 3 for t2r_val in rows)  # t0..t2 excluded
    assert used == ["Jangle Pop"]


def test_resolver_warns_and_returns_none_below_support(tmp_path, caplog):
    db = _make_db(tmp_path)
    t2r = {f"t{i}": i for i in range(40)}
    import logging
    with caplog.at_level(logging.WARNING):
        rows, n, used = resolve_tag_sonic_prototype_rows(
            ["jangle pop"], metadata_db_path=db, track_id_to_row=t2r,
            exclude_artist="Seed", min_support=100)   # 37 < 100
    assert rows is None
    assert any("sonic prototype" in r.message.lower() for r in caplog.records)


def test_resolver_empty_tags_returns_none(tmp_path):
    db = _make_db(tmp_path)
    rows, n, used = resolve_tag_sonic_prototype_rows(
        [], metadata_db_path=db, track_id_to_row={}, min_support=25)
    assert rows is None and n == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_tag_steering.py -k resolver -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_tag_sonic_prototype_rows'`.

- [ ] **Step 3: Implement the resolver**

```python
# src/playlist/tag_steering.py  (append; add `import sqlite3` at top)
def resolve_tag_sonic_prototype_rows(
    tags: Sequence[str],
    *,
    metadata_db_path: str,
    track_id_to_row: dict,
    exclude_artist: Optional[str] = None,
    min_support: int = 25,
) -> tuple[Optional[list], int, list]:
    """Library row indices carrying ANY selected tag (authority-effective genres),
    seed-artist excluded. Rows index into the caller's bundle track ordering.
    Returns (rows | None, support_n, tags_used); None + WARN below min_support."""
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return None, 0, []
    con = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        ph = ",".join("?" for _ in wanted)
        cur.execute(
            f"SELECT DISTINCT track_id FROM track_effective_genres WHERE lower(genre) IN ({ph})",
            [t.lower() for t in wanted],
        )
        tids = [str(r[0]) for r in cur.fetchall()]
        excl = set()
        if exclude_artist:
            cur.execute("SELECT track_id FROM tracks WHERE artist=?", (exclude_artist,))
            excl = {str(r[0]) for r in cur.fetchall()}
    finally:
        con.close()
    rows = [track_id_to_row[t] for t in tids if t in track_id_to_row and t not in excl]
    if len(rows) < int(min_support):
        logger.warning(
            "Tag steering sonic prototype: only %d library tracks carry %s "
            "(min_support=%d) — sonic prototype disabled, falling back to the "
            "genre-dense signal for this run.",
            len(rows), wanted, int(min_support),
        )
        return None, len(rows), wanted
    return rows, len(rows), wanted
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_tag_steering.py -k resolver -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/tag_steering.py tests/unit/test_tag_steering.py
git commit --only -- src/playlist/tag_steering.py tests/unit/test_tag_steering.py -m "feat(tag-steering): resolve tag -> library sonic-prototype rows (authority, seed-excluded, support guard)"
```

---

### Task 3: Pier sonic term (piers pick on-tag seeds)

**Files:**
- Modify: `src/playlist/artist_style.py` (`cluster_artist_tracks`, ~L695 signature and ~L911 `tag_slice` block)
- Modify: `src/playlist_generator.py` (~L1784 prototype resolution; ~L1930 `cluster_affinities`; ~L1897 `cluster_artist_tracks` call)
- Modify: `config.example.yaml` (~L327 pier_bridge block)
- Test: `tests/integration/test_tag_steering_sonic_prototype_live.py` (create)

**Interfaces:**
- Consumes: `sonic_prototype_from_rows`, `sonic_global_mean`, `resolve_tag_sonic_prototype_rows` (Tasks 1–2).
- Produces: `cluster_artist_tracks(..., sonic_tag_affinity: Optional[np.ndarray] = None, sonic_tag_weight: float = 0.0)` — a bundle-aligned `(N,)` affinity vector and its weight; when present, added into the medoid `tag_slice`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_tag_steering_sonic_prototype_live.py  (create)
"""Live: the sonic prototype lifts Eno's ambient-pier affinity above genre-dense alone."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path  # noqa: E402

from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.artist_style import ArtistStyleConfig, cluster_artist_tracks  # noqa: E402
from src.playlist.tag_steering import (  # noqa: E402
    resolve_tag_steering_target,
    resolve_tag_sonic_prototype_rows,
    sonic_prototype_from_rows,
    sonic_global_mean,
)

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB = "data/metadata.db"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    resolved_artifact_path()
    return load_artifact_bundle(str(ARTIFACT))


def _sonic_tag_affinity(bundle, tags):
    xmq = np.asarray(bundle.X_sonic_muq, dtype=np.float64)
    t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        tags, metadata_db_path=DB, track_id_to_row=t2r,
        exclude_artist="Brian Eno", min_support=25)
    assert rows is not None, "ambient must have enough library support"
    gm = sonic_global_mean(xmq)
    proto, cohesion, _ = sonic_prototype_from_rows(xmq, rows, global_mean=gm)
    xmn = xmq / (np.linalg.norm(xmq, axis=1, keepdims=True) + 1e-12)
    return (xmn - gm) @ proto


def _mean_pier_genre_aff(bundle, medoids, target):
    xgd = np.asarray(bundle.X_genre_dense, dtype=np.float64)
    return float(np.mean([xgd[m] @ target for m in medoids])) if medoids else 0.0


def test_sonic_prototype_raises_ambient_pier_affinity(bundle):
    if getattr(bundle, "genre_emb", None) is None or getattr(bundle, "X_sonic_muq", None) is None:
        pytest.skip("dense genre or MuQ sidecar absent")
    tags = ["ambient", "drone", "dark ambient", "space ambient"]
    target, mapped, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    if target is None:
        pytest.skip("ambient tags did not map")
    aff_vec = _sonic_tag_affinity(bundle, tags)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)

    # genre-dense only (baseline)
    _, med_base, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB)
    # + sonic prototype term
    _, med_sonic, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=aff_vec, sonic_tag_weight=0.5)

    # Piers chosen with the sonic term should have higher MEAN sonic ambient affinity.
    base_sonic = float(np.mean([aff_vec[m] for m in med_base])) if med_base else 0.0
    with_sonic = float(np.mean([aff_vec[m] for m in med_sonic])) if med_sonic else 0.0
    assert with_sonic > base_sonic, (
        f"sonic-term piers ambient affinity {with_sonic:.3f} !> genre-only {base_sonic:.3f}")


def test_no_sonic_affinity_is_byte_identical_piers(bundle):
    tags = ["ambient"]
    target, _, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)
    _, med_a, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB)
    _, med_b, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=None, sonic_tag_weight=0.5)   # None => no effect
    assert med_a == med_b
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/integration/test_tag_steering_sonic_prototype_live.py -k raises_ambient -v`
Expected: FAIL with `TypeError: cluster_artist_tracks() got an unexpected keyword argument 'sonic_tag_affinity'`.

- [ ] **Step 3a: Add params to `cluster_artist_tracks` and enrich `tag_slice`**

In `src/playlist/artist_style.py`, add to the `cluster_artist_tracks` signature (near the existing `steering_target: Optional[np.ndarray] = None,` at ~L700):

```python
    sonic_tag_affinity: Optional[np.ndarray] = None,   # bundle-aligned (N,) sonic prototype affinity
    sonic_tag_weight: float = 0.0,
```

Then extend the `tag_slice` block (~L911–916) to add the additive sonic term:

```python
        tag_slice: Optional[np.ndarray] = None
        _xgd = getattr(bundle, "X_genre_dense", None)
        if steering_target is not None and _xgd is not None and cfg.medoid_tag_weight > 0:
            tag_slice = np.asarray(_xgd, dtype=float)[members_eligible] @ np.asarray(
                steering_target, dtype=float
            )
        # Sonic-prototype term (track-level resolution). Additive: a flat genre
        # term drops out of the ranking, so this decides for genre-blended artists.
        if (
            sonic_tag_affinity is not None
            and float(sonic_tag_weight) > 0.0
            and cfg.medoid_tag_weight > 0
        ):
            _sonic_slice = float(sonic_tag_weight) * np.asarray(
                sonic_tag_affinity, dtype=float
            )[members_eligible]
            tag_slice = _sonic_slice if tag_slice is None else (tag_slice + _sonic_slice)
```

- [ ] **Step 3b: Resolve the prototype in the generator and thread it in**

In `src/playlist_generator.py`, just after `steering_target` is resolved (~L1786–1795), compute the bundle-aligned sonic affinity vector + weight:

```python
                sonic_tag_affinity = None
                sonic_tag_weight = 0.0
                _xmq = getattr(bundle, "X_sonic_muq", None)
                if _steering_tag_list and _xmq is not None:
                    from src.playlist.tag_steering import (
                        resolve_tag_sonic_prototype_rows,
                        sonic_prototype_from_rows,
                        sonic_global_mean,
                    )
                    _pb = (ds_cfg.get("pier_bridge", {}) or {})
                    _min_support = int(_pb.get("tag_steering_prototype_min_support", 25))
                    _t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
                    _rows, _n, _ = resolve_tag_sonic_prototype_rows(
                        _steering_tag_list,
                        metadata_db_path=resolve_database_path(self.config),
                        track_id_to_row=_t2r,
                        exclude_artist=artist_name,
                        min_support=_min_support,
                    )
                    if _rows is not None:
                        _xmq_arr = np.asarray(_xmq, dtype=np.float64)
                        _gm = sonic_global_mean(_xmq_arr)
                        _proto, _cohesion, _ = sonic_prototype_from_rows(
                            _xmq_arr, _rows, global_mean=_gm
                        )
                        _min_coh = float(_pb.get("tag_steering_prototype_min_cohesion", 0.15))
                        if _proto is not None and _cohesion < _min_coh:
                            logger.warning(
                                "Tag steering sonic prototype: cohesion %.3f < %.2f (sonically "
                                "multimodal tag) — sonic term disabled, using genre-dense only.",
                                _cohesion, _min_coh,
                            )
                            _proto = None
                        if _proto is not None:
                            _xmn = _xmq_arr / (np.linalg.norm(_xmq_arr, axis=1, keepdims=True) + 1e-12)
                            sonic_tag_affinity = (_xmn - _gm) @ _proto
                            sonic_tag_weight = float(_pb.get("tag_steering_sonic_weight", 0.5))
                            logger.info(
                                "Tag steering sonic prototype: support=%d cohesion=%.3f weight=%.2f",
                                _n, _cohesion, sonic_tag_weight,
                            )
```

Pass both into the `cluster_artist_tracks(...)` call (~L1897) by adding:

```python
                    sonic_tag_affinity=sonic_tag_affinity,
                    sonic_tag_weight=sonic_tag_weight,
```

And fold the sonic term into `cluster_affinities` (~L1930–1934) so pier-slot allocation also uses it:

```python
                    _xgd = np.asarray(_xgd, dtype=float)
                    _tgt = np.asarray(steering_target, dtype=float)
                    cluster_affinities = [
                        (float(np.mean(_xgd[members] @ _tgt))
                         + (sonic_tag_weight * float(np.mean(sonic_tag_affinity[members]))
                            if sonic_tag_affinity is not None else 0.0))
                        if len(members) else 0.0
                        for members in clusters
                    ]
```

- [ ] **Step 3c: Add config knobs**

In `config.example.yaml`, in the `pier_bridge` block (near `tag_steering_pier_weight`, ~L328), add:

```yaml
      tag_steering_sonic_weight: 0.5          # weight of the library-learned sonic-prototype term in pier scoring
      tag_steering_prototype_min_support: 25  # min library tracks/tag to trust the sonic prototype (else genre-dense only)
      tag_steering_prototype_min_cohesion: 0.15  # min intra-set cohesion; below this the tag is sonically multimodal -> genre-dense only
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/integration/test_tag_steering_sonic_prototype_live.py -k "raises_ambient or byte_identical_piers" -v`
Expected: PASS (2 tests). If `raises_ambient` fails, read the per-cluster affinities in the log before concluding — do not just bump the weight.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py src/playlist_generator.py config.example.yaml tests/integration/test_tag_steering_sonic_prototype_live.py
git commit --only -- src/playlist/artist_style.py src/playlist_generator.py config.example.yaml tests/integration/test_tag_steering_sonic_prototype_live.py -m "feat(tag-steering): sonic-prototype term in pier selection"
```

---

### Task 4: Sonic candidate-pool lever (bridges pull from an on-tag pool)

**Files:**
- Modify: `src/playlist/candidate_pool.py` (`build_candidate_pool` signature ~L563; sonic block ~L647–651)
- Modify: `src/playlist/pipeline/core.py` (~L554 resolve; ~L627 pass-through)
- Modify: `config.example.yaml` (pier_bridge block)
- Test: `tests/integration/test_tag_steering_sonic_prototype_live.py` (add)

**Interfaces:**
- Consumes: `resolve_tag_sonic_prototype_rows`, `sonic_prototype_from_rows`, `sonic_global_mean` (Tasks 1–2).
- Produces: `build_candidate_pool(..., sonic_prototype: Optional[np.ndarray] = None, sonic_blend: float = 0.35)` — blends `sonic_prototype` (built in `X_sonic` space) into each seed's sonic admission vector.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_tag_steering_sonic_prototype_live.py  (add)
import inspect
from src.playlist.candidate_pool import build_candidate_pool


def test_build_candidate_pool_accepts_sonic_prototype():
    sig = inspect.signature(build_candidate_pool)
    assert "sonic_prototype" in sig.parameters
    assert "sonic_blend" in sig.parameters


def test_sonic_prototype_blend_shifts_admitted_sonic_affinity(bundle):
    """Synthetic seeds: blending an ambient prototype into the sonic centroid
    raises the admitted pool's mean affinity to that prototype."""
    if getattr(bundle, "X_sonic_muq", None) is None:
        pytest.skip("MuQ absent")
    from src.playlist.config import CandidatePoolConfig
    xmq = np.asarray(bundle.X_sonic_muq, dtype=np.float64)
    t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        ["ambient"], metadata_db_path=DB, track_id_to_row=t2r, min_support=25)
    gm = sonic_global_mean(xmq)
    proto, _, _ = sonic_prototype_from_rows(xmq, rows, global_mean=gm)
    xmn = xmq / (np.linalg.norm(xmq, axis=1, keepdims=True) + 1e-12)
    aff = (xmn - gm) @ proto

    cfg = CandidatePoolConfig()   # production defaults
    seed = int(np.argmax(aff))    # a strongly-ambient seed
    common = dict(seed_idx=seed, seed_indices=[seed], embedding=xmq,
                  artist_keys=bundle.artist_keys, track_ids=bundle.track_ids,
                  cfg=cfg, random_seed=0, X_sonic=xmq)
    base = build_candidate_pool(**common)
    steered = build_candidate_pool(**common, sonic_prototype=proto, sonic_blend=0.5)

    def mean_aff(res):
        ids = [t2r[str(t)] for t in res.candidate_track_ids if str(t) in t2r]
        return float(np.mean([aff[i] for i in ids])) if ids else 0.0
    assert mean_aff(steered) >= mean_aff(base)
```

(If `CandidatePoolResult`'s pool attribute is not `candidate_track_ids`, adjust to the actual field — check `CandidatePoolResult` in `candidate_pool.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/integration/test_tag_steering_sonic_prototype_live.py -k "accepts_sonic_prototype or blend_shifts" -v`
Expected: FAIL — `sonic_prototype` not in signature.

- [ ] **Step 3a: Add params + blend into the sonic seed vectors**

In `src/playlist/candidate_pool.py`, add to the signature after `steering_blend: float = 0.5,` (~L564):

```python
    # Tag steering (sonic): blend a library-learned sonic prototype into each
    # seed's sonic admission vector. None = off (byte-identical legacy behavior).
    sonic_prototype: Optional[np.ndarray] = None,
    sonic_blend: float = 0.35,
```

Then in the sonic block (~L647–652), steer the seed sonic vectors before computing similarity:

```python
    if X_sonic is not None:
        sonic_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
        seed_vecs_sonic = sonic_norm[seed_list]
        if sonic_prototype is not None:
            _sb = float(np.clip(sonic_blend, 0.0, 1.0))
            _proto = np.asarray(sonic_prototype, dtype=seed_vecs_sonic.dtype)
            seed_vecs_sonic = (1.0 - _sb) * seed_vecs_sonic + _sb * _proto
            _n = np.linalg.norm(seed_vecs_sonic, axis=1, keepdims=True)
            seed_vecs_sonic = seed_vecs_sonic / (_n + 1e-12)
            logger.info("Tag steering sonic pool lever: blend=%.2f applied to seed sonic vectors", _sb)
        sonic_seed_sim_matrix = np.dot(sonic_norm, seed_vecs_sonic.T)
        sonic_seed_sim = np.max(sonic_seed_sim_matrix, axis=1)
        sonic_seed_sim[seed_mask] = -1.0
```

Add an audit line after the admitted set is known (near the end of the function, mirroring the genre lever's affinity log). If a convenient admitted-set is available as a boolean mask `eligible_mask`, emit:

```python
        if sonic_prototype is not None:
            _padm = ((sonic_norm - 0.0) @ np.asarray(sonic_prototype, dtype=np.float64))
            _sel = _padm[eligible_mask] if 'eligible_mask' in dir() else _padm
            if _sel.size:
                logger.info(
                    "Tag steering sonic pool affinity (admitted set): p10=%.3f p50=%.3f p90=%.3f n=%d",
                    float(np.percentile(_sel, 10)), float(np.percentile(_sel, 50)),
                    float(np.percentile(_sel, 90)), int(_sel.size),
                )
```

(Place the audit line where the eligible/admitted list is finalized; use the actual admitted-index array in scope there.)

- [ ] **Step 3b: Resolve + wire in `pipeline/core.py`**

After the genre target resolution (~L563–569), add sonic prototype resolution:

```python
    _tag_sonic_prototype = None
    try:
        _tag_sonic_blend = float(pb_overrides.get("tag_steering_sonic_blend", 0.35))
    except (TypeError, ValueError):
        _tag_sonic_blend = 0.35
    if _tag_steering_tags:
        _xsonic = embedding.X_sonic_for_embed
        _xmuq = getattr(bundle, "X_sonic_muq", None)
        if _xsonic is not None and _xmuq is not None:
            from src.playlist.tag_steering import (
                resolve_tag_sonic_prototype_rows, sonic_prototype_from_rows, sonic_global_mean,
            )
            _min_support = int(pb_overrides.get("tag_steering_prototype_min_support", 25))
            _t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
            _rows, _n, _ = resolve_tag_sonic_prototype_rows(
                _tag_steering_tags, metadata_db_path=_meta_db,
                track_id_to_row=_t2r, min_support=_min_support,
            )
            if _rows is not None:
                _xs = np.asarray(_xsonic, dtype=np.float64)
                _gm = sonic_global_mean(_xs)
                _tag_sonic_prototype, _coh, _ = sonic_prototype_from_rows(_xs, _rows, global_mean=_gm)
                _min_coh = float(pb_overrides.get("tag_steering_prototype_min_cohesion", 0.15))
                if _tag_sonic_prototype is not None and _coh < _min_coh:
                    logger.warning(
                        "Tag steering sonic pool prototype: cohesion %.3f < %.2f — sonic pool "
                        "lever disabled (genre pool lever unaffected).", _coh, _min_coh,
                    )
                    _tag_sonic_prototype = None
```

(`_meta_db` is already computed at ~L581 — move that computation above this block, or reference it after; ensure it is defined before use.)

Pass through in the `build_candidate_pool(...)` call (~L627, after `steering_blend=_tag_steering_blend,`):

```python
            sonic_prototype=_tag_sonic_prototype,
            sonic_blend=_tag_sonic_blend,
```

Note: `build_candidate_pool`'s sonic prototype is built from `X_sonic` here (the pool's own space), while Task 3's pier prototype is built from `X_sonic_muq`. This is intentional — each prototype lives in the space it is scored against (Global Constraints: space-consistency).

- [ ] **Step 3c: Add the pool knob to `config.example.yaml`**

```yaml
      tag_steering_sonic_blend: 0.35          # blend of the sonic prototype into the sonic-admission centroid (pool)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/integration/test_tag_steering_sonic_prototype_live.py -k "accepts_sonic_prototype or blend_shifts" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py config.example.yaml tests/integration/test_tag_steering_sonic_prototype_live.py
git commit --only -- src/playlist/candidate_pool.py src/playlist/pipeline/core.py config.example.yaml tests/integration/test_tag_steering_sonic_prototype_live.py -m "feat(tag-steering): sonic candidate-pool lever"
```

---

### Task 5: End-to-end validation, calibration, and docs

**Files:**
- Create: `scripts/research/tag_steering_sonic_eval.py` (diagnostic; not shipped in the warm path)
- Modify: `docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md` (append the sonic-prototype recipe)
- Modify: `config.example.yaml` (only if calibration moves a default)

**Interfaces:**
- Consumes: all prior tasks + the real artist path (`PlaylistApp.create_playlist_for_artist`).

- [ ] **Step 1: Write the end-to-end eval harness**

Adapt `scratchpad/steer_diag.py` (baseline vs ambient vs art-rock Brian Eno + Real Estate jangle/dream) into `scripts/research/tag_steering_sonic_eval.py`, running each through `PlaylistApp` with a temp config that sets `tag_steering_tags` (+ overrides the new knobs), at INFO. Emit for each run: the tag-steering log lines, and per-track sonic-prototype affinity of the realized playlist (map result track_ids via the artifact `track_ids`).

- [ ] **Step 2: Run the acceptance comparison (read the logs, per playlist-testing)**

Run: `python scripts/research/tag_steering_sonic_eval.py 2>&1 | tee scratchpad/sonic_eval.txt`
Acceptance criteria (compare sonic-on vs the genre-dense-only baseline captured pre-Task-3):
- Eno ambient: mean **pier** sonic-ambient affinity materially above baseline (baseline piers were `[0.32, 0.205, 0.205, -0.009]`); the forced off-pole pier (−0.009) should rise or be displaced.
- Eno ambient: admitted-**pool** sonic affinity p50 clearly above the genre-lever baseline (`p50=0.079`).
- Real Estate jangle: realized playlist de-selects the "Sting"-type beatless interludes as piers.
- **No-tag** run is byte-identical to a pre-change no-tag run (same track_ids).

- [ ] **Step 3: Calibrate the two weights**

Sweep `tag_steering_sonic_weight ∈ {0.35, 0.5, 0.7}` and `tag_steering_sonic_blend ∈ {0.25, 0.35, 0.5}`. Pick the pair that maximizes on-tag lean while keeping the worst-edge (min-T) and repair count within one notch of baseline (report min/p10/p50/p90 of edge T, per the eval reporting rules — distributions, not means). Honor the "sonic leads" lean: prefer the lower `sonic_blend` when two are within noise. If a default moves, update `config.example.yaml`.

- [ ] **Step 4: Full fast suite + write the tuning recipe**

Run: `python -m pytest -q -m "not slow"`
Expected: PASS (no regressions). Quote the real pass/fail counts.
Then append to `docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md`: the three knobs, their calibrated values, the acceptance numbers, and the cohesion-guard fallback behavior.

- [ ] **Step 5: Commit**

```bash
git add scripts/research/tag_steering_sonic_eval.py docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md config.example.yaml
git commit --only -- scripts/research/tag_steering_sonic_eval.py docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md config.example.yaml -m "feat(tag-steering): sonic-prototype e2e eval + calibrated tuning recipe"
```

---

## Self-Review

**Spec coverage:**
- Prototype resolver (support guard + warn, seed-artist exclusion, centering, multi-tag union) → Tasks 1–2. ✓
- Combined additive signal at piers → Task 3. ✓
- Sonic pool lever mirroring the genre lever → Task 4. ✓
- New knobs (`sonic_weight`, `sonic_blend`, `prototype_min_support`) → Tasks 3–4 config edits. ✓
- Byte-identical no-tag → guarded in every consumer + explicit test (Task 3 Step 1; Task 5 Step 2). ✓
- Centering null-bias check → Task 5 Step 3 (sweep) + the `global_mean` path (Task 1). ✓
- Validation via real artist path → Task 5. ✓
- Cohesion guard (multimodal-tag fallback) → resolver returns cohesion (Task 1); the `tag_steering_prototype_min_cohesion` (0.15) floor *disables* the sonic term + warns in both consumers (Task 3 Step 3b/3c pier; Task 4 Step 3b pool). ✓ (gap closed inline)

**Placeholder scan:** No TBD/TODO; all code steps show real code. Two "adjust to actual field" notes (CandidatePoolResult pool attribute; admitted-set mask) are explicit verification instructions, not placeholders — the implementer confirms the exact attribute name in `candidate_pool.py` before wiring.

**Type consistency:** `sonic_tag_affinity` is a bundle-aligned `(N,)` float vector everywhere; `sonic_prototype` is a `(D,)` unit vector in the consumer's sonic space; both resolver helpers keep the `(value|None, cohesion, support_n)` / `(rows|None, support_n, tags)` shapes across tasks. `resolve_tag_sonic_prototype_rows` / `sonic_prototype_from_rows` / `sonic_global_mean` names match between definition (Tasks 1–2) and use (Tasks 3–4). ✓
