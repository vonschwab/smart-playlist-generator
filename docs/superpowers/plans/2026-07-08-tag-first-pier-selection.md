# Tag-First Pier Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In artist mode with a genre/tag selected, choose the piers (seeds) from the artist's authority-defined on-tag tracks instead of sonic-cluster-first medoids, so the piers are the artist's most on-tag tracks.

**Architecture:** A new authority read returns the seed artist's on-tag tracks (union over selected genres, with a multi-tag hit count). A pure helper builds the on-tag member set `M` (members + soft top-up by combined affinity; empty → `None` = legacy fallback). `cluster_artist_tracks` gains a `restrict_to_track_ids` param so all existing pier machinery (bridgeability veto, clustering, medoids, tag-skew allocation, arc ordering) runs over `M`. A three-mode dispatch in `playlist_generator.py` routes OFF → tag-first arc over `M`, ON → most-popular-within-`M`, Fire → unchanged.

**Tech Stack:** Python 3.11, numpy, sqlite3, pytest. Reads the published genre authority via `src/genre/authority.py`.

**Design + findings:** `docs/superpowers/specs/2026-07-08-tag-first-pier-selection.md` (this plan implements it); `docs/superpowers/specs/2026-07-08-genre-mode-design-notes.md` (finding F2 is the bug).

## Global Constraints

- **Genre authority One Rule:** "on-tag" is read ONLY from `release_effective_genres` via `src/genre/authority.py`. Match the chip source `resolved_genres_for_artist` exactly: `assignment_layer IN ('observed_leaf','legacy')` and artist match `LOWER(TRIM(artist)) = LOWER(TRIM(?))`. Never read `track_effective_genres`/`*_genres`.
- **Multi-tag = union** across selected genres; ranking rewards multi-tag hits.
- **Live default (#22):** `tag_first_pier_selection` defaults `true`; keep a config rollback to the legacy path. Never merge inactive.
- **No silent no-op:** if the knob is on but `steering_target`/`X_genre_dense` is absent at runtime, take the legacy path and log at WARNING.
- **Tests mirror production:** all generation tests go through `tests/support/gui_fidelity.py::generate_like_gui` with multi-pier seeds (playlist-testing skill). Never hand-built overrides, never single-seed.
- **Sub-agent models:** dispatch Tasks 1, 3 on a cheap model (mechanical); Tasks 2, 4, 5, 6 on a standard model (judgment/integration). Never inherit the session model.
- **Bit-identical when off:** `restrict_to_track_ids=None` and `tag_first_pier_selection=false` must leave today's behavior byte-identical.

---

## File Structure

- `src/genre/authority.py` — add `on_tag_track_ids_for_artist(conn, artist_name, genre_ids)`. Pure authority read.
- `src/playlist/tag_steering.py` — factor the chip→canonical-id mapping into `_canonical_genre_ids_for_tags(con, tags)`; reuse it in `resolve_tag_sonic_prototype_rows`; add `resolve_artist_on_tag_membership(...)` and the pure `build_tag_first_pier_members(...)`.
- `src/playlist/artist_style.py` — add `restrict_to_track_ids` param to `cluster_artist_tracks`.
- `src/playlist_generator.py` — compute `M`, restructure the pier dispatch (~1952–2025).
- `config.example.yaml` — document `tag_first_pier_selection`, `tag_first_topup_mult`.
- `tests/unit/test_tag_first_piers.py` — new unit tests (Tasks 1–4).
- `tests/integration/test_gui_fidelity_regressions.py` — integration cases (Task 6).

---

### Task 1: Authority read — `on_tag_track_ids_for_artist`

**Files:**
- Modify: `src/genre/authority.py` (add function after `resolved_genres_for_artist`, ~line 207)
- Test: `tests/unit/test_tag_first_piers.py`

**Interfaces:**
- Produces: `on_tag_track_ids_for_artist(conn: sqlite3.Connection, artist_name: str, genre_ids: set[str]) -> dict[str, int]` — `{track_id: hit_count}`, hit_count = number of distinct selected genre_ids the track's album carries. `{}` for empty inputs / unknown artist / missing tables.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tag_first_piers.py
import sqlite3
import pytest
from src.genre.authority import on_tag_track_ids_for_artist


def _db():
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, assignment_layer TEXT);
        INSERT INTO tracks VALUES ('t1','Boards of Canada','alb_h'),
                                  ('t2','Boards of Canada','alb_h'),
                                  ('t3','Boards of Canada','alb_e'),
                                  ('t4','Aphex Twin','alb_a');
        INSERT INTO release_effective_genres VALUES
            ('alb_h','hauntology','observed_leaf'),
            ('alb_h','kosmische','observed_leaf'),
            ('alb_e','electronica','observed_leaf'),
            ('alb_e','hauntology','inferred_family'),  -- inferred: must NOT count
            ('alb_a','hauntology','observed_leaf');
        """
    )
    return c


def test_on_tag_union_and_hitcount():
    c = _db()
    m = on_tag_track_ids_for_artist(c, "boards of canada", {"hauntology", "kosmische"})
    assert m == {"t1": 2, "t2": 2}          # union; both tags hit; t3 inferred-only excluded


def test_on_tag_single_genre_and_case_insensitive_artist():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "BOARDS OF CANADA", {"hauntology"}) == {"t1": 1, "t2": 1}


def test_on_tag_empty_inputs():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "Boards of Canada", set()) == {}
    assert on_tag_track_ids_for_artist(c, "Nobody", {"hauntology"}) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: FAIL (ImportError: cannot import name `on_tag_track_ids_for_artist`).

- [ ] **Step 3: Implement**

```python
# src/genre/authority.py
def on_tag_track_ids_for_artist(
    conn: sqlite3.Connection, artist_name: str, genre_ids: set[str]
) -> dict[str, int]:
    """The seed artist's tracks whose album is published (observed_leaf/legacy) with ANY
    of ``genre_ids``, mapped to the count of distinct selected genres that album carries
    (for multi-tag ranking). Union semantics. Same layer + artist-match filter as
    ``resolved_genres_for_artist`` (the chip source), so 'on-tag' == 'would show this chip'.
    Returns {} for empty inputs / unknown artist / absent tables — callers fall back, never crash.
    """
    name = (artist_name or "").strip()
    gids = {str(g) for g in (genre_ids or set()) if str(g)}
    if not name or not gids:
        return {}
    ph = ",".join("?" for _ in gids)
    try:
        rows = conn.execute(
            f"SELECT t.track_id, COUNT(DISTINCT reg.genre_id) AS hits "
            f"FROM tracks t JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            f"WHERE LOWER(TRIM(t.artist)) = LOWER(TRIM(?)) "
            f"  AND reg.genre_id IN ({ph}) "
            f"  AND reg.assignment_layer IN ('observed_leaf', 'legacy') "
            f"GROUP BY t.track_id",
            (name, *gids),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): int(r[1]) for r in rows}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/genre/authority.py tests/unit/test_tag_first_piers.py
git commit --only -- src/genre/authority.py tests/unit/test_tag_first_piers.py -m "feat(genre): authority read for an artist's on-tag tracks (union + hit count)"
```

---

### Task 2: Chip→id mapping refactor + `resolve_artist_on_tag_membership`

**Files:**
- Modify: `src/playlist/tag_steering.py`
- Test: `tests/unit/test_tag_first_piers.py`

**Interfaces:**
- Consumes: `on_tag_track_ids_for_artist` (Task 1).
- Produces:
  - `_canonical_genre_ids_for_tags(con: sqlite3.Connection, tags: Sequence[str]) -> set[str]` — the existing mapping (canonical names/ids + aliases), extracted verbatim from `resolve_tag_sonic_prototype_rows`.
  - `resolve_artist_on_tag_membership(tags, artist_name, *, metadata_db_path, track_id_to_row) -> dict[int, int]` — `{bundle_row: hit_count}`, seed-INCLUDED. `{}` when nothing maps or no on-tag tracks.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_tag_first_piers.py`)

```python
def _authority_db_with_taxonomy(tmp_path):
    import sqlite3
    p = tmp_path / "meta.db"
    c = sqlite3.connect(p)
    c.executescript(
        """
        CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, assignment_layer TEXT);
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        CREATE TABLE genre_graph_aliases (alias TEXT, canonical_genre_id TEXT);
        INSERT INTO tracks VALUES ('t1','Boards of Canada','alb_h'),('t2','Boards of Canada','alb_h');
        INSERT INTO release_effective_genres VALUES ('alb_h','hauntology','observed_leaf');
        INSERT INTO genre_graph_canonical_genres VALUES ('hauntology','hauntology');
        """
    )
    c.commit(); c.close()
    return str(p)


def test_resolve_artist_on_tag_membership(tmp_path):
    from src.playlist.tag_steering import resolve_artist_on_tag_membership
    dbp = _authority_db_with_taxonomy(tmp_path)
    t2r = {"t1": 5, "t2": 9, "other": 3}
    m = resolve_artist_on_tag_membership(
        ["hauntology"], "Boards of Canada", metadata_db_path=dbp, track_id_to_row=t2r
    )
    assert m == {5: 1, 9: 1}


def test_resolve_artist_on_tag_membership_unmapped(tmp_path):
    from src.playlist.tag_steering import resolve_artist_on_tag_membership
    dbp = _authority_db_with_taxonomy(tmp_path)
    assert resolve_artist_on_tag_membership(
        ["not_a_genre"], "Boards of Canada", metadata_db_path=dbp, track_id_to_row={"t1": 0}
    ) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: FAIL (ImportError: `resolve_artist_on_tag_membership`).

- [ ] **Step 3: Implement** — extract the mapping (currently `resolve_tag_sonic_prototype_rows` lines ~143–156) into a shared helper, call it from both functions:

```python
# src/playlist/tag_steering.py
def _canonical_genre_ids_for_tags(con: "sqlite3.Connection", tags: Sequence[str]) -> set:
    """Chip names/ids (space or underscore form) -> canonical genre_ids via
    genre_graph_canonical_genres + genre_graph_aliases. {} if none map."""
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return set()
    lower = [t.lower() for t in wanted]
    under = [t.lower().replace(" ", "_") for t in wanted]
    ph = ",".join("?" for _ in wanted)
    cur = con.cursor()
    gids: set = set()
    cur.execute(
        f"SELECT genre_id FROM genre_graph_canonical_genres "
        f"WHERE lower(name) IN ({ph}) OR lower(genre_id) IN ({ph})",
        lower + under,
    )
    gids.update(str(r[0]) for r in cur.fetchall())
    cur.execute(
        f"SELECT canonical_genre_id FROM genre_graph_aliases WHERE lower(alias) IN ({ph})",
        lower,
    )
    gids.update(str(r[0]) for r in cur.fetchall())
    return gids


def resolve_artist_on_tag_membership(
    tags: Sequence[str],
    artist_name: str,
    *,
    metadata_db_path: str,
    track_id_to_row: dict,
) -> dict:
    """{bundle_row: tag_hit_count} for the SEED artist's authority on-tag tracks
    (seed-included; union over the selected tags). {} when nothing maps or no on-tag
    tracks. Reads the authority via on_tag_track_ids_for_artist (One Rule)."""
    from src.genre.authority import on_tag_track_ids_for_artist
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return {}
    con = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        gids = _canonical_genre_ids_for_tags(con, wanted)
        if not gids:
            logger.warning(
                "Tag-first piers: none of %s map to a canonical genre — legacy pier "
                "selection for this run.", wanted,
            )
            return {}
        hits = on_tag_track_ids_for_artist(con, artist_name, gids)
    finally:
        con.close()
    return {track_id_to_row[t]: int(n) for t, n in hits.items() if t in track_id_to_row}
```

Then replace the inline mapping block in `resolve_tag_sonic_prototype_rows` with `gids = _canonical_genre_ids_for_tags(con, wanted)` (keep its existing empty-`gids` WARN + return). Verify no behavior change.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_tag_first_piers.py tests/unit/test_tag_steering.py -q`
Expected: PASS (new tests + the existing 16 resolver tests still green after the refactor).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/tag_steering.py tests/unit/test_tag_first_piers.py
git commit --only -- src/playlist/tag_steering.py tests/unit/test_tag_first_piers.py -m "feat(tag-steering): artist on-tag membership resolver + shared chip->id mapping"
```

---

### Task 3: `restrict_to_track_ids` param on `cluster_artist_tracks`

**Files:**
- Modify: `src/playlist/artist_style.py` (`cluster_artist_tracks`, signature ~688, filter after the `excluded_track_ids` block ~716–727)
- Test: `tests/unit/test_tag_first_piers.py`

**Interfaces:**
- Produces: `cluster_artist_tracks(..., restrict_to_track_ids: Optional[set[str]] = None)`. When provided, only artist tracks whose `track_id` is in the set are clustered. `None` = unchanged.

- [ ] **Step 1: Write the failing test** — use the existing artist-style unit fixtures. Add a small synthetic-bundle test:

```python
def test_cluster_restrict_to_track_ids_subsets_members(monkeypatch):
    import numpy as np
    from src.playlist import artist_style
    from src.playlist.artist_style import cluster_artist_tracks, ArtistStyleConfig

    class B:
        track_ids = np.array([f"t{i}" for i in range(12)])
        artist_keys = np.array(["boc"] * 12)
        X_sonic = np.random.default_rng(0).normal(size=(12, 8))
        durations_ms = np.array([200000] * 12)
        track_titles = np.array([f"Title {i}" for i in range(12)])
    monkeypatch.setattr(artist_style, "_artist_indices_in_bundle",
                        lambda bundle, name, include_collaborations=False: list(range(12)))
    keep = {f"t{i}" for i in range(6)}
    cfg = ArtistStyleConfig(pier_bridgeability_enabled=False, dedupe_versions=False)
    clusters, medoids, _mbc, _xn = cluster_artist_tracks(
        bundle=B(), artist_name="boc", cfg=cfg, random_seed=0, restrict_to_track_ids=keep,
    )
    picked = {str(B.track_ids[i]) for c in clusters for i in c}
    assert picked <= keep and picked  # only restricted members clustered
```

(If `ArtistStyleConfig` field names differ, read the dataclass and set the two flags that disable bridgeability + dedupe. Keep the test to the restriction behavior only.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_tag_first_piers.py::test_cluster_restrict_to_track_ids_subsets_members -q`
Expected: FAIL (unexpected keyword argument `restrict_to_track_ids`).

- [ ] **Step 3: Implement** — add the param and the intersection. In the signature (~703) add `restrict_to_track_ids: Optional[set[str]] = None`. After the `excluded_track_ids` block (~727), insert:

```python
    if restrict_to_track_ids is not None:
        before = len(artist_indices)
        keep = {str(tid) for tid in restrict_to_track_ids}
        artist_indices = [i for i in artist_indices if str(bundle.track_ids[i]) in keep]
        logger.info(
            "Tag-first piers: restricted clustering to %d/%d on-tag member(s) of %s",
            len(artist_indices), before, artist_name,
        )
```

(The existing `len(artist_indices) < max(3, cfg.cluster_k_min)` guard already handles an over-restricted set by raising — the caller's top-up floor prevents that in practice.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/artist_style.py tests/unit/test_tag_first_piers.py
git commit --only -- src/playlist/artist_style.py tests/unit/test_tag_first_piers.py -m "feat(artist-style): restrict_to_track_ids param on cluster_artist_tracks"
```

---

### Task 4: Pure `build_tag_first_pier_members` (member set + top-up)

**Files:**
- Modify: `src/playlist/tag_steering.py`
- Test: `tests/unit/test_tag_first_piers.py`

**Interfaces:**
- Produces: `build_tag_first_pier_members(membership: dict, combined_affinity, artist_indices: Sequence[int], *, target_pier_count: int, cluster_k_min: int, topup_mult: float) -> Optional[set]`. Returns `M` (bundle indices) or `None` when `membership` is empty (legacy fallback signal).

- [ ] **Step 1: Write the failing test**

```python
def test_build_members_empty_is_none():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    assert build_tag_first_pier_members(
        {}, np.zeros(10), list(range(10)), target_pier_count=4, cluster_k_min=3, topup_mult=2.0
    ) is None


def test_build_members_large_set_unchanged():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {i: 1 for i in range(18)}          # 18 on-tag >> floor
    M = build_tag_first_pier_members(
        membership, np.zeros(30), list(range(30)),
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert M == set(range(18))


def test_build_members_topup_by_affinity():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {0: 1, 1: 1}                        # 2 members, floor = max(3, ceil(2*4)) = 8
    aff = np.full(20, -9.0); aff[0] = aff[1] = 5.0
    aff[7] = 4.0; aff[3] = 3.0; aff[5] = 2.0; aff[9] = 1.0; aff[2] = 0.5; aff[8] = 0.4
    M = build_tag_first_pier_members(
        membership, aff, list(range(20)),
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert {0, 1} <= M and len(M) == 8              # topped up to floor
    assert M == {0, 1, 7, 3, 5, 9, 2, 8}            # highest-affinity non-members added


def test_build_members_floor_capped_at_artist_count():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {0: 1}
    M = build_tag_first_pier_members(
        membership, np.array([5.0, 4.0, 3.0]), [0, 1, 2],
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert M == {0, 1, 2}                            # floor capped at 3 available tracks
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: FAIL (ImportError: `build_tag_first_pier_members`).

- [ ] **Step 3: Implement**

```python
import math

def build_tag_first_pier_members(
    membership: dict,
    combined_affinity,
    artist_indices: Sequence[int],
    *,
    target_pier_count: int,
    cluster_k_min: int,
    topup_mult: float,
) -> Optional[set]:
    """On-tag pier member set M (bundle indices) or None (=> legacy fallback).

    members = keys(membership) (authority on-tag, seed-included). Empty -> None: we go
    tag-first ONLY when the artist actually has authority on-tag tracks, never fabricate
    membership from a sonic proxy. If len(members) < floor, top up with the artist's
    highest-combined-affinity NON-member tracks. floor = min(len(artist_indices),
    max(cluster_k_min, ceil(topup_mult * target_pier_count)))."""
    members = set(int(i) for i in membership.keys())
    if not members:
        return None
    artist_set = [int(i) for i in artist_indices]
    floor = min(len(artist_set), max(int(cluster_k_min), math.ceil(float(topup_mult) * int(target_pier_count))))
    if len(members) >= floor:
        return members
    aff = np.asarray(combined_affinity, dtype=np.float64)
    candidates = [i for i in artist_set if i not in members]
    candidates.sort(key=lambda i: (-float(aff[i]), i))   # highest affinity first, index tiebreak
    for i in candidates:
        if len(members) >= floor:
            break
        members.add(i)
    return members
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_tag_first_piers.py -q`
Expected: PASS (all Task 1–4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/tag_steering.py tests/unit/test_tag_first_piers.py
git commit --only -- src/playlist/tag_steering.py tests/unit/test_tag_first_piers.py -m "feat(tag-steering): build_tag_first_pier_members (member set + soft top-up)"
```

---

### Task 5: Wire the dispatch + config knobs

**Files:**
- Modify: `src/playlist_generator.py` (~1856–2025 pier block)
- Modify: `config.example.yaml` (~333, next to `pier_tag_skew`)
- Test: covered by Task 6 (integration). Add no unit test here.

**Interfaces:**
- Consumes: `resolve_artist_on_tag_membership`, `build_tag_first_pier_members` (Tasks 2, 4); `cluster_artist_tracks(restrict_to_track_ids=...)` (Task 3); existing `select_popular_piers`, `allocate_piers_by_tag_affinity`, `order_clusters`.

- [ ] **Step 1: Add config knobs** (`config.example.yaml`, under `pier_bridge:`, beside `pier_tag_skew`):

```yaml
      tag_first_pier_selection: true   # steer piers from the artist's authority on-tag tracks (rollback: false = legacy tag-skew allocation)
      tag_first_topup_mult: 2.0        # top-up floor = max(cluster_k_min, ceil(mult * target_pier_count)) on-tag candidates before topping up by affinity
```

- [ ] **Step 2: Compute `M` before `cluster_artist_tracks`** — insert immediately before the `clusters, medoids, ... = cluster_artist_tracks(` call (~1952). `_t2r`, `_steering_tag_list`, `steering_target`, `sonic_tag_affinity`, `sonic_tag_weight`, `target_pier_count`, `style_cfg`, `_pb` are all already in scope above.

```python
                # --- Tag-first pier member set M (authority on-tag; None => legacy) ---
                _M_ids = None       # track_ids to restrict clustering to (None = legacy full-artist)
                _tag_first_on = bool(_pb.get("tag_first_pier_selection", True))
                _xgd_bundle = getattr(bundle, "X_genre_dense", None)
                if (
                    _tag_first_on
                    and steering_target is not None
                    and _xgd_bundle is not None
                    and popular_seeds_mode != "fire"
                ):
                    from src.playlist.tag_steering import (
                        resolve_artist_on_tag_membership,
                        build_tag_first_pier_members,
                    )
                    _membership = resolve_artist_on_tag_membership(
                        _steering_tag_list, artist_name,
                        metadata_db_path=resolve_database_path(self.config),
                        track_id_to_row=_t2r,
                    )
                    if _membership:
                        _combined = np.asarray(_xgd_bundle, dtype=np.float64) @ np.asarray(
                            steering_target, dtype=np.float64)
                        if sonic_tag_affinity is not None:
                            _combined = _combined + float(sonic_tag_weight) * np.asarray(
                                sonic_tag_affinity, dtype=np.float64)
                        _all_artist_idx = _artist_indices_in_bundle(
                            bundle, artist_name, include_collaborations=include_collaborations)
                        _M = build_tag_first_pier_members(
                            _membership, _combined, _all_artist_idx,
                            target_pier_count=target_pier_count,
                            cluster_k_min=int(style_cfg.cluster_k_min),
                            topup_mult=float(_pb.get("tag_first_topup_mult", 2.0)),
                        )
                        if _M is not None:
                            _M_ids = {str(bundle.track_ids[i]) for i in _M}
                            logger.info(
                                "Tag-first piers: %d authority on-tag member(s) (+top-up to %d) for %s",
                                len(_membership), len(_M), artist_name,
                            )
                    else:
                        logger.info(
                            "Tag-first piers: %s has no authority on-tag tracks for %s — "
                            "legacy tag-skew pier selection.", artist_name, _steering_tag_list,
                        )
                elif _tag_first_on and steering_target is not None and _xgd_bundle is None:
                    logger.warning(
                        "Tag-first piers enabled but X_genre_dense absent — legacy pier selection.")
```

- [ ] **Step 3: Pass `restrict_to_track_ids` to the cluster call** — in the existing `cluster_artist_tracks(...)` call (~1952), add `restrict_to_track_ids=_M_ids,`.

- [ ] **Step 4: Restructure the post-cluster dispatch** — replace the block from the `# 🔥 Pure-hits piers` comment (~1967) through the tag-allocation `if/else` (~2025) with the four-way dispatch. `clusters` already contains only `M` members when `_M_ids` is set, so popular/allocation over `clusters` is automatically "within M":

```python
                if not medoids:
                    raise ValueError("Style clustering returned no medoids")
                _all_members = [i for _c in clusters for i in _c]
                _xgd = getattr(bundle, "X_genre_dense", None)
                _on_within_tag = (
                    _M_ids is not None and popular_seeds_mode == "on"
                    and popularity_values is not None
                )
                if popular_seeds_mode == "fire" and popularity_values is not None:
                    _fire = select_popular_piers(_all_members, popularity_values, target_pier_count)
                    if _fire:
                        logger.info("Popular Seeds 🔥: overriding %d medoid piers with top-%d popular",
                                    len(medoids), len(_fire))
                        medoids = _fire
                    else:
                        logger.warning("Popular Seeds 🔥: no popular piers (uncached?) — medoid piers")
                    ordered_medoids = _cap_order(medoids, X_norm, target_pier_count)
                elif _on_within_tag:
                    _pop = select_popular_piers(_all_members, popularity_values, target_pier_count)
                    if _pop:
                        logger.info("Tag-first piers (ON): top-%d popular WITHIN %d on-tag members",
                                    len(_pop), len(_all_members))
                        medoids = _pop
                    ordered_medoids = _cap_order(medoids, X_norm, target_pier_count)
                elif steering_target is not None and _xgd is not None:
                    _xgd = np.asarray(_xgd, dtype=float)
                    _tgt = np.asarray(steering_target, dtype=float)
                    cluster_affinities = [
                        (float(np.mean(_xgd[members] @ _tgt))
                         + (sonic_tag_weight * float(np.mean(sonic_tag_affinity[members]))
                            if sonic_tag_affinity is not None else 0.0))
                        if len(members) else 0.0
                        for members in clusters
                    ]
                    pier_tag_skew = float(_pb.get("pier_tag_skew", 0.6))
                    selected = allocate_piers_by_tag_affinity(
                        medoids_by_cluster, cluster_affinities, target_pier_count, pier_tag_skew)
                    ordered_medoids = order_clusters(selected, X_norm)
                    logger.info("Tag steering pier allocation: skew=%.2f cluster_affinities=%s selected=%d/%d",
                                pier_tag_skew, [round(a, 3) for a in cluster_affinities],
                                len(selected), len(medoids))
                else:
                    ordered_medoids = _cap_order(medoids, X_norm, target_pier_count)
```

Add a tiny local helper near the top of the method (or module-level) to DRY the cap+order used by three branches (preserving today's cap log):

```python
def _cap_order(medoids, X_norm, target):
    ordered = order_clusters(medoids, X_norm)
    if len(ordered) > target:
        logger.info("Capping medoids from %d to target_pier_count=%d", len(ordered), target)
        ordered = ordered[:target]
    return ordered
```

- [ ] **Step 5: Verify off-path byte-identical** — with `tag_first_pier_selection: false`, `_M_ids` stays `None`, `restrict_to_track_ids=None`, and the dispatch collapses to the original fire / allocation / else paths. Sanity-run:

Run: `python -m pytest tests/test_gui_fidelity.py -q`
Expected: PASS (fast fidelity guards unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/playlist_generator.py config.example.yaml
git commit --only -- src/playlist_generator.py config.example.yaml -m "feat(tag-steering): tag-first pier dispatch (off/on/fire) wired live-default"
```

---

### Task 6: Integration validation (gui_fidelity harness)

**Files:**
- Modify: `tests/integration/test_gui_fidelity_regressions.py`
- Test: itself (mark `@pytest.mark.integration @pytest.mark.slow`, skip if artifact absent)

**Interfaces:**
- Consumes: the whole feature end-to-end via `generate_like_gui` with an artist seed + `tag_steering_tags`.

- [ ] **Step 1: Write integration cases** — follow the existing file's skip-if-artifact-absent pattern. Assert on-tag pier membership by re-reading the authority for the realized BoC piers (mirror `scratchpad/pier_check.py`). Cases:
  - BoC + `["hauntology"]`, `popular_seeds_mode="off"`: ≥3/4 piers are authority-hauntology BoC tracks (up from 1/4); worst-edge min-T ≥ (no-steer baseline − 0.05).
  - BoC + `["hauntology"]`, `popular_seeds_mode="off"`, `tag_first_pier_selection: false`: reproduces the legacy 1/4 (guards the rollback + that the fix is what changed it).
  - BoC + `["hauntology","kosmische"]` (multi-tag union): piers on-tag; a track hitting both ranks in.
  - Real Estate + `["jangle pop"]`, off: distinct-artist count and worst-edge within one notch of today (M ≈ all RE → no regression).
  - Artist with zero on-tag tracks for the tag: log asserts "no authority on-tag tracks … legacy".

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_gui_fidelity_regressions.py -q -m "integration and not slow" ` (and the slow ones locally with the artifact)
Expected: PASS, or SKIP where the live artifact is unavailable in CI.

- [ ] **Step 3: Manual verification (record results, do not skip)** — run `scratchpad/pier_check.py` (BoC+hauntology) and confirm ≥3/4 piers now authority-hauntology; run one BoC+hauntology and one RE+jangle through the real path and read the INFO log for the "Tag-first piers" lines + worst-edge. Quote the numbers in the task report.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_gui_fidelity_regressions.py
git commit --only -- tests/integration/test_gui_fidelity_regressions.py -m "test(tag-steering): tag-first pier selection integration (BoC/hauntology, multi-tag, RE, fallback)"
```

---

## Self-Review (completed)

- **Spec coverage:** on-tag member set (T1/T2), top-up + empty fallback (T4), restrict clustering (T3), 3-mode dispatch + knobs (T5), multi-tag union + hit count (T1/T2), unchanged bridges (untouched), acceptance/tests (T6). All covered.
- **Placeholders:** none — every code step has real code; the one read-and-adapt note (ArtistStyleConfig flag names in T3) is bounded to two flags.
- **Type consistency:** `membership: dict[bundle_row:int -> hit_count:int]` produced by T2, consumed by T4; `restrict_to_track_ids: set[str]` produced in T5 (`_M_ids`), consumed by T3; `_cap_order` defined once, used in T5.
