# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the actionable code-review findings from the latest push while explicitly deferring the duplicate `pier_bridge/assemble.py` implementation issue.

**Architecture:** Keep the fixes small and local: separate explicit track blacklists from scoped artist/album blacklists, make GUI diversity policy pass through existing caps unless the hard mode is selected, enforce the hard non-seed artist cap inside edge repair, update active docs to current config names, and clean the whitespace gate. Do not edit `src/playlist/pier_bridge/assemble.py` in this plan.

**Tech Stack:** Python 3.11+, SQLite via `sqlite3`, pytest, PySide6 policy models.

---

## Deferred Finding

- Do not fix the duplicate/stale `src/playlist/pier_bridge/assemble.py` implementation in this pass.
- Do not import from it, delete it, or redirect callers to it.
- Leave a follow-up note in the final implementation summary that this finding is intentionally deferred.

## File Structure

- Modify `src/metadata_client.py`: add explicit per-track blacklist persistence and recompute effective `tracks.is_blacklisted` from explicit + artist-scope + album-scope sources.
- Modify `tests/unit/test_blacklist_scopes.py`: add regression coverage for manual track blacklist persistence across artist/album scope removal.
- Modify `src/playlist_gui/policy.py`: only emit `max_non_seed_tracks_per_artist` when the UI selects the hard one-per-artist mode.
- Modify `tests/unit/test_gui_policy.py`: update weighted-mode expectations and add merge regression coverage.
- Modify `src/playlist/repair/edge_repair.py`: refuse repair candidates that would violate `max_non_seed_tracks_per_artist`.
- Modify `src/playlist/pier_bridge_builder.py`: pass the hard cap into edge repair.
- Modify `tests/unit/test_edge_repair.py`: add a repair regression for the hard cap.
- Modify `docs/CONFIG.md`: replace stale `genre_conflict_*` config docs with `genre_compatibility_*`.
- Modify `src/playlist/repair/__init__.py`: remove the extra blank EOF line.

---

### Task 1: Preserve Manual Track Blacklists Across Scoped Unblacklist

**Files:**
- Modify: `src/metadata_client.py`
- Test: `tests/unit/test_blacklist_scopes.py`

- [ ] **Step 1: Add failing blacklist persistence tests**

Append these tests to `tests/unit/test_blacklist_scopes.py`:

```python
def test_artist_scope_removal_preserves_manual_track_blacklist(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "154", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_artist_blacklisted("Wire", True)
    metadata.set_artist_blacklisted("Wire", False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}


def test_album_scope_removal_preserves_manual_track_blacklist(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "Chairs Missing", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_album_blacklisted("Wire", "Chairs Missing", True)
    metadata.set_album_blacklisted("Wire", "Chairs Missing", False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}


def test_manual_unblacklist_under_active_artist_scope_waits_for_scope_removal(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)

    metadata.set_blacklisted(["1"], True)
    metadata.set_artist_blacklisted("Wire", True)
    metadata.set_blacklisted(["1"], False)

    assert metadata.fetch_blacklisted_track_ids() == {"1"}

    metadata.set_artist_blacklisted("Wire", False)
    assert metadata.fetch_blacklisted_track_ids() == set()
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/unit/test_blacklist_scopes.py -q
```

Expected before implementation: at least `test_artist_scope_removal_preserves_manual_track_blacklist` fails because `set_artist_blacklisted(..., False)` clears `tracks.is_blacklisted` for the manually blacklisted track.

- [ ] **Step 3: Add explicit track blacklist storage**

In `src/metadata_client.py`, add a new table in `_init_database()` near the existing scope tables:

```python
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_blacklist (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_blacklist_track_id ON track_blacklist(track_id)")
```

Add these helpers after `_is_scope_blacklisted`:

```python
    def _is_track_blacklisted(self, track_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM track_blacklist WHERE track_id = ?",
            (str(track_id),),
        )
        return cursor.fetchone() is not None

    def _is_effectively_blacklisted(self, track_id: str, artist_key: str, album: str) -> bool:
        return self._is_track_blacklisted(track_id) or self._is_scope_blacklisted(artist_key, album)

    def _backfill_track_blacklist(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(1) AS count FROM track_blacklist")
        row = cursor.fetchone()
        if row is not None and int(row["count"] if "count" in row.keys() else row[0]) > 0:
            return

        cursor.execute(
            """
            SELECT track_id, title, artist, artist_key, album
            FROM tracks
            WHERE is_blacklisted = 1
            """
        )
        rows = cursor.fetchall()
        inserts = []
        for row in rows:
            track_id = str(row["track_id"] or "")
            artist_key = str(row["artist_key"] or "")
            album = str(row["album"] or "")
            if not track_id:
                continue
            if self._is_scope_blacklisted(artist_key, album):
                continue
            inserts.append((track_id, row["title"] or "", row["artist"] or "", album))

        if inserts:
            cursor.executemany(
                """
                INSERT OR IGNORE INTO track_blacklist
                (track_id, title, artist, album, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                inserts,
            )
            self.conn.commit()
```

Call `_backfill_track_blacklist()` after `ensure_blacklist_schema(self.conn, logger=logger)` in `_init_database()`.

- [ ] **Step 4: Recompute effective blacklist state from explicit + scoped sources**

Change `_apply_scoped_blacklist_for_rows()` to use the new effective helper:

```python
            should_blacklist = self._is_effectively_blacklisted(
                str(row["track_id"] or ""),
                str(row["artist_key"] or ""),
                str(row["album"] or ""),
            )
```

Include `track_id` in that helper's row query:

```python
                "SELECT track_id, artist_key, album FROM tracks WHERE rowid = ?",
```

In `add_track()`, compute `is_blacklisted` with the track id:

```python
        is_blacklisted = 1 if self._is_effectively_blacklisted(track_id, artist_key, album or "") else 0
```

In the `ON CONFLICT` update, set the recomputed value directly:

```sql
                is_blacklisted=excluded.is_blacklisted,
```

- [ ] **Step 5: Make `set_blacklisted()` maintain explicit track state**

Replace the body after the empty-list guard with:

```python
        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in track_ids)
        if value:
            cursor.execute(
                f"""
                SELECT track_id, title, artist, album
                FROM tracks
                WHERE track_id IN ({placeholders})
                """,
                tuple(str(t) for t in track_ids),
            )
            rows = cursor.fetchall()
            cursor.executemany(
                """
                INSERT INTO track_blacklist
                (track_id, title, artist, album, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(track_id) DO UPDATE SET
                    title=excluded.title,
                    artist=excluded.artist,
                    album=excluded.album,
                    last_updated=CURRENT_TIMESTAMP
                """,
                [
                    (
                        str(row["track_id"]),
                        row["title"] or "",
                        row["artist"] or "",
                        row["album"] or "",
                    )
                    for row in rows
                ],
            )
        else:
            cursor.execute(
                f"DELETE FROM track_blacklist WHERE track_id IN ({placeholders})",
                tuple(str(t) for t in track_ids),
            )

        cursor.execute(
            f"SELECT rowid FROM tracks WHERE track_id IN ({placeholders})",
            tuple(str(t) for t in track_ids),
        )
        rowids = [int(row["rowid"]) for row in cursor.fetchall()]
        self.conn.commit()
        return self._apply_scoped_blacklist_for_rows(rowids)
```

- [ ] **Step 6: Run focused blacklist tests**

Run:

```bash
pytest tests/unit/test_blacklist_scopes.py -q
```

Expected: all tests in `tests/unit/test_blacklist_scopes.py` pass.

---

### Task 2: Make Weighted GUI Diversity Preserve Existing Hard Caps

**Files:**
- Modify: `src/playlist_gui/policy.py`
- Test: `tests/unit/test_gui_policy.py`

- [ ] **Step 1: Update/add policy tests**

In `tests/unit/test_gui_policy.py`, replace `test_weighted_diversity_clears_pier_bridge_cap` with:

```python
    def test_weighted_diversity_does_not_emit_pier_bridge_cap_override(self):
        state = UIStateModel(artist_diversity_mode="weighted")

        decisions = derive_runtime_config(state)

        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
        ) is None
```

Add this test to `TestArtistDiversity`:

```python
    def test_weighted_diversity_preserves_user_pier_bridge_cap_when_merged(self):
        user_overrides = {
            "playlists": {
                "ds_pipeline": {
                    "pier_bridge": {
                        "max_non_seed_tracks_per_artist": 1,
                    }
                }
            }
        }
        state = UIStateModel(artist_diversity_mode="weighted")

        decisions = derive_runtime_config(state)
        merged = merge_overrides(user_overrides, decisions.overrides)

        assert _get_nested(
            merged,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
        ) == 1
```

- [ ] **Step 2: Run the focused policy tests and confirm failure**

Run:

```bash
pytest tests/unit/test_gui_policy.py::TestArtistDiversity -q
```

Expected before implementation: the merge test fails because weighted mode sets the cap to `None`.

- [ ] **Step 3: Stop owning/clearing the hard cap in weighted mode**

In `src/playlist_gui/policy.py`, remove this entry from `POLICY_OWNED_KEYS`:

```python
    "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
```

Then delete the `else` block in `derive_runtime_config()` that writes `None`:

```python
    else:
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
            None,
        )
```

Keep the `one_per_artist` branch that writes `1`.

- [ ] **Step 4: Run focused policy tests**

Run:

```bash
pytest tests/unit/test_gui_policy.py::TestArtistDiversity -q
```

Expected: `TestArtistDiversity` passes.

---

### Task 3: Make Edge Repair Respect Hard Non-Seed Artist Caps

**Files:**
- Modify: `src/playlist/repair/edge_repair.py`
- Modify: `src/playlist/pier_bridge_builder.py`
- Test: `tests/unit/test_edge_repair.py`

- [ ] **Step 1: Extend the repair test helper**

In `tests/unit/test_edge_repair.py`, update `_repair_bundle()` to accept an optional matrix:

```python
def _repair_bundle(
    *,
    titles: list[str] | None = None,
    artists: list[str] | None = None,
    X: list[list[float]] | None = None,
) -> ArtifactBundle:
    X_arr = np.array(
        X
        if X is not None
        else [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    n = int(X_arr.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array(artists or [f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array(artists or [f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array(titles or [f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_arr,
        X_sonic_start=X_arr,
        X_sonic_mid=X_arr,
        X_sonic_end=X_arr,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )
```

- [ ] **Step 2: Add the hard-cap repair regression test**

Append this test:

```python
def test_edge_repair_refuses_candidate_that_exceeds_non_seed_artist_cap():
    bundle = _repair_bundle(
        X=[
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        artists=["Pier A", "Broken Artist", "Pier B", "Duplicate Artist", "Duplicate Artist"],
    )
    ctx = _context(bundle)

    result = repair_playlist_edges(
        final_indices=[0, 1, 3, 2],
        candidate_indices=[4],
        metric_context=ctx,
        bundle=bundle,
        seed_indices={0, 2},
        pier_positions={0, 3},
        transition_floor=0.2,
        centered_cos_floor=-0.5,
        margin=0.05,
        max_non_seed_tracks_per_artist=1,
    )

    assert result.indices == [0, 1, 3, 2]
    assert "max_non_seed_artist_cap" in {entry["reason"] for entry in result.swap_log}
```

- [ ] **Step 3: Run the focused repair test and confirm failure**

Run:

```bash
pytest tests/unit/test_edge_repair.py::test_edge_repair_refuses_candidate_that_exceeds_non_seed_artist_cap -q
```

Expected before implementation: fails with `TypeError` because `repair_playlist_edges()` does not accept `max_non_seed_tracks_per_artist`.

- [ ] **Step 4: Add cap-aware refusal logic**

In `src/playlist/repair/edge_repair.py`, add `Optional` is already imported. Extend `_candidate_refusal_reasons()` with:

```python
    max_non_seed_tracks_per_artist: Optional[int],
```

Add this helper near `_candidate_refusal_reasons()`:

```python
def _non_seed_artist_counts_after_replacement(
    *,
    candidate: int,
    current_indices: Sequence[int],
    replace_position: int,
    bundle: ArtifactBundle,
    seed_indices: set[int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pos, idx in enumerate(current_indices):
        effective_idx = int(candidate) if int(pos) == int(replace_position) else int(idx)
        if effective_idx in seed_indices:
            continue
        try:
            artist_key = identity_keys_for_index(bundle, effective_idx).artist_key
        except Exception:
            artist_key = ""
        if artist_key:
            counts[str(artist_key)] = counts.get(str(artist_key), 0) + 1
    return counts
```

Inside `_candidate_refusal_reasons()`, after the duplicate/disallowed-artist checks, add:

```python
    if isinstance(max_non_seed_tracks_per_artist, int) and max_non_seed_tracks_per_artist > 0:
        counts = _non_seed_artist_counts_after_replacement(
            candidate=candidate,
            current_indices=current_indices,
            replace_position=replace_position,
            bundle=bundle,
            seed_indices=seed_indices,
        )
        if any(count > int(max_non_seed_tracks_per_artist) for count in counts.values()):
            reasons.append("max_non_seed_artist_cap")
```

Extend `repair_playlist_edges()` signature with:

```python
    max_non_seed_tracks_per_artist: Optional[int] = None,
```

Pass the new argument into `_candidate_refusal_reasons()`.

- [ ] **Step 5: Pass the configured cap from the active builder**

In `src/playlist/pier_bridge_builder.py`, update the `repair_playlist_edges()` call:

```python
            max_non_seed_tracks_per_artist=getattr(cfg, "max_non_seed_tracks_per_artist", None),
```

Do not edit `src/playlist/pier_bridge/assemble.py`.

- [ ] **Step 6: Run focused repair tests**

Run:

```bash
pytest tests/unit/test_edge_repair.py -q
```

Expected: all edge-repair tests pass.

---

### Task 4: Update Active Config Docs for Genre Compatibility Names

**Files:**
- Modify: `docs/CONFIG.md`

- [ ] **Step 1: Replace stale config block**

In `docs/CONFIG.md`, replace the `genre_conflict_*` block under `candidate_pool` with:

```yaml
      # Raw genre-compatibility penalty.
      # Applies against the raw artifact genre vocabulary. The old
      # genre_conflict_min_confidence hard gate was deleted because it rejected
      # legitimate candidates with the current high-dimensional identity
      # affinity vocabulary. The soft penalty demotes off-axis tracks without
      # blocking them.
      genre_compatibility_enabled: true
      genre_compatibility_penalty_strength: 0.20
      genre_compatibility_compatible_threshold: 0.35
      genre_compatibility_conflict_threshold: 0.15
```

Update nearby prose that says `candidate_pool.genre_conflict_min_confidence` so it refers to the deleted historical gate, not an active key:

```markdown
      # Size raised from 500 to 1500 in v4.1; min_confidence set to null
      # for the same reason the old candidate-pool genre-conflict hard gate
      # was deleted.
```

- [ ] **Step 2: Verify active docs and code no longer advertise stale active keys**

Run:

```bash
rg -n "genre_conflict_" docs/CONFIG.md config.example.yaml README.md src tests
```

Expected: no matches.

---

### Task 5: Clean Whitespace Gate Failure

**Files:**
- Modify: `src/playlist/repair/__init__.py`

- [ ] **Step 1: Remove the extra blank EOF line**

Make `src/playlist/repair/__init__.py` exactly:

```python
"""Playlist repair passes."""
```

- [ ] **Step 2: Verify diff whitespace**

Run:

```bash
git diff --check 2c09ffc342973df8b5484d9f838780a66a4e9ecc..HEAD
```

Expected: no output and exit code `0`.

---

### Task 6: Final Verification

**Files:**
- No code edits.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
pytest tests/unit/test_blacklist_scopes.py tests/unit/test_gui_policy.py::TestArtistDiversity tests/unit/test_edge_repair.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m compileall -q src tests
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run full unit suite**

Run:

```bash
pytest -q --maxfail=1
```

Expected: all tests pass. If the local shell lacks `pytest`, use the repository's Python environment or install contributor dependencies with `pip install -e .[gui,dev]` before rerunning.

- [ ] **Step 4: Confirm the deferred finding stayed untouched**

Run:

```bash
git diff -- src/playlist/pier_bridge/assemble.py
```

Expected: no output.

- [ ] **Step 5: Review changed files**

Run:

```bash
git diff --stat
```

Expected changed implementation/test/docs files only:

```text
src/metadata_client.py
src/playlist_gui/policy.py
src/playlist/repair/edge_repair.py
src/playlist/pier_bridge_builder.py
src/playlist/repair/__init__.py
tests/unit/test_blacklist_scopes.py
tests/unit/test_gui_policy.py
tests/unit/test_edge_repair.py
docs/CONFIG.md
```

## Self-Review

- Spec coverage: fixes are planned for manual blacklist persistence, GUI weighted diversity pass-through, edge repair hard-cap preservation, stale config docs, and whitespace gate. The stale `assemble.py` duplicate implementation is explicitly deferred per request.
- Placeholder scan: no task contains TBD/TODO/later placeholders.
- Type consistency: new repair parameter is named `max_non_seed_tracks_per_artist` in tests, function signature, refusal helper, and builder call.
