# Edit Enriched Genres Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user manually edit enriched genres for a release directly from the playlist viewer's right-click context menu, with edits surviving `build-enriched` re-runs.

**Architecture:** Manual edits live in a new `ai_genre_user_overrides` table (add/remove sets keyed by `release_key`). `build-enriched` reads overrides as a final layer when writing `enriched_genre_signatures`. `EnrichedGenreResolver.get_enriched_genres` applies overrides on read so edits take effect immediately without rebuilding. The GUI exposes "Edit genres for this album..." via the existing track-table context menu, opens a modal `EditGenresDialog` with the current signature in a textarea, computes the add/remove diff on Apply, and writes through a new worker `edit_genres` command. After write, the playlist viewer refreshes the affected rows' Genres column from the resolver.

**Tech Stack:** PySide6 (existing GUI), SQLite (sidecar DB), existing IPC pattern (NDJSON over stdout via worker_client → worker handlers).

---

## File Structure

**Storage layer (sidecar DB):**
- Modify: `src/ai_genre_enrichment/storage.py` — add `ai_genre_user_overrides` table to `initialize()`; add `set_user_override`, `get_user_override`, `delete_user_override` methods; modify `build_enriched_for_release` (line 1180-1280 area) to read and apply the override when writing the signature.
- Modify: `src/ai_genre_enrichment/genre_resolver.py` — `get_enriched_genres` reads the override row when present and applies add/remove on top of the stored signature, so edits take effect even when the signature row hasn't been re-baked.

**Worker / IPC:**
- Modify: `src/playlist_gui/worker.py` — add `handle_edit_genres` command handler that writes the override and emits `done`. Register in `TRACKED_COMMAND_HANDLERS`.
- Modify: `src/playlist_gui/worker_client.py` — add `edit_genres(artist, album, genres)` method that sends the command.

**GUI:**
- Create: `src/playlist_gui/widgets/edit_genres_dialog.py` — `EditGenresDialog(QDialog)` with album label, multi-line text edit, Save / Cancel buttons. Signal `genres_committed(artist, album, list[str])`.
- Modify: `src/playlist_gui/widgets/track_table.py` — add "Edit genres for *Album*..." action to the context menu when single-album selection; emit new signal `edit_genres_requested(dict[str,str])` carrying `{artist, album}`.
- Modify: `src/playlist_gui/main_window.py` — wire `edit_genres_requested` to open the dialog and `genres_committed` to dispatch through `worker_client.edit_genres`; on done, call `_track_table.refresh_genres_for_album(artist, album)`.
- Modify: `src/playlist_gui/widgets/track_table.py` — add `refresh_genres_for_album` which re-runs `_resolve_track_genres` (already in worker.py:561 — extract a small helper in `src/ai_genre_enrichment/genre_resolver.py` so it can be called GUI-side too).

**Tests:**
- Create: `tests/unit/test_user_overrides_storage.py`
- Create: `tests/unit/test_resolver_with_overrides.py`
- Create: `tests/unit/test_edit_genres_dialog.py`
- Create: `tests/unit/test_worker_edit_genres.py`
- Modify: `tests/unit/test_track_table.py` (if it exists; else create) — context menu action visibility under selection state.

---

## Bite-Sized Tasks

### Task 1: User Overrides Schema + Storage Methods

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py` (`SidecarStore.initialize` near line 294; add new methods near other CRUD helpers around line 1100-1300)
- Test: `tests/unit/test_user_overrides_storage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_user_overrides_storage.py
import json
from src.ai_genre_enrichment.storage import SidecarStore


def test_user_override_round_trip(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    store.set_user_override(
        release_key="autechre::amber",
        normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"],
        genres_remove=["warp"],
    )

    override = store.get_user_override("autechre::amber")
    assert override is not None
    assert set(override["genres_add"]) == {"modular synthesizer"}
    assert set(override["genres_remove"]) == {"warp"}


def test_user_override_replaces_on_repeat_set(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["x"], genres_remove=[],
    )
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["y"], genres_remove=["z"],
    )
    override = store.get_user_override("a::b")
    assert set(override["genres_add"]) == {"y"}
    assert set(override["genres_remove"]) == {"z"}


def test_user_override_delete(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["x"], genres_remove=[],
    )
    store.delete_user_override("a::b")
    assert store.get_user_override("a::b") is None


def test_get_user_override_missing_returns_none(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    assert store.get_user_override("never::seen") is None


def test_user_override_casefolds_genres(tmp_path):
    """Genres are stored casefolded so 'IDM' and 'idm' collapse to 'idm'."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["IDM", "idm", "Glitch"],
        genres_remove=["Warp"],
    )
    override = store.get_user_override("a::b")
    assert set(override["genres_add"]) == {"idm", "glitch"}
    assert set(override["genres_remove"]) == {"warp"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_user_overrides_storage.py -v`
Expected: FAIL with `AttributeError: 'SidecarStore' object has no attribute 'set_user_override'`

- [ ] **Step 3: Add the schema migration**

In `src/ai_genre_enrichment/storage.py`, inside the `initialize()` method's `executescript(...)` block, add this CREATE statement next to the other tables (near line 301, after `enriched_genre_signatures`):

```python
                CREATE TABLE IF NOT EXISTS ai_genre_user_overrides (
                    release_key TEXT PRIMARY KEY,
                    normalized_artist TEXT NOT NULL,
                    normalized_album TEXT NOT NULL,
                    genres_add_json TEXT NOT NULL DEFAULT '[]',
                    genres_remove_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
```

- [ ] **Step 4: Add the storage methods**

Append the following methods to `SidecarStore` (near where other CRUD helpers live):

```python
    def set_user_override(
        self,
        *,
        release_key: str,
        normalized_artist: str,
        normalized_album: str,
        genres_add: list[str],
        genres_remove: list[str],
    ) -> None:
        """Upsert the manual override for a release. Replaces prior entry."""
        import json
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_genre_user_overrides (
                    release_key, normalized_artist, normalized_album,
                    genres_add_json, genres_remove_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_key) DO UPDATE SET
                    genres_add_json = excluded.genres_add_json,
                    genres_remove_json = excluded.genres_remove_json,
                    updated_at = excluded.updated_at
                """,
                (
                    release_key, normalized_artist, normalized_album,
                    json.dumps(sorted({g.casefold() for g in genres_add})),
                    json.dumps(sorted({g.casefold() for g in genres_remove})),
                    now,
                ),
            )
            conn.commit()

    def get_user_override(self, release_key: str) -> dict | None:
        """Return the override dict or None when no override exists."""
        import json
        with self.connect() as conn:
            row = conn.execute(
                "SELECT genres_add_json, genres_remove_json, updated_at "
                "FROM ai_genre_user_overrides WHERE release_key = ?",
                (release_key,),
            ).fetchone()
        if not row:
            return None
        return {
            "genres_add": json.loads(row["genres_add_json"]),
            "genres_remove": json.loads(row["genres_remove_json"]),
            "updated_at": row["updated_at"],
        }

    def delete_user_override(self, release_key: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM ai_genre_user_overrides WHERE release_key = ?",
                (release_key,),
            )
            conn.commit()
```

- [ ] **Step 5: Run tests to verify passing**

Run: `pytest tests/unit/test_user_overrides_storage.py -v`
Expected: PASS, 4 tests passing

- [ ] **Step 6: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_user_overrides_storage.py
git commit -m "feat: add ai_genre_user_overrides table for manual genre edits"
```

---

### Task 2: Resolver Applies User Overrides on Read

**Files:**
- Modify: `src/ai_genre_enrichment/genre_resolver.py` (line 24, `get_enriched_genres`)
- Test: `tests/unit/test_resolver_with_overrides.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_resolver_with_overrides.py
import json
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


def _seed_signature(store, release_key, normalized_artist, normalized_album, genres):
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (release_key, normalized_artist, normalized_album, None,
             json.dumps({"genres": genres, "sources": []}), "2026-05-28"),
        )
        conn.commit()


def test_resolver_applies_override_add(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber", ["idm", "glitch"])
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", genres_add=["modular synthesizer"], genres_remove=[],
    )

    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    genres = resolver.get_enriched_genres(artist="Autechre", album="Amber")
    assert set(genres) == {"idm", "glitch", "modular synthesizer"}


def test_resolver_applies_override_remove(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber",
                    ["idm", "glitch", "warp"])
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", genres_add=[], genres_remove=["warp"],
    )

    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    genres = resolver.get_enriched_genres(artist="Autechre", album="Amber")
    assert set(genres) == {"idm", "glitch"}


def test_resolver_works_without_override(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber", ["idm"])
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    assert set(resolver.get_enriched_genres(artist="Autechre", album="Amber")) == {"idm"}


def test_override_without_signature_returns_overrides_as_signature(tmp_path):
    """Edge case: user creates an override for a release that was never enriched.
    Treat the add list as the signature so manual genres still flow to playlists."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="rare::release", normalized_artist="rare",
        normalized_album="release", genres_add=["field recordings"], genres_remove=[],
    )
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    assert resolver.get_enriched_genres(artist="Rare", album="Release") == ["field recordings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_resolver_with_overrides.py -v`
Expected: FAIL — resolver returns the raw signature unmodified.

- [ ] **Step 3: Modify `get_enriched_genres` to apply overrides**

In `src/ai_genre_enrichment/genre_resolver.py`, replace the body of `get_enriched_genres` with:

```python
    def get_enriched_genres(self, *, artist: str, album: str | None) -> list[str] | None:
        if not album:
            return None
        release_key = self._release_key(artist, album)
        with self._connect() as conn:
            sig_row = conn.execute(
                "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
                (release_key,),
            ).fetchone()
            override_row = conn.execute(
                "SELECT genres_add_json, genres_remove_json "
                "FROM ai_genre_user_overrides WHERE release_key = ?",
                (release_key,),
            ).fetchone() if self._user_overrides_exist(conn) else None

        sig_genres: list[str] = []
        if sig_row:
            payload = json.loads(sig_row["signature_json"])
            sig_genres = list(payload.get("genres") or [])

        if not override_row:
            return sig_genres if sig_genres else None

        # Both sets are stored casefolded (normalised at write time in set_user_override).
        add = list(json.loads(override_row["genres_add_json"]))
        remove_lower = set(json.loads(override_row["genres_remove_json"]))
        combined: list[str] = []
        combined_lower: set[str] = set()
        for g in sig_genres:
            gk = g.casefold()
            if gk not in remove_lower and gk not in combined_lower:
                combined.append(g)
                combined_lower.add(gk)
        for g in add:
            gk = g.casefold()
            if gk not in combined_lower:
                combined.append(g)
                combined_lower.add(gk)
        return combined if combined else None

    @staticmethod
    def _user_overrides_exist(conn) -> bool:
        # Backwards-compat: empty in-memory fallback DB doesn't have the table.
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_genre_user_overrides'"
        ).fetchone()
        return row is not None
```

Also update the empty in-memory fallback in `_connect` (line 92-95) to include the overrides table so the schema is consistent:

```python
            conn.execute(
                "CREATE TABLE ai_genre_user_overrides("
                "release_key TEXT, normalized_artist TEXT, normalized_album TEXT, "
                "genres_add_json TEXT, genres_remove_json TEXT, updated_at TEXT)"
            )
```

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest tests/unit/test_resolver_with_overrides.py tests/unit/test_similarity_calc_enriched.py tests/unit/test_artifact_builder_enriched.py tests/unit/test_ai_genre_enrichment.py -v`
Expected: All pass (the override-aware resolver must not regress existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/genre_resolver.py tests/unit/test_resolver_with_overrides.py
git commit -m "feat: resolver applies user overrides on top of enriched signatures"
```

---

### Task 3: build-enriched Preserves Overrides

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py` (`build_enriched_for_release` around line 1180-1280; specifically the place that computes `expanded_genres` and writes `signature_json`)
- Test: `tests/unit/test_user_overrides_storage.py` (add tests to the file from Task 1)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_user_overrides_storage.py`:

```python
def test_build_enriched_preserves_overrides_in_signature(tmp_path):
    """After build-enriched re-bakes a release, the signature should still
    reflect any manual overrides on disk."""
    import json
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    # Seed source page + tags + classifications so build_enriched_for_release works
    page_id = store.upsert_source_page(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", album_id=None,
        source_url="lastfm://artist/autechre/album/amber",
        source_type="lastfm_tags", identity_status="confirmed",
        identity_confidence=1.0, evidence_summary="lastfm",
    )
    store.replace_source_tags(page_id, ["idm", "glitch"])
    store.classify_source_tags(page_id, adjudicate=False)

    # Set an override that adds "modular synthesizer" and removes "glitch"
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"], genres_remove=["glitch"],
    )

    # Re-bake the signature
    store.build_enriched_for_release("autechre::amber")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
            ("autechre::amber",),
        ).fetchone()

    payload = json.loads(row["signature_json"])
    genres = set(payload["genres"])
    assert "modular synthesizer" in genres, f"add not preserved: {genres}"
    assert "glitch" not in genres, f"remove not preserved: {genres}"
    assert "idm" in genres
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_user_overrides_storage.py::test_build_enriched_preserves_overrides_in_signature -v`
Expected: FAIL — the signature still has `glitch` and lacks `modular synthesizer`.

- [ ] **Step 3: Modify `build_enriched_for_release` to apply overrides**

In `src/ai_genre_enrichment/storage.py`, find the block around line 1242 that builds the signature payload. Locate:

```python
            if metadata_row and rows:
                signature = {
                    "genres": sorted(expanded_genres),
                    "sources": _signature_sources(source_rows),
                }
```

Replace with:

```python
            if metadata_row and rows:
                override_row = conn.execute(
                    "SELECT genres_add_json, genres_remove_json "
                    "FROM ai_genre_user_overrides WHERE release_key = ?",
                    (release_key,),
                ).fetchone()
                final_genres = set(expanded_genres)
                if override_row:
                    import json as _json
                    final_genres -= set(_json.loads(override_row["genres_remove_json"]))
                    final_genres |= set(_json.loads(override_row["genres_add_json"]))
                signature = {
                    "genres": sorted(final_genres),
                    "sources": _signature_sources(source_rows),
                }
```

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest tests/unit/test_user_overrides_storage.py tests/unit/test_ai_genre_enrichment.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_user_overrides_storage.py
git commit -m "feat: build-enriched preserves user overrides in signatures"
```

---

### Task 4: Worker `edit_genres` Command

**Files:**
- Modify: `src/playlist_gui/worker.py` (add handler near `handle_enrich_artist`; register in `TRACKED_COMMAND_HANDLERS` around line 1913)
- Modify: `src/playlist_gui/worker_client.py` (add method near `enrich_artist` line 553)
- Test: `tests/unit/test_worker_edit_genres.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_worker_edit_genres.py
from pathlib import Path
import json
from src.playlist_gui.worker import handle_edit_genres
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


def test_edit_genres_handler_writes_override(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    # Seed a signature so we can compute the diff
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("autechre::amber", "autechre", "amber", None,
             json.dumps({"genres": ["idm", "glitch", "warp"], "sources": []}),
             "2026-05-28"),
        )
        conn.commit()

    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))

    cmd = {
        "cmd": "edit_genres",
        "request_id": "r1",
        "artist": "Autechre",
        "album": "Amber",
        "genres": ["idm", "glitch", "modular synthesizer"],  # user removes 'warp', adds 'modular synthesizer'
    }
    handle_edit_genres(cmd)

    override = store.get_user_override("autechre::amber")
    assert override is not None
    assert set(override["genres_add"]) == {"modular synthesizer"}
    assert set(override["genres_remove"]) == {"warp"}

    # And the resolver sees the new genres
    resolver = EnrichedGenreResolver(str(sidecar))
    new_genres = set(resolver.get_enriched_genres(artist="Autechre", album="Amber"))
    assert new_genres == {"idm", "glitch", "modular synthesizer"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_worker_edit_genres.py -v`
Expected: FAIL with `ImportError: cannot import name 'handle_edit_genres'`

- [ ] **Step 3: Add the worker handler**

Add to `src/playlist_gui/worker.py` near `handle_enrich_artist`:

```python
def handle_edit_genres(cmd_data: Dict[str, Any]) -> None:
    """Write a user override for (artist, album) computed from a target genre list.

    Diff is computed against the current resolved genres so the override stores
    the minimal add/remove set rather than the full signature.
    """
    request_id = cmd_data.get("request_id", "")
    try:
        artist = (cmd_data.get("artist") or "").strip()
        album = (cmd_data.get("album") or "").strip()
        target_genres = [str(g).strip() for g in cmd_data.get("genres") or [] if str(g).strip()]
        if not artist or not album:
            raise ValueError("artist and album are required")

        from src.ai_genre_enrichment.storage import SidecarStore
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        from src.ai_genre_enrichment.tag_classification import normalize_source_tag

        resolver = EnrichedGenreResolver(SIDECAR_DB_PATH)
        current = set(resolver.get_enriched_genres(artist=artist, album=album) or [])
        target = set(target_genres)
        add = target - current
        remove = current - target

        store = SidecarStore(SIDECAR_DB_PATH)
        store.initialize()
        release_key = f"{normalize_source_tag(artist)}::{normalize_source_tag(album)}"
        store.set_user_override(
            release_key=release_key,
            normalized_artist=normalize_source_tag(artist),
            normalized_album=normalize_source_tag(album),
            genres_add=sorted(add),
            genres_remove=sorted(remove),
        )

        emit_result("edit_genres", {
            "artist": artist, "album": album,
            "genres": sorted(target),
            "added": sorted(add), "removed": sorted(remove),
        }, request_id=request_id)
        emit_done("edit_genres", True, "ok", request_id=request_id)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb, request_id=request_id)
        emit_done("edit_genres", False, str(e), request_id=request_id)
```

Register in `TRACKED_COMMAND_HANDLERS` (around line 1913) by adding `"edit_genres": handle_edit_genres,` to the dict.

- [ ] **Step 4: Add the worker_client method**

In `src/playlist_gui/worker_client.py`, add near line 553:

```python
    def edit_genres(self, artist: str, album: str, genres: List[str], job_id: Optional[str] = None) -> Optional[str]:
        """Write a user override for (artist, album). Returns the request_id."""
        return self.send_command(
            {
                "cmd": "edit_genres",
                "artist": artist,
                "album": album,
                "genres": list(genres),
            },
            job_id=job_id,
        )
```

- [ ] **Step 5: Run tests to verify passing**

Run: `pytest tests/unit/test_worker_edit_genres.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/worker.py src/playlist_gui/worker_client.py tests/unit/test_worker_edit_genres.py
git commit -m "feat: add edit_genres worker command"
```

---

### Task 5: EditGenresDialog Widget

**Files:**
- Create: `src/playlist_gui/widgets/edit_genres_dialog.py`
- Test: `tests/unit/test_edit_genres_dialog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_edit_genres_dialog.py
import pytest

pytestmark = pytest.mark.gui


def test_dialog_emits_committed_with_normalized_genres(qtbot):
    from PySide6.QtWidgets import QDialogButtonBox
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(
        artist="Autechre",
        album="Amber",
        current_genres=["idm", "glitch", "warp"],
    )
    qtbot.addWidget(dialog)

    # User removes 'warp' and adds 'modular synthesizer'
    dialog.set_text("idm\nglitch\nmodular synthesizer\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append((artist, album, genres)))

    dialog._on_save_clicked()
    assert captured == [("Autechre", "Amber", ["idm", "glitch", "modular synthesizer"])]


def test_dialog_strips_empty_lines_and_whitespace(qtbot):
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("  ambient  \n\n  drone \n\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    assert captured == [["ambient", "drone"]]


def test_dialog_deduplicates(qtbot):
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog

    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("idm\nIDM\nidm")
    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    # Lowercase-deduplicated, original first occurrence kept
    assert captured == [["idm"]]


def test_dialog_rejects_empty_save(qtbot, monkeypatch):
    """Saving with all lines blank must NOT emit genres_committed."""
    from src.playlist_gui.widgets.edit_genres_dialog import EditGenresDialog
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
    dialog = EditGenresDialog(artist="X", album="Y", current_genres=["a"])
    qtbot.addWidget(dialog)
    dialog.set_text("   \n\n\n")

    captured = []
    dialog.genres_committed.connect(lambda artist, album, genres: captured.append(genres))
    dialog._on_save_clicked()
    assert captured == []  # signal must not fire; at least one genre required
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_edit_genres_dialog.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Create the dialog**

Create `src/playlist_gui/widgets/edit_genres_dialog.py`:

```python
"""Dialog for editing the enriched genre signature of a release."""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EditGenresDialog(QDialog):
    """Modal dialog for editing enriched genres for a single release.

    The dialog operates on a (artist, album) pair. The text edit shows one
    genre per line; on save we deduplicate (casefold) preserving first
    occurrence and emit the cleaned list. The caller is responsible for
    computing the add/remove diff and persisting via the worker.
    """

    genres_committed = Signal(str, str, list)  # artist, album, genres

    def __init__(
        self,
        *,
        artist: str,
        album: str,
        current_genres: Iterable[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._artist = artist
        self._album = album
        self.setWindowTitle(f"Edit genres — {artist} / {album}")
        self.setMinimumSize(420, 360)

        layout = QVBoxLayout(self)
        label = QLabel(
            f"<b>{artist} / {album}</b><br>"
            "One genre per line. Lines you remove will be marked as user-removed; "
            "lines you add will be marked as user-added. Edits are preserved across "
            "future re-runs of build-enriched.<br>"
            "<i>After saving, run <b>Tools → Build Artifacts</b> to apply changes "
            "to playlist generation.</i>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        self._text = QPlainTextEdit(self)
        self._text.setPlainText("\n".join(list(current_genres)))
        layout.addWidget(self._text, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_clicked)
        button_box.addButton(save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def set_text(self, text: str) -> None:
        self._text.setPlainText(text)

    def _on_save_clicked(self) -> None:
        raw_lines = [line.strip() for line in self._text.toPlainText().splitlines()]
        seen: set[str] = set()
        cleaned: list[str] = []
        for line in raw_lines:
            if not line:
                continue
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(line)
        if not cleaned:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No genres",
                "At least one genre is required. "
                "To remove enriched genres entirely, delete the override via the pipeline.",
            )
            return
        self.genres_committed.emit(self._artist, self._album, cleaned)
        self.accept()
```

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest tests/unit/test_edit_genres_dialog.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/widgets/edit_genres_dialog.py tests/unit/test_edit_genres_dialog.py
git commit -m "feat: add EditGenresDialog widget for manual genre edits"
```

---

### Task 6: Track Table Context Menu Wiring

**Files:**
- Modify: `src/playlist_gui/widgets/track_table.py` (add signal near line 71; add action inside `_show_context_menu` after the "Blacklist Album" action)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_edit_genres_dialog.py` (or create a new file for track table context menu tests):

```python
def test_track_table_emits_edit_genres_for_single_album_selection(qtbot, monkeypatch):
    """When a single track from one album is selected, the context menu must
    expose an Edit Genres action that emits the artist/album dict."""
    from src.playlist_gui.widgets.track_table import TrackTable

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber", "title": "Foil"},
    ])
    table.select_row(0)  # TrackTable public helper — QTableView has no selectRow()

    captured = []
    table.edit_genres_requested.connect(lambda payload: captured.append(payload))

    # Look up the action by name (avoids the modal exec)
    action = table._build_edit_genres_action_for_selection()
    assert action is not None
    action.trigger()
    assert captured == [{"artist": "Autechre", "album": "Amber"}]


def test_track_table_no_edit_genres_when_mixed_albums(qtbot):
    from src.playlist_gui.widgets.track_table import TrackTable

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber", "title": "Foil"},
        {"position": 2, "artist": "Stereolab", "album": "Emperor", "title": "Cybele"},
    ])
    table._table.selectAll()

    action = table._build_edit_genres_action_for_selection()
    assert action is None  # mixed-album selection disables the action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_edit_genres_dialog.py::test_track_table_emits_edit_genres_for_single_album_selection -v`
Expected: FAIL — signal and helper don't exist.

- [ ] **Step 3: Add signal and helper to TrackTable**

In `src/playlist_gui/widgets/track_table.py`:

Add to signal declarations near line 71:

```python
    edit_genres_requested = Signal(dict)  # {"artist": str, "album": str}
```

Add a helper method (this isolates the action-build logic so we can unit-test it without `exec()`):

```python
    def _build_edit_genres_action_for_selection(self):
        """Return a QAction that emits edit_genres_requested for the selected
        album, or None if selection spans multiple albums.

        Extracted so tests don't need to drive QMenu.exec.
        """
        from PySide6.QtGui import QAction
        selected = self.get_selected_tracks()
        if not selected:
            return None
        artists = {(t.get("artist") or "").strip() for t in selected}
        albums = {(t.get("album") or "").strip() for t in selected}
        if len(artists) != 1 or len(albums) != 1:
            return None
        artist = next(iter(artists))
        album = next(iter(albums))
        if not artist or not album:
            return None
        action = QAction(f"Edit genres for album: {album}", self)
        action.triggered.connect(
            lambda checked=False, a=artist, b=album:
                self.edit_genres_requested.emit({"artist": a, "album": b})
        )
        return action
```

Modify `_show_context_menu` to use the helper after the existing "Blacklist Album" action (around line 545):

```python
            edit_action = self._build_edit_genres_action_for_selection()
            if edit_action is not None:
                menu.addSeparator()
                menu.addAction(edit_action)
```

- [ ] **Step 4: Run tests to verify passing**

Run: `pytest tests/unit/test_edit_genres_dialog.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/widgets/track_table.py tests/unit/test_edit_genres_dialog.py
git commit -m "feat: track table emits edit_genres_requested for single-album selections"
```

---

### Task 7: Main Window Wiring + Refresh

**Files:**
- Modify: `src/playlist_gui/main_window.py` (connect signal near line 336 where `replace_track_requested` is wired; add `_on_edit_genres_requested`, `_on_genres_committed`, `_on_edit_genres_done` handlers)
- Modify: `src/playlist_gui/widgets/track_table.py` (add `refresh_genres_for_album` method that re-resolves enriched genres for matching rows)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_edit_genres_dialog.py
def test_refresh_genres_for_album_updates_matching_rows(qtbot, tmp_path, monkeypatch):
    """After committing edits, the table's Genres column should reflect the new resolved genres."""
    import json
    from src.playlist_gui.widgets.track_table import TrackTable
    from src.ai_genre_enrichment.storage import SidecarStore

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar))
    store.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("autechre::amber", "autechre", "amber", None,
             json.dumps({"genres": ["idm"], "sources": []}), "2026-05-28"),
        )
        conn.commit()

    table = TrackTable()
    qtbot.addWidget(table)
    table.set_tracks([
        {"position": 1, "artist": "Autechre", "album": "Amber",
         "title": "Foil", "genres": ["idm"]},
        {"position": 2, "artist": "Other", "album": "Different",
         "title": "X", "genres": ["jazz"]},
    ])

    # Update the signature on disk
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"], genres_remove=[],
    )
    monkeypatch.setattr(
        "src.playlist_gui.widgets.track_table.SIDECAR_DB_PATH",
        str(sidecar),
        raising=False,
    )

    table.refresh_genres_for_album(artist="Autechre", album="Amber")

    rows = table.get_tracks()
    assert set(rows[0]["genres"]) == {"idm", "modular synthesizer"}
    # Untouched
    assert rows[1]["genres"] == ["jazz"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_edit_genres_dialog.py::test_refresh_genres_for_album_updates_matching_rows -v`
Expected: FAIL — `refresh_genres_for_album` doesn't exist.

- [ ] **Step 3: Implement refresh in TrackTable**

In `src/playlist_gui/widgets/track_table.py`, add module-level constant and method:

```python
SIDECAR_DB_PATH = "data/ai_genre_enrichment.db"


# ... inside class TrackTable:
    def refresh_genres_for_album(self, *, artist: str, album: str) -> None:
        """Re-resolve enriched genres for all rows matching (artist, album)
        and update the model in place. No-op if the resolver returns None."""
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        resolver = EnrichedGenreResolver(SIDECAR_DB_PATH)
        enriched = resolver.get_enriched_genres(artist=artist, album=album)
        if enriched is None:
            return

        tracks = list(self._model.get_tracks())
        for track in tracks:
            if (track.get("artist") or "").strip() == artist and \
               (track.get("album") or "").strip() == album:
                track["genres"] = list(enriched)
        self._model.set_tracks(tracks)
```

- [ ] **Step 4: Wire up main_window**

In `src/playlist_gui/main_window.py`:

Add an import:
```python
from .widgets.edit_genres_dialog import EditGenresDialog
```

In the section near line 336 where signals are wired, add:
```python
        self._track_table.edit_genres_requested.connect(self._on_edit_genres_requested)
```

Add the handler methods (near `_open_replace_dialog`):
```python
    @Slot(dict)
    def _on_edit_genres_requested(self, payload: dict) -> None:
        artist = payload.get("artist") or ""
        album = payload.get("album") or ""
        if not artist or not album:
            return
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        resolver = EnrichedGenreResolver("data/ai_genre_enrichment.db")
        current = resolver.get_enriched_genres(artist=artist, album=album) or []
        dialog = EditGenresDialog(
            artist=artist, album=album, current_genres=current, parent=self
        )
        dialog.genres_committed.connect(self._on_genres_committed)
        dialog.exec()

    @Slot(str, str, list)
    def _on_genres_committed(self, artist: str, album: str, genres: list) -> None:
        self._pending_genre_edit = (artist, album)
        self._worker_client.edit_genres(artist, album, genres)

    @Slot(dict)
    def _on_edit_genres_done(self, payload: dict) -> None:
        """Wired into the worker's done signal for the edit_genres command."""
        if not getattr(self, "_pending_genre_edit", None):
            return
        artist, album = self._pending_genre_edit
        self._pending_genre_edit = None
        self._track_table.refresh_genres_for_album(artist=artist, album=album)
```

Find the existing `_on_worker_done` (or similar dispatcher) and add a branch:
```python
        if payload.get("cmd") == "edit_genres":
            self._on_edit_genres_done(payload)
            return
```

- [ ] **Step 5: Run tests to verify passing**

Run: `pytest tests/unit/test_edit_genres_dialog.py -v`
Expected: All pass.

- [ ] **Step 6: Manual smoke test**

Run: `python -m playlist_gui.app`
- Generate a playlist that includes at least one track from an enriched release (e.g., Autechre).
- Right-click that track → "Edit genres for album: Amber"
- Remove a genre, add a new one, click Save
- Confirm the Genres column updates immediately for all tracks of that album
- Quit and relaunch; confirm the edits persisted
- Run `python scripts/ai_genre_enrich.py --sidecar-db data/ai_genre_enrichment.db build-enriched`
- Relaunch GUI; confirm edits still present (this validates Task 3)

- [ ] **Step 7: Commit**

```bash
git add src/playlist_gui/main_window.py src/playlist_gui/widgets/track_table.py tests/unit/test_edit_genres_dialog.py
git commit -m "feat: wire EditGenresDialog into main window with table refresh"
```

---

## Self-Review Notes

1. **Spec coverage:** All four feature areas (storage, resolver, build-enriched preservation, GUI dialog + context menu + refresh) have at least one task. ✅
2. **Placeholder scan:** No "TBD"/"implement later" — every code block contains the actual content. ✅
3. **Type consistency:** Signal `edit_genres_requested` is `dict` in both the declaration and the test consumer; `genres_committed` is `(str, str, list)` consistently. ✅
4. **Open question for execution-time review:** the `refresh_genres_for_album` test uses `monkeypatch.setattr` on `track_table.SIDECAR_DB_PATH`. If the implementer prefers to inject the path via the TrackTable constructor instead, that's cleaner — but it requires touching every caller. The current approach is intentional: keep TrackTable's constructor signature stable, accept the small testability cost.
5. **Why no override-as-vocab promotion:** the user may add genres not in the curated `data/genre_vocabulary.yaml`. We deliberately don't auto-promote — the override system trusts the user's text verbatim. If a manual addition later proves common, it can be promoted to the YAML separately via `graduate-ai`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-edit-enriched-genres.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between, fast iteration
2. **Inline Execution** — execute tasks in this session using executing-plans

Which approach?
