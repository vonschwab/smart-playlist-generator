# AI Genre Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize genre enrichment before model-prior work by restoring a legacy-default artifact boundary, fixing source-policy bugs, making extraction dry runs API-free, and rejecting stale dense sidecars.

**Architecture:** Put source rules and policy versions in a focused `policy.py` module. Keep historical sidecar rows readable as `legacy-v0`, while new and explicitly refreshed signatures use `genre-enrichment-v2`. Route artifact builds through an explicit source-mode helper and share one deterministic dense-sidecar validator between runtime loading and library-analysis verification.

**Tech Stack:** Python 3.11+, SQLite, NumPy NPZ artifacts, PyYAML, pytest

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ai_genre_enrichment/policy.py` | New source canonicalization, evidence basis, Last.fm quarantine, and policy-version constants. |
| `src/ai_genre_enrichment/storage.py` | Add nullable policy columns, rebuild signatures under a selected policy, and report policy/source counts. |
| `scripts/ai_genre_enrich.py` | Store canonical Bandcamp source types, make extraction dry runs network-free, and expose explicit artifact source modes. |
| `src/ai_genre_enrichment/artifact_modes.py` | New artifact-mode enum, config resolution, resolver construction, sidecar snapshot hashing, and isolated shadow path helper. |
| `scripts/build_beat3tower_artifacts.py` | Default to `legacy`; only load the sidecar resolver for explicit enriched or shadow builds. |
| `src/config_loader.py` | Add `get_ds_genre_source()`. |
| `src/playlist_gui/worker.py` | Make GUI artifact rebuilds explicitly `legacy`. |
| `scripts/analyze_library.py` | Make full library-analysis artifact rebuilds explicitly `legacy`; reuse dense-sidecar validation. |
| `src/genre/artifact_identity.py` | New deterministic sparse genre-artifact identity and dense-sidecar schema validator. |
| `scripts/build_genre_embedding.py` | Persist sparse identity and dense schema version in sidecars. |
| `src/features/artifacts.py` | Reject sidecars when track IDs, vocab, sparse identity, or schema version do not match. |
| `config.example.yaml` | Document `playlists.ds_pipeline.genre_source: legacy`. |
| `docs/AI_GENRE_ENRICHMENT.md` | Correct stale sidecar-consumption and Bandcamp documentation. |
| `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` | Document artifact source modes and dense-sidecar validation. |

## Task 1: Add Policy Constants And Forward-Only Schema Versioning

**Files:**
- Create: `src/ai_genre_enrichment/policy.py`
- Modify: `src/ai_genre_enrichment/storage.py:161-390`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing schema migration tests**

Append:

```python
def test_sidecar_initializes_forward_only_enrichment_policy_columns(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()

    with sqlite3.connect(db_path) as conn:
        genre_cols = {row[1] for row in conn.execute("PRAGMA table_info(enriched_genres)")}
        sig_cols = {row[1] for row in conn.execute("PRAGMA table_info(enriched_genre_signatures)")}

    assert "enrichment_policy_version" in genre_cols
    assert "enrichment_policy_version" in sig_cols


def test_existing_signature_without_policy_is_reported_as_legacy_v0(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO enriched_genre_signatures(
                release_key, normalized_artist, normalized_album, album_id,
                signature_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("artist::album", "artist", "album", "a1", '{"genres":["rock"],"sources":[]}', "2026-01-01"),
        )

    assert store.report()["signature_policy_counts"] == {"legacy-v0": 1}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_sidecar_initializes_forward_only_enrichment_policy_columns tests/unit/test_ai_genre_enrichment.py::test_existing_signature_without_policy_is_reported_as_legacy_v0 -q --basetemp C:\tmp\genre-policy-schema -o cache_dir=C:\tmp\genre-policy-schema-cache
```

Expected: FAIL because the policy columns and report key do not exist.

- [ ] **Step 3: Add the policy module**

Create:

```python
"""Stable source-policy rules for sidecar genre enrichment."""

from __future__ import annotations

LEGACY_POLICY_VERSION = "legacy-v0"
STABILIZED_POLICY_VERSION = "genre-enrichment-v2"
CANONICAL_BANDCAMP_SOURCE_TYPE = "bandcamp_release"


def canonical_source_type(source_type: str) -> str:
    """Map historical source names to the current read-side contract."""
    return CANONICAL_BANDCAMP_SOURCE_TYPE if source_type == "bandcamp_tags" else source_type


def evidence_basis(source_type: str) -> str:
    """Return the durable enriched_genres basis for a canonicalized source."""
    canonical = canonical_source_type(source_type)
    if canonical == "lastfm_tags":
        return "lastfm_tags"
    if canonical == "local_metadata":
        return "local_metadata"
    return "authoritative_source"


def can_seed_signature(source_type: str) -> bool:
    """Last.fm may corroborate an existing term but cannot create one."""
    return canonical_source_type(source_type) != "lastfm_tags"
```

- [ ] **Step 4: Add nullable policy columns and report counts**

In `SidecarStore.initialize()`, add `enrichment_policy_version TEXT` to both table declarations and idempotent migrations:

```python
_ensure_column(conn, "enriched_genres", "enrichment_policy_version", "TEXT")
_ensure_column(conn, "enriched_genre_signatures", "enrichment_policy_version", "TEXT")
```

In `SidecarStore.report()`, query and return:

```python
signature_policy_counts = {
    row["policy_version"]: row["count"]
    for row in conn.execute(
        """
        SELECT COALESCE(enrichment_policy_version, 'legacy-v0') AS policy_version,
               COUNT(*) AS count
        FROM enriched_genre_signatures
        GROUP BY COALESCE(enrichment_policy_version, 'legacy-v0')
        ORDER BY policy_version
        """
    )
}
```

Add:

```python
"signature_policy_counts": signature_policy_counts,
```

- [ ] **Step 5: Run tests and verify they pass**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/policy.py src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: version genre enrichment sidecar policy"
```

## Task 2: Canonicalize Bandcamp And Quarantine Last.fm For Refreshed Signatures

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py:1153-1295`
- Modify: `scripts/ai_genre_enrich.py:647-720`
- Modify: `src/ai_genre_enrichment/models.py:28-39`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing source-policy regressions**

Append:

```python
def test_historical_bandcamp_tags_rows_contribute_after_refresh(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="bandcamp://artist/duster/album/stratosphere",
        source_type="bandcamp_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="Historical CLI Bandcamp row.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("duster::stratosphere")

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT genre FROM enriched_genres").fetchone()[0] == "slowcore"
        payload = json.loads(conn.execute("SELECT signature_json FROM enriched_genre_signatures").fetchone()[0])
        version = conn.execute("SELECT enrichment_policy_version FROM enriched_genre_signatures").fetchone()[0]

    assert payload["sources"][0]["source_type"] == "bandcamp_release"
    assert version == "genre-enrichment-v2"


def test_lastfm_only_release_does_not_create_current_policy_signature(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="a1",
        source_url="lastfm://artist/slowdive/album/souvlaki",
        source_type="lastfm_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="Last.fm top tags.",
    )
    store.replace_source_tags(page_id, ["shoegaze"])
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("slowdive::souvlaki")

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM enriched_genres").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM enriched_genre_signatures").fetchone()[0] == 0


def test_lastfm_may_corroborate_non_lastfm_genre(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    for source_url, source_type in [
        ("local://album/a1", "local_metadata"),
        ("lastfm://artist/slowdive/album/souvlaki", "lastfm_tags"),
    ]:
        page_id = store.upsert_source_page(
            release_key="slowdive::souvlaki",
            normalized_artist="slowdive",
            normalized_album="souvlaki",
            album_id="a1",
            source_url=source_url,
            source_type=source_type,
            identity_status="confirmed",
            identity_confidence=0.9,
            evidence_summary=source_type,
        )
        store.replace_source_tags(page_id, ["shoegaze"])
        store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("slowdive::souvlaki")

    with sqlite3.connect(db_path) as conn:
        bases = {row[0] for row in conn.execute("SELECT basis FROM enriched_genres")}

    assert bases == {"local_metadata", "lastfm_tags"}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_historical_bandcamp_tags_rows_contribute_after_refresh tests/unit/test_ai_genre_enrichment.py::test_lastfm_only_release_does_not_create_current_policy_signature tests/unit/test_ai_genre_enrichment.py::test_lastfm_may_corroborate_non_lastfm_genre -q --basetemp C:\tmp\genre-source-policy -o cache_dir=C:\tmp\genre-source-policy-cache
```

Expected: FAIL because `bandcamp_tags` is omitted, Last.fm seeds signatures, and policy rows are unstamped.

- [ ] **Step 3: Rebuild rows through policy helpers**

Update `rebuild_enriched_genres_for_release()` to:

```python
from .policy import STABILIZED_POLICY_VERSION, can_seed_signature, canonical_source_type, evidence_basis

def rebuild_enriched_genres_for_release(
    self,
    release_key: str,
    *,
    enrichment_policy_version: str = STABILIZED_POLICY_VERSION,
) -> None:
```

Include both historical and canonical Bandcamp names in the SQL source filter:

```sql
'bandcamp_release',
'bandcamp_tags',
```

Expand candidate genres first, then keep Last.fm rows only when a non-Last.fm row supports the same genre:

```python
expanded_rows: list[tuple[sqlite3.Row, str]] = []
for row in source_rows:
    canonical = _vocab.resolve_alias(row["normalized_tag"])
    decomposed = _vocab.decompose_tag(canonical)
    for genre in [g for g in (decomposed or [canonical]) if g not in ALWAYS_PRUNE_GENRES]:
        expanded_rows.append((row, genre))

seeded_genres = {
    genre for row, genre in expanded_rows if can_seed_signature(row["source_type"])
}
for row, genre in expanded_rows:
    if not can_seed_signature(row["source_type"]) and genre not in seeded_genres:
        continue
    basis = evidence_basis(row["source_type"])
```

Stamp `enrichment_policy_version` in both inserts. Canonicalize source types inside `_signature_sources()` before serializing provenance.

- [ ] **Step 4: Store canonical Bandcamp rows and stop treating local metadata as authoritative web evidence**

In `cmd_extract_bandcamp()`, change:

```python
source_type="bandcamp_release",
```

In `src/ai_genre_enrichment/models.py`, remove `"local_metadata"` from `AUTHORITATIVE_SOURCE_TYPES`. Keep it in `SOURCE_TYPES` because it remains a valid baseline source.

- [ ] **Step 5: Run targeted regressions**

Run the Step 2 command, then:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\genre-source-policy-full -o cache_dir=C:\tmp\genre-source-policy-full-cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/policy.py src/ai_genre_enrichment/storage.py src/ai_genre_enrichment/models.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "fix: stabilize genre enrichment source policy"
```

## Task 3: Make Extraction Dry Runs Strictly API-Free And Sidecar-Free

**Files:**
- Modify: `scripts/ai_genre_enrich.py:566-720`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing API-free dry-run tests**

Append:

```python
def test_extract_lastfm_dry_run_skips_api_and_sidecar(monkeypatch, tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    def explode(**_kwargs):
        raise AssertionError("Last.fm API must not be called")

    monkeypatch.setattr("src.ai_genre_enrichment.lastfm_enrichment.fetch_lastfm_tags", explode)
    rc = ai_genre_cli.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "extract-lastfm", "--artist", "Slowdive", "--dry-run",
    ])

    assert rc == 0
    assert not sidecar_db.exists()


def test_extract_bandcamp_dry_run_skips_openai_fetch_and_sidecar(monkeypatch, tmp_path: Path):
    metadata_db = _metadata_db(tmp_path)
    sidecar_db = tmp_path / "sidecar.db"

    def explode(**_kwargs):
        raise AssertionError("Bandcamp locator and fetch must not be called")

    monkeypatch.setattr("src.ai_genre_enrichment.bandcamp_enrichment.fetch_bandcamp_tags", explode)
    rc = ai_genre_cli.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar_db),
        "extract-bandcamp", "--artist", "Slowdive", "--dry-run",
    ])

    assert rc == 0
    assert not sidecar_db.exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_extract_lastfm_dry_run_skips_api_and_sidecar tests/unit/test_ai_genre_enrichment.py::test_extract_bandcamp_dry_run_skips_openai_fetch_and_sidecar -q --basetemp C:\tmp\genre-api-free-dryrun -o cache_dir=C:\tmp\genre-api-free-dryrun-cache
```

Expected: FAIL because both fetchers run before their current dry-run checks.

- [ ] **Step 3: Return dry-run routing output before key lookup, initialization, or network calls**

At the start of both extraction commands, discover releases first and branch:

```python
releases = _discover(args)
if not releases:
    print("No matching release found.")
    return 1
if getattr(args, "dry_run", False):
    for release in releases:
        print(json.dumps({
            "release_key": release.release_key,
            "source_type": "lastfm_tags",
            "network_calls": 0,
            "sidecar_writes": 0,
            "dry_run": True,
        }, ensure_ascii=False, sort_keys=True))
    return 0
```

Use this Bandcamp payload in `cmd_extract_bandcamp()`:

```python
{
    "release_key": release.release_key,
    "source_type": "bandcamp_release",
    "route": ["openai_source_locator", "bandcamp_release_html", "classify_tags"],
    "network_calls": 0,
    "sidecar_writes": 0,
    "dry_run": True,
}
```

Only resolve API keys, create `SidecarStore`, call `initialize()`, set vocabulary, and invoke fetchers after the dry-run return.

- [ ] **Step 4: Run targeted and full enrichment tests**

Run the Step 2 command, then:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\genre-api-free-dryrun-full -o cache_dir=C:\tmp\genre-api-free-dryrun-full-cache
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "fix: make genre extraction dry runs API free"
```

## Task 4: Restore Explicit Artifact Source Modes With Legacy Default

**Files:**
- Create: `src/ai_genre_enrichment/artifact_modes.py`
- Modify: `scripts/build_beat3tower_artifacts.py:63-130,600-625`
- Modify: `scripts/ai_genre_enrich.py:204-240,1509-1540`
- Modify: `src/config_loader.py:114-120`
- Modify: `src/playlist_gui/worker.py:1567-1610`
- Modify: `scripts/analyze_library.py:1122-1160`
- Modify: `config.example.yaml:99-103`
- Test: `tests/unit/test_artifact_builder_enriched.py`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing artifact-mode tests**

Append to `tests/unit/test_artifact_builder_enriched.py`:

```python
def test_beat3tower_builder_legacy_default_does_not_auto_load_sidecar(tmp_path, monkeypatch):
    from argparse import Namespace
    from scripts import build_beat3tower_artifacts as builder

    called = []
    monkeypatch.setattr(builder, "load_tracks_with_beat3tower", lambda *_args: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr("src.ai_genre_enrichment.genre_resolver.EnrichedGenreResolver", lambda *_args: called.append("resolver"))

    args = Namespace(
        db_path="metadata.db", config="config.yaml", output=str(tmp_path / "out.npz"),
        genre_sim_path=None, max_tracks=0, no_pca=False, pca_variance=0.95,
        clip_sigma=3.0, random_seed=42, no_genre_normalization=False,
        sidecar_db=str(tmp_path / "sidecar.db"), verbose=False,
    )
    (tmp_path / "sidecar.db").touch()
    with pytest.raises(RuntimeError, match="stop"):
        builder.build_artifacts(args)
    assert called == []
```

Append to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_artifact_mode_defaults_to_legacy():
    from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource
    assert GenreArtifactSource.resolve(None) is GenreArtifactSource.LEGACY
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_artifact_builder_enriched.py::test_beat3tower_builder_legacy_default_does_not_auto_load_sidecar tests/unit/test_ai_genre_enrichment.py::test_artifact_mode_defaults_to_legacy -q --basetemp C:\tmp\genre-artifact-mode -o cache_dir=C:\tmp\genre-artifact-mode-cache
```

Expected: FAIL because the mode helper does not exist and the builder auto-loads the resolver.

- [ ] **Step 3: Add the artifact source enum and resolver factory**

Create:

```python
"""Explicit artifact genre-source modes."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class GenreArtifactSource(str, Enum):
    LEGACY = "legacy"
    ENRICHED = "enriched"
    HYBRID_SHADOW = "hybrid_shadow"

    @classmethod
    def resolve(cls, value: str | None) -> "GenreArtifactSource":
        return cls(value or cls.LEGACY.value)


def make_resolver(mode: GenreArtifactSource, sidecar_db: str | Path) -> Any | None:
    if mode is GenreArtifactSource.LEGACY:
        return None
    from .genre_resolver import EnrichedGenreResolver
    return EnrichedGenreResolver(sidecar_db)
```

- [ ] **Step 4: Gate the beat3tower resolver behind `--genre-source`**

Add parser argument:

```python
parser.add_argument(
    "--genre-source",
    choices=["legacy", "enriched", "hybrid_shadow"],
    default=None,
    help="Genre source for artifact matrices. Defaults to config, then legacy.",
)
```

In `build_artifacts()`:

```python
from src.config_loader import Config
from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource, make_resolver

config_genre_source = (
    Config(args.config).config.get("playlists", {}).get("ds_pipeline", {}).get("genre_source")
)
genre_source = GenreArtifactSource.resolve(getattr(args, "genre_source", None) or config_genre_source)
if enriched_resolver is None:
    enriched_resolver = make_resolver(genre_source, getattr(args, "sidecar_db", "data/ai_genre_enrichment.db"))
logger.info("Artifact genre source: %s", genre_source.value)
```

Store `"genre_source": genre_source.value` inside the saved `build_config`.

- [ ] **Step 5: Wire legacy defaults through GUI, analysis, config, and CLI rebuilds**

Add to `Config`:

```python
def get_ds_genre_source(self) -> str:
    """Get artifact-build genre source. Legacy is the backward-compatible default."""
    return self._get_ds_pipeline("genre_source", default="legacy")
```

Add under `playlists.ds_pipeline` in `config.example.yaml`:

```yaml
    # Artifact builds remain backward-compatible unless explicitly opted in.
    genre_source: legacy  # legacy | enriched | hybrid_shadow
```

Add `genre_source="legacy"` to the GUI worker and `scripts/analyze_library.py` `Namespace` objects.

Add `--genre-source` to `rebuild-artifacts` and only construct `EnrichedGenreResolver` when it is not `legacy`:

```python
rebuild.add_argument(
    "--genre-source",
    choices=["legacy", "enriched", "hybrid_shadow"],
    default="legacy",
)
```

- [ ] **Step 6: Run artifact tests**

Run:

```powershell
pytest tests/unit/test_artifact_builder_enriched.py tests/unit/test_worker_enrich_artist.py tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\genre-artifact-mode-full -o cache_dir=C:\tmp\genre-artifact-mode-full-cache
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add -- src/ai_genre_enrichment/artifact_modes.py scripts/build_beat3tower_artifacts.py scripts/ai_genre_enrich.py src/config_loader.py src/playlist_gui/worker.py scripts/analyze_library.py config.example.yaml tests/unit/test_artifact_builder_enriched.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: default artifact genre source to legacy"
```

## Task 5: Isolate `hybrid_shadow` Output Paths

**Files:**
- Modify: `src/ai_genre_enrichment/artifact_modes.py`
- Modify: `scripts/ai_genre_enrich.py:1509-1540`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing deterministic shadow-path tests**

Append:

```python
def test_shadow_output_path_is_fingerprinted_and_isolated(tmp_path: Path):
    from src.ai_genre_enrichment.artifact_modes import shadow_output_paths

    paths = shadow_output_paths(
        artifacts_dir=tmp_path,
        policy_version="genre-enrichment-v2",
        signature_snapshot="sig-123",
        prior_snapshot="none",
        sparse_input_identity="artifact-456",
        dense_config={"dim": 64, "skip_prior": True},
    )

    assert paths.root.parent == tmp_path / "shadow"
    assert paths.sparse_artifact == paths.root / "data_matrices_step1.npz"
    assert paths.dense_sidecar == paths.root / "data_matrices_step1_genre_emb_dim64.npz"
    assert paths.report == paths.root / "comparison_report.json"
    assert paths.root.name
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_shadow_output_path_is_fingerprinted_and_isolated -q --basetemp C:\tmp\genre-shadow-path -o cache_dir=C:\tmp\genre-shadow-path-cache
```

Expected: FAIL because `shadow_output_paths()` does not exist.

- [ ] **Step 3: Add deterministic shadow paths**

Append to `artifact_modes.py`:

```python
import hashlib
import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowOutputPaths:
    root: Path
    sparse_artifact: Path
    dense_sidecar: Path
    report: Path


def signature_snapshot_identity(sidecar_db: str | Path) -> str:
    resolved = Path(sidecar_db)
    if not resolved.exists():
        return "missing-sidecar"
    uri = f"file:{resolved.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT release_key, signature_json,
                   COALESCE(enrichment_policy_version, 'legacy-v0')
            FROM enriched_genre_signatures
            ORDER BY release_key
            """
        ).fetchall()
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_identity(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shadow_output_paths(
    *,
    artifacts_dir: str | Path,
    policy_version: str,
    signature_snapshot: str,
    prior_snapshot: str,
    sparse_input_identity: str,
    dense_config: dict[str, Any],
) -> ShadowOutputPaths:
    payload = {
        "genre_source": GenreArtifactSource.HYBRID_SHADOW.value,
        "policy_version": policy_version,
        "signature_snapshot": signature_snapshot,
        "prior_snapshot": prior_snapshot,
        "sparse_input_identity": sparse_input_identity,
        "dense_config": dense_config,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    root = Path(artifacts_dir) / "shadow" / fingerprint
    return ShadowOutputPaths(
        root=root,
        sparse_artifact=root / "data_matrices_step1.npz",
        dense_sidecar=root / "data_matrices_step1_genre_emb_dim64.npz",
        report=root / "comparison_report.json",
    )
```

- [ ] **Step 4: Reject unsafe shadow output overrides**

In `cmd_rebuild_artifacts()`, resolve `hybrid_shadow` output through `shadow_output_paths()` and refuse an active-path overwrite:

```python
from src.ai_genre_enrichment.artifact_modes import (
    GenreArtifactSource,
    file_identity,
    shadow_output_paths,
    signature_snapshot_identity,
)
from src.ai_genre_enrichment.policy import STABILIZED_POLICY_VERSION
from src.ai_genre_enrichment.storage import SidecarStore

genre_source = GenreArtifactSource.resolve(getattr(args, "genre_source", None))
artifacts_dir = _Path(getattr(args, "artifacts_dir", "data/artifacts/beat3tower_32k"))
active_path = artifacts_dir / "data_matrices_step1.npz"
if genre_source is GenreArtifactSource.HYBRID_SHADOW:
    SidecarStore(args.sidecar_db).initialize()
    paths = shadow_output_paths(
        artifacts_dir=artifacts_dir,
        policy_version=STABILIZED_POLICY_VERSION,
        signature_snapshot=signature_snapshot_identity(args.sidecar_db),
        prior_snapshot="none",
        sparse_input_identity=file_identity(active_path),
        dense_config={"dim": 64, "skip_prior": True},
    )
    out_path = paths.sparse_artifact
    out_path.parent.mkdir(parents=True, exist_ok=True)
else:
    out_path = active_path
```

Use `"none"` for `prior_snapshot` until the shadow-integration plan adds model-prior terms.

- [ ] **Step 5: Run targeted test**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/artifact_modes.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: isolate hybrid shadow artifact paths"
```

## Task 6: Persist And Validate Dense Sidecar Identity

**Files:**
- Create: `src/genre/artifact_identity.py`
- Modify: `scripts/build_genre_embedding.py:59-168`
- Modify: `src/features/artifacts.py:127-154`
- Modify: `scripts/analyze_library.py:1235-1252`
- Test: `tests/integration/test_dense_genre_integration.py`

- [ ] **Step 1: Write failing dense-sidecar identity tests**

Add synthetic helpers and tests:

```python
def test_loader_rejects_dense_sidecar_when_vocab_differs(tmp_path, caplog):
    from src.features.artifacts import load_artifact_bundle
    from src.genre.artifact_identity import genre_artifact_identity

    track_ids = np.array(["t1"], dtype=object)
    vocab = np.array(["rock"], dtype=object)
    X_raw = np.array([[1.0]], dtype=np.float32)
    artifact = tmp_path / "mini.npz"
    np.savez(
        artifact, track_ids=track_ids, artist_keys=np.array(["a"], dtype=object),
        track_artists=np.array(["A"], dtype=object), track_titles=np.array(["T"], dtype=object),
        X_sonic=np.array([[1.0]], dtype=np.float32), X_genre_raw=X_raw,
        X_genre_smoothed=X_raw, genre_vocab=vocab,
    )
    np.savez(
        tmp_path / "mini_genre_emb_dim64.npz",
        X_genre_dense=np.zeros((1, 64), dtype=np.float32),
        genre_emb=np.zeros((1, 64), dtype=np.float32),
        genre_vocab=np.array(["jazz"], dtype=object),
        track_ids=track_ids,
        emb_config={
            "schema_version": "dense-genre-sidecar-v2",
            "sparse_genre_identity": genre_artifact_identity(track_ids, vocab, X_raw),
        },
    )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "vocabulary mismatch" in caplog.text


def test_loader_rejects_dense_sidecar_when_sparse_identity_differs(tmp_path, caplog):
    from src.features.artifacts import load_artifact_bundle

    track_ids = np.array(["t1"], dtype=object)
    vocab = np.array(["rock"], dtype=object)
    X_raw = np.array([[1.0]], dtype=np.float32)
    artifact = tmp_path / "mini.npz"
    np.savez(
        artifact, track_ids=track_ids, artist_keys=np.array(["a"], dtype=object),
        track_artists=np.array(["A"], dtype=object), track_titles=np.array(["T"], dtype=object),
        X_sonic=np.array([[1.0]], dtype=np.float32), X_genre_raw=X_raw,
        X_genre_smoothed=X_raw, genre_vocab=vocab,
    )
    np.savez(
        tmp_path / "mini_genre_emb_dim64.npz",
        X_genre_dense=np.zeros((1, 64), dtype=np.float32),
        genre_emb=np.zeros((1, 64), dtype=np.float32),
        genre_vocab=vocab,
        track_ids=track_ids,
        emb_config={
            "schema_version": "dense-genre-sidecar-v2",
            "sparse_genre_identity": "wrong",
        },
    )

    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(artifact)
    assert bundle.X_genre_dense is None
    assert "sparse genre identity mismatch" in caplog.text
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/integration/test_dense_genre_integration.py::test_loader_rejects_dense_sidecar_when_vocab_differs tests/integration/test_dense_genre_integration.py::test_loader_rejects_dense_sidecar_when_sparse_identity_differs -q --basetemp C:\tmp\genre-dense-identity -o cache_dir=C:\tmp\genre-dense-identity-cache
```

Expected: FAIL because runtime loading accepts vocab drift and has no sparse identity contract.

- [ ] **Step 3: Create the shared identity helper**

Create:

```python
"""Deterministic identity checks for sparse genre artifacts and dense sidecars."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from pathlib import Path

import numpy as np

DENSE_SIDECAR_SCHEMA_VERSION = "dense-genre-sidecar-v2"


def genre_artifact_identity(track_ids: np.ndarray, genre_vocab: np.ndarray, X_genre_raw: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([str(v) for v in track_ids], separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps([str(v) for v in genre_vocab], separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(X_genre_raw).tobytes())
    return digest.hexdigest()


def dense_sidecar_mismatch_reason(*, artifact: Any, sidecar: Any) -> str | None:
    if not np.array_equal(sidecar["track_ids"], artifact["track_ids"]):
        return "track_ids mismatch"
    if not np.array_equal(sidecar["genre_vocab"], artifact["genre_vocab"]):
        return "vocabulary mismatch"
    config = sidecar["emb_config"].item()
    if config.get("schema_version") != DENSE_SIDECAR_SCHEMA_VERSION:
        return "schema version mismatch"
    current = genre_artifact_identity(artifact["track_ids"], artifact["genre_vocab"], artifact["X_genre_raw"])
    if config.get("sparse_genre_identity") != current:
        return "sparse genre identity mismatch"
    return None


def dense_sidecar_mismatch_reason_from_paths(*, artifact_path: str | Path, sidecar_path: str | Path) -> str | None:
    artifact = np.load(Path(artifact_path), allow_pickle=True)
    sidecar = np.load(Path(sidecar_path), allow_pickle=True)
    return dense_sidecar_mismatch_reason(artifact=artifact, sidecar=sidecar)
```

- [ ] **Step 4: Save identity metadata and reject mismatches**

In `build_genre_embedding_sidecar()`, add:

```python
from src.genre.artifact_identity import DENSE_SIDECAR_SCHEMA_VERSION, genre_artifact_identity

emb_config["schema_version"] = DENSE_SIDECAR_SCHEMA_VERSION
emb_config["sparse_genre_identity"] = genre_artifact_identity(track_ids, np.array(genre_vocab, dtype=object), X_genre_raw)
```

In `src/features/artifacts.py`, replace the track-only check with:

```python
from src.genre.artifact_identity import dense_sidecar_mismatch_reason

reason = dense_sidecar_mismatch_reason(artifact=data, sidecar=_sc)
if reason is None:
    X_genre_dense = _sc["X_genre_dense"].astype(np.float32)
    genre_emb = _sc["genre_emb"].astype(np.float32)
else:
    logger.warning(
        "Genre embedding sidecar %s %s - ignoring. Re-run scripts/build_genre_embedding.py to rebuild.",
        _sidecar_path.name,
        reason,
    )
```

Use the same helper in `scripts/analyze_library.py::stage_verify()`.

- [ ] **Step 5: Update synthetic sidecar fixtures to include schema and sparse identity**

Where test fixtures write `emb_config`, use:

```python
from src.genre.artifact_identity import DENSE_SIDECAR_SCHEMA_VERSION, genre_artifact_identity

emb_config={
    "dim": 64,
    "schema_version": DENSE_SIDECAR_SCHEMA_VERSION,
    "sparse_genre_identity": genre_artifact_identity(track_ids, np.array(vocab, dtype=object), X_raw),
}
```

- [ ] **Step 6: Run dense fast tests**

Run:

```powershell
pytest tests/integration/test_dense_genre_integration.py -m "not slow" -q --basetemp C:\tmp\genre-dense-identity-full -o cache_dir=C:\tmp\genre-dense-identity-full-cache
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add -- src/genre/artifact_identity.py scripts/build_genre_embedding.py src/features/artifacts.py scripts/analyze_library.py tests/integration/test_dense_genre_integration.py
git commit -m "fix: reject stale dense genre sidecars"
```

## Task 7: Extend Reports And Correct Documentation

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py:585-755`
- Modify: `docs/AI_GENRE_ENRICHMENT.md`
- Modify: `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`
- Modify: `config.example.yaml`
- Test: `tests/unit/test_ai_genre_enrichment.py`

- [ ] **Step 1: Write failing report test**

Append:

```python
def test_report_includes_source_policy_metrics(tmp_path: Path):
    db_path = tmp_path / "sidecar.db"
    store = SidecarStore(db_path)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere", normalized_artist="duster",
        normalized_album="stratosphere", album_id="a1",
        source_url="bandcamp://artist/duster/album/stratosphere",
        source_type="bandcamp_tags", identity_status="confirmed",
        identity_confidence=0.9, evidence_summary="Historical Bandcamp row.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)
    store.rebuild_enriched_genres_for_release("duster::stratosphere")

    report = store.report()
    assert report["source_page_counts"]["bandcamp_release"] == 1
    assert report["accepted_basis_counts"]["authoritative_source"] == 1
    assert report["signature_policy_counts"]["genre-enrichment-v2"] == 1
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_report_includes_source_policy_metrics -q --basetemp C:\tmp\genre-policy-report -o cache_dir=C:\tmp\genre-policy-report-cache
```

Expected: FAIL because the new report keys do not exist.

- [ ] **Step 3: Add canonical source and accepted-basis report metrics**

Inside `SidecarStore.report()`:

```python
from .policy import canonical_source_type

source_page_counts: Counter[str] = Counter()
for row in conn.execute("SELECT source_type, COUNT(*) AS count FROM ai_genre_source_pages GROUP BY source_type"):
    source_page_counts[canonical_source_type(row["source_type"])] += row["count"]
accepted_basis_counts = {
    row["basis"]: row["count"]
    for row in conn.execute(
        "SELECT basis, COUNT(*) AS count FROM enriched_genres GROUP BY basis ORDER BY basis"
    )
}
```

Return:

```python
"source_page_counts": dict(source_page_counts),
"accepted_basis_counts": accepted_basis_counts,
```

- [ ] **Step 4: Correct docs**

Update `docs/AI_GENRE_ENRICHMENT.md` to state:

```markdown
- Normal artifact builds default to `playlists.ds_pipeline.genre_source: legacy`.
- `enriched` builds explicitly consume accepted sidecar signatures.
- `hybrid_shadow` writes isolated artifacts for evaluation.
- Bandcamp release tags are extracted deterministically after release-specific URL confirmation.
- Last.fm is weak corroboration only for current-policy signatures.
- Existing untouched sidecar signatures remain readable as `legacy-v0`.
```

Update `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` with the same artifact source modes and the four dense-sidecar validation checks.

- [ ] **Step 5: Run report and docs tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\genre-policy-report-full -o cache_dir=C:\tmp\genre-policy-report-full-cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py docs/AI_GENRE_ENRICHMENT.md docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md config.example.yaml tests/unit/test_ai_genre_enrichment.py
git commit -m "docs: explain stabilized genre enrichment policy"
```

## Task 8: Stabilization Verification Checkpoint

**Files:**
- No code changes.

- [ ] **Step 1: Run focused stabilization suite**

```powershell
pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_artifact_builder_enriched.py tests/unit/test_worker_enrich_artist.py tests/unit/test_genre_vocabulary.py tests/unit/test_user_overrides_storage.py tests/integration/test_dense_genre_integration.py -m "not slow" -q --basetemp C:\tmp\genre-stabilization -o cache_dir=C:\tmp\genre-stabilization-cache
```

Expected: PASS.

- [ ] **Step 2: Run the live dense baseline**

Rebuild the generated dense sidecar once so it carries the new validation schema:

```powershell
python scripts/build_genre_embedding.py --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz --skip-prior
```

Expected: `data_matrices_step1_genre_emb_dim64.npz` is regenerated without API calls.

Then run:

```powershell
pytest tests/integration/test_dense_genre_integration.py -q --basetemp C:\tmp\genre-stabilization-live -o cache_dir=C:\tmp\genre-stabilization-live-cache
```

Expected: the same three recorded live-data failures unless calibration is handled separately.

- [ ] **Step 3: Run dry-run smoke checks**

```powershell
python scripts/ai_genre_enrich.py extract-lastfm --artist "Slowdive" --dry-run
python scripts/ai_genre_enrich.py extract-bandcamp --artist "Slowdive" --dry-run
python scripts/build_beat3tower_artifacts.py --help
```

Expected: extraction commands print `network_calls: 0` routing JSON without requiring API keys; builder help lists `--genre-source`.

- [ ] **Step 4: Inspect the diff**

```powershell
git status --short
git diff --stat
```

Expected: no uncommitted stabilization files. Existing user-owned dirty files may remain.
