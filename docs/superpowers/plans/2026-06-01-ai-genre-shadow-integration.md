# AI Genre Shadow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow mapped provisional album-prior terms to participate only in isolated `hybrid_shadow` artifacts and generate comparison evidence for an adoption decision.

**Architecture:** Add a dedicated `HybridShadowGenreResolver` rather than weakening `EnrichedGenreResolver`. Fingerprint shadow outputs from accepted signature and prior snapshots plus the active sparse artifact identity and dense config. Build a separate sparse artifact, dense sidecar, and JSON report; normal `legacy` and explicit `enriched` builds remain unchanged.

**Tech Stack:** Python 3.11+, SQLite, NumPy NPZ artifacts, subprocess benchmark harness, pytest

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ai_genre_enrichment/genre_resolver.py` | Add shadow-only resolver that merges accepted provisional prior terms into read results. |
| `src/ai_genre_enrichment/storage.py` | Add accepted-signature and accepted-prior snapshot identities plus shadow-report counters. |
| `src/ai_genre_enrichment/artifact_modes.py` | Construct `HybridShadowGenreResolver` for `hybrid_shadow`; use snapshot identities in output fingerprints. |
| `src/ai_genre_enrichment/shadow_report.py` | New comparison report builder for coverage, source policy, fixed-seed genre metrics, and playlist audit deltas. |
| `scripts/ai_genre_enrich.py` | Build isolated sparse and dense shadow artifacts and write a report. |
| `scripts/run_genre_shadow_benchmarks.py` | New fixed-seed subprocess harness for legacy and shadow playlist audits. |
| `docs/AI_GENRE_ENRICHMENT.md` | Document shadow workflow and interpretation. |
| `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` | Document isolated artifact paths and report fields. |
| `tests/unit/test_ai_genre_shadow.py` | New shadow resolver, fingerprint, report, and isolation tests. |

## Task 1: Add A Shadow-Only Resolver For Provisional Prior Terms

**Files:**
- Modify: `src/ai_genre_enrichment/genre_resolver.py`
- Test: `tests/unit/test_ai_genre_shadow.py`

- [ ] **Step 1: Write failing resolver isolation tests**

Create:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from src.ai_genre_enrichment.storage import SidecarStore


def _seed_signature_and_prior(db_path: Path) -> None:
    store = SidecarStore(db_path)
    store.initialize()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO enriched_genre_signatures(
                release_key, normalized_artist, normalized_album, album_id,
                signature_json, updated_at, enrichment_policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("duster::stratosphere", "duster", "stratosphere", "a1",
             '{"genres":["slowcore"],"sources":[]}', "2026-06-01", "genre-enrichment-v2"),
        )
    store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster",
        normalized_album="stratosphere", album_id="a1", provider="openai",
        model="gpt-4o-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="h",
        status="complete", response_json={"genres": [], "warnings": []},
        warnings=[], error_message=None, token_usage={}, estimated_cost_usd=None,
        mapped_terms=[
            {"raw_term": "space rock", "normalized_term": "space rock", "canonical_slug": "space rock",
             "confidence": 0.8, "specificity": "subgenre", "taxonomy_role": "secondary_style",
             "mapping_status": "mapped", "accepted_for_shadow": 1, "auto_apply_eligible": 0, "notes": ""},
            {"raw_term": "cosmic", "normalized_term": "cosmic", "canonical_slug": None,
             "confidence": 0.8, "specificity": "broad", "taxonomy_role": "secondary_style",
             "mapping_status": "unmapped", "accepted_for_shadow": 0, "auto_apply_eligible": 0, "notes": ""},
        ],
    )


def test_normal_resolver_excludes_model_prior_terms(tmp_path: Path):
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    db_path = tmp_path / "sidecar.db"
    _seed_signature_and_prior(db_path)
    resolver = EnrichedGenreResolver(db_path)
    assert resolver.get_enriched_genres(artist="Duster", album="Stratosphere") == ["slowcore"]


def test_hybrid_shadow_resolver_includes_only_accepted_prior_terms(tmp_path: Path):
    from src.ai_genre_enrichment.genre_resolver import HybridShadowGenreResolver

    db_path = tmp_path / "sidecar.db"
    _seed_signature_and_prior(db_path)
    resolver = HybridShadowGenreResolver(db_path)
    assert resolver.get_enriched_genres(artist="Duster", album="Stratosphere") == ["slowcore", "space rock"]


def test_hybrid_shadow_resolver_uses_latest_complete_prior_only(tmp_path: Path):
    from src.ai_genre_enrichment.genre_resolver import HybridShadowGenreResolver

    db_path = tmp_path / "sidecar.db"
    _seed_signature_and_prior(db_path)
    store = SidecarStore(db_path)
    store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster",
        normalized_album="stratosphere", album_id="a1", provider="openai",
        model="gpt-4.1-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="h2",
        status="complete", response_json={"genres": [], "warnings": []},
        warnings=[], error_message=None, token_usage={}, estimated_cost_usd=None,
        mapped_terms=[
            {"raw_term": "dream pop", "normalized_term": "dream pop", "canonical_slug": "dream pop",
             "confidence": 0.8, "specificity": "subgenre", "taxonomy_role": "secondary_style",
             "mapping_status": "mapped", "accepted_for_shadow": 1, "auto_apply_eligible": 0, "notes": ""},
        ],
    )

    resolver = HybridShadowGenreResolver(db_path)
    assert resolver.get_enriched_genres(artist="Duster", album="Stratosphere") == ["slowcore", "dream pop"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py::test_normal_resolver_excludes_model_prior_terms tests/unit/test_ai_genre_shadow.py::test_hybrid_shadow_resolver_includes_only_accepted_prior_terms -q --basetemp C:\tmp\genre-shadow-resolver -o cache_dir=C:\tmp\genre-shadow-resolver-cache
```

Expected: first test PASS, second FAIL because `HybridShadowGenreResolver` does not exist.

- [ ] **Step 3: Add the explicit shadow resolver**

Append to `genre_resolver.py`:

```python
class HybridShadowGenreResolver(EnrichedGenreResolver):
    """Read accepted signatures plus mapped provisional prior terms for shadow builds only."""

    def get_enriched_genres(self, *, artist: str, album: str | None) -> list[str] | None:
        genres = super().get_enriched_genres(artist=artist, album=album) or []
        if not album:
            return genres or None
        release_key = self.make_release_key(artist, album)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT t.canonical_slug
                FROM ai_genre_model_prior_terms t
                JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
                WHERE t.release_key = ?
                  AND p.status = 'complete'
                  AND p.prior_id = (
                      SELECT MAX(p2.prior_id)
                      FROM ai_genre_model_priors p2
                      WHERE p2.release_key = p.release_key AND p2.status = 'complete'
                  )
                  AND t.accepted_for_shadow = 1
                  AND t.auto_apply_eligible = 0
                  AND t.mapping_status = 'mapped'
                  AND t.canonical_slug IS NOT NULL
                ORDER BY t.canonical_slug
                """,
                (release_key,),
            ).fetchall()
        combined = list(genres)
        seen = {genre.casefold() for genre in combined}
        for row in rows:
            genre = row["canonical_slug"]
            if genre.casefold() not in seen:
                combined.append(genre)
                seen.add(genre.casefold())
        return combined or None

    def get_all_enrichment(self) -> dict[str, dict]:
        result = super().get_all_enrichment()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.release_key, t.canonical_slug
                FROM ai_genre_model_prior_terms t
                JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
                WHERE p.status = 'complete'
                  AND p.prior_id = (
                      SELECT MAX(p2.prior_id)
                      FROM ai_genre_model_priors p2
                      WHERE p2.release_key = p.release_key AND p2.status = 'complete'
                  )
                  AND t.accepted_for_shadow = 1
                  AND t.auto_apply_eligible = 0
                  AND t.mapping_status = 'mapped'
                  AND t.canonical_slug IS NOT NULL
                ORDER BY t.release_key, t.canonical_slug
                """
            ).fetchall()
        for row in rows:
            rec = result.setdefault(row["release_key"], {"genres": [], "add": [], "remove": []})
            genres = rec["genres"] if rec["genres"] is not None else []
            if row["canonical_slug"].casefold() not in {genre.casefold() for genre in genres}:
                genres.append(row["canonical_slug"])
            rec["genres"] = genres
        return result
```

Extend the in-memory fallback schema in `_connect()` with empty model-prior tables so a missing sidecar still falls back safely.

- [ ] **Step 4: Run resolver tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- src/ai_genre_enrichment/genre_resolver.py tests/unit/test_ai_genre_shadow.py
git commit -m "feat: add hybrid shadow genre resolver"
```

## Task 2: Fingerprint Accepted Signature And Prior Snapshots

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Modify: `src/ai_genre_enrichment/artifact_modes.py`
- Test: `tests/unit/test_ai_genre_shadow.py`

- [ ] **Step 1: Write failing snapshot tests**

Append:

```python
def test_shadow_snapshot_changes_when_accepted_prior_changes(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    db_path = tmp_path / "sidecar.db"
    _seed_signature_and_prior(db_path)
    store = SidecarStore(db_path)
    before = store.model_prior_snapshot_identity()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE ai_genre_model_prior_terms SET accepted_for_shadow = 0 WHERE canonical_slug = 'space rock'"
        )
    after = store.model_prior_snapshot_identity()
    assert before != after
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py::test_shadow_snapshot_changes_when_accepted_prior_changes -q --basetemp C:\tmp\genre-shadow-snapshot -o cache_dir=C:\tmp\genre-shadow-snapshot-cache
```

Expected: FAIL because the snapshot method does not exist.

- [ ] **Step 3: Add stable query hashing**

In `storage.py`, add:

```python
import hashlib


def _hash_query_rows(rows: list[sqlite3.Row]) -> str:
    payload = [dict(row) for row in rows]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
```

Add:

```python
def signature_snapshot_identity(self) -> str:
    with self.connect() as conn:
        rows = list(conn.execute(
            """
            SELECT release_key, signature_json,
                   COALESCE(enrichment_policy_version, 'legacy-v0') AS enrichment_policy_version
            FROM enriched_genre_signatures
            ORDER BY release_key
            """
        ))
    return _hash_query_rows(rows)


def model_prior_snapshot_identity(self) -> str:
    with self.connect() as conn:
        rows = list(conn.execute(
            """
            SELECT t.release_key, t.canonical_slug, t.confidence, t.specificity, t.taxonomy_role
            FROM ai_genre_model_prior_terms t
            JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
            WHERE p.status = 'complete'
              AND p.prior_id = (
                  SELECT MAX(p2.prior_id)
                  FROM ai_genre_model_priors p2
                  WHERE p2.release_key = p.release_key AND p2.status = 'complete'
              )
              AND t.accepted_for_shadow = 1
              AND t.auto_apply_eligible = 0
              AND t.mapping_status = 'mapped'
              AND t.canonical_slug IS NOT NULL
            ORDER BY t.release_key, t.canonical_slug
            """
        ))
    return _hash_query_rows(rows)
```

- [ ] **Step 4: Construct the shadow resolver from the explicit mode**

Update `make_resolver()`:

```python
if mode is GenreArtifactSource.LEGACY:
    return None
if mode is GenreArtifactSource.HYBRID_SHADOW:
    from .genre_resolver import HybridShadowGenreResolver
    return HybridShadowGenreResolver(sidecar_db)
from .genre_resolver import EnrichedGenreResolver
return EnrichedGenreResolver(sidecar_db)
```

- [ ] **Step 5: Run snapshot and resolver tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py -q --basetemp C:\tmp\genre-shadow-snapshot-full -o cache_dir=C:\tmp\genre-shadow-snapshot-full-cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py src/ai_genre_enrichment/artifact_modes.py tests/unit/test_ai_genre_shadow.py
git commit -m "feat: fingerprint accepted genre shadow inputs"
```

## Task 3: Build Sparse And Dense Shadow Artifacts Without Touching Production

**Files:**
- Modify: `scripts/ai_genre_enrich.py:1509-1540`
- Modify: `src/ai_genre_enrichment/artifact_modes.py`
- Test: `tests/unit/test_ai_genre_shadow.py`

- [ ] **Step 1: Write failing build-isolation test**

Append:

```python
def test_hybrid_shadow_rebuild_writes_only_shadow_paths(monkeypatch, tmp_path: Path):
    from argparse import Namespace
    from scripts import ai_genre_enrich

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    active = artifacts_dir / "data_matrices_step1.npz"
    active.write_bytes(b"active")
    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    built = []

    def fake_build(args):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_bytes(b"shadow")
        built.append(Path(args.output))

    def fake_dense(path, **_kwargs):
        out = Path(path).parent / "data_matrices_step1_genre_emb_dim64.npz"
        out.write_bytes(b"dense")
        return out

    monkeypatch.setattr("scripts.build_beat3tower_artifacts.build_artifacts", fake_build)
    monkeypatch.setattr("scripts.build_genre_embedding.build_genre_embedding_sidecar", fake_dense)
    monkeypatch.setattr("src.genre.artifact_identity.genre_artifact_identity_from_path", lambda _path: "active-id")

    rc = ai_genre_enrich.cmd_rebuild_artifacts(Namespace(
        sidecar_db=sidecar, metadata_db=tmp_path / "metadata.db", artifacts_dir=artifacts_dir,
        config="config.yaml", genre_sim_path=None, genre_source="hybrid_shadow",
    ))

    assert rc == 0
    assert active.read_bytes() == b"active"
    assert built[0].parent.parent == artifacts_dir / "shadow"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py::test_hybrid_shadow_rebuild_writes_only_shadow_paths -q --basetemp C:\tmp\genre-shadow-build -o cache_dir=C:\tmp\genre-shadow-build-cache
```

Expected: FAIL because rebuilds still target the active artifact and do not build a dense shadow sidecar.

- [ ] **Step 3: Add path identity helper**

Append to `src/genre/artifact_identity.py`:

```python
from pathlib import Path


def genre_artifact_identity_from_path(path: str | Path) -> str:
    data = np.load(Path(path), allow_pickle=True)
    return genre_artifact_identity(data["track_ids"], data["genre_vocab"], data["X_genre_raw"])
```

- [ ] **Step 4: Build the explicit shadow branch**

In `cmd_rebuild_artifacts()`:

```python
from argparse import Namespace
from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource, make_resolver, shadow_output_paths
from src.ai_genre_enrichment.policy import STABILIZED_POLICY_VERSION
from src.genre.artifact_identity import (
    dense_sidecar_mismatch_reason_from_paths,
    genre_artifact_identity_from_path,
)
from scripts.build_beat3tower_artifacts import build_artifacts
from scripts.build_genre_embedding import build_genre_embedding_sidecar

mode = GenreArtifactSource.resolve(args.genre_source)
artifacts_dir = _Path(args.artifacts_dir)
active_path = artifacts_dir / "data_matrices_step1.npz"
store = SidecarStore(args.sidecar_db)
store.initialize()
resolver = make_resolver(mode, args.sidecar_db)
if mode is GenreArtifactSource.HYBRID_SHADOW:
    paths = shadow_output_paths(
        artifacts_dir=artifacts_dir,
        policy_version=STABILIZED_POLICY_VERSION,
        signature_snapshot=store.signature_snapshot_identity(),
        prior_snapshot=store.model_prior_snapshot_identity(),
        sparse_input_identity=genre_artifact_identity_from_path(active_path),
        dense_config={"dim": 64, "skip_prior": True},
    )
    output_path = paths.sparse_artifact
else:
    output_path = active_path
build_artifacts(Namespace(
    db_path=str(args.metadata_db), config=args.config, output=str(output_path),
    genre_sim_path=args.genre_sim_path, max_tracks=0, no_pca=False,
    pca_variance=0.95, clip_sigma=3.0, random_seed=42,
    no_genre_normalization=False, sidecar_db=str(args.sidecar_db),
    genre_source=mode.value, verbose=False,
), enriched_resolver=resolver)
dense_path = build_genre_embedding_sidecar(output_path, skip_prior=True)
print(f"Rebuilt {mode.value} artifacts at {output_path} dense_sidecar={dense_path}")
return 0
```

Do not expose `hybrid_shadow` through the GUI build button.

- [ ] **Step 5: Run isolation test**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/genre/artifact_identity.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_shadow.py
git commit -m "feat: build isolated sparse and dense genre shadows"
```

## Task 4: Emit Shadow Comparison Reports

**Files:**
- Create: `src/ai_genre_enrichment/shadow_report.py`
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_shadow.py`

- [ ] **Step 1: Write failing report-shape test**

Append:

```python
def test_shadow_report_contains_policy_coverage_and_provisional_counts(tmp_path: Path):
    from src.ai_genre_enrichment.shadow_report import build_shadow_report

    db_path = tmp_path / "sidecar.db"
    _seed_signature_and_prior(db_path)
    report = build_shadow_report(
        sidecar_db=db_path,
        active_artifact=tmp_path / "active.npz",
        shadow_artifact=tmp_path / "shadow.npz",
        shadow_dense_sidecar=tmp_path / "shadow_genre_emb_dim64.npz",
        policy_version="genre-enrichment-v2",
        fixed_seed_metrics={"legacy": {}, "shadow": {}},
        playlist_benchmarks=[],
    )

    assert report["policy_version"] == "genre-enrichment-v2"
    assert report["provisional_prior_terms_included"] == 1
    assert report["signature_policy_counts"]["genre-enrichment-v2"] == 1
    assert report["top_changed_genres"] == [("space rock", 1)]
    assert report["largest_signature_changes"] == [("duster::stratosphere", 1)]
    assert "stale_sidecar_validation_status" in report
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py::test_shadow_report_contains_policy_coverage_and_provisional_counts -q --basetemp C:\tmp\genre-shadow-report -o cache_dir=C:\tmp\genre-shadow-report-cache
```

Expected: FAIL because `shadow_report.py` does not exist.

- [ ] **Step 3: Add the report builder**

Create:

```python
"""Comparison report for isolated hybrid genre shadows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.genre.artifact_identity import genre_artifact_identity_from_path

from .storage import SidecarStore


def build_shadow_report(
    *,
    sidecar_db: str | Path,
    active_artifact: str | Path,
    shadow_artifact: str | Path,
    shadow_dense_sidecar: str | Path,
    policy_version: str,
    fixed_seed_metrics: dict[str, Any],
    playlist_benchmarks: list[dict[str, Any]],
) -> dict[str, Any]:
    store = SidecarStore(sidecar_db)
    report = store.report()
    prior = store.model_prior_report()
    active_path = Path(active_artifact)
    shadow_path = Path(shadow_artifact)
    artifact_identities = {
        "active": genre_artifact_identity_from_path(active_path) if active_path.exists() else None,
        "shadow": genre_artifact_identity_from_path(shadow_path) if shadow_path.exists() else None,
    }
    track_coverage = {"active": None, "shadow": None, "delta": None}
    if active_path.exists() and shadow_path.exists():
        active_data = np.load(active_path, allow_pickle=True)
        shadow_data = np.load(shadow_path, allow_pickle=True)
        active_coverage = int((active_data["X_genre_raw"] > 0).any(axis=1).sum())
        shadow_coverage = int((shadow_data["X_genre_raw"] > 0).any(axis=1).sum())
        track_coverage = {
            "active": active_coverage,
            "shadow": shadow_coverage,
            "delta": shadow_coverage - active_coverage,
        }
    dense_status = "missing"
    dense_path = Path(shadow_dense_sidecar)
    if shadow_path.exists() and dense_path.exists():
        dense_status = dense_sidecar_mismatch_reason_from_paths(
            artifact_path=shadow_path,
            sidecar_path=dense_path,
        ) or "ok"
    with store.connect() as conn:
        accepted_bandcamp = conn.execute(
            """
            SELECT COUNT(*)
            FROM enriched_genres g
            JOIN ai_genre_source_pages p ON p.source_page_id = g.source_page_id
            WHERE p.source_type IN ('bandcamp_release', 'bandcamp_tags')
            """
        ).fetchone()[0]
        lastfm_only_suppressed = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT t.normalized_tag
                FROM ai_genre_source_tags t
                JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
                JOIN ai_genre_tag_classifications c ON c.source_tag_id = t.source_tag_id
                WHERE p.source_type = 'lastfm_tags'
                  AND c.classification = 'genre_style'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM ai_genre_source_tags t2
                      JOIN ai_genre_source_pages p2 ON p2.source_page_id = t2.source_page_id
                      JOIN ai_genre_tag_classifications c2 ON c2.source_tag_id = t2.source_tag_id
                      WHERE t2.normalized_tag = t.normalized_tag
                        AND p2.source_type != 'lastfm_tags'
                        AND c2.classification = 'genre_style'
                  )
            )
            """
        ).fetchone()[0]
        changed_rows = conn.execute(
            """
            SELECT t.release_key, t.canonical_slug
            FROM ai_genre_model_prior_terms t
            JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
            WHERE p.status = 'complete'
              AND p.prior_id = (
                  SELECT MAX(p2.prior_id)
                  FROM ai_genre_model_priors p2
                  WHERE p2.release_key = p.release_key AND p2.status = 'complete'
              )
              AND t.accepted_for_shadow = 1
              AND t.mapping_status = 'mapped'
              AND t.canonical_slug IS NOT NULL
            ORDER BY t.release_key, t.canonical_slug
            """
        ).fetchall()
    genre_counts: dict[str, int] = {}
    release_counts: dict[str, int] = {}
    for row in changed_rows:
        genre_counts[row["canonical_slug"]] = genre_counts.get(row["canonical_slug"], 0) + 1
        release_counts[row["release_key"]] = release_counts.get(row["release_key"], 0) + 1
    return {
        "policy_version": policy_version,
        "active_artifact": str(Path(active_artifact)),
        "shadow_artifact": str(Path(shadow_artifact)),
        "shadow_dense_sidecar": str(Path(shadow_dense_sidecar)),
        "artifact_identities": artifact_identities,
        "track_coverage": track_coverage,
        "signature_policy_counts": report["signature_policy_counts"],
        "signature_count": sum(report["signature_policy_counts"].values()),
        "accepted_basis_counts": report["accepted_basis_counts"],
        "source_page_counts": report["source_page_counts"],
        "accepted_bandcamp_contributions": accepted_bandcamp,
        "lastfm_only_terms_suppressed": lastfm_only_suppressed,
        "provisional_prior_terms_included": len(changed_rows),
        "mapping_status_counts": prior["mapping_status_counts"],
        "top_changed_genres": sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))[:20],
        "largest_signature_changes": sorted(release_counts.items(), key=lambda item: (-item[1], item[0]))[:20],
        "fixed_seed_genre_metrics": fixed_seed_metrics,
        "playlist_benchmarks": playlist_benchmarks,
        "stale_sidecar_validation_status": dense_status,
    }
```

- [ ] **Step 4: Write the report beside each shadow artifact**

After the shadow dense build:

```python
report = build_shadow_report(
    sidecar_db=args.sidecar_db,
    active_artifact=active_path,
    shadow_artifact=output_path,
    shadow_dense_sidecar=dense_path,
    policy_version=STABILIZED_POLICY_VERSION,
    fixed_seed_metrics={"legacy": {}, "shadow": {}},
    playlist_benchmarks=[],
)
paths.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(f"Shadow report: {paths.report}")
```

Task 5 fills the fixed-seed and playlist benchmark sections.

- [ ] **Step 5: Run report tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py -q --basetemp C:\tmp\genre-shadow-report-full -o cache_dir=C:\tmp\genre-shadow-report-full-cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/shadow_report.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_shadow.py
git commit -m "feat: emit genre shadow comparison report"
```

## Task 5: Add Fixed-Seed Genre And Playlist Benchmark Deltas

**Files:**
- Create: `scripts/run_genre_shadow_benchmarks.py`
- Modify: `src/ai_genre_enrichment/shadow_report.py`
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_shadow.py`

- [ ] **Step 1: Write failing delta test**

Append:

```python
def test_shadow_report_computes_playlist_benchmark_deltas():
    from src.ai_genre_enrichment.shadow_report import playlist_benchmark_deltas

    result = playlist_benchmark_deltas(
        legacy=[{"seed": "duster", "candidate_pool_size": 100, "min_transition": 0.2, "distinct_artists": 20}],
        shadow=[{"seed": "duster", "candidate_pool_size": 130, "min_transition": 0.25, "distinct_artists": 22}],
    )

    assert result == [{
        "seed": "duster",
        "candidate_pool_size_delta": 30,
        "min_transition_delta": 0.05,
        "distinct_artists_delta": 2,
    }]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py::test_shadow_report_computes_playlist_benchmark_deltas -q --basetemp C:\tmp\genre-shadow-deltas -o cache_dir=C:\tmp\genre-shadow-deltas-cache
```

Expected: FAIL because `playlist_benchmark_deltas()` does not exist.

- [ ] **Step 3: Add deterministic delta aggregation**

Append to `shadow_report.py`:

```python
def playlist_benchmark_deltas(*, legacy: list[dict[str, Any]], shadow: list[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy_by_seed = {row["seed"]: row for row in legacy}
    result: list[dict[str, Any]] = []
    for row in shadow:
        base = legacy_by_seed[row["seed"]]
        result.append({
            "seed": row["seed"],
            "candidate_pool_size_delta": row["candidate_pool_size"] - base["candidate_pool_size"],
            "min_transition_delta": round(row["min_transition"] - base["min_transition"], 6),
            "distinct_artists_delta": row["distinct_artists"] - base["distinct_artists"],
        })
    return result
```

- [ ] **Step 4: Add fixed-seed benchmark harness**

Create `scripts/run_genre_shadow_benchmarks.py` with four fixed seed IDs matching `scripts/measure_genre_baseline.py::REFERENCE_CASES`. For each artifact:

1. Load `config.yaml` with `yaml.safe_load`.
2. Write a temporary config with `playlists.ds_pipeline.artifact_path` set to the selected artifact.
3. Run:

```powershell
python main_app.py --config $tempConfig --artist $caseArtist --tracks 30 --dry-run --verbose --anchor-seed-ids $caseTrackId
```

4. Parse the existing JSON blocks using:

```python
from scripts.sweep_pier_bridge_dials import _parse_json_block

summary = _parse_json_block(stdout, "2) Playlist Summary")
pool = _parse_json_block(stdout, "3) Pool / Gating Summary")
row = {
    "seed": case["name"],
    "candidate_pool_size": pool["candidate_pool_stats"]["pool_size"],
    "min_transition": summary["min_transition"],
    "mean_transition": summary["mean_transition"],
    "distinct_artists": summary["distinct_artists"],
}
```

Write:

```json
{
  "legacy": [],
  "shadow": [],
  "deltas": []
}
```

Use `playlist_benchmark_deltas()` for `deltas`.

- [ ] **Step 5: Add artifact-level fixed-seed genre metrics**

Reuse:

```python
from scripts.measure_genre_baseline import REFERENCE_CASES, measure_case
```

Load active and shadow NPZ files and call `measure_case()` for each reference case. Add the resulting `legacy`, `shadow`, and per-floor deltas to `fixed_seed_genre_metrics`.

- [ ] **Step 6: Run unit tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_shadow.py -q --basetemp C:\tmp\genre-shadow-deltas-full -o cache_dir=C:\tmp\genre-shadow-deltas-full-cache
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add -- scripts/run_genre_shadow_benchmarks.py src/ai_genre_enrichment/shadow_report.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_shadow.py
git commit -m "feat: compare fixed-seed genre shadow quality"
```

## Task 6: Document And Verify Shadow Evaluation

**Files:**
- Modify: `docs/AI_GENRE_ENRICHMENT.md`
- Modify: `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`

- [ ] **Step 1: Document the explicit workflow**

Add:

```markdown
## Hybrid Shadow Evaluation

`python scripts/ai_genre_enrich.py rebuild-artifacts --genre-source hybrid_shadow`
writes a fingerprinted sparse artifact, dense sidecar, and comparison report under
`data/artifacts/beat3tower_32k/shadow/<fingerprint>/`.

Normal GUI and library-analysis builds remain `legacy`. Shadow reports separate
`legacy-v0` signatures from current-policy rows and identify provisional
model-prior terms. Shadow output is evaluation-only.
```

- [ ] **Step 2: Run full focused rollout suite**

```powershell
pytest tests/unit/test_ai_genre_enrichment.py tests/unit/test_ai_genre_model_prior.py tests/unit/test_ai_genre_shadow.py tests/unit/test_artifact_builder_enriched.py tests/unit/test_worker_enrich_artist.py tests/unit/test_genre_vocabulary.py tests/unit/test_user_overrides_storage.py tests/integration/test_dense_genre_integration.py -m "not slow" -q --basetemp C:\tmp\genre-shadow-rollout -o cache_dir=C:\tmp\genre-shadow-rollout-cache
```

Expected: PASS.

- [ ] **Step 3: Run live dense baseline**

```powershell
pytest tests/integration/test_dense_genre_integration.py -q --basetemp C:\tmp\genre-shadow-live -o cache_dir=C:\tmp\genre-shadow-live-cache
```

Expected: either the recorded three pre-existing failures or an explicitly reviewed calibration change.

- [ ] **Step 4: Back up the live sidecar before a manual shadow run**

```powershell
Copy-Item -LiteralPath data\ai_genre_enrichment.db -Destination ("data\ai_genre_enrichment.db.bak_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
```

Expected: timestamped sidecar backup exists. This does not modify `data/metadata.db`.

- [ ] **Step 5: Run manual shadow evaluation**

```powershell
python scripts/ai_genre_enrich.py rebuild-artifacts --genre-source hybrid_shadow
```

Expected: fingerprinted sparse artifact, dense sidecar, and `comparison_report.json` under `data/artifacts/beat3tower_32k/shadow/`. Active production artifacts retain their previous hashes.

- [ ] **Step 6: Commit docs**

```powershell
git add -- docs/AI_GENRE_ENRICHMENT.md docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md
git commit -m "docs: explain hybrid genre shadow evaluation"
```
