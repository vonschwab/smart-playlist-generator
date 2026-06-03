# Hybrid Genre Evidence Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real deterministic/LLM hybrid genre enrichment model that fuses Bandcamp, Last.fm, local metadata, curated taxonomy, and album-level LLM hypotheses into usable album genre signatures with explicit noise suppression.

**Architecture:** Treat the existing `model-prior` code as one evidence source, not the product. Add a focused `hybrid_evidence.py` fusion layer that consumes source tags plus latest model-prior terms, assigns source-aware decisions, and emits an explainable report. First ship `hybrid-enrich-one --dry-run`; only after that works, add explicit sidecar persistence and explicit signature application.

**Tech Stack:** Python 3.11+, SQLite sidecar DB, existing `SidecarStore`, existing `GenreVocabulary`, existing OpenAI Responses model-prior client, pytest.

---

## Non-Negotiable Scope Guard

This plan must not become another stabilization loop.

Do not do these in this milestone:

- Do not rebuild sparse or dense artifacts.
- Do not modify `data/metadata.db`.
- Do not scrape RYM or Google UI.
- Do not add GUI controls.
- Do not redesign existing artifact modes.
- Do not run broad live enrichment batches.
- Do not rewrite the old enrichment pipeline before the new CLI report exists.

Allowed work:

- Fix direct blockers in the already-built `model-prior` CLI.
- Add a hybrid evidence fusion module.
- Add focused storage read/write helpers for hybrid reports.
- Add `hybrid-enrich-one`.
- Add tests that prove Last.fm noise is suppressed and Bandcamp/model/local evidence is fused.

## Correct Vocabulary

Use these names in code and docs:

- **Model prior:** album-level LLM hypothesis. One signal only.
- **Hybrid evidence model:** the real enrichment model.
- **Source evidence:** Bandcamp, Last.fm, local/Discogs/MusicBrainz tags already present in sidecar/local metadata.
- **Fused decision:** accepted, rejected_noise, provisional, or needs_review genre decision.

## File Structure

| File | Responsibility |
|---|---|
| `src/ai_genre_enrichment/hybrid_evidence.py` | New deterministic fusion policy, source weights, evidence dataclasses, and report builder. |
| `src/ai_genre_enrichment/storage.py` | Add read helpers for source evidence and latest model-prior terms; later persist hybrid reports. |
| `scripts/ai_genre_enrich.py` | Add `hybrid-enrich-one`; fix model-prior cache order and subcommand model flags. |
| `tests/unit/test_ai_genre_hybrid_evidence.py` | Focused fusion-policy tests. |
| `tests/unit/test_ai_genre_hybrid_cli.py` | CLI dry-run and no-side-effect tests. |
| `docs/AI_GENRE_ENRICHMENT.md` | Document the hybrid model, source hierarchy, and review-reduction behavior. |

## Success Shape

The first successful command should be:

```powershell
python scripts/ai_genre_enrich.py hybrid-enrich-one --artist "Duster" --album "Stratosphere" --dry-run
```

It should print JSON shaped like:

```json
{
  "release_key": "duster::stratosphere",
  "dry_run": true,
  "accepted_genres": [
    {
      "term": "slowcore",
      "confidence": 0.94,
      "basis": "bandcamp_release+model_prior+taxonomy",
      "sources": ["bandcamp_release", "model_prior"],
      "reason": "Strong release-specific Bandcamp evidence corroborated by model taxonomy."
    }
  ],
  "provisional_genres": [],
  "rejected_noise": [
    {
      "term": "seen live",
      "sources": ["lastfm_tags"],
      "reason": "Last.fm-only non-genre noise."
    }
  ],
  "needs_review": []
}
```

## Task 1: Fix Direct Model-Prior Blockers

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Add failing cache-order and model-flag tests**

Append to `tests/unit/test_ai_genre_model_prior.py`:

```python
def test_model_prior_missing_only_skips_before_api_call(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    release = ai_genre_enrich._discover(type("Args", (), {
        "metadata_db": metadata_db,
        "limit": 1,
        "artist": "Duster",
        "album": "Stratosphere",
        "generic_only": False,
        "min_existing_specific_genres": None,
    })())[0]

    payload = ai_genre_enrich.build_model_prior_payload(release)
    store.record_model_prior(
        release_key=release.release_key,
        normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album,
        album_id=release.album_id,
        provider="openai",
        model="gpt-4o-mini",
        prompt_version=ai_genre_enrich.MODEL_PRIOR_PROMPT_VERSION,
        taxonomy_version=ai_genre_enrich.MODEL_PRIOR_TAXONOMY_VERSION,
        schema_version=ai_genre_enrich.MODEL_PRIOR_SCHEMA_VERSION,
        enrichment_policy_version=ai_genre_enrich.STABILIZED_POLICY_VERSION,
        input_hash=ai_genre_enrich.stable_input_hash(payload),
        status="complete",
        response_json={"genres": [], "warnings": []},
        warnings=[],
        error_message=None,
        token_usage={},
        estimated_cost_usd=None,
        mapped_terms=[],
    )

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called before cache check")),
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "model-prior",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--missing-only",
    ])

    assert rc == 0
    assert "existing-model-prior" in capsys.readouterr().out


def test_model_prior_subcommands_accept_model_after_command():
    from scripts.ai_genre_enrich import build_parser

    args = build_parser().parse_args([
        "model-prior-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--model", "gpt-4.1-mini",
    ])

    assert args.model == "gpt-4.1-mini"
```

- [ ] **Step 2: Run the two tests and verify failure**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_model_prior_missing_only_skips_before_api_call tests/unit/test_ai_genre_model_prior.py::test_model_prior_subcommands_accept_model_after_command -q --basetemp C:\tmp\hybrid-prior-blockers -o cache_dir=C:\tmp\hybrid-prior-blockers-cache
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Move cache check before the API call**

In `scripts/ai_genre_enrich.py`, update `_run_model_prior_release()` so it computes payload and input hash, initializes the store, checks cache, and only then creates `OpenAIEnrichmentClient`.

The top of the function should follow this order:

```python
def _run_model_prior_release(args: argparse.Namespace, release: ReleasePayload) -> int:
    payload = build_model_prior_payload(release)
    input_hash = stable_input_hash(payload)

    if not args.dry_run:
        store = SidecarStore(args.sidecar_db)
        store.initialize()
        cached = store.find_model_prior(
            release_key=release.release_key,
            provider="openai",
            model=args.model,
            prompt_version=MODEL_PRIOR_PROMPT_VERSION,
            taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION,
            schema_version=MODEL_PRIOR_SCHEMA_VERSION,
            enrichment_policy_version=STABILIZED_POLICY_VERSION,
            input_hash=input_hash,
        )
        if cached and getattr(args, "missing_only", False) and not args.force:
            print(f"existing-model-prior {release.release_key}")
            return 0
        if cached and cached["status"] == "complete" and not args.force:
            print(f"cached-model-prior {release.release_key}")
            return 0
    else:
        store = None

    client = OpenAIEnrichmentClient(model=args.model, dry_run=args.dry_run, web_mode="off")
```

Keep the existing `record_model_prior()` call after the API result. Assert `store is not None` before recording.

- [ ] **Step 4: Add `--model` to the model-prior subcommands**

In `build_parser()`, add:

```python
model_prior_one.add_argument("--model", default=DEFAULT_MODEL)
model_prior.add_argument("--model", default=DEFAULT_MODEL)
```

- [ ] **Step 5: Run focused model-prior tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py -q --basetemp C:\tmp\hybrid-prior-fixed -o cache_dir=C:\tmp\hybrid-prior-fixed-cache
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add -- scripts/ai_genre_enrich.py tests/unit/test_ai_genre_model_prior.py
git commit -m "fix: make album model-prior cache effective"
```

## Task 2: Add The Hybrid Evidence Fusion Core

**Files:**
- Create: `src/ai_genre_enrichment/hybrid_evidence.py`
- Create: `tests/unit/test_ai_genre_hybrid_evidence.py`

- [ ] **Step 1: Write failing fusion tests**

Create `tests/unit/test_ai_genre_hybrid_evidence.py`:

```python
from __future__ import annotations

from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence


def test_bandcamp_and_model_accepts_specific_genre():
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="slowcore", source_type="bandcamp_release", confidence=0.90),
            EvidenceTerm(term="slowcore", source_type="model_prior", confidence=0.88),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["slowcore"]
    assert report.accepted_genres[0].basis == "bandcamp_release+model_prior+taxonomy"
    assert report.accepted_genres[0].confidence >= 0.90


def test_lastfm_only_is_rejected_noise():
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[EvidenceTerm(term="seen live", source_type="lastfm_tags", confidence=0.70)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    assert report.rejected_noise[0].term == "seen live"
    assert "Last.fm-only" in report.rejected_noise[0].reason


def test_model_only_high_confidence_sparse_release_is_provisional():
    report = fuse_hybrid_evidence(
        release_key="obscure::album",
        evidence=[EvidenceTerm(term="ambient americana", source_type="model_prior", confidence=0.86)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    assert report.provisional_genres[0].term == "ambient americana"
    assert report.provisional_genres[0].basis == "model_prior+taxonomy"


def test_local_and_model_can_accept_when_no_stronger_conflict():
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[
            EvidenceTerm(term="dream pop", source_type="local_metadata", confidence=0.65),
            EvidenceTerm(term="dream pop", source_type="model_prior", confidence=0.82),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres[0].term == "dream pop"
    assert "local_metadata" in report.accepted_genres[0].sources
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_evidence.py -q --basetemp C:\tmp\hybrid-core-red -o cache_dir=C:\tmp\hybrid-core-red-cache
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Implement the fusion dataclasses and policy**

Create `src/ai_genre_enrichment/hybrid_evidence.py`:

```python
"""Deterministic fusion policy for hybrid album genre evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DecisionKind = Literal["accepted", "provisional", "rejected_noise", "needs_review"]

SOURCE_WEIGHTS: dict[str, float] = {
    "bandcamp_release": 0.95,
    "official_release": 0.95,
    "discogs": 0.78,
    "musicbrainz": 0.76,
    "local_metadata": 0.70,
    "model_prior": 0.68,
    "lastfm_tags": 0.25,
}

LASTFM_SOURCE_TYPES = {"lastfm_tags", "lastfm"}
MODEL_SOURCE_TYPES = {"model_prior"}
STRONG_SOURCE_TYPES = {"bandcamp_release", "official_release"}
MEDIUM_SOURCE_TYPES = {"local_metadata", "discogs", "musicbrainz"}


@dataclass(frozen=True)
class EvidenceTerm:
    term: str
    source_type: str
    confidence: float
    canonical_slug: str | None = None
    mapping_status: str = "mapped"
    notes: str = ""


@dataclass(frozen=True)
class FusedGenreDecision:
    term: str
    confidence: float
    basis: str
    sources: list[str]
    reason: str


@dataclass(frozen=True)
class HybridGenreReport:
    release_key: str
    accepted_genres: list[FusedGenreDecision]
    provisional_genres: list[FusedGenreDecision]
    rejected_noise: list[FusedGenreDecision]
    needs_review: list[FusedGenreDecision]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fuse_hybrid_evidence(
    *,
    release_key: str,
    evidence: list[EvidenceTerm],
    sparse_release: bool,
) -> HybridGenreReport:
    grouped: dict[str, list[EvidenceTerm]] = {}
    for item in evidence:
        term = _decision_term(item)
        if not term:
            continue
        grouped.setdefault(term, []).append(item)

    accepted: list[FusedGenreDecision] = []
    provisional: list[FusedGenreDecision] = []
    rejected: list[FusedGenreDecision] = []
    review: list[FusedGenreDecision] = []

    for term in sorted(grouped):
        items = grouped[term]
        sources = sorted({item.source_type for item in items})
        score = _score(items)

        if all(source in LASTFM_SOURCE_TYPES for source in sources):
            rejected.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="lastfm_only",
                sources=sources,
                reason="Last.fm-only signal is treated as noisy corroboration, not accepted evidence.",
            ))
            continue

        if any(source in STRONG_SOURCE_TYPES for source in sources):
            basis = _basis(sources)
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.90),
                basis=basis,
                sources=sources,
                reason="Strong release-specific source evidence supports this mapped genre.",
            ))
            continue

        if "model_prior" in sources and any(source in MEDIUM_SOURCE_TYPES for source in sources):
            accepted.append(FusedGenreDecision(
                term=term,
                confidence=max(score, 0.78),
                basis=_basis(sources),
                sources=sources,
                reason="Model taxonomy agrees with existing non-Last.fm metadata.",
            ))
            continue

        if sources == ["model_prior"] and sparse_release and score >= 0.58:
            provisional.append(FusedGenreDecision(
                term=term,
                confidence=score,
                basis="model_prior+taxonomy",
                sources=sources,
                reason="Sparse release has a high-confidence mapped model taxonomy signal.",
            ))
            continue

        review.append(FusedGenreDecision(
            term=term,
            confidence=score,
            basis=_basis(sources),
            sources=sources,
            reason="Evidence is mapped but not strong enough for automatic acceptance.",
        ))

    return HybridGenreReport(
        release_key=release_key,
        accepted_genres=accepted,
        provisional_genres=provisional,
        rejected_noise=rejected,
        needs_review=review,
    )


def _decision_term(item: EvidenceTerm) -> str:
    if item.mapping_status not in {"mapped", "canonical", "alias"}:
        return ""
    return item.canonical_slug or item.term.strip().casefold()


def _score(items: list[EvidenceTerm]) -> float:
    weighted = 0.0
    total_weight = 0.0
    for item in items:
        weight = SOURCE_WEIGHTS.get(item.source_type, 0.40)
        weighted += weight * max(0.0, min(1.0, item.confidence))
        total_weight += weight
    if total_weight == 0:
        return 0.0
    agreement_bonus = min(0.10, 0.03 * max(0, len({item.source_type for item in items}) - 1))
    return min(1.0, (weighted / total_weight) + agreement_bonus)


def _basis(sources: list[str]) -> str:
    ordered = [
        source
        for source in [
            "bandcamp_release",
            "official_release",
            "discogs",
            "musicbrainz",
            "local_metadata",
            "model_prior",
            "lastfm_tags",
        ]
        if source in sources
    ]
    extra = sorted(source for source in sources if source not in ordered)
    return "+".join(ordered + extra + ["taxonomy"])
```

- [ ] **Step 4: Run fusion tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_evidence.py -q --basetemp C:\tmp\hybrid-core-green -o cache_dir=C:\tmp\hybrid-core-green-cache
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/ai_genre_enrichment/hybrid_evidence.py tests/unit/test_ai_genre_hybrid_evidence.py
git commit -m "feat: add hybrid genre evidence fusion"
```

## Task 3: Collect Source Evidence And Model-Prior Terms

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Modify: `src/ai_genre_enrichment/hybrid_evidence.py`
- Test: `tests/unit/test_ai_genre_hybrid_evidence.py`

- [ ] **Step 1: Add failing collection test**

Append to `tests/unit/test_ai_genre_hybrid_evidence.py`:

```python
from pathlib import Path


def test_collect_hybrid_evidence_reads_sidecar_sources_and_prior(tmp_path: Path):
    from src.ai_genre_enrichment.hybrid_evidence import collect_hybrid_evidence
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://example.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    store.record_model_prior(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        provider="openai",
        model="gpt-4o-mini",
        prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1",
        schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2",
        input_hash="hash-1",
        status="complete",
        response_json={"genres": [], "warnings": []},
        warnings=[],
        error_message=None,
        token_usage={},
        estimated_cost_usd=None,
        mapped_terms=[{
            "raw_term": "slowcore",
            "normalized_term": "slowcore",
            "canonical_slug": "slowcore",
            "confidence": 0.86,
            "specificity": "subgenre",
            "taxonomy_role": "core_style",
            "mapping_status": "mapped",
            "accepted_for_shadow": 1,
            "auto_apply_eligible": 0,
            "notes": "",
        }],
    )

    evidence = collect_hybrid_evidence(store, "duster::stratosphere")
    source_types = sorted({item.source_type for item in evidence if item.term == "slowcore"})

    assert source_types == ["bandcamp_release", "model_prior"]
```

- [ ] **Step 2: Run the collection test and verify failure**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_evidence.py::test_collect_hybrid_evidence_reads_sidecar_sources_and_prior -q --basetemp C:\tmp\hybrid-collect-red -o cache_dir=C:\tmp\hybrid-collect-red-cache
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Add storage readers**

Add these methods to `SidecarStore` in `src/ai_genre_enrichment/storage.py`:

```python
def hybrid_source_terms_for_release(self, release_key: str) -> list[dict[str, Any]]:
    with self.connect() as conn:
        rows = conn.execute(
            """
            SELECT
                p.source_type,
                t.normalized_tag AS term,
                t.canonical_genre AS canonical_slug,
                t.classification AS mapping_status,
                t.confidence,
                p.identity_confidence
            FROM ai_genre_source_tags t
            JOIN ai_genre_source_pages p ON p.source_page_id = t.source_page_id
            WHERE p.release_key = ?
            ORDER BY p.source_type, t.normalized_tag
            """,
            (release_key,),
        ).fetchall()
        return [dict(row) for row in rows]


def latest_model_prior_terms_for_release(self, release_key: str) -> list[dict[str, Any]]:
    with self.connect() as conn:
        rows = conn.execute(
            """
            SELECT t.*
            FROM ai_genre_model_prior_terms t
            JOIN ai_genre_model_priors p ON p.prior_id = t.prior_id
            WHERE t.release_key = ?
              AND p.status = 'complete'
              AND p.updated_at = (
                  SELECT MAX(p2.updated_at)
                  FROM ai_genre_model_priors p2
                  WHERE p2.release_key = p.release_key
                    AND p2.status = 'complete'
              )
            ORDER BY t.normalized_term
            """,
            (release_key,),
        ).fetchall()
        return [dict(row) for row in rows]
```

If the actual source tag column names differ, inspect `SidecarStore.initialize()` and adjust the SQL to use the real names. Keep the returned dict keys exactly as shown.

- [ ] **Step 4: Add `collect_hybrid_evidence()`**

Append to `src/ai_genre_enrichment/hybrid_evidence.py`:

```python
def collect_hybrid_evidence(store: object, release_key: str) -> list[EvidenceTerm]:
    evidence: list[EvidenceTerm] = []

    for row in store.hybrid_source_terms_for_release(release_key):
        mapping = str(row.get("mapping_status") or "")
        if mapping == "genre_style":
            mapping = "mapped"
        confidence = float(row.get("confidence") or row.get("identity_confidence") or 0.50)
        evidence.append(EvidenceTerm(
            term=str(row["term"]),
            source_type=str(row["source_type"]),
            confidence=confidence,
            canonical_slug=row.get("canonical_slug") or row["term"],
            mapping_status=mapping,
        ))

    for row in store.latest_model_prior_terms_for_release(release_key):
        evidence.append(EvidenceTerm(
            term=str(row["normalized_term"]),
            source_type="model_prior",
            confidence=float(row["confidence"]),
            canonical_slug=row.get("canonical_slug") or row["normalized_term"],
            mapping_status=str(row["mapping_status"]),
            notes=str(row.get("notes") or ""),
        ))

    return evidence
```

- [ ] **Step 5: Run collection and fusion tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_evidence.py -q --basetemp C:\tmp\hybrid-collect-green -o cache_dir=C:\tmp\hybrid-collect-green-cache
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py src/ai_genre_enrichment/hybrid_evidence.py tests/unit/test_ai_genre_hybrid_evidence.py
git commit -m "feat: collect hybrid genre evidence"
```

## Task 4: Add `hybrid-enrich-one --dry-run`

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Create: `tests/unit/test_ai_genre_hybrid_cli.py`

- [ ] **Step 1: Write failing CLI dry-run test**

Create `tests/unit/test_ai_genre_hybrid_cli.py`:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_hybrid_enrich_one_dry_run_fuses_existing_sidecar_without_api(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://example.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI should not be called in dry-run")),
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--dry-run",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["release_key"] == "duster::stratosphere"
    assert output["dry_run"] is True
    assert output["accepted_genres"][0]["term"] == "slowcore"
```

- [ ] **Step 2: Run CLI test and verify failure**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_cli.py -q --basetemp C:\tmp\hybrid-cli-red -o cache_dir=C:\tmp\hybrid-cli-red-cache
```

Expected:

```text
FAILED
```

- [ ] **Step 3: Add parser and command routing**

In `scripts/ai_genre_enrich.py`, import:

```python
from src.ai_genre_enrichment.hybrid_evidence import collect_hybrid_evidence, fuse_hybrid_evidence
```

Add command routing near the existing command dispatch:

```python
if args.command == "hybrid-enrich-one":
    return cmd_hybrid_enrich_one(args)
```

Add parser:

```python
hybrid_one = sub.add_parser("hybrid-enrich-one", help="Fuse source evidence and model prior into one album genre report")
add_release_filters(hybrid_one)
hybrid_one.add_argument("--dry-run", action="store_true")
hybrid_one.add_argument("--include-provisional", action="store_true")
```

- [ ] **Step 4: Implement command**

Append near the model-prior command functions:

```python
def cmd_hybrid_enrich_one(args: argparse.Namespace) -> int:
    releases = _discover(args)
    if len(releases) != 1:
        print(f"Expected exactly one release, found {len(releases)}.")
        return 2

    release = releases[0]
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    evidence = collect_hybrid_evidence(store, release.release_key)
    sparse_release = not release.existing_genres_by_source
    report = fuse_hybrid_evidence(
        release_key=release.release_key,
        evidence=evidence,
        sparse_release=sparse_release,
    ).to_dict()
    report["dry_run"] = bool(args.dry_run)
    report["evidence_count"] = len(evidence)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0
```

Dry-run and non-dry-run are intentionally identical in this task. Persistence comes later.

- [ ] **Step 5: Run CLI tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_hybrid_evidence.py -q --basetemp C:\tmp\hybrid-cli-green -o cache_dir=C:\tmp\hybrid-cli-green-cache
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add -- scripts/ai_genre_enrich.py tests/unit/test_ai_genre_hybrid_cli.py
git commit -m "feat: add hybrid genre enrichment dry-run"
```

## Stop Point A: User Review

Stop here and run one or two real dry-run examples against temporary sidecars only.

Do not implement persistence until the JSON output is inspected.

Required review question:

```text
Does this report shape expose enough evidence to trust the accepted/rejected/provisional decisions?
```

## Task 5: Persist Hybrid Reports Without Applying Signatures

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_hybrid_cli.py`

- [ ] **Step 1: Add failing persistence test**

Append to `tests/unit/test_ai_genre_hybrid_cli.py`:

```python
def test_hybrid_enrich_one_live_persists_report_without_signature_mutation(tmp_path: Path):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://example.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
    ])

    assert rc == 0
    persisted = SidecarStore(sidecar).latest_hybrid_report("duster::stratosphere")
    assert persisted is not None
    assert persisted["accepted_genres"][0]["term"] == "slowcore"
    with sqlite3.connect(sidecar) as conn:
        assert conn.execute("SELECT COUNT(*) FROM enriched_genre_signatures").fetchone()[0] == 0
```

- [ ] **Step 2: Add hybrid report tables**

Inside `SidecarStore.initialize()` add:

```sql
CREATE TABLE IF NOT EXISTS ai_genre_hybrid_reports (
    hybrid_report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_key TEXT NOT NULL,
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    report_json TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    accepted_count INTEGER NOT NULL,
    provisional_count INTEGER NOT NULL,
    rejected_noise_count INTEGER NOT NULL,
    needs_review_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ai_genre_hybrid_reports_release
    ON ai_genre_hybrid_reports (release_key, created_at);
```

- [ ] **Step 3: Add persistence helpers**

Add to `SidecarStore`:

```python
def record_hybrid_report(
    self,
    *,
    release_key: str,
    normalized_artist: str,
    normalized_album: str,
    album_id: str | None,
    report: dict[str, Any],
    evidence_count: int,
) -> int:
    now = _now_iso()
    with self.connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_genre_hybrid_reports (
                release_key, normalized_artist, normalized_album, album_id,
                report_json, evidence_count, accepted_count, provisional_count,
                rejected_noise_count, needs_review_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                release_key,
                normalized_artist,
                normalized_album,
                album_id,
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                evidence_count,
                len(report.get("accepted_genres", [])),
                len(report.get("provisional_genres", [])),
                len(report.get("rejected_noise", [])),
                len(report.get("needs_review", [])),
                now,
            ),
        )
        return int(cur.lastrowid)


def latest_hybrid_report(self, release_key: str) -> dict[str, Any] | None:
    with self.connect() as conn:
        row = conn.execute(
            """
            SELECT report_json
            FROM ai_genre_hybrid_reports
            WHERE release_key = ?
            ORDER BY created_at DESC, hybrid_report_id DESC
            LIMIT 1
            """,
            (release_key,),
        ).fetchone()
        return json.loads(row["report_json"]) if row else None
```

- [ ] **Step 4: Persist only on non-dry-run**

In `cmd_hybrid_enrich_one()`, after building the report:

```python
if not args.dry_run:
    store.record_hybrid_report(
        release_key=release.release_key,
        normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album,
        album_id=release.album_id,
        report=report,
        evidence_count=len(evidence),
    )
```

- [ ] **Step 5: Run hybrid CLI tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_cli.py -q --basetemp C:\tmp\hybrid-persist-green -o cache_dir=C:\tmp\hybrid-persist-green-cache
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_hybrid_cli.py
git commit -m "feat: persist hybrid genre reports"
```

## Task 6: Explicitly Apply Hybrid Signatures

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_ai_genre_hybrid_cli.py`

- [ ] **Step 1: Add failing apply test**

Append to `tests/unit/test_ai_genre_hybrid_cli.py`:

```python
def test_hybrid_enrich_one_apply_writes_accepted_signature_only(tmp_path: Path):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://example.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--apply",
    ])

    assert rc == 0
    with sqlite3.connect(sidecar) as conn:
        genres = [row[0] for row in conn.execute(
            "SELECT genre FROM enriched_genres WHERE release_key = ? ORDER BY genre",
            ("duster::stratosphere",),
        )]
    assert genres == ["slowcore"]
```

- [ ] **Step 2: Add explicit `--apply` parser flag**

In the `hybrid-enrich-one` parser:

```python
hybrid_one.add_argument(
    "--apply",
    action="store_true",
    help="Write accepted hybrid decisions to enriched sidecar signatures. Never writes metadata.db.",
)
```

- [ ] **Step 3: Add an apply helper that writes only accepted decisions**

Add to `SidecarStore`:

```python
def replace_hybrid_signature(
    self,
    *,
    release_key: str,
    normalized_artist: str,
    normalized_album: str,
    album_id: str | None,
    accepted_genres: list[dict[str, Any]],
) -> None:
    now = _now_iso()
    signature = {
        "genres": [item["term"] for item in accepted_genres],
        "sources": sorted({source for item in accepted_genres for source in item.get("sources", [])}),
        "basis": "hybrid_evidence",
        "updated_at": now,
    }
    with self.connect() as conn:
        conn.execute("DELETE FROM enriched_genres WHERE release_key = ?", (release_key,))
        conn.executemany(
            """
            INSERT INTO enriched_genres (
                release_key, normalized_artist, normalized_album, album_id,
                genre, confidence, basis, source_count, created_at, enrichment_policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, 'hybrid_evidence', ?, ?, ?)
            """,
            [
                (
                    release_key,
                    normalized_artist,
                    normalized_album,
                    album_id,
                    item["term"],
                    float(item["confidence"]),
                    len(item.get("sources", [])),
                    now,
                    "genre-enrichment-v2",
                )
                for item in accepted_genres
            ],
        )
        conn.execute("DELETE FROM enriched_genre_signatures WHERE release_key = ?", (release_key,))
        conn.execute(
            """
            INSERT INTO enriched_genre_signatures (
                release_key, normalized_artist, normalized_album, album_id,
                signature_json, created_at, enrichment_policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                release_key,
                normalized_artist,
                normalized_album,
                album_id,
                json.dumps(signature, ensure_ascii=False, sort_keys=True),
                now,
                "genre-enrichment-v2",
            ),
        )
```

If the table columns differ in this branch, inspect the existing inserts in `rebuild_enriched_genres_for_release()` and match them exactly. Preserve `basis='hybrid_evidence'`.

- [ ] **Step 4: Call the helper only when `--apply` is set**

In `cmd_hybrid_enrich_one()`:

```python
if args.apply:
    store.replace_hybrid_signature(
        release_key=release.release_key,
        normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album,
        album_id=release.album_id,
        accepted_genres=list(report["accepted_genres"]),
    )
```

Do not apply `provisional_genres` in this task.

- [ ] **Step 5: Run apply and existing enrichment tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\hybrid-apply-green -o cache_dir=C:\tmp\hybrid-apply-green-cache
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```powershell
git add -- scripts/ai_genre_enrich.py src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_hybrid_cli.py
git commit -m "feat: apply accepted hybrid genre signatures"
```

## Task 7: Add Batch Reporting Without Live APIs

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
- Modify: `src/ai_genre_enrichment/storage.py`
- Test: `tests/unit/test_ai_genre_hybrid_cli.py`

- [ ] **Step 1: Add command**

Add parser:

```python
hybrid_report = sub.add_parser("hybrid-report", help="Summarize hybrid genre report coverage and decisions")
hybrid_report.add_argument("--limit", type=int)
```

Add dispatch:

```python
if args.command == "hybrid-report":
    return cmd_hybrid_report(args)
```

- [ ] **Step 2: Add report aggregation**

Add to `SidecarStore`:

```python
def hybrid_report_summary(self) -> dict[str, Any]:
    with self.connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS reports,
                COALESCE(SUM(accepted_count), 0) AS accepted,
                COALESCE(SUM(provisional_count), 0) AS provisional,
                COALESCE(SUM(rejected_noise_count), 0) AS rejected_noise,
                COALESCE(SUM(needs_review_count), 0) AS needs_review
            FROM ai_genre_hybrid_reports
            """
        ).fetchone()
        return dict(row)
```

Add command:

```python
def cmd_hybrid_report(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    print(json.dumps(store.hybrid_report_summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0
```

- [ ] **Step 3: Add a focused report test**

Append to `tests/unit/test_ai_genre_hybrid_cli.py`:

```python
def test_hybrid_report_summarizes_persisted_reports(tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    store.record_hybrid_report(
        release_key="a::b",
        normalized_artist="a",
        normalized_album="b",
        album_id=None,
        report={
            "accepted_genres": [{"term": "slowcore"}],
            "provisional_genres": [],
            "rejected_noise": [{"term": "seen live"}],
            "needs_review": [],
        },
        evidence_count=2,
    )

    rc = ai_genre_enrich.main(["--sidecar-db", str(sidecar), "hybrid-report"])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reports"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected_noise"] == 1
```

- [ ] **Step 4: Run hybrid tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_hybrid_evidence.py -q --basetemp C:\tmp\hybrid-report-green -o cache_dir=C:\tmp\hybrid-report-green-cache
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

```powershell
git add -- scripts/ai_genre_enrich.py src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_hybrid_cli.py
git commit -m "feat: report hybrid genre enrichment decisions"
```

## Task 8: Documentation And Final Focused Verification

**Files:**
- Modify: `docs/AI_GENRE_ENRICHMENT.md`
- Test: no new test file

- [ ] **Step 1: Document the actual new model**

Add a section to `docs/AI_GENRE_ENRICHMENT.md`:

```markdown
## Hybrid Genre Evidence Model

`hybrid-enrich-one` is the first usable deterministic/LLM hybrid enrichment path.
It fuses confirmed sidecar source tags, local metadata signals, and cached album
model-prior terms into explainable decisions:

- `accepted_genres`: safe to write with `--apply`;
- `provisional_genres`: useful model/taxonomy signals for sparse releases, not applied yet;
- `rejected_noise`: Last.fm-only or non-genre noise;
- `needs_review`: mapped but insufficiently supported terms.

Source hierarchy:

1. `bandcamp_release` and `official_release` are strong release-specific evidence.
2. Discogs, MusicBrainz, and local metadata are medium evidence.
3. `model_prior` is clean taxonomy evidence but not source authority.
4. Last.fm is weak corroboration only; Last.fm-only terms are rejected or routed to review.

The system does not scrape RYM. RYM-like behavior means standard hierarchical
music-taxonomy reasoning through the curated taxonomy and the album model-prior
prompt. Real RYM data requires a separate approved source strategy.
```

- [ ] **Step 2: Run focused verification**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py tests/unit/test_ai_genre_hybrid_evidence.py tests/unit/test_ai_genre_hybrid_cli.py tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\hybrid-final -o cache_dir=C:\tmp\hybrid-final-cache
```

Expected:

```text
passed
```

- [ ] **Step 3: Confirm no metadata DB writes**

Run:

```powershell
git status --short
```

Expected:

```text
only intended source, test, and doc files are modified
```

- [ ] **Step 4: Commit**

```powershell
git add -- docs/AI_GENRE_ENRICHMENT.md
git commit -m "docs: explain hybrid genre evidence model"
```

## Final Definition Of Done

This milestone is complete only when:

1. `model-prior` cache checks happen before API calls.
2. `hybrid-enrich-one --dry-run` prints accepted, provisional, rejected-noise, and review decisions.
3. Bandcamp + model evidence accepts mapped specific genres.
4. Last.fm-only evidence is rejected or routed away from acceptance.
5. Model-only high-confidence evidence for sparse releases is provisional, not silently authoritative.
6. `hybrid-enrich-one` live mode persists a report but does not alter signatures.
7. `hybrid-enrich-one --apply` explicitly writes accepted decisions to sidecar enriched signatures.
8. No command writes `data/metadata.db`.
9. Focused hybrid and existing enrichment tests pass.
10. The next user-visible artifact is a real example report, not another design document.

## What Comes After This Milestone

Only after this milestone is reviewed with real album examples:

1. Tune fusion thresholds against 20 to 50 representative releases.
2. Add batch `hybrid-enrich --limit --missing-only`.
3. Add optional provisional inclusion for sparse releases.
4. Feed hybrid signatures into isolated `hybrid_shadow` artifacts.
5. Compare candidate pools and playlist transitions before any default behavior changes.
