# AI Genre Model-Prior Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI-only, no-web album-level model-prior lane that stores taxonomy-shaped provisional genre hypotheses without changing normal enriched signatures or artifacts.

**Architecture:** Extend the existing OpenAI Responses wrapper with a generic structured-output method while preserving the current refinement API. Put the album-prior schema, prompt, validation, and taxonomy mapping in a focused `model_prior.py` module. Store prior runs and normalized terms in dedicated sidecar tables keyed by provider, model, prompt, taxonomy, schema, policy, and input hash.

**Tech Stack:** Python 3.11+, SQLite sidecar DB, OpenAI Responses API, pytest

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ai_genre_enrichment/client.py` | Add reusable validated structured-output method; preserve `enrich()` compatibility. |
| `src/ai_genre_enrichment/model_prior.py` | New prior prompt, JSON schema, validation, payload hashing, and taxonomy mapping. |
| `src/ai_genre_enrichment/storage.py` | Add prior tables, cache lookup, persistence, batch candidates, and report metrics. |
| `scripts/ai_genre_enrich.py` | Add `model-prior-one`, `model-prior`, and `model-prior-report`. |
| `docs/AI_GENRE_ENRICHMENT.md` | Document CLI-only provisional model-prior behavior and provenance limits. |
| `tests/unit/test_ai_genre_model_prior.py` | New focused tests for the prior contract, storage, cache, and CLI. |

## Task 1: Add A Generic Structured Responses Method

**Files:**
- Modify: `src/ai_genre_enrichment/client.py:26-135`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Write failing wrapper tests**

Create:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def test_request_structured_uses_supplied_validator(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    seen = []

    class Response:
        output_text = '{"genres":[],"warnings":[]}'
        usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", api_key="test", max_retries=0)
    monkeypatch.setattr(client, "_call_openai", lambda *_args, **_kwargs: Response())

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: seen.append(value) or value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "complete"
    assert result.response_json == {"genres": [], "warnings": []}
    assert seen == [{"genres": [], "warnings": []}]


def test_request_structured_dry_run_does_not_call_openai(monkeypatch):
    from src.ai_genre_enrichment.client import OpenAIEnrichmentClient

    client = OpenAIEnrichmentClient(model="gpt-4o-mini", dry_run=True)
    monkeypatch.setattr(
        client,
        "_call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )

    result = client.request_structured(
        payload={"artist": "Duster"},
        prompt="classify",
        response_format={"type": "json_schema", "name": "prior", "schema": {}},
        validator=lambda value: value,
        instructions="No web.",
        estimated_output_tokens=300,
    )

    assert result.status == "skipped"
    assert result.response_json["dry_run"] is True
    assert result.response_json["estimated_output_tokens"] == 300
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_request_structured_uses_supplied_validator tests/unit/test_ai_genre_model_prior.py::test_request_structured_dry_run_does_not_call_openai -q --basetemp C:\tmp\genre-prior-client -o cache_dir=C:\tmp\genre-prior-client-cache
```

Expected: FAIL because `request_structured()` does not exist.

- [ ] **Step 3: Extract generic validation from `enrich()`**

Add:

```python
from collections.abc import Callable


def request_structured(
    self,
    *,
    payload: dict[str, Any],
    prompt: str,
    response_format: dict[str, Any],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    instructions: str,
    estimated_output_tokens: int,
) -> EnrichmentResult:
    if self.dry_run:
        estimated_chars = len(prompt)
        estimated_prompt_tokens = max(1, estimated_chars // 4)
        return EnrichmentResult(
            status="skipped",
            response_json={
                "dry_run": True,
                "model": self.model,
                "payload": payload,
                "web_mode": self.web_mode.value,
                "estimated_prompt_chars": estimated_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_output_tokens": estimated_output_tokens,
            },
            token_usage={
                "estimated_prompt_chars": estimated_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_output_tokens": estimated_output_tokens,
            },
            estimated_cost_usd=estimate_cost_usd(
                self.model,
                input_tokens=estimated_prompt_tokens,
                output_tokens=estimated_output_tokens,
            ),
        )
    api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return EnrichmentResult(status="failed", response_json={}, token_usage={}, error_message="OPENAI_API_KEY is not set")
    try:
        response = self._call_openai(prompt, response_format, instructions=instructions)
        response_json = validator(_extract_response_json(response))
        token_usage = _extract_token_usage(response)
        return EnrichmentResult(
            status="complete",
            response_json=response_json,
            token_usage=token_usage,
            estimated_cost_usd=estimate_cost_usd(
                self.model,
                input_tokens=token_usage.get("input_tokens", 0),
                output_tokens=token_usage.get("output_tokens", 0),
            ),
        )
    except Exception as exc:
        return EnrichmentResult(status="failed", response_json={}, token_usage={}, error_message=str(exc))
```

Keep `enrich()` as the existing retrying refinement path. Do not change its public behavior.

- [ ] **Step 4: Run client tests**

Run the Step 2 command and:

```powershell
pytest tests/unit/test_ai_genre_enrichment.py::test_dry_run_client_does_not_call_openai tests/unit/test_ai_genre_enrichment.py::test_client_retries_once_after_response_validation_failure -q --basetemp C:\tmp\genre-prior-client-compat -o cache_dir=C:\tmp\genre-prior-client-compat-cache
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- src/ai_genre_enrichment/client.py tests/unit/test_ai_genre_model_prior.py
git commit -m "refactor: expose structured genre enrichment requests"
```

## Task 2: Define The Album-Prior Contract And Taxonomy Mapping

**Files:**
- Create: `src/ai_genre_enrichment/model_prior.py`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Write failing validation and mapping tests**

Append:

```python
def test_validate_model_prior_response_normalizes_terms():
    from src.ai_genre_enrichment.model_prior import validate_model_prior_response

    result = validate_model_prior_response({
        "genres": [{
            "term": "  Ambient Americana ",
            "confidence": 0.82,
            "specificity": "subgenre",
            "taxonomy_role": "core_style",
            "notes": "Taxonomic fit.",
        }],
        "warnings": [],
    })
    assert result["genres"][0]["term"] == "ambient americana"


def test_validate_model_prior_response_rejects_source_claims():
    from src.ai_genre_enrichment.model_prior import validate_model_prior_response

    with pytest.raises(ValueError, match="source authority"):
        validate_model_prior_response({
            "genres": [{
                "term": "slowcore", "confidence": 0.9, "specificity": "subgenre",
                "taxonomy_role": "core_style", "notes": "Bandcamp says this is slowcore.",
            }],
            "warnings": [],
        })


def test_map_model_prior_terms_accepts_known_style_and_rejects_descriptor():
    from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
    from src.ai_genre_enrichment.model_prior import map_model_prior_terms

    mapped = map_model_prior_terms(
        [
            {"term": "slowcore", "confidence": 0.9, "specificity": "subgenre", "taxonomy_role": "core_style", "notes": ""},
            {"term": "instrumental", "confidence": 0.8, "specificity": "broad", "taxonomy_role": "secondary_style", "notes": ""},
        ],
        GenreVocabulary(),
    )

    assert mapped[0]["mapping_status"] == "mapped"
    assert mapped[0]["accepted_for_shadow"] == 1
    assert mapped[0]["auto_apply_eligible"] == 0
    assert mapped[1]["mapping_status"] == "descriptor"
    assert mapped[1]["accepted_for_shadow"] == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_validate_model_prior_response_normalizes_terms tests/unit/test_ai_genre_model_prior.py::test_validate_model_prior_response_rejects_source_claims tests/unit/test_ai_genre_model_prior.py::test_map_model_prior_terms_accepts_known_style_and_rejects_descriptor -q --basetemp C:\tmp\genre-prior-contract -o cache_dir=C:\tmp\genre-prior-contract-cache
```

Expected: FAIL because `model_prior.py` does not exist.

- [ ] **Step 3: Create the model-prior contract**

Create `src/ai_genre_enrichment/model_prior.py` with:

```python
"""No-web album-level model-prior contract and taxonomy mapping."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .tag_classification import normalize_source_tag

MODEL_PRIOR_PROMPT_VERSION = "album-model-prior-v1"
MODEL_PRIOR_SCHEMA_VERSION = "album-model-prior-response-v1"
MODEL_PRIOR_TAXONOMY_VERSION = "genre-vocabulary-v1"
MODEL_PRIOR_INSTRUCTIONS = (
    "Classify the album into a compact multi-genre signature using only the supplied payload. "
    "Do not use web search. Do not claim that any external source says something. "
    "Return taxonomic hypotheses, not authoritative evidence."
)
SPECIFICITIES = {"broad", "genre", "subgenre", "microgenre"}
TAXONOMY_ROLES = {"parent", "core_style", "secondary_style", "edge_case"}
SOURCE_CLAIM_MARKERS = ("bandcamp says", "discogs says", "musicbrainz says", "last.fm says", "official source")

MODEL_PRIOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "genres": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "confidence": {"type": "number"},
                    "specificity": {"type": "string", "enum": sorted(SPECIFICITIES)},
                    "taxonomy_role": {"type": "string", "enum": sorted(TAXONOMY_ROLES)},
                    "notes": {"type": "string"},
                },
                "required": ["term", "confidence", "specificity", "taxonomy_role", "notes"],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["genres", "warnings"],
    "additionalProperties": False,
}


def model_prior_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "name": "ai_genre_album_model_prior", "strict": True, "schema": deepcopy(MODEL_PRIOR_RESPONSE_SCHEMA)}


def validate_model_prior_response(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"genres", "warnings"}:
        raise ValueError("model prior response must contain genres and warnings")
    if not isinstance(data["genres"], list) or not isinstance(data["warnings"], list):
        raise ValueError("model prior genres and warnings must be lists")
    normalized: list[dict[str, Any]] = []
    for item in data["genres"]:
        term = normalize_source_tag(str(item.get("term", "")))
        confidence = item.get("confidence")
        specificity = item.get("specificity")
        role = item.get("taxonomy_role")
        notes = str(item.get("notes", "")).strip()
        if not term or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("model prior term and confidence are invalid")
        if specificity not in SPECIFICITIES or role not in TAXONOMY_ROLES:
            raise ValueError("model prior specificity or taxonomy role is invalid")
        if any(marker in notes.casefold() for marker in SOURCE_CLAIM_MARKERS):
            raise ValueError("model prior notes must not claim source authority")
        normalized.append({"term": term, "confidence": float(confidence), "specificity": specificity, "taxonomy_role": role, "notes": notes})
    return {"genres": normalized, "warnings": [str(value) for value in data["warnings"]]}


def map_model_prior_terms(items: list[dict[str, Any]], vocabulary: Any) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in items:
        term = normalize_source_tag(item["term"])
        non_genre = vocabulary.classify_non_genre(term)
        genre = vocabulary.classify_genre(term)
        conditional = item["taxonomy_role"] == "edge_case" or item["confidence"] < 0.70
        if non_genre:
            status, slug, accepted = non_genre, None, 0
        elif genre and conditional:
            status, slug, accepted = "conditional", genre.genre, 0
        elif genre:
            status, slug, accepted = "mapped", genre.genre, 1
        else:
            status, slug, accepted = "unmapped", None, 0
        mapped.append({**item, "raw_term": item["term"], "normalized_term": term, "canonical_slug": slug, "mapping_status": status, "accepted_for_shadow": accepted, "auto_apply_eligible": 0})
    return mapped


def stable_input_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run contract tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- src/ai_genre_enrichment/model_prior.py tests/unit/test_ai_genre_model_prior.py
git commit -m "feat: define album genre model-prior contract"
```

## Task 3: Add Dedicated Prior Tables And Cache Methods

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py:161-390`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Write failing persistence test**

Append:

```python
def test_store_records_and_reuses_model_prior_cache(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    prior_id = store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster",
        normalized_album="stratosphere", album_id="a1", provider="openai",
        model="gpt-4o-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="hash-1",
        status="complete", response_json={"genres": [], "warnings": []},
        warnings=[], error_message=None, token_usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        estimated_cost_usd=0.00001, mapped_terms=[],
    )

    cached = store.find_model_prior(
        release_key="duster::stratosphere", provider="openai", model="gpt-4o-mini",
        prompt_version="album-model-prior-v1", taxonomy_version="genre-vocabulary-v1",
        schema_version="album-model-prior-response-v1", enrichment_policy_version="genre-enrichment-v2",
        input_hash="hash-1",
    )

    assert prior_id > 0
    assert cached["status"] == "complete"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_store_records_and_reuses_model_prior_cache -q --basetemp C:\tmp\genre-prior-storage -o cache_dir=C:\tmp\genre-prior-storage-cache
```

Expected: FAIL because the prior storage methods do not exist.

- [ ] **Step 3: Add schema from the approved design**

Inside `SidecarStore.initialize()`, add:

```sql
CREATE TABLE IF NOT EXISTS ai_genre_model_priors (
    prior_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_key TEXT NOT NULL,
    normalized_artist TEXT NOT NULL,
    normalized_album TEXT NOT NULL,
    album_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    enrichment_policy_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT,
    warnings_json TEXT,
    error_message TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    estimated_cost_usd REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (
        release_key, provider, model, prompt_version, taxonomy_version,
        schema_version, enrichment_policy_version, input_hash
    )
);

CREATE TABLE IF NOT EXISTS ai_genre_model_prior_terms (
    prior_term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prior_id INTEGER NOT NULL,
    release_key TEXT NOT NULL,
    raw_term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    canonical_slug TEXT,
    confidence REAL NOT NULL,
    specificity TEXT NOT NULL,
    taxonomy_role TEXT NOT NULL,
    mapping_status TEXT NOT NULL,
    accepted_for_shadow INTEGER NOT NULL DEFAULT 0,
    auto_apply_eligible INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prior_id) REFERENCES ai_genre_model_priors(prior_id) ON DELETE CASCADE
);
```

Add indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_ai_genre_model_priors_release
    ON ai_genre_model_priors (release_key);
CREATE INDEX IF NOT EXISTS idx_ai_genre_model_priors_provider_model
    ON ai_genre_model_priors (provider, model);
CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_release
    ON ai_genre_model_prior_terms (release_key);
CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_normalized
    ON ai_genre_model_prior_terms (normalized_term);
CREATE INDEX IF NOT EXISTS idx_ai_genre_model_prior_terms_mapping
    ON ai_genre_model_prior_terms (mapping_status, accepted_for_shadow);
```

- [ ] **Step 4: Add `find_model_prior()` and `record_model_prior()`**

Implement:

```python
def find_model_prior(self, *, release_key: str, provider: str, model: str, prompt_version: str,
                     taxonomy_version: str, schema_version: str, enrichment_policy_version: str,
                     input_hash: str) -> dict[str, Any] | None:
    with self.connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM ai_genre_model_priors
            WHERE release_key = ? AND provider = ? AND model = ? AND prompt_version = ?
              AND taxonomy_version = ? AND schema_version = ?
              AND enrichment_policy_version = ? AND input_hash = ?
            """,
            (release_key, provider, model, prompt_version, taxonomy_version, schema_version,
             enrichment_policy_version, input_hash),
        ).fetchone()
        return dict(row) if row else None
```

Implement `record_model_prior()` as one transaction:

```python
now = _now_iso()
with self.connect() as conn:
    conn.execute(
        """
        INSERT INTO ai_genre_model_priors (
            release_key, normalized_artist, normalized_album, album_id,
            provider, model, prompt_version, taxonomy_version, schema_version,
            enrichment_policy_version, input_hash, status, response_json,
            warnings_json, error_message, input_tokens, output_tokens,
            total_tokens, estimated_cost_usd, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (
            release_key, provider, model, prompt_version, taxonomy_version,
            schema_version, enrichment_policy_version, input_hash
        ) DO UPDATE SET
            status = excluded.status,
            response_json = excluded.response_json,
            warnings_json = excluded.warnings_json,
            error_message = excluded.error_message,
            input_tokens = excluded.input_tokens,
            output_tokens = excluded.output_tokens,
            total_tokens = excluded.total_tokens,
            estimated_cost_usd = excluded.estimated_cost_usd,
            updated_at = excluded.updated_at
        """,
        (
            release_key, normalized_artist, normalized_album, album_id,
            provider, model, prompt_version, taxonomy_version, schema_version,
            enrichment_policy_version, input_hash, status,
            json.dumps(response_json, sort_keys=True) if response_json is not None else None,
            json.dumps(warnings, sort_keys=True), error_message,
            token_usage.get("input_tokens"), token_usage.get("output_tokens"),
            token_usage.get("total_tokens"), estimated_cost_usd, now, now,
        ),
    )
    prior_id = int(conn.execute(
        """
        SELECT prior_id FROM ai_genre_model_priors
        WHERE release_key = ? AND provider = ? AND model = ? AND prompt_version = ?
          AND taxonomy_version = ? AND schema_version = ?
          AND enrichment_policy_version = ? AND input_hash = ?
        """,
        (release_key, provider, model, prompt_version, taxonomy_version,
         schema_version, enrichment_policy_version, input_hash),
    ).fetchone()["prior_id"])
    conn.execute("DELETE FROM ai_genre_model_prior_terms WHERE prior_id = ?", (prior_id,))
    conn.executemany(
        """
        INSERT INTO ai_genre_model_prior_terms (
            prior_id, release_key, raw_term, normalized_term, canonical_slug,
            confidence, specificity, taxonomy_role, mapping_status,
            accepted_for_shadow, auto_apply_eligible, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        [
            (
                prior_id, release_key, term["raw_term"], term["normalized_term"],
                term["canonical_slug"], term["confidence"], term["specificity"],
                term["taxonomy_role"], term["mapping_status"],
                term["accepted_for_shadow"], term["notes"], now,
            )
            for term in mapped_terms
        ],
    )
return prior_id
```

Always persist `auto_apply_eligible=0`.

- [ ] **Step 5: Run storage test**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_model_prior.py
git commit -m "feat: store album genre model priors"
```

## Task 4: Add `model-prior-one` With API-Free Dry Run

**Files:**
- Modify: `src/ai_genre_enrichment/model_prior.py`
- Modify: `scripts/ai_genre_enrich.py:36-240`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Write failing CLI dry-run test**

Append:

```python
def test_model_prior_one_dry_run_is_api_free_and_sidecar_free(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )
    sidecar = tmp_path / "sidecar.db"
    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db), "--sidecar-db", str(sidecar),
        "model-prior-one", "--artist", "Duster", "--album", "Stratosphere", "--dry-run",
    ])

    assert rc == 0
    assert not sidecar.exists()
    assert '"dry_run": true' in capsys.readouterr().out
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_model_prior_one_dry_run_is_api_free_and_sidecar_free -q --basetemp C:\tmp\genre-prior-one -o cache_dir=C:\tmp\genre-prior-one-cache
```

Expected: FAIL because the CLI command does not exist.

- [ ] **Step 3: Add payload and prompt helpers**

Append to `model_prior.py`:

```python
def build_model_prior_payload(release: Any) -> dict[str, Any]:
    baseline = release.existing_genres_by_source
    return {
        "release_key": release.release_key,
        "artist": release.normalized_artist,
        "album": release.normalized_album,
        "album_id": release.album_id,
        "identifiers": getattr(release, "identifiers", {}) or {},
        "year": getattr(release, "year", None),
        "track_titles": list(release.track_titles[:8]),
        "baseline_genres_by_source": baseline,
        "known_tags": sorted({tag for tags in baseline.values() for tag in tags}),
        "prompt_version": MODEL_PRIOR_PROMPT_VERSION,
        "taxonomy_version": MODEL_PRIOR_TAXONOMY_VERSION,
        "schema_version": MODEL_PRIOR_SCHEMA_VERSION,
    }


def build_model_prior_prompt(payload: dict[str, Any]) -> str:
    return "Return a compact album genre prior for this local payload:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
```

- [ ] **Step 4: Add parser and command**

Add parser:

```python
model_prior_one = sub.add_parser("model-prior-one", help="Generate or preview one no-web album model prior")
model_prior_one.add_argument("--artist", required=True)
model_prior_one.add_argument("--album", required=True)
model_prior_one.add_argument("--dry-run", action="store_true")
model_prior_one.add_argument("--force", action="store_true")
```

Add command routing and implement a shared one-release runner:

```python
from src.ai_genre_enrichment.genre_vocabulary import GenreVocabulary
from src.ai_genre_enrichment.model_prior import (
    MODEL_PRIOR_INSTRUCTIONS,
    MODEL_PRIOR_PROMPT_VERSION,
    MODEL_PRIOR_SCHEMA_VERSION,
    MODEL_PRIOR_TAXONOMY_VERSION,
    build_model_prior_payload,
    build_model_prior_prompt,
    map_model_prior_terms,
    model_prior_response_format,
    stable_input_hash,
    validate_model_prior_response,
)
from src.ai_genre_enrichment.policy import STABILIZED_POLICY_VERSION


def _run_model_prior_release(args: argparse.Namespace, release: ReleasePayload) -> int:
    payload = build_model_prior_payload(release)
    input_hash = stable_input_hash(payload)
    client = OpenAIEnrichmentClient(model=args.model, dry_run=args.dry_run, web_mode="off")
    result = client.request_structured(
        payload=payload,
        prompt=build_model_prior_prompt(payload),
        response_format=model_prior_response_format(),
        validator=validate_model_prior_response,
        instructions=MODEL_PRIOR_INSTRUCTIONS,
        estimated_output_tokens=300,
    )
    if args.dry_run:
        print(json.dumps(result.response_json, ensure_ascii=False, sort_keys=True))
        return 0

    store = SidecarStore(args.sidecar_db)
    store.initialize()
    cached = store.find_model_prior(
        release_key=release.release_key, provider="openai", model=args.model,
        prompt_version=MODEL_PRIOR_PROMPT_VERSION, taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION,
        schema_version=MODEL_PRIOR_SCHEMA_VERSION, enrichment_policy_version=STABILIZED_POLICY_VERSION,
        input_hash=input_hash,
    )
    if cached and getattr(args, "missing_only", False) and not args.force:
        print(f"existing-model-prior {release.release_key}")
        return 0
    if cached and cached["status"] == "complete" and not args.force:
        print(f"cached-model-prior {release.release_key}")
        return 0

    mapped_terms = []
    if result.status == "complete":
        vocabulary = GenreVocabulary(library_db_path=args.metadata_db)
        mapped_terms = map_model_prior_terms(result.response_json["genres"], vocabulary)
    store.record_model_prior(
        release_key=release.release_key, normalized_artist=release.normalized_artist,
        normalized_album=release.normalized_album, album_id=release.album_id,
        provider="openai", model=args.model, prompt_version=MODEL_PRIOR_PROMPT_VERSION,
        taxonomy_version=MODEL_PRIOR_TAXONOMY_VERSION, schema_version=MODEL_PRIOR_SCHEMA_VERSION,
        enrichment_policy_version=STABILIZED_POLICY_VERSION, input_hash=input_hash,
        status=result.status, response_json=result.response_json or None,
        warnings=result.response_json.get("warnings", []), error_message=result.error_message,
        token_usage=result.token_usage, estimated_cost_usd=result.estimated_cost_usd,
        mapped_terms=mapped_terms,
    )
    print(f"{result.status}-model-prior {release.release_key}")
    return 0 if result.status == "complete" else 1


def cmd_model_prior_one(args: argparse.Namespace) -> int:
    releases = _discover(args)
    if len(releases) != 1:
        print(f"Expected exactly one release, found {len(releases)}.")
        return 2
    return _run_model_prior_release(args, releases[0])
```

The live path records failed validation or API calls as failed prior rows. The batch command in Task 5 continues after a non-zero per-release result and reports the failed count.

- [ ] **Step 5: Run CLI dry-run test**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/model_prior.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_model_prior.py
git commit -m "feat: add single album genre model-prior CLI"
```

## Task 5: Add Batch Generation And Prior Reporting

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py`
- Modify: `scripts/ai_genre_enrich.py`
- Test: `tests/unit/test_ai_genre_model_prior.py`

- [ ] **Step 1: Write failing batch and report tests**

Append:

```python
def test_model_prior_report_counts_mapping_statuses(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.record_model_prior(
        release_key="duster::stratosphere", normalized_artist="duster", normalized_album="stratosphere",
        album_id="a1", provider="openai", model="gpt-4o-mini", prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1", schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2", input_hash="h", status="complete",
        response_json={"genres": [], "warnings": []}, warnings=[], error_message=None, token_usage={},
        estimated_cost_usd=None, mapped_terms=[
            {"raw_term": "slowcore", "normalized_term": "slowcore", "canonical_slug": "slowcore",
             "confidence": 0.9, "specificity": "subgenre", "taxonomy_role": "core_style",
             "mapping_status": "mapped", "accepted_for_shadow": 1, "auto_apply_eligible": 0, "notes": ""},
        ],
    )

    report = store.model_prior_report()
    assert report["mapping_status_counts"] == {"mapped": 1}
    assert report["accepted_for_shadow"] == 1
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py::test_model_prior_report_counts_mapping_statuses -q --basetemp C:\tmp\genre-prior-report -o cache_dir=C:\tmp\genre-prior-report-cache
```

Expected: FAIL because `model_prior_report()` does not exist.

- [ ] **Step 3: Add storage report aggregation**

Implement:

```python
def model_prior_report(self) -> dict[str, Any]:
    with self.connect() as conn:
        status_counts = {row["status"]: row["count"] for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM ai_genre_model_priors GROUP BY status ORDER BY status"
        )}
        mapping_status_counts = {row["mapping_status"]: row["count"] for row in conn.execute(
            "SELECT mapping_status, COUNT(*) AS count FROM ai_genre_model_prior_terms GROUP BY mapping_status ORDER BY mapping_status"
        )}
        accepted = conn.execute(
            "SELECT COUNT(*) FROM ai_genre_model_prior_terms WHERE accepted_for_shadow = 1"
        ).fetchone()[0]
        return {
            "status_counts": status_counts,
            "mapping_status_counts": mapping_status_counts,
            "accepted_for_shadow": accepted,
        }
```

- [ ] **Step 4: Add batch and report subcommands**

Add:

```python
model_prior = sub.add_parser("model-prior", help="Generate no-web album model priors in a bounded batch")
add_release_filters(model_prior)
model_prior.add_argument("--dry-run", action="store_true")
model_prior.add_argument("--missing-only", action="store_true")
model_prior.add_argument("--force", action="store_true")
sub.add_parser("model-prior-report", help="Report album model-prior coverage and mapping status")
```

Implement `cmd_model_prior(args)` by iterating `_discover(args)` and calling the same one-release helper used by `cmd_model_prior_one()`. `--missing-only` uses the normal cache path and skips complete cached rows:

```python
def cmd_model_prior(args: argparse.Namespace) -> int:
    failures = 0
    for release in _discover(args):
        rc = _run_model_prior_release(args, release)
        if rc != 0:
            failures += 1
    print(f"model-prior batch complete failures={failures}")
    return 1 if failures else 0


def cmd_model_prior_report(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    print(json.dumps(store.model_prior_report(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0
```

- [ ] **Step 5: Run focused model-prior tests**

Run:

```powershell
pytest tests/unit/test_ai_genre_model_prior.py -q --basetemp C:\tmp\genre-prior-prototype -o cache_dir=C:\tmp\genre-prior-prototype-cache
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- src/ai_genre_enrichment/storage.py scripts/ai_genre_enrich.py tests/unit/test_ai_genre_model_prior.py
git commit -m "feat: add batch album genre priors and report"
```

## Task 6: Document And Verify The CLI-Only Prototype

**Files:**
- Modify: `docs/AI_GENRE_ENRICHMENT.md`
- Modify: `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`

- [ ] **Step 1: Add model-prior documentation**

Document:

```markdown
## Album Model Prior

`model-prior-one`, `model-prior`, and `model-prior-report` are CLI-only.
The model defaults to OpenAI `gpt-4o-mini`, uses `web_mode=off`, and receives
only local release metadata. Prior terms are provisional classifier signals:
they are never authoritative, never auto-apply eligible, and never enter normal
`enriched_genre_signatures`.
```

- [ ] **Step 2: Run prototype suite**

```powershell
pytest tests/unit/test_ai_genre_model_prior.py tests/unit/test_ai_genre_enrichment.py -q --basetemp C:\tmp\genre-prior-prototype-full -o cache_dir=C:\tmp\genre-prior-prototype-full-cache
```

Expected: PASS.

- [ ] **Step 3: Run dry-run smoke test**

```powershell
python scripts/ai_genre_enrich.py model-prior-one --artist "Duster" --album "Stratosphere" --dry-run
```

Expected: JSON preview with `"dry_run": true`, prompt-size estimate, output-size estimate, and no sidecar mutation.

- [ ] **Step 4: Commit**

```powershell
git add -- docs/AI_GENRE_ENRICHMENT.md docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md
git commit -m "docs: explain CLI album genre model priors"
```
