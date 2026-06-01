# AI Genre Model Prior And Stabilization Design

**Status:** Approved 2026-06-01.
**Source proposal:** `C:/Users/Dylan/Downloads/AI_GENRE_MODEL_PRIOR_SPEC.md`
**Scope:** Stabilize the existing offline genre-enrichment system, restore an explicit backward-compatible artifact boundary, and add a CLI-only album-level model-prior lane for shadow evaluation.

---

## 1. Goal

Improve album-level genre signatures while preserving local-first playlist generation and making data-policy changes measurable before they influence normal generation.

The implementation proceeds incrementally:

1. Stabilize the existing enrichment workflow.
2. Make legacy artifact generation the default.
3. Add reproducible enriched shadow artifacts and comparison reports.
4. Add a separate no-web album-level model-prior lane.
5. Allow mapped model-prior terms to participate only in shadow artifacts as explicitly provisional terms.
6. Evaluate results before considering GUI exposure or production adoption.

This design does not authorize writes to `data/metadata.db`. Existing sidecar signatures are preserved unless the user explicitly refreshes a release.

---

## 2. Current Architecture Findings

The downloaded proposal correctly identifies the need for a clean album-level model prior, but the repository already contains several building blocks that should be reused.

### 2.1 Existing enrichment flow

The GUI worker currently runs:

```text
ingest-local
-> extract-lastfm
-> extract-bandcamp
-> classify-tags --adjudicate
-> build-enriched
```

The CLI entry point is `scripts/ai_genre_enrich.py`. The active AI model is configurable and defaults to OpenAI `gpt-4o-mini`.

### 2.2 Existing sidecar tables

`src/ai_genre_enrichment/storage.py::SidecarStore.initialize()` already creates:

- `ai_genre_release_checks`
- `ai_genre_suggestions`
- `ai_genre_run_log`
- `ai_genre_source_pages`
- `ai_genre_source_tags`
- `ai_genre_tag_classifications`
- `ai_tag_adjudication_cache`
- `ai_genre_review_decisions`
- `ai_genre_user_overrides`
- `enriched_genres`
- `enriched_genre_signatures`

The implementation must extend these conventions rather than replace them.

### 2.3 Existing taxonomy learning

The current system already provides:

- curated taxonomy categories in `data/genre_vocabulary.yaml`;
- aliases and decomposition rules;
- deterministic source-tag classification;
- AI adjudication of unknown source tags;
- reusable AI adjudication cache;
- human review decisions;
- graduation of reviewed or repeatedly AI-adjudicated terms into the YAML vocabulary;
- release-specific user overrides.

The first model-prior implementation does not add a second general taxonomy authority. Broader scoped taxonomy decisions remain a later extension if shadow evaluation demonstrates a concrete need.

### 2.4 Existing artifact consumption

Enriched signatures already influence artifacts when a resolver is provided:

- `scripts/build_beat3tower_artifacts.py` prefers sidecar signatures for enriched releases.
- The GUI artifact-build command currently supplies `data/ai_genre_enrichment.db`.
- `src/features/artifacts.py` automatically loads a 64-dimensional dense genre sidecar when present.
- `src/playlist/candidate_pool.py` prefers dense PMI-SVD genre similarity when available.

Therefore, model-prior work requires an explicit artifact source boundary before provisional terms are introduced.

### 2.5 Existing embedding prior is separate

`src/genre/llm_prior.py` is an optional vocabulary-embedding prior. It asks an LLM to rate similarity between genre labels and anchor genres so rare vocabulary terms can be placed in the PMI-SVD coordinate space.

That feature is separate from this design's album-level model prior:

- **Embedding prior:** vocabulary-token similarity, artifact-build concern.
- **Album model prior:** provisional album genre signature, enrichment concern.

They require separate prompts, caches, storage, provenance, and reporting.

---

## 3. Stabilization Requirements

Stabilization is a prerequisite for the model-prior lane.

### 3.1 Artifact source modes

Add an explicit artifact genre-source mode:

```yaml
playlists:
  ds_pipeline:
    genre_source: legacy | enriched | hybrid_shadow
```

Behavior:

| Mode | Behavior |
|---|---|
| `legacy` | Default. Build artifacts exclusively from legacy `metadata.db` genre tables. Ignore sidecar signatures and model-prior terms. |
| `enriched` | Explicit opt-in. Build artifacts from accepted enriched signatures when present, falling back to legacy metadata for other releases. Never include model-prior-only terms. |
| `hybrid_shadow` | Build reproducible parallel shadow artifacts from accepted enriched signatures plus eligible provisional model-prior terms. Do not replace the active artifact or silently change normal playlist generation. |

Expose the same choice as `--genre-source` on artifact-building CLI paths. The GUI's normal **Build Artifacts** action uses `legacy`. The first implementation does not add a GUI selector for shadow mode.

### 3.2 Shadow output isolation

`hybrid_shadow` writes a separate sparse artifact and dense sidecar under a shadow output path. It never overwrites:

```text
data/artifacts/beat3tower_32k/data_matrices_step1.npz
data/artifacts/beat3tower_32k/data_matrices_step1_genre_emb_dim64.npz
```

Write each shadow build under:

```text
data/artifacts/beat3tower_32k/shadow/<shadow_fingerprint>/
```

`shadow_fingerprint` is a stable hash of the selected genre-source mode, enrichment policy version, accepted-signature snapshot identity, model-prior snapshot identity, sparse artifact input identity, and dense embedding config. Write the sparse artifact, dense sidecar, and comparison report in that directory. Print the resolved paths in command output. The output must be usable for repeatable comparison runs, but normal generation does not automatically select it.

### 3.3 Bandcamp source typing

Fix the current mismatch:

- `extract-bandcamp` stores source pages as `bandcamp_tags`.
- `rebuild_enriched_genres_for_release()` accepts `bandcamp_release`.

Use `bandcamp_release` as the canonical source type for confirmed artist/label Bandcamp release tags. Preserve compatibility by normalizing existing `bandcamp_tags` rows to `bandcamp_release` on read and whenever a release is explicitly refreshed. Do not bulk-rewrite historical rows.

Bandcamp evidence remains authoritative only when the release URL is confirmed and release-specific.

### 3.4 Last.fm quarantine

Under the stabilized policy:

- Last.fm remains available as weak corroboration.
- Last.fm cannot create a canonical term by itself.
- Last.fm cannot produce an accepted signature by itself.
- Last.fm cannot make a term auto-apply eligible.
- Last.fm cannot support pruning.
- Last.fm cannot override official, Bandcamp, MusicBrainz, Discogs, or local evidence.
- Known canonical Last.fm terms may add a small corroboration signal when another eligible source supports the term.

Existing signatures are not bulk-rewritten. New or explicitly refreshed releases use the stabilized policy.

### 3.5 Policy versioning and forward migration

Add an enrichment policy version to signatures and relevant stored outputs.

Add nullable `enrichment_policy_version TEXT` columns to `enriched_genre_signatures` and `enriched_genres`. Treat `NULL` as `legacy-v0`. Stamp new and explicitly refreshed rows with the current stabilized policy version.

Required semantics:

- Existing untouched signatures remain readable and are treated as legacy-policy rows.
- New releases use the current stabilized policy version.
- Explicit refresh of an existing release rebuilds it under the current policy.
- No automatic bulk migration runs.
- Reports distinguish legacy-policy and current-policy signatures.

### 3.6 Strict API-free dry runs

All enrichment and model-prior `--dry-run` paths must avoid external API and network calls.

In particular:

- `extract-lastfm --dry-run` does not call Last.fm.
- `extract-bandcamp --dry-run` does not call OpenAI and does not fetch Bandcamp HTML.
- model-prior dry runs do not call OpenAI.

Dry runs do not initialize or mutate persistent sidecar state. They emit routing, payload shape, estimated prompt size, estimated output size, and estimated cost where available.

### 3.7 Dense sidecar identity validation

`src/features/artifacts.py` currently validates dense-sidecar `track_ids`, but not vocabulary identity.

Before loading a dense sidecar, validate at least:

- track ID sequence;
- genre vocabulary sequence;
- sparse artifact identity or an equivalent deterministic genre-artifact hash;
- dense embedding configuration schema version.

Reject stale or mismatched dense sidecars with an actionable warning. Do not silently load them.

---

## 4. Album-Level Model-Prior Lane

### 4.1 Purpose

Add a cheap, no-web model-prior pass that generates compact album-level genre/style hypotheses for sparse, generic, or under-tagged releases.

The prior is:

```text
a taxonomy-shaped provisional classifier signal
```

It is not:

```text
authoritative source evidence
```

### 4.2 Provider and model strategy

Reuse the existing OpenAI Responses API integration.

- Provider: OpenAI for the first milestone.
- Default model: `gpt-4o-mini`.
- CLI override: supported.
- Model benchmarking: evaluate alternatives on a fixed album sample before changing the default.
- GUI integration: deferred until shadow evaluation passes.

Do not add Anthropic album-prior support in the first implementation. The separate offline vocabulary tooling may continue to support Anthropic.

### 4.3 Input

The model-prior request receives:

- canonical artist;
- canonical album;
- release identifiers when available;
- year when available;
- compact track-title sample;
- deterministic baseline genres grouped by source;
- normalized known tags after deterministic cleanup;
- prompt, taxonomy, schema, and policy versions.

The request does not use web search.

### 4.4 Structured response

Use strict structured output equivalent to:

```json
{
  "genres": [
    {
      "term": "ambient americana",
      "confidence": 0.82,
      "specificity": "subgenre",
      "taxonomy_role": "core_style",
      "notes": "Short taxonomic rationale without source claims."
    }
  ],
  "warnings": []
}
```

Validation rules:

- terms are normalized to lowercase before taxonomy mapping;
- confidence must be in `[0, 1]`;
- specificity is one of `broad`, `genre`, `subgenre`, `microgenre`;
- taxonomy role is one of `parent`, `core_style`, `secondary_style`, `edge_case`;
- notes must not claim source authority;
- invalid JSON or schema violations are recorded safely and do not produce terms.

### 4.5 Dedicated storage

Add:

```sql
ai_genre_model_priors (
  prior_id INTEGER PRIMARY KEY,
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
```

```sql
ai_genre_model_prior_terms (
  prior_term_id INTEGER PRIMARY KEY,
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
  FOREIGN KEY (prior_id) REFERENCES ai_genre_model_priors(prior_id)
    ON DELETE CASCADE
);
```

Required indexes:

- prior cache identity;
- release key;
- normalized term;
- mapping status;
- provider and model.

### 4.6 Taxonomy mapping

Map model-prior terms through the existing curated taxonomy:

- known canonical term -> `mapped`;
- known alias -> resolve to canonical slug and mark `mapped`;
- known descriptor, instrument, place, format, mood/function, or rejected term -> classify and reject for shadow scoring;
- unknown term -> `unmapped`;
- conditionally useful term -> `conditional`, excluded from shadow until reviewed.

Only mapped usable genre/style terms may set `accepted_for_shadow=1`.

### 4.7 Provenance and safety rules

- Model-prior-only terms are never authoritative.
- Model-prior-only terms are never auto-apply eligible.
- Model-prior-only terms never enter normal `enriched_genre_signatures`.
- Unmapped or conditional terms never enter artifacts.
- Mapped model-prior terms may enter `hybrid_shadow` signatures as provisional terms.
- Shadow signatures retain basis and confidence so reports can separate accepted source terms from provisional model-prior terms.
- Official, Bandcamp, Discogs, MusicBrainz, and local evidence outrank the model prior.
- Last.fm remains weaker than the model prior for cleanliness, but the model prior still does not become authority.

### 4.8 CLI

Add:

```bash
python scripts/ai_genre_enrich.py model-prior-one \
  --artist "The Beets" \
  --album "Let The Poison Out" \
  --dry-run

python scripts/ai_genre_enrich.py model-prior \
  --limit 100 \
  --missing-only

python scripts/ai_genre_enrich.py model-prior-report
```

Support:

- `--dry-run`;
- `--limit`;
- `--missing-only`;
- `--artist`;
- `--album`;
- `--model`;
- `--force`;
- prompt-size and cost estimates.

The first release is CLI-only.

---

## 5. Shadow Artifact Flow

### 5.1 Inputs

`hybrid_shadow` combines:

1. accepted enriched source signatures under the selected policy;
2. mapped model-prior terms with `accepted_for_shadow=1`;
3. legacy metadata fallback for releases without a shadow signature.

Production `enriched` mode excludes model-prior-only terms.

Legacy-policy signatures remain eligible for explicit `enriched` and `hybrid_shadow` builds during gradual migration. Reports separate `legacy-v0` and current-policy signature counts so evaluation does not imply that untouched historical rows already satisfy the stabilized Last.fm rules.

### 5.2 Output

Build:

- a separate sparse shadow artifact;
- a separate 64-dimensional dense PMI-SVD sidecar;
- a JSON or Markdown comparison report.

Do not change the active production artifact path.

### 5.3 Comparison report

Report:

- policy version;
- input artifact and sidecar identities;
- output shadow paths;
- legacy, enriched, and shadow signature counts;
- track coverage delta;
- accepted Bandcamp contribution;
- Last.fm-only terms suppressed;
- provisional model-prior terms included;
- unmapped prior terms;
- top changed genres;
- releases with the largest signature changes;
- candidate-pool deltas for fixed benchmark seeds;
- weakest-edge and transition-quality deltas;
- distinct-artist deltas;
- stale-sidecar validation status.

Use fixed benchmark seeds so changes remain comparable between runs.

---

## 6. Migration Policy

### 6.1 Existing sidecar rows

Do not bulk-rebuild current signatures.

Current rows remain usable for explicit `enriched` builds and are reported as legacy policy when they lack an explicit policy version.

### 6.2 New and refreshed rows

Apply the stabilized policy to:

- newly enriched releases;
- manually refreshed releases;
- explicit targeted rebuild commands.

This provides gradual migration without rewriting the existing sidecar.

### 6.3 Metadata database safety

This design does not write, migrate, or alter `data/metadata.db`.

The existing uncommitted `scripts/normalize_genre_vocab.py --apply` pathway proposes writes to `genre_canonical_token`. That work is outside this design and must not be run against `data/metadata.db` without the repository-required explicit instruction, second confirmation, and backup.

---

## 7. Reporting

Extend reporting to show:

- signature count by enrichment policy version;
- signature count by evidence basis;
- Bandcamp pages, extracted tags, and accepted contributions;
- Last.fm pages, accepted corroborations, and quarantined Last.fm-only terms;
- model-prior cache hits and misses;
- model-prior failures and invalid responses;
- token usage and estimated cost;
- top mapped prior terms;
- top unmapped prior terms;
- terms with the largest potential review payoff;
- prior terms accepted for shadow;
- releases routed to manual review;
- shadow artifact comparison metrics.

---

## 8. Error Handling

- API failures record a failed prior row and continue processing other releases.
- Invalid model JSON records a schema-validation failure and contributes no terms.
- Missing API keys return actionable errors for live commands and are irrelevant for dry runs.
- Missing sidecar DB falls back safely to legacy artifact behavior.
- Stale dense sidecar identity causes rejection with an actionable rebuild message.
- Shadow build failures leave the active artifact untouched.
- Missing taxonomy mapping causes `unmapped`, not implicit acceptance.
- Existing legacy-policy signatures remain readable.

---

## 9. Test Requirements

### 9.1 Stabilization

Add tests for:

- canonical Bandcamp source typing;
- compatibility with existing `bandcamp_tags` rows;
- Bandcamp genre inclusion after refresh;
- Last.fm quarantine under the stabilized policy;
- Last.fm-only rows cannot create a current-policy signature;
- Last.fm may corroborate an already supported canonical term;
- legacy-policy signature compatibility;
- policy upgrade on explicit refresh;
- API-free `extract-lastfm --dry-run`;
- API-free `extract-bandcamp --dry-run`;
- `legacy` as the artifact-build default;
- explicit `enriched` behavior;
- isolated `hybrid_shadow` paths;
- dense-sidecar rejection when genre vocab differs;
- dense-sidecar rejection when sparse artifact identity differs.

### 9.2 Model prior

Add tests for:

- schema migration idempotency;
- model-prior cache identity;
- model-prior dry run performs no API calls and writes no prior rows;
- valid mocked Responses API output;
- invalid JSON handling;
- invalid schema handling;
- lowercase normalization;
- alias mapping;
- descriptor rejection;
- unmapped-term reporting;
- `auto_apply_eligible=0` for every model-prior term;
- mapped terms accepted for shadow;
- unmapped terms excluded from shadow;
- model-only terms excluded from normal enriched signatures;
- CLI `model-prior-one`;
- CLI `model-prior --limit --missing-only`;
- CLI `model-prior-report`.

### 9.3 Shadow reports

Add tests for:

- shadow artifact output isolation;
- comparison report shape;
- accepted Bandcamp counts;
- quarantined Last.fm-only counts;
- provisional prior counts;
- fixed benchmark candidate-pool deltas;
- dense-sidecar identity status.

### 9.4 Existing baseline

Before implementation, preserve the measured baseline:

- enrichment, taxonomy, override, GUI-worker, PMI-SVD, and artifact-builder units: `213 passed`;
- dense live integration: `23 passed`, `3 failed`.

The three existing dense integration failures must not be attributed to model-prior changes:

- Beach Boys pool size is `148`, below the expected minimum `200`;
- Charli XCX dense p50 similarity is below the existing expectation;
- Charli XCX dense p90 similarity is below the existing sparse-baseline expectation.

Implementation work should either address these in a dedicated stabilization task or keep them clearly recorded as pre-existing failures.

---

## 10. Documentation

Update:

- `docs/AI_GENRE_ENRICHMENT.md`
- `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`
- `config.example.yaml`

Document:

- artifact source modes;
- legacy default behavior;
- shadow artifact isolation;
- stabilized Bandcamp handling;
- Last.fm quarantine;
- enrichment policy versioning;
- forward-only migration;
- album-level model prior;
- distinction between album model prior and vocabulary-embedding LLM prior;
- model-prior provenance limitations;
- CLI commands;
- API-free dry-run semantics;
- shadow comparison reports;
- dense-sidecar identity validation.

---

## 11. Implementation Sequence

### Milestone 0: Stabilization

1. Add policy versioning.
2. Fix Bandcamp source typing with compatibility for existing rows.
3. Quarantine Last.fm for current-policy signatures.
4. Make extraction dry runs API-free.
5. Add artifact source modes with `legacy` default.
6. Add isolated `hybrid_shadow` sparse artifact builds.
7. Add dense-sidecar identity validation.
8. Add shadow dense-sidecar build support and basic comparison reporting.
9. Update stabilization docs and tests.

### Milestone 1: Model-Prior Prototype

1. Add model-prior schema and indexes.
2. Add structured prompt and validator.
3. Reuse the configurable OpenAI Responses client with `gpt-4o-mini` default.
4. Add cache support.
5. Add `model-prior-one`.
6. Add API-free dry-run estimates.
7. Add `model-prior-report`.
8. Add tests and docs.

### Milestone 2: Batch And Shadow Integration

1. Add `model-prior --limit --missing-only`.
2. Map prior terms through the existing taxonomy.
3. Include eligible provisional terms in `hybrid_shadow` signatures only.
4. Add fixed-seed candidate-pool and transition-quality comparison reports.
5. Evaluate benchmark results before production adoption.

### Deferred

- GUI model-prior controls;
- GUI model-prior reporting;
- automatic background batches;
- production use of model-prior-only terms;
- full generalized taxonomy relationships graph;
- broad scoped-review schema expansion;
- automatic bulk migration of existing signatures;
- Anthropic support for album-level model priors.

---

## 12. Acceptance Criteria

The design is implemented successfully when:

1. Normal artifact builds default to `legacy` and remain backward-compatible.
2. Explicit enriched builds consume accepted sidecar signatures but never model-prior-only terms.
3. Shadow builds write isolated sparse and dense artifacts without replacing active artifacts.
4. Confirmed Bandcamp release tags contribute correctly after refresh.
5. New-policy Last.fm-only tags cannot create accepted signatures.
6. Existing untouched signatures remain readable and distinguishable as legacy policy.
7. Refreshed signatures adopt the stabilized policy.
8. Dense-sidecar identity mismatches are rejected safely.
9. Album model priors are stored with strict provenance and cache identity.
10. Model-prior dry runs are API-free.
11. Mapped provisional model-prior terms can participate in shadow builds only.
12. Reports provide enough evidence to decide whether later production adoption is justified.
