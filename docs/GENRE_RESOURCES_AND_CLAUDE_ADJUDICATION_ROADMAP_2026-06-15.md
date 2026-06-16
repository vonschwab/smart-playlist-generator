# Genre resources audit & Claude-adjudication roadmap

**Date:** 2026-06-15
**Status:** Decision-support / roadmap. No pipeline changes proposed for *this* round — each roadmap phase below becomes its own spec → plan → implementation cycle.
**Decision taken:** Move from the current scrape → classify → fuse → `gpt-4o-mini` adjudicate pipeline to **Claude (on the Max subscription) as the primary genre adjudicator**, with the scraped tags and file tags as *evidence Claude reasons over* rather than inputs to a mechanical fusion. Sequencing: **eval-gated pilot, then drain** (Approach A).

---

## 1. Executive summary

The genre subsystem is not short on *data* — it has eight distinct genre layers across two databases, a 762-record curated taxonomy graph, and 50k+ release-level assignments. It is short on a **trustworthy adjudication policy** that escalates only genuinely hard cases to the human.

Two numbers frame the whole problem:

- **`ai_genre_review_queue`: 34,476 rows, 34,273 still `pending`.**
- **`ai_genre_review_decisions`: 93 decisions ever made.**

That backlog is the "boatload of manual review." But its *composition* (§4) shows the overwhelming majority is **policy- or coverage-driven, not per-item taste judgment** — e.g. 12,398 rows are simply "term not in the taxonomy." Those don't need a human; they need a better adjudicator and a coverage loop.

The headline finding for sizing the work: **"Claude as primary adjudicator" is ~70% already built and dormant** (§5). `claude_client.py` is a production-grade, Max-backed, batch-cached Agent-SDK client; `model_prior.py` is a complete "LLM states its own genre belief" contract with the exact anti-noise guards we keep getting bitten by. They were built, wired to `ai_genre_model_priors`, and never run (that table has 0 rows). The roadmap is therefore *evolve + validate + flip*, not *build from scratch*.

**Metric this effort moves:** review-queue `pending` from ~34k down to a small genuinely-ambiguous residue, **and** published-genre correctness up (fewer bandcamp/Last.fm-collision errors), validated by a blind eval gate before anything writes to the authority.

---

## 2. Resource inventory ("real stock")

Library scale: **40,427 tracks · 3,428 albums · 2,010 artists.**

### 2.1 `data/metadata.db` (794 MB) — the durable store

| Layer (table) | Rows | Role | Authority status |
|---|---|---|---|
| `track_genres` | 27,590 | Raw file/MusicBrainz tags (track level) | **Input** |
| `album_genres` | 20,094 | Raw album tags incl. Discogs | **Input** |
| `artist_genres` | 5,968 | Raw artist-level tags | **Input** |
| `track_effective_genres` | 256,178 | Legacy track-level resolved tags | Input (legacy) |
| `genre_raw_map` / `genre_canonical_token` | 857 / 696 | Raw→canonical mapping tables | Internal |
| `genre_graph_canonical_genres` | 455 | Taxonomy genres mirrored into DB | Structure |
| `genre_graph_canonical_facets` | 52 | Taxonomy facets | Structure |
| `genre_graph_edges` | 1,965 | Taxonomy edges | Structure |
| `genre_graph_aliases` | 218 | Alias → canonical | Structure |
| `genre_graph_release_genre_assignments` | 50,678 | Graph-resolved per-release assignments (layer, confidence, `rejected_by_user`) | Pre-publish working |
| `genre_graph_release_facet_assignments` | 1,410 | Per-release facet assignments | Pre-publish working |
| **`release_effective_genres`** | **43,815** | ✅ **THE published authority** (graph-resolved + user overrides) | **Authoritative** |

Read the authority only via `src/genre/authority.py`. Everything above it in the table is input or working data (per the `genre-data-authority` skill).

### 2.2 `data/ai_genre_enrichment.db` (81 MB) — collection + adjudication working data

| Layer (table) | Rows | Role |
|---|---|---|
| `ai_genre_source_pages` | 5,926 | Located evidence pages (with `identity_status`/confidence) |
| `ai_genre_source_tags` | 49,749 | Tags scraped off those pages |
| `ai_genre_tag_classifications` | 49,749 | Each tag classified (genre/descriptor/place/…); `deterministic` vs `cached_ai` |
| `ai_tag_adjudication_cache` | 2,991 | AI verdicts on unknown tags, cached by normalized tag |
| `enriched_genres` | 12,229 | Fused per-release genre evidence (pre-publish) |
| `ai_genre_release_checks` | 4,121 | **LLM adjudications — all `gpt-4o-mini`, web off** |
| `ai_genre_suggestions` | 22,863 | LLM genre/prune suggestions from those checks |
| **`ai_genre_review_queue`** | **34,476** | ⚠️ human-review backlog (34,273 pending) |
| `ai_genre_review_decisions` | **93** | human decisions ever recorded |
| `ai_genre_user_overrides` | 2,009 | add/remove overrides per release |
| `ai_genre_model_priors` / `_prior_terms` | **0 / 0** | dormant "LLM prior belief" feature (see §5) |
| `enriched_genre_signatures` | 1,784 | **Deprecated** bandcamp-era signatures (stale; never read by new consumers) |

### 2.3 Taxonomy graph — `data/layered_genre_taxonomy.yaml`

Version **`0.12.1-group1-pass9-edge-upgrade`**, **762 records**:

| kind | count | | kind | count |
|---|---|---|---|---|
| family | 16 | | facet | 52 |
| umbrella | 26 | | alias | 235 |
| genre | 152 | | reject | 20 |
| subgenre | 258 | | **canonical genres total** | **455** |
| microgenre | 3 | | status (all 762): active 443 · review 64 · alias_only 235 · rejected 20 | |

Grown via the SP3a loop (`taxonomy-growth` skill). The git-tracked YAML is the *structure* source of truth; it's mirrored into both DBs' `genre_graph_*` tables.

### 2.4 The sonic side (for completeness — not genre, but co-located)

Genre vectors are baked into `data/artifacts/beat3tower_32k/data_matrices_step1.npz` as `X_genre_*` at artifact-build time, sourced from the authority (`genre_source: graph`). Generation reads the artifact, never the live DB.

---

## 3. Pipeline map — how a genre flows today

Orchestrated by `scripts/analyze_library.py`, stages: **`scan → genres → discogs → lastfm → enrich → publish → artifacts`**.

```
scan        tracks/albums/artists discovered
  │
genres      file + MusicBrainz tags ──► track_genres / album_genres / artist_genres   [INPUT]
discogs     Discogs genres ───────────► album_genres
lastfm      Last.fm tags ─────────────► (lastfm evidence)
  │
enrich  (src/ai_genre_enrichment/)  ── the heavy stage ──────────────────────────────
  │   source_locator      → ai_genre_source_pages   (identity-checked evidence pages)
  │   source_extraction   → ai_genre_source_tags    (raw scraped tags)
  │   tag_classification  → ai_genre_tag_classifications   (genre vs noise; deterministic + cached_ai)
  │   tag_adjudicator     → ai_tag_adjudication_cache      (AI verdict on unknown tags)
  │   hybrid_evidence     → enriched_genres          (fuse file tags + scraped evidence)
  │   client (OpenAI)     → ai_genre_release_checks   (gpt-4o-mini release adjudication, WEB OFF)
  │   review_queue        → ai_genre_review_queue     (anything not auto-acceptable)
  │   model_prior         → ai_genre_model_priors      (DORMANT — never run, shadow-only)
  │
publish (src/genre/genre_publish.py)  ── ONLY writer of the authority ──
  │   maps graph/enriched assignments → release_effective_genres   [AUTHORITY]
  │   (timestamped metadata.db backup on first run)
  │
artifacts (scripts/build_beat3tower_artifacts.py)
      bakes authority → X_genre_* in the npz   (genre_source: graph)
        │
   generation / GUI / export  read artifact + authority (authority.py)
```

**Where quality leaks (each a documented incident):**

- **`source_extraction` / `source_locator`** — name-collision identity failures: Last.fm tags fetched by artist-name string assigned "Green-House" (ambient) a Ukrainian hip-hop act's tags; `identity_status` stamped `confirmed` unconditionally (~76 generic-name artists affected).
- **`hybrid_evidence` (fusion)** — trusted a stranger over the user: a Bandcamp *label storefront* tagged a hardcore record "indie rock/pop" at 0.95 and the fusion **replaced** the user's correct file tags (which were shunted to the unapplied queue). Same page double-counted.
- **`client` adjudication** — runs on `gpt-4o-mini` with **web grounding off**; ungrounded, weak, and the source of most `needs_review` checks.
- **`publish` / artifact build** — over-inclusion: inferred hub-families baked into `X_genre_*` drove random-pair cosine p50 ≈ 0.42 (genre signal near-useless) until inferred layers were excluded from the vectors.

The pattern: most leaks are **collection-identity** and **fusion-trust** problems that a strong reasoning adjudicator with the right contract is well-positioned to catch — *if* it's given file tags as ground truth and disambiguating context.

---

## 4. Quality diagnosis — why noise + a 34k backlog

**Queue composition (the key evidence):**

By `basis`: `hybrid_fusion` 11,880 · `hybrid_provisional` 11,812 · `layered_taxonomy` 10,784.

By `reason` (top):

| reason | rows | nature |
|---|---|---|
| Unknown layered taxonomy term | **12,398** | **coverage gap** — term isn't in the 455-genre graph |
| High-confidence Last.fm signal usable provisionally w/ release evidence | 10,836 | **policy threshold** |
| Evidence mapped but not strong enough for auto-accept | 5,720 | **policy threshold** |
| Last.fm-only mapped signal needs review unless corroborated | 4,834 | **policy threshold** |
| Known taxonomy term marked for review | 592 | review flag |
| retail_bucket / source_noise / label / joke_tag | ~76 | noise |

**Reading:** ~33.8k of 34.3k pending fall into *coverage* (one taxonomy gap) or *corroboration-policy* buckets. These are not 34k independent taste decisions — they are a handful of **policy rules and one coverage gap**, multiplied across the library. That is exactly the shape that a confident, grounded adjudicator + a taxonomy-growth feedback loop collapses.

**Four root findings:**

1. **The backlog is real but mostly mechanical.** Coverage + policy, not judgment.
2. **The live adjudicator is ungrounded `gpt-4o-mini`.** All 4,121 `release_checks` used it, web off; 1,312 came back `needs_review`. The genres your library runs on today were inferred by the weakest available model with no grounding.
3. **The "model priors" feature — an independent LLM genre opinion — is built but never run.** 0 rows. It already encodes the anti-noise guards we keep rediscovering.
4. **Identity is the upstream root of the worst errors.** Green-House and the bandcamp-label incident are both *who is this / who said this* failures, not *what genre is this* failures. The adjudicator contract must treat identity as a first-class, escapable ("uncertain → escalate") decision.

---

## 5. What already exists for Claude (the "70% built" finding)

### 5.1 `src/ai_genre_enrichment/claude_client.py` — `ClaudeCodeEnrichmentClient`

- Runs on the **Claude Max subscription via the Agent SDK** — no API billing, no key to rotate.
- **Provider-neutral**: mirrors `OpenAIEnrichmentClient`'s surface (`call_structured`, `enrich`, `request_structured`, `call_structured_batch`), so existing call sites can switch providers without rework. This is what makes a clean A/B between `gpt-4o-mini` and Claude possible.
- **Batch path is already cost-engineered**: `call_structured_batch` chunks items and runs *all chunks in one SDK session* (`ClaudeSDKClient`), so the shared prompt prefix stays cached across turns. Per-item validation with single-call fallback.
- **Fails loudly** if the SDK/auth is missing (project discipline: a knob that can't act is an error). Model is a constructor arg (`haiku` default → `sonnet`/`opus` per stage).

### 5.2 `src/ai_genre_enrichment/model_prior.py` — the dormant adjudicator scaffold

This is, almost verbatim, "Claude states its own genre belief about a release":

- Contract `album-model-prior-v1`: output schema is `genres[{term, confidence, specificity, taxonomy_role, notes}] + warnings`.
- **Evidence-richness confidence cap**: 0.30 (blind) → 0.80 (tags + tracks + identifiers). Caps Claude's confidence by how much real evidence it had — directly counters overconfident sparse guesses.
- **Anti-noise guards already written**: rejects responses that claim source authority ("bandcamp says…") and that reason from demographics/aesthetics ("from japan", "name suggests…"). These are the exact failure modes from the 2026-06-12 findings.
- **Taxonomy mapping** (`map_model_prior_terms`): maps the model's terms to canonical vocabulary as `mapped` / `conditional` / `unmapped`, with `accepted_for_shadow` and `auto_apply_eligible: 0` (shadow-only — never auto-applies today).
- Writes to `ai_genre_model_priors` / `_prior_terms` — empty because **it was never wired into a live run.**

### 5.3 The gap to "primary adjudicator"

`model_prior` is a deliberately **no-web, no-evidence blind prior** — designed as a *second* opinion, not the decider. To make Claude the *primary* adjudicator we need a sibling contract — call it `album-adjudicator-v1` / `artist-adjudicator-v1` — that **sees the evidence and adjudicates it**:

- **Inputs:** file tags (**ground truth, never silently dropped**) + scraped evidence (tags with source + identity confidence) + the taxonomy candidate list + (for albums) the artist-level prior + disambiguating context (titles, identifiers, year).
- **Output:** a proposed canonical genre set with per-genre confidence, layer (`observed_leaf` / `inferred_family`), rationale, and an explicit **`escalate` flag** with conflict notes (file-tag conflict, uncertain identity, novel term).
- **Reuses:** the whole `claude_client` batch path, `model_prior`'s confidence-cap + guard logic, and the taxonomy-mapping helper.

So Phase 1 is *evolve `model_prior` into a grounded adjudicator contract*, not greenfield.

---

## 6. Target architecture — Claude as primary adjudicator

```
            ┌─────────────────────────── evidence (read-only) ───────────────────────────┐
            │  file tags (GROUND TRUTH)   scraped tags + source/identity conf             │
            │  taxonomy candidates        artist-level prior   identifiers / titles / year │
            └──────────────────────────────────────────────┬───────────────────────────────┘
                                                            ▼
                                   ┌────────────────────────────────────────┐
   artist pass (2,010)  ──────────►   Claude adjudicator (Max, Agent SDK)    │
   anchors album pass             │   album-adjudicator-v1 / artist-…-v1     │
                                   │   → genre set + confidence + layer       │
                                   │   → escalate flag + conflict notes       │
                                   └───────────────┬──────────────┬──────────┘
                                                   │              │
                              auto-apply (safe)    │              │  escalate (true conflict /
                              confident + corrob.  │              │  uncertain identity / low conf)
                              + no file-tag conflict▼              ▼
                                   publish stage (genre_publish)   small review queue (you)
                                   → release_effective_genres            │
                                   (metadata.db backup)                  ▼
                                          │                     decisions feed back as overrides
                                          ▼
                                   artifacts → generation / GUI
                              ┌──── unmapped terms ────► SP3a taxonomy growth (Claude-assisted) ────┐
                              └──────────────────────────────────────────────────────────────────────┘
```

**Principles baked into the contract:**

- **File tags are ground truth.** The adjudicator may *demote* or *contextualize* a file tag but never silently drops one; dropping requires an `escalate`. (Carries `hybrid_evidence`'s never-drop-local rule forward.)
- **Identity is escapable.** Given disambiguating context, Claude may return "uncertain identity → escalate" rather than guess. This is the structural fix for the Green-House / bandcamp-label class.
- **Two granularities.** An **artist pass** (2,010 — cheap, Claude's strongest footing) builds a per-artist genre profile that anchors the **album pass** (3,428). Open question §9: whether the artist profile is also published as a display layer or used only as a prior.
- **No web in the pilot.** Evidence is already collected; reintroducing live web search reintroduces the stranger-trust + latency problems. Revisit only if the eval shows real evidence gaps.
- **Write-back is publish-only, backed up.** The adjudicator never touches `metadata.db` directly; it produces assignments that the `publish` stage writes, with the existing timestamped backup discipline.

---

## 7. Roadmap (Approach A: eval-gated pilot, then drain)

Each phase is its own spec → plan → implementation cycle. Phases 1–3 touch **no authoritative data** (shadow only); the authority is written only at Phase 4, behind the Phase-2 eval gate.

### Phase 0 — Baseline & eval corpus
- **Do:** this document; plus a curated eval corpus of ~40–60 releases: (a) known failure cases (bandcamp-label, Green-House + a sample of the ~76 generic-name artists, an inferred-hub-saturated release), (b) correctly-tagged controls, (c) sparse-evidence releases. Hand-label gold genres (file tags + manual).
- **Exit:** corpus committed; metrics defined — correctness precision/recall vs gold, noise rate, **file-tag-preservation rate**, escalation rate.
- **Effort:** S.

### Phase 1 — Grounded adjudicator contract
- **Do:** build `album-adjudicator-v1` + `artist-adjudicator-v1` (evolve `model_prior`): evidence-grounded inputs, escalate flag, identity-uncertainty path. Run **shadow on the Phase-0 corpus only**, model = `sonnet`.
- **Exit:** valid structured output across the corpus; SDK/auth clean; tokens + throughput measured.
- **Effort:** M. Touches `model_prior.py` (new sibling), `claude_client.py` (reuse), `prompt.py`, `models.py`.

### Phase 2 — Eval gate (blind A/B)
- **Do:** via the `evaluation-methodology` harness, blind-score Claude's proposed set vs current `release_effective_genres` vs gold, on the corpus.
- **GATE:** Claude must beat the current pipeline on correctness **and** not drop correct file tags. Fail → iterate the contract; **do not proceed to any write-back.** Record the decision.
- **Effort:** S–M.

### Phase 3 — Trust policy + shadow-at-scale
- **Do:** derive auto-apply vs escalate thresholds from Phase-2 data (auto-apply ⇐ Claude-confident **and** corroborated by ≥1 independent source **and** no file-tag conflict; else escalate). Run the adjudicator across the **full library in shadow** — write to `ai_genre_model_priors` / a new adjudication table, **not** the authority. Measure projected queue collapse, auto-apply rate, escalation volume; spot-check that escalations are genuinely ambiguous.
- **Exit:** shadow numbers show a small residue and clean escalations.
- **Effort:** M.

### Phase 4 — Flip to live: drain + publish
- **Do:** wire adjudicator output into `genre_publish` (backup discipline intact). Auto-apply the safe set → `release_effective_genres`; route conflicts to the now-small review queue. Re-publish, rebuild the artifact, re-run generation eval to confirm no regression (genre-vector / T sanity).
- **Exit:** queue at small residue; authority correctness up; generation unaffected or improved.
- **Effort:** M–L. Highest-risk phase (touches the authority) — gated by Phases 2–3.

### Phase 5 — Taxonomy coverage feedback loop
- **Do:** the unmapped terms Claude surfaces (the 12,398 "unknown term" driver) batch into SP3a taxonomy-growth handoffs (Claude-assisted placement), grow the graph, re-map, shrink next cycle's escalations. Ongoing.
- **Effort:** ongoing; per the `taxonomy-growth` skill.

---

## 8. Risks & guardrails

- **Never trust a stranger over the user.** File tags are ground truth; dropping one requires an explicit escalation, never a silent overwrite. (The single worst past incident.)
- **Eval before trust.** Phase 2 is a hard gate. Nothing writes to the authority before a documented win.
- **Shadow before live.** Phases 1–3 write only to non-authoritative tables.
- **Data safety.** `publish` is the only authority writer; `metadata.db` is irreplaceable and backed up on first publish; the adjudicator never writes the DB directly. The sidecar/working DB is rebuildable.
- **Identity grounding.** Give the adjudicator disambiguating context and an explicit "uncertain identity → escalate" path; do not let it guess identity.
- **Cost / throughput.** Max is non-metered but rate-limited. Use the batch prefix-cache path; pilot on `sonnet`, decide `haiku`-for-bulk from Phase-2 quality/token data. ~2,010 artist + ~3,428 album calls is tractable.
- **Reversibility.** Keep the `gpt-4o-mini` path until Claude proves out; the provider-neutral surface already supports running both for A/B and rollback.

---

## 9. Open questions (resolve before Phase 1 kickoff)

1. **Artist genres as a published layer, or prior-only?** Lean: artist profile *informs* album adjudication; optionally surface artist genres as a separate display facet, but the album remains the authority unit.
2. **Bulk model tier:** `sonnet` for the pilot; `haiku` for the full-library bulk *iff* Phase-2 quality holds. Decide from data.
3. **Retire or keep `gpt-4o-mini`?** Keep as an optional cross-check during transition, or cut once Claude clears the gate? Lean: keep through Phase 4, cut after.
4. **Web grounding:** stay no-web (rely on already-collected evidence) for the pilot; revisit only if the eval shows systematic evidence gaps.
5. **Escalation surface:** does the shrunken review queue stay in the existing Genre Review GUI, or get a purpose-built "conflicts only" view? (Defer to Phase 4.)

---

## Appendix — provenance of every number

All counts pulled 2026-06-15 from `data/metadata.db`, `data/ai_genre_enrichment.db`, and `data/layered_genre_taxonomy.yaml` via direct `SELECT COUNT(*)` / `GROUP BY` and a YAML record count. Code references: `scripts/analyze_library.py` (stage registry), `src/genre/genre_publish.py` (authority writer), `src/genre/authority.py` (authority reader), `src/ai_genre_enrichment/claude_client.py`, `src/ai_genre_enrichment/model_prior.py`.
