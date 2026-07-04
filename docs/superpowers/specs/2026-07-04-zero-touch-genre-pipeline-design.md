# Zero-touch genre pipeline — design

**Date:** 2026-07-04
**Status:** Approved design (brainstorm complete; implementation plan to follow)
**Goal:** Make genre collection automatic enough to ship in a downloadable product. An end user points the app at their library and gets confidence-weighted canonical genres for every release with **zero keys, zero review, zero queues** — while Dylan's curation work (taxonomy, tag mappings) ships as a compounding data asset.

Supersedes nothing; **folds in** `docs/GENRE_RESOURCES_AND_CLAUDE_ADJUDICATION_ROADMAP_2026-06-15.md` (its Phases 3–4 become Milestone 3 here). The genre-data-authority layer rules are unchanged.

---

## Decisions taken (with Dylan, 2026-07-04)

1. **Offline core + optional LLM.** The zero-key baseline must produce good genres (file tags + MusicBrainz + shipped taxonomy/mapping data, deterministic). A BYO Anthropic/OpenAI key unlocks the LLM adjudicator as a quality tier — never a requirement.
2. **Zero-touch + optional curation.** The pipeline always auto-publishes its best answer for every release. The Genre Review GUI survives as an opt-in curation surface; nothing ever waits on it.
3. **One pipeline, dogfood first.** The current pipeline evolves in place; Dylan's library is user #0 and the acceptance test. No separate "product mode" fork.

### Rationale in one paragraph

The 34k review queue is not 34k judgments: ~12.4k rows are one coverage gap ("term not in taxonomy") and ~21k are three corroboration-policy rules, multiplied across the library (2026-06-15 roadmap §4). The pipeline was designed to defer uncertainty to a curator; the product inverts that posture. **Every uncertainty becomes a confidence weight, not a queue row.** Genre is already a soft axis in generation (never-fail principle) — a slightly-wrong 0.6-confidence genre degrades gracefully; an unpublished release is invisible to steering, which is worse.

---

## Architecture: three tiers, one pipeline

- **Tier 0 — build-time (Dylan, ships as data):** the curated taxonomy (`data/layered_genre_taxonomy.yaml`, 455 canonical genres) plus a new **tag-mapping pack**: a generated, versioned lookup mapping the wild-tag long tail (Discogs styles, MusicBrainz genres, Last.fm top tags, sidecar unknowns) onto the canonical vocabulary.
- **Tier 1 — runtime, zero-key (the floor):** file tags + embedded MBIDs → deterministic canonicalization (taxonomy → pack) → keyless MusicBrainz enrichment in the background → confidence-weighted fusion → **always publish**.
- **Tier 2 — runtime, optional LLM:** with an API key, the grounded adjudicator (`album_adjudicator.py` contract) runs on thin/conflicted releases only; `model_prior` is primary for evidence-less releases, confidence-capped by evidence richness.

Unchanged invariants: `publish` (src/genre/genre_publish.py) remains the **only** writer of `release_effective_genres`, with backup discipline; consumers read via `src/genre/authority.py`; file tags are ground truth and are never silently dropped.

---

## Milestone 1 — the policy flip (always-publish)

Replace queue-routing with confidence weighting in the fusion → publish path:

| Today (queue row) | New behavior |
|---|---|
| "Evidence mapped but not strong enough for auto-accept" | Publish at its actual confidence |
| Last.fm-only signal | Publish, confidence capped low (~0.4) |
| "High-confidence Last.fm usable provisionally" | Publish at capped confidence |
| Unknown taxonomy term | Resolve via pack (M2); still unknown → drop from genres, log to local unmapped-terms report |
| Broad-term blocklist (rock, pop, …) | Unchanged — excluded from `observed_leaf`; derived by the graph |
| File-tag conflict with external evidence | **User's tags win and publish**; conflict recorded in the audit trail, surfaced in optional curation UI |

- `ai_genre_review_queue` / `adjudication_escalations` become an **audit trail** (curation surface reads them); no blocking semantics anywhere.
- The never-drop-file-tags rule survives structurally: in zero-touch mode a conflict cannot wait for a human, so local metadata outranks strangers by default — also the taste-fidelity-correct answer (Layer 1 #3).

**Acceptance (eval-gated before touching the authority, per `evaluation-methodology`):**
- Blind eval on Dylan's library: always-publish authority vs current authority vs gold corpus (corpus must include the 268-class evidence-less releases and the known failure cases — bandcamp-label, Green-House).
- Artifact genre-vector sanity after re-publish (random-pair cosine percentiles must not regress toward the over-dense failure mode documented 2026-06-16).
- Publish-side changes are additive/surgical — never wholesale re-derivation (2026-06-15 roadmap §10.4).

## Milestone 2 — coverage program (build-time bulk tag mapping)

Bulk-map the wild-tag space onto the taxonomy using the existing `taxonomy_term_adjudicator` contract, Claude-batched:

- **Inputs:** unique unknown terms from Dylan's sidecar (the 12.4k-row driver; see `docs/SIDECAR_UNIQUE_GENRE_TERMS.md`), Discogs style list (closed, ~600), MusicBrainz genre list (~2k), Last.fm top tags (~15k covers the Zipf head).
- **Verdicts:** canonical-match / alias-of / facet(kind) / reject(reason) — the SP3a shape.
- **Human review only on the head:** terms covering ~80% of occurrence mass go through the Taxonomy panel; the tail ships at Claude confidence with provenance recorded.
- **Storage:** the curated YAML stays human-scale. Bulk output lives in a separate **generated pack file** (versioned; per-entry provenance + confidence) consulted by canonicalization as a secondary alias layer *after* the YAML. Pack version folds into the publish/artifact fingerprints (same self-heal discipline as taxonomy versions, 01d7b1c).
- Terms needing a genuinely **new canonical node** surface to the SP3a growth loop — rare, and they stay Dylan's.
- End users never grow anything; post-pack unmapped terms accumulate in a local report the user can voluntarily export. **No telemetry in v1.**

**Acceptance:** re-run canonicalization over Dylan's library — the 12.4k "unknown term" rows resolve mechanically; unmapped-term residue is small and genuinely obscure.

## Milestone 3 — adjudicator tier activation (folds in 2026-06-15 roadmap Phases 3–4)

- **New plain API-key client** (Anthropic HTTP + the existing OpenAI client) alongside the Agent-SDK client, behind the existing provider-neutral surface. Dylan's Max/Agent-SDK path remains as his personal instance of the same seam.
- **Trigger policy:** adjudicator runs only on releases `routing.py` classifies as thin/conflicted (`SKIP_WELL_TAGGED` stays skipped); a cost estimate is shown before any run.
- **`model_prior` becomes primary for evidence-less releases** (the 268-class), confidence-capped by evidence richness as designed. No web grounding.
- Shadow-at-scale on Dylan's library → flip, per the original roadmap's gates. The dogfood is the Phase 3/4 validation.
- **Retire the `gpt-4o-mini` path after the flip** (roadmap open question 3: resolved — cut).

**Acceptance:** shadow numbers show a small genuinely-ambiguous residue; escalations (now audit-trail entries) are spot-checked as genuinely ambiguous; authority correctness up on the eval corpus; generation eval unaffected or improved.

## Milestone 4 — cold start on a stranger's library

First-run flow: scan → file-tag canonicalization → **publish immediately** (instant, offline-capable baseline) → background MusicBrainz fetch (keyless, 1 req/s; re-publishes in waves with progress UI) → artifact build.

- **Identity is MBID-first:** embedded MBIDs from file tags (Picard-tagged libraries) anchor lookups; name-only matches get confidence-capped. This is the structural fix for the Green-House class — identity failures become low confidence, not wrong-at-0.95.
- **Last.fm:** embed a free app key if ToS permits (verify at M4 start); otherwise skip in the zero-key tier (it's the 0.25-weight source).
- **Discogs:** optional per-user token, off by default.
- **Caching by identity:** enrichment results keyed by (release identity, taxonomy+pack version) — re-scans touch only new/changed releases. This also keeps the door open for a v2 hosted shared cache without redesign.
- **No network → file-tags-only publish still works.** Genre never gates generation (Layer 2 #14).

**Acceptance:** end-to-end integration test on a synthetic fresh-library fixture (zero keys, network on/off variants); a real cold-start run on a second machine/library.

---

## Deletions and demotions (discipline #22 — activate fixes, delete legacy)

- Queue-**blocking** semantics: deleted at M1 (tables remain as audit trail).
- `gpt-4o-mini` adjudication path: deleted after M3 flip.
- Web-scraping lane (`source_locator` / Bandcamp extraction): **demoted to off-by-default** — fragile on strangers' machines and the source of the stranger-trust incidents. Stays available behind a knob for Dylan's build-time curation. (Deliberate demote-not-delete: it still earns its keep at build time.)

## Risks

- **Always-publish publishes garbage on junk-tagged libraries** → mitigated by the M1 blind eval gate and confidence weighting; worst case matches today's *unpublished* (invisible) state.
- **MusicBrainz coverage gaps on obscure libraries** → exactly the LLM tier's job; zero-key users get the honest file-tag baseline.
- **Bulk pack quality (tail shipped at Claude confidence)** → provenance + confidence per entry; wrong tail mappings are data fixes in the next pack release, not code changes.
- **Licensing** only bites if v2 ships dump-derived data (knowledge pack); v1 runtime API fetching sidesteps it. Verify MusicBrainz supplementary-data and Discogs dump licenses before any v2 pack work.

## Out of scope (v2 candidates)

- Hosted shared adjudication cache (album adjudicated once globally).
- Shipped release/artist→genre knowledge pack built from open-data dumps (licensing check first).
- Opt-in telemetry for unmapped terms (v1: local report only).
- Local-model adjudication tier.

## Open items to resolve during implementation planning

- Pack file format + loader seam in `layered_taxonomy.py` canonicalization (secondary alias layer).
- Exact confidence caps/weights for the M1 policy table (derive from the eval corpus, not a priori).
- Whether artifact build consumes per-genre confidence directly or keeps layer-based weighting initially (start: preserve existing behavior; calibrate later).
